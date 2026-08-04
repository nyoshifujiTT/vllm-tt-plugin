# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""TT single-process executor that overlaps decode read-back.

Background
----------
The TT decode path returns an ``AsyncModelRunnerOutput`` whose ``get_output()``
performs the device -> host read-back of the decode logits (for Qwen3-ASR that
is a full ``[1, 1, 32, 151936]`` bf16 tensor, ~100 ms). vLLM's async scheduling
is meant to *overlap* that read-back with the next scheduling / device step.

Older vLLM (the fork this plugin migrated from) overlapped it by submitting
``get_output`` to a dedicated ``async_output_thread`` inside
``UniProcExecutor.collective_rpc``. Upstream ``vllm==0.24.0`` dropped that
thread from ``UniProcExecutor`` and now resolves ``AsyncOutputFuture.result()``
by calling ``get_output()`` inline on the engine thread. For the TT backend
that serializes every decode step behind its own read-back, roughly quadrupling
per-token decode latency versus the fork (measured on Qwen3-ASR / p150:
``finalize_decode`` 116 ms vs 17 ms, decode ~6 vs ~24 tok/s/user).

``TTUniProcExecutor`` restores the overlap: it owns a one-worker
``ThreadPoolExecutor`` and, in non-blocking ``collective_rpc``, submits the
``AsyncModelRunnerOutput.get_output`` read-back to that thread instead of
running it inline. The engine thread proceeds to schedule / dispatch the next
step while the previous step's read-back completes in the background, which is
exactly the fork's behaviour and independent of any specific vLLM release.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from vllm.v1.executor.uniproc_executor import UniProcExecutor
from vllm.v1.outputs import AsyncModelRunnerOutput
from vllm.v1.serial_utils import run_method

from vllm_tt_plugin.logger import init_tt_logger

logger = init_tt_logger(__name__)


class TTUniProcExecutor(UniProcExecutor):
    """UniProc executor that offloads decode read-back to a background thread.

    Identical to the upstream single-process executor except that a completed
    ``AsyncModelRunnerOutput`` is finalized on a dedicated worker thread, so the
    TT device -> host decode read-back overlaps the engine loop instead of
    blocking it.
    """

    def _init_executor(self) -> None:
        super()._init_executor()
        # ``async_scheduling`` implies ``max_concurrent_batches > 1`` for the TT
        # backend, i.e. a batch queue is active and read-back overlap is
        # meaningful. Guard on that so a strictly synchronous config keeps the
        # simple inline path. Upstream's ``UniProcExecutor`` does not expose a
        # ``max_concurrent_batches`` attribute (the engine reads it from
        # ``VllmConfig``), so derive the same condition from the scheduler
        # config instead.
        self._tt_async_output_thread: ThreadPoolExecutor | None = None
        if bool(getattr(self.scheduler_config, "async_scheduling", False)):
            self._tt_async_output_thread = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="TTWorkerAsyncOutput"
            )
            logger.info(
                "TTUniProcExecutor: decode read-back overlaps on a background "
                "thread (restores async output overlap dropped from upstream "
                "UniProcExecutor)."
            )

    def shutdown(self) -> None:  # pragma: no cover - lifecycle glue
        thread = getattr(self, "_tt_async_output_thread", None)
        if thread is not None:
            thread.shutdown(wait=False)
            self._tt_async_output_thread = None
        parent_shutdown = getattr(super(), "shutdown", None)
        if callable(parent_shutdown):
            parent_shutdown()

    def collective_rpc(  # type: ignore[override]
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: bool = False,
        single_value: bool = False,
    ) -> Any:
        if kwargs is None:
            kwargs = {}

        thread = getattr(self, "_tt_async_output_thread", None)
        # Only intercept the non-blocking path whose result may be an
        # AsyncModelRunnerOutput; everything else defers to upstream so we do
        # not fork behaviour we do not need to.
        if not non_block or thread is None:
            return super().collective_rpc(
                method,
                timeout=timeout,
                args=args,
                kwargs=kwargs,
                non_block=non_block,
                single_value=single_value,
            )

        try:
            result = run_method(self.driver_worker, method, args, kwargs)
            if isinstance(result, AsyncModelRunnerOutput):
                if single_value:
                    return thread.submit(result.get_output)

                def _get_output_list() -> list[Any]:
                    return [result.get_output()]

                return thread.submit(_get_output_list)
            future: Future[Any] = Future()
            future.set_result(result if single_value else [result])
        except Exception as exc:  # noqa: BLE001 - surfaced via future
            future = Future()
            future.set_exception(exc)
        return future
