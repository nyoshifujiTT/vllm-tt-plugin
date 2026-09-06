# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""Host-only tests for TTUniProcExecutor's decode read-back overlap.

The TT decode path returns an ``AsyncModelRunnerOutput`` whose ``get_output()``
performs the device->host read-back. Upstream ``UniProcExecutor`` finalizes it
inline on the engine thread; ``TTUniProcExecutor`` restores the fork behaviour
of offloading that read-back to a background thread so it overlaps the engine
loop. These tests exercise the routing logic without importing vLLM by stubbing
the base class.
"""

import sys
import types
from concurrent.futures import Future


def _install_fake_vllm(monkeypatch):
    """Provide the minimal vllm modules TTUniProcExecutor imports."""

    class _AsyncModelRunnerOutput:
        pass

    def _run_method(worker, method, args, kwargs):
        # Tests replace this via monkeypatch on the instance path; default no-op.
        raise AssertionError("run_method should be patched per-test")

    class _UniProcExecutor:
        # Base stub: records that inline finalization would occur.
        def __init__(self):
            self.scheduler_config = types.SimpleNamespace(async_scheduling=True)
            self.driver_worker = object()

        def _init_executor(self):
            pass

        def collective_rpc(self, method, timeout=None, args=(), kwargs=None,
                           non_block=False, single_value=False):
            return ("INLINE", method)

    uni_mod = types.ModuleType("vllm.v1.executor.uniproc_executor")
    uni_mod.UniProcExecutor = _UniProcExecutor
    out_mod = types.ModuleType("vllm.v1.outputs")
    out_mod.AsyncModelRunnerOutput = _AsyncModelRunnerOutput
    serial_mod = types.ModuleType("vllm.v1.serial_utils")
    serial_mod.run_method = _run_method

    for name, mod in {
        "vllm.v1.executor.uniproc_executor": uni_mod,
        "vllm.v1.outputs": out_mod,
        "vllm.v1.serial_utils": serial_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return uni_mod, out_mod, serial_mod


def _load_executor(monkeypatch):
    uni_mod, out_mod, serial_mod = _install_fake_vllm(monkeypatch)
    # Provide a lightweight logger to avoid importing the real plugin logger deps.
    log_mod = types.ModuleType("vllm_tt_plugin.logger")
    log_mod.init_tt_logger = lambda _n: types.SimpleNamespace(info=lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "vllm_tt_plugin.logger", log_mod)
    sys.modules.pop("vllm_tt_plugin.executor", None)
    # The module imported below binds the stub UniProcExecutor as its base
    # class. Leaving it in sys.modules would hand that stub-derived class to
    # every later test, so drop it again when this test finishes; monkeypatch
    # restores the stubbed vllm modules at the same point.
    monkeypatch.delitem(sys.modules, "vllm_tt_plugin.executor", raising=False)
    import importlib
    mod = importlib.import_module("vllm_tt_plugin.executor")
    return mod, out_mod, serial_mod


def test_async_output_offloaded_to_background_thread(monkeypatch):
    mod, out_mod, serial_mod = _load_executor(monkeypatch)

    ex = mod.TTUniProcExecutor.__new__(mod.TTUniProcExecutor)
    ex.scheduler_config = types.SimpleNamespace(async_scheduling=True)
    ex.driver_worker = object()
    # emulate _init_executor's thread creation
    mod.TTUniProcExecutor._init_executor(ex)
    assert ex._tt_async_output_thread is not None

    sentinel = out_mod.AsyncModelRunnerOutput()
    sentinel.get_output = lambda: "FINALIZED"
    # The executor binds ``run_method`` at import time, so patch the name on the
    # executor module (not the source module).
    monkeypatch.setattr(mod, "run_method", lambda *a, **k: sentinel)

    fut = ex.collective_rpc("execute_model", non_block=True, single_value=True)
    assert isinstance(fut, Future)
    assert fut.result(timeout=5) == "FINALIZED"


def test_blocking_path_delegates_to_super(monkeypatch):
    mod, _out, serial_mod = _load_executor(monkeypatch)
    ex = mod.TTUniProcExecutor.__new__(mod.TTUniProcExecutor)
    ex.scheduler_config = types.SimpleNamespace(async_scheduling=True)
    ex.driver_worker = object()
    mod.TTUniProcExecutor._init_executor(ex)

    # non_block=False must defer to the (stub) base collective_rpc.
    out = ex.collective_rpc("execute_model", non_block=False, single_value=True)
    assert out == ("INLINE", "execute_model")


def test_no_thread_when_async_scheduling_disabled(monkeypatch):
    mod, out_mod, serial_mod = _load_executor(monkeypatch)
    ex = mod.TTUniProcExecutor.__new__(mod.TTUniProcExecutor)
    ex.scheduler_config = types.SimpleNamespace(async_scheduling=False)
    ex.driver_worker = object()
    mod.TTUniProcExecutor._init_executor(ex)
    assert ex._tt_async_output_thread is None
    # non_block with no thread -> delegate to base (inline).
    out = ex.collective_rpc("execute_model", non_block=True, single_value=True)
    assert out == ("INLINE", "execute_model")


class TestUniprocBackendRecognition:
    """The TT executor must satisfy checks that ask for a uniproc executor.

    check_and_update_config swaps distributed_executor_backend to the TT
    executor early, so any later constraint that spells the uniproc set inline
    as (None, "uni") rejects a value the plugin itself installed. The
    block-output guard did exactly that and failed 20 tests with
    "Block-output models require the uniproc executor; got
    distributed_executor_backend='vllm_tt_plugin.executor.TTUniProcExecutor'".
    """

    def test_vllm_spellings_are_uniproc(self):
        from vllm_tt_plugin.platform import _is_uniproc_executor_backend

        assert _is_uniproc_executor_backend(None)
        assert _is_uniproc_executor_backend("uni")

    def test_the_tt_executor_is_uniproc(self):
        from vllm_tt_plugin.platform import (
            TT_UNIPROC_EXECUTOR_BACKEND,
            _is_uniproc_executor_backend,
        )

        assert _is_uniproc_executor_backend(TT_UNIPROC_EXECUTOR_BACKEND)

    def test_multiprocess_backends_are_not_uniproc(self):
        from vllm_tt_plugin.platform import _is_uniproc_executor_backend

        assert not _is_uniproc_executor_backend("mp")
        assert not _is_uniproc_executor_backend("ray")


def test_the_docstring_does_not_read_the_32_as_concurrency():
    """[1, 1, 32, 151936] invites "so it serves 32 users". It does not.

    Qwen3-ASR serves at max_num_seqs = 4. The 32 is tile_padded_batch_rows --
    TILE_SIZE * ceil(max_batch_size / TILE_SIZE) in tt_transformers' model
    config -- so every width from 1 to 32 pads to the same 32 rows.

    That detail is load-bearing for this file's argument: because the read-back
    costs the same regardless of concurrency, serialising it hurts most at low
    concurrency, where there is no other work to hide it behind. Left
    unexplained, a reader can conclude the overlap only matters for large
    batches and drop it for a 4-way deployment.
    """
    import os

    src = open(
        os.path.join(os.path.dirname(__file__), "..", "src", "vllm_tt_plugin", "executor.py")
    ).read()
    head = src[: src.index('"""', src.index('"""') + 3)]
    flat = " ".join(head.split())

    assert "tile_padded_batch_rows" in flat, (
        "name what the 32 actually is, or it reads as the concurrency"
    )
    assert "max_num_seqs = 4" in flat, "state the real serving width"
    assert "TILE_SIZE * ceil(max_batch_size / TILE_SIZE)" in flat, (
        "give the formula, so the 32 can be re-derived for another model"
    )
    # the consequence, which is why this matters here
    assert "same at conc=1 and conc=4" in flat


def test_the_docstring_quantifies_the_read_back():
    """"~100 ms" is only checkable if the size it moves is stated.

    [1, 1, 32, 151936] bf16 is 9.3 MiB. Without that, the reader cannot tell
    whether 100 ms is plausible or a typo, and cannot scale it to another
    vocabulary size.
    """
    import os

    src = open(
        os.path.join(os.path.dirname(__file__), "..", "src", "vllm_tt_plugin", "executor.py")
    ).read()
    head = src[: src.index('"""', src.index('"""') + 3)]
    flat = " ".join(head.split())

    assert "9.3 MiB" in flat, "give the transfer size behind the ~100 ms"
    # and it must match the shape actually quoted
    assert "[1, 1, 32, 151936]" in flat
    mib = 1 * 1 * 32 * 151936 * 2 / 1024 / 1024
    assert abs(mib - 9.3) < 0.05, (
        f"the quoted size no longer matches the shape ({mib:.1f} MiB)"
    )
