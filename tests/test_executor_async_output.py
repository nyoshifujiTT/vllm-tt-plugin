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


def test_shutdown_stops_the_background_thread(monkeypatch):
    """A surviving worker thread keeps the engine process alive.

    ThreadPoolExecutor's workers are non-daemon, so an executor that is torn
    down without shutting the pool leaves a thread joined at interpreter exit.
    This was marked "pragma: no cover - lifecycle glue" and had no test, even
    though this plugin already treats shutdown paths as testable (see
    test_worker_shutdown_closes_mesh_once_across_shutdown_and_del).
    """
    mod, _out, _serial = _load_executor(monkeypatch)
    ex = mod.TTUniProcExecutor.__new__(mod.TTUniProcExecutor)
    ex.scheduler_config = types.SimpleNamespace(async_scheduling=True)
    ex.driver_worker = object()
    mod.TTUniProcExecutor._init_executor(ex)

    thread = ex._tt_async_output_thread
    assert thread is not None
    calls = []
    thread.shutdown = lambda **kwargs: calls.append(kwargs)

    ex.shutdown()

    assert calls == [{"wait": False}], (
        "the pool must be shut down, and not waited on -- shutdown runs on the "
        "engine thread and a pending read-back would block it"
    )
    assert ex._tt_async_output_thread is None, (
        "the handle must be cleared, or a second shutdown submits to a dead pool"
    )


def test_shutdown_is_idempotent(monkeypatch):
    """Called twice, the second call must not touch the pool again."""
    mod, _out, _serial = _load_executor(monkeypatch)
    ex = mod.TTUniProcExecutor.__new__(mod.TTUniProcExecutor)
    ex.scheduler_config = types.SimpleNamespace(async_scheduling=True)
    ex.driver_worker = object()
    mod.TTUniProcExecutor._init_executor(ex)

    calls = []
    ex._tt_async_output_thread.shutdown = lambda **kwargs: calls.append(kwargs)

    ex.shutdown()
    ex.shutdown()

    assert len(calls) == 1, f"the pool was shut down {len(calls)} times"


def test_shutdown_without_a_thread_is_not_an_error(monkeypatch):
    """async_scheduling off means no pool was ever created."""
    mod, _out, _serial = _load_executor(monkeypatch)
    ex = mod.TTUniProcExecutor.__new__(mod.TTUniProcExecutor)
    ex.scheduler_config = types.SimpleNamespace(async_scheduling=False)
    ex.driver_worker = object()
    mod.TTUniProcExecutor._init_executor(ex)
    assert ex._tt_async_output_thread is None

    ex.shutdown()  # must not raise


def test_shutdown_calls_the_base_class_teardown(monkeypatch):
    """Our pool is not the only thing that needs releasing.

    The base executor owns the driver worker; skipping its teardown would leak
    whatever it holds. It is called through getattr because upstream has
    shipped UniProcExecutor without a shutdown() -- so the call has to be
    conditional, and the conditional has to be exercised both ways.
    """
    mod, _out, _serial = _load_executor(monkeypatch)

    base_calls = []
    monkeypatch.setattr(
        mod.UniProcExecutor,
        "shutdown",
        lambda self: base_calls.append(True),
        raising=False,
    )

    ex = mod.TTUniProcExecutor.__new__(mod.TTUniProcExecutor)
    ex.scheduler_config = types.SimpleNamespace(async_scheduling=True)
    ex.driver_worker = object()
    mod.TTUniProcExecutor._init_executor(ex)

    ex.shutdown()

    assert base_calls == [True], "the base class teardown must run"


def test_shutdown_survives_a_base_class_without_one(monkeypatch):
    """The getattr guard exists for upstream versions with no shutdown()."""
    mod, _out, _serial = _load_executor(monkeypatch)
    monkeypatch.delattr(mod.UniProcExecutor, "shutdown", raising=False)

    ex = mod.TTUniProcExecutor.__new__(mod.TTUniProcExecutor)
    ex.scheduler_config = types.SimpleNamespace(async_scheduling=True)
    ex.driver_worker = object()
    mod.TTUniProcExecutor._init_executor(ex)

    ex.shutdown()  # must not raise
    assert ex._tt_async_output_thread is None


def test_no_lifecycle_method_is_excused_from_coverage():
    """"pragma: no cover" on our own code is a claim, so state the rule.

    shutdown() carried that marker and therefore had no test, while the same
    file's routing logic was covered thoroughly. Keep the marker for the one
    place it is honest -- an import fallback for vLLM versions this
    environment cannot install -- and nowhere else in what we added.
    """
    import os

    src_dir = os.path.join(os.path.dirname(__file__), "..", "src", "vllm_tt_plugin")
    with open(os.path.join(src_dir, "executor.py")) as fh:
        assert "pragma: no cover" not in fh.read(), (
            "executor.py is host-only Python; nothing in it needs excusing"
        )
