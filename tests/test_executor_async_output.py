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
