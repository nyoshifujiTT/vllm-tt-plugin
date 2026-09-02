# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""The TT executor really is a UniProcExecutor.

``_is_uniproc_executor_backend`` accepts the TT executor by name. That is only
sound if the class it names actually subclasses ``UniProcExecutor`` -- otherwise
the block-output guard would be waved through for an executor that does not run
the worker in the engine process.

This lives apart from test_executor_async_output.py, which stubs out vLLM to
test the routing logic in isolation; here the real class hierarchy is the point.
"""


def test_the_tt_executor_subclasses_uniproc():
    import importlib
    import sys

    # test_executor_async_output.py imports this module against a stubbed
    # vllm.v1.executor.uniproc_executor. monkeypatch restores sys.modules
    # afterwards, but a module object already bound to the stub base class can
    # linger, so re-import both against the real vLLM here.
    for name in (
        "vllm_tt_plugin.executor",
        "vllm.v1.executor.uniproc_executor",
    ):
        sys.modules.pop(name, None)

    from vllm.v1.executor.uniproc_executor import UniProcExecutor

    executor_mod = importlib.import_module("vllm_tt_plugin.executor")

    assert issubclass(executor_mod.TTUniProcExecutor, UniProcExecutor)
