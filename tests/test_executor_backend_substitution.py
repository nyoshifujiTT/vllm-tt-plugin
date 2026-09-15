# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""``check_and_update_config`` really installs the TT executor.

``_is_uniproc_executor_backend`` only decides *whether* a backend counts as
uniproc; the substitution that puts ``TTUniProcExecutor`` into
``parallel_config.distributed_executor_backend`` is a separate statement. A
mutation that deleted that assignment left every existing test green, so the
decode read-back would have silently gone back to running inline on the engine
thread. These tests pin the assignment itself.
"""

from types import SimpleNamespace

import pytest
from vllm.v1.core.sched import interface as sched_interface

from vllm_tt_plugin.platform import TT_UNIPROC_EXECUTOR_BACKEND, TTPlatform

if not hasattr(sched_interface, "PauseState"):
    sched_interface.PauseState = type("PauseState", (), {})


def _run_check_and_update_config(
    monkeypatch: pytest.MonkeyPatch, vllm_config: SimpleNamespace
) -> None:
    """Drive the real hook with model registration/loading stubbed out."""

    dummy_model_class = type(
        "DummyModel",
        (),
        {"__module__": "models.tt_transformers.tt.generator_vllm"},
    )
    with monkeypatch.context() as m:
        m.setattr(
            "vllm_tt_plugin.platform.register_tt_models",
            lambda *args, **kwargs: None,
        )
        m.setattr(
            "vllm_tt_plugin.platform._resolve_standard_dp_visible_device_groups",
            lambda _cfg: None,
        )
        m.setattr(
            "vllm.model_executor.models.registry.ModelRegistry.get_supported_archs",
            lambda: ["TTDummyModel"],
        )
        m.setattr(
            "vllm.model_executor.model_loader.utils.get_model_architecture",
            lambda _model_config: (dummy_model_class, None),
        )
        TTPlatform.check_and_update_config(vllm_config)


class TestExecutorBackendSubstitution:
    @pytest.mark.parametrize("backend", [None, "uni"])
    def test_uniproc_backends_are_replaced_by_the_tt_executor(
        self, monkeypatch: pytest.MonkeyPatch, vllm_config: SimpleNamespace, backend
    ) -> None:
        vllm_config.parallel_config.distributed_executor_backend = backend

        _run_check_and_update_config(monkeypatch, vllm_config)

        assert (
            vllm_config.parallel_config.distributed_executor_backend
            == TT_UNIPROC_EXECUTOR_BACKEND
        )

    def test_the_substitution_is_idempotent(
        self, monkeypatch: pytest.MonkeyPatch, vllm_config: SimpleNamespace
    ) -> None:
        # vLLM runs the hook again in the engine-core process, so a second
        # pass over an already-updated config must not disturb the choice.
        vllm_config.parallel_config.distributed_executor_backend = (
            TT_UNIPROC_EXECUTOR_BACKEND
        )

        _run_check_and_update_config(monkeypatch, vllm_config)

        assert (
            vllm_config.parallel_config.distributed_executor_backend
            == TT_UNIPROC_EXECUTOR_BACKEND
        )

    @pytest.mark.parametrize("backend", ["mp", "ray"])
    def test_multiprocess_backends_are_left_alone(
        self, monkeypatch: pytest.MonkeyPatch, vllm_config: SimpleNamespace, backend
    ) -> None:
        # "mp"/"ray" have their own async output handling; overriding them
        # would silently collapse a multiprocess launch to one process.
        vllm_config.parallel_config.distributed_executor_backend = backend

        _run_check_and_update_config(monkeypatch, vllm_config)

        assert vllm_config.parallel_config.distributed_executor_backend == backend
