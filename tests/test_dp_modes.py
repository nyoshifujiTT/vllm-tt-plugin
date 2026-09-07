# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

"""Unit tests for how ``check_and_update_config`` routes a DP config.

Engine-core/launcher/scheduler selection and the worker's device lifecycle. The
``TT_VISIBLE_DEVICES`` group lifecycle lives in ``test_visible_devices.py``, and
the DP-to-lanes conversion in ``test_galaxy_dp_conversion.py``.
"""

from types import SimpleNamespace

import pytest
import ttnn
from vllm.v1.core.sched import interface as sched_interface

from vllm_tt_plugin import worker
from vllm_tt_plugin.platform import TTPlatform
from vllm_tt_plugin.worker import TTWorker

if not hasattr(sched_interface, "PauseState"):
    sched_interface.PauseState = type("PauseState", (), {})


class TestDPModes:
    @pytest.fixture
    def dummy_model_class(self) -> type:
        return type(
            "DummyModel",
            (),
            {"__module__": "models.tt_transformers.tt.generator_vllm"},
        )

    @staticmethod
    def register_dummy_model(
        monkeypatch: pytest.MonkeyPatch,
        vllm_config: SimpleNamespace,
        dummy_model_class: type,
        visible_device_groups: list[str] | None = None,
    ) -> None:
        with monkeypatch.context() as m:
            m.setattr(
                "vllm_tt_plugin.platform.register_tt_models",
                lambda *args, **kwargs: None,
            )
            m.setattr(
                "vllm_tt_plugin.platform._resolve_standard_dp_visible_device_groups",
                lambda _cfg: visible_device_groups,
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

    @pytest.mark.parametrize("original_max_model_len", [8192, -1, None])
    def test_check_and_update_config_never_rewrites_max_model_len(
        self,
        monkeypatch: pytest.MonkeyPatch,
        vllm_config: SimpleNamespace,
        dummy_model_class: type,
        original_max_model_len: int | None,
    ) -> None:
        """The TT platform leaves vLLM's max_model_len policy alone.

        A numeric value must reach upstream's override-aware capacity check
        unchanged (so an oversized value fails loudly instead of being silently
        clamped), an explicit -1 must stay -1 so upstream auto-fits, and an
        omitted value must stay None so upstream keeps its HF-derived default.
        """
        vllm_config.model_config.original_max_model_len = original_max_model_len

        self.register_dummy_model(monkeypatch, vllm_config, dummy_model_class)

        assert vllm_config.model_config.original_max_model_len == original_max_model_len

    def test_check_and_update_config_forces_eager(
        self,
        monkeypatch: pytest.MonkeyPatch,
        vllm_config: SimpleNamespace,
        dummy_model_class: type,
    ) -> None:
        """TT never uses vLLM's compiled graph, so the platform pins eager.

        ``enforce_eager`` must be flipped on and the already-built
        ``compilation_config`` must be pinned to a non-compiling mode, because
        ``VllmConfig.__post_init__`` derives the compilation mode from
        ``enforce_eager`` before this hook runs. Flipping the flag alone leaves
        a config whose mode was already computed from the old value.

        This test and the pin it covers were both added in bf6185b/47bcc8a and
        silently lost in the first upstream merge; the branch then ran for weeks
        with enforce_eager set but the compilation mode unpinned.
        """
        from vllm.config import CompilationMode, CUDAGraphMode

        assert vllm_config.model_config.enforce_eager is False

        self.register_dummy_model(monkeypatch, vllm_config, dummy_model_class)

        assert vllm_config.model_config.enforce_eager is True
        assert vllm_config.compilation_config.mode == CompilationMode.NONE
        assert vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.NONE

    def test_a_missing_compilation_mode_is_the_only_quiet_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
        vllm_config: SimpleNamespace,
        dummy_model_class: type,
    ) -> None:
        """The guard is for the import, not for the assignments.

        Both were inside one try/except Exception, so an assignment that failed
        -- a None compilation_config, a frozen one, or only CUDAGraphMode
        having moved -- was reported as "could not pin" too. The CUDAGraphMode
        case was the worst: `mode` had already been set and `cudagraph_mode`
        had not, so the config was half-pinned and the log said neither.

        Pinning is worth ~2.3x on decode (2.68 -> 6.21 tok/s/user measured on
        this model), so it losing effect quietly is a silent halving.
        """
        import vllm.config as vllm_config_module

        # `from vllm.config import CompilationMode, CUDAGraphMode` raises
        # ImportError when either name is absent -- so both names are resolved
        # before anything is assigned, which is exactly the property under
        # test. Previously the assignments shared the try, so a failure after
        # the first one left the config half-pinned.
        monkeypatch.delattr(vllm_config_module, "CUDAGraphMode", raising=False)

        self.register_dummy_model(monkeypatch, vllm_config, dummy_model_class)

        assert vllm_config.compilation_config.mode is None, (
            "nothing may be assigned when a name cannot be resolved; a "
            "half-pinned config is worse than an unpinned one"
        )
        assert vllm_config.compilation_config.cudagraph_mode is None
        assert vllm_config.model_config.enforce_eager is True, (
            "the documented fallback is enforce_eager alone; it must hold"
        )

    def test_the_import_failure_still_leaves_eager_on(
        self,
        monkeypatch: pytest.MonkeyPatch,
        vllm_config: SimpleNamespace,
        dummy_model_class: type,
    ) -> None:
        """When the names are gone entirely, enforce_eager must still be set.

        That is the fallback the warning claims ("relying on enforce_eager
        alone"), and it was never checked.
        """
        import builtins

        real_import = builtins.__import__

        def _no_compilation_mode(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "vllm.config" and fromlist and "CompilationMode" in fromlist:
                raise ImportError("no CompilationMode in this vLLM")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", _no_compilation_mode)

        self.register_dummy_model(monkeypatch, vllm_config, dummy_model_class)

        assert vllm_config.model_config.enforce_eager is True
        assert vllm_config.compilation_config.mode is None

    def test_update_max_model_len_syncs_worker_model_config(self) -> None:
        worker_instance = TTWorker.__new__(TTWorker)
        worker_instance.model_config = SimpleNamespace(max_model_len=262_144)

        TTWorker.update_max_model_len(worker_instance, 131_072)

        assert worker_instance.model_config.max_model_len == 131_072

    def test_engine_core_classes_are_left_to_upstream(
        self,
        monkeypatch: pytest.MonkeyPatch,
        vllm_config: SimpleNamespace,
        dummy_model_class: type,
    ) -> None:
        # ``ParallelConfig`` accepts arbitrary attribute writes, so naming an
        # engine-core class upstream does not define reads as configuration but
        # does nothing. The plugin selects a worker and a scheduler; engine-core
        # selection is upstream's.
        self.register_dummy_model(monkeypatch, vllm_config, dummy_model_class)

        assert not hasattr(vllm_config.parallel_config, "engine_core_cls")
        assert not hasattr(vllm_config.parallel_config, "engine_core_proc_cls")
        assert (
            vllm_config.parallel_config.dp_engine_core_proc_cls
            == "vllm.v1.engine.core.DPEngineCoreProc"
        )

    def test_lane_mode_keeps_upstream_dp_engine_core(
        self,
        monkeypatch: pytest.MonkeyPatch,
        vllm_config: SimpleNamespace,
        dummy_model_class: type,
    ) -> None:
        vllm_config.additional_config = {"_tt_resolved_lane_count": 2}
        vllm_config.parallel_config.data_parallel_size = 1

        self.register_dummy_model(monkeypatch, vllm_config, dummy_model_class)

        assert not hasattr(vllm_config.parallel_config, "engine_core_cls")
        assert not hasattr(vllm_config.parallel_config, "engine_core_proc_cls")
        assert (
            vllm_config.parallel_config.dp_engine_core_proc_cls
            == "vllm.v1.engine.core.DPEngineCoreProc"
        )

    def test_collapsed_standard_dp_rank_warms_up_model(self) -> None:
        worker = TTWorker.__new__(TTWorker)
        worker.enable_model_warmup = True
        worker.parallel_config = SimpleNamespace(
            data_parallel_size=1,
            data_parallel_rank_local=7,
            data_parallel_index=7,
        )
        warmup_calls: list[str] = []
        worker.model_runner = SimpleNamespace(
            warmup_model=lambda: warmup_calls.append("warmup")
        )

        timings = TTWorker.compile_or_warm_up_model(worker)

        assert warmup_calls == ["warmup"]
        assert timings.language_model >= 0.0

    def test_single_host_standard_dp_leaves_the_launcher_to_upstream(
        self,
        monkeypatch: pytest.MonkeyPatch,
        vllm_config: SimpleNamespace,
        dummy_model_class: type,
    ) -> None:
        vllm_config.parallel_config.data_parallel_size = 4

        self.register_dummy_model(
            monkeypatch,
            vllm_config,
            dummy_model_class,
            visible_device_groups=["24,25", "26,27", "3,2", "1,0"],
        )

        assert not hasattr(vllm_config.parallel_config, "engine_core_launcher_cls")
        assert TTPlatform._standard_dp_visible_device_groups == [
            "24,25",
            "26,27",
            "3,2",
            "1,0",
        ]

    @pytest.mark.parametrize(
        ("tt_config", "parallel_overrides"),
        [
            ({"rank_binding": "/tmp/rank_binding.yaml"}, {}),
            ({"mpi_args": "--host hostA"}, {}),
            ({}, {"nnodes": 2}),
            ({}, {"node_rank": 1}),
        ],
    )
    def test_explicit_tt_launch_is_rejected_without_the_launcher_hook(
        self,
        monkeypatch: pytest.MonkeyPatch,
        vllm_config: SimpleNamespace,
        dummy_model_class: type,
        tt_config: dict,
        parallel_overrides: dict,
    ) -> None:
        # Upstream vLLM defines no ``engine_core_launcher_cls``, and writing one
        # anyway is silently ignored, which would quietly single-host a run the
        # user asked to spread over several nodes. Every explicit-MPI trigger
        # (rank_binding, mpi_args, nnodes > 1, node_rank > 0) must fail fast.
        vllm_config.parallel_config.data_parallel_size = 4
        vllm_config.additional_config = {"tt": tt_config}
        for key, value in parallel_overrides.items():
            setattr(vllm_config.parallel_config, key, value)

        with pytest.raises(NotImplementedError, match="engine-core launcher hook"):
            self.register_dummy_model(monkeypatch, vllm_config, dummy_model_class)

    def test_rank_binding_selects_the_tt_launcher_when_the_hook_exists(
        self,
        monkeypatch: pytest.MonkeyPatch,
        vllm_config: SimpleNamespace,
        dummy_model_class: type,
    ) -> None:
        vllm_config.parallel_config.data_parallel_size = 4
        vllm_config.parallel_config.engine_core_launcher_cls = "auto"
        vllm_config.additional_config = {
            "tt": {"rank_binding": "/tmp/rank_binding.yaml"}
        }

        self.register_dummy_model(monkeypatch, vllm_config, dummy_model_class)

        assert (
            vllm_config.parallel_config.engine_core_launcher_cls
            == "vllm_tt_plugin.launcher.TTCoreEngineLauncher"
        )
        assert TTPlatform._standard_dp_visible_device_groups is None

    def test_tt_platform_set_device_uses_ttnn_default_device(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        assigned_devices: list[object] = []

        monkeypatch.setattr(ttnn, "GetDefaultDevice", lambda: None, raising=False)
        monkeypatch.setattr(
            ttnn,
            "SetDefaultDevice",
            lambda device: assigned_devices.append(device),
            raising=False,
        )

        mesh_device = object()
        TTPlatform.set_device(None)
        TTPlatform.set_device(mesh_device)

        assert assigned_devices == [mesh_device]

    def test_init_device_tracks_mesh_as_worker_device(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mesh_device = SimpleNamespace(get_num_devices=lambda: 8)
        model_runner = SimpleNamespace()

        parallel_config = SimpleNamespace(
            data_parallel_size=4,
            data_parallel_rank_local=0,
            data_parallel_index=0,
            assigned_physical_gpu_ids=None,
        )

        worker_instance = TTWorker.__new__(TTWorker)
        worker_instance.vllm_config = SimpleNamespace(
            additional_config={}, parallel_config=parallel_config
        )
        # WorkerBase aliases the field; keep the test's view identical.
        worker_instance.parallel_config = parallel_config
        worker_instance.device_config = SimpleNamespace(device=None)
        worker_instance.trace_mode = "all"
        worker_instance.enable_model_warmup = True

        monkeypatch.setattr(TTPlatform, "check_and_update_config", lambda _cfg: None)
        monkeypatch.setattr(worker, "get_tt_config", lambda _cfg: {})
        monkeypatch.setattr(
            worker,
            "open_mesh_device",
            lambda _tt_config, _trace_mode, _local_dp_rank: mesh_device,
        )

        # The KV pool is sized and --max-model-len settled during init_device so
        # the model sees the fitted length when load_model runs next.
        steps: list[str] = []

        def size_pool(_cfg, _num_devices):
            steps.append("size")
            return 7

        def fit_max_model_len(_cfg, _num_tt_blocks):
            steps.append("fit")

        def build_runner(**_kwargs):
            steps.append("runner")
            return model_runner

        monkeypatch.setattr(worker, "get_num_available_blocks_tt", size_pool)
        monkeypatch.setattr(
            worker, "_fit_block_output_max_model_len", fit_max_model_len
        )
        monkeypatch.setattr(worker, "TTModelRunner", build_runner)

        try:
            TTWorker.init_device(worker_instance)

            assert worker_instance.mesh_device is mesh_device
            assert worker_instance.device is mesh_device
            assert worker_instance.device_config.device is mesh_device
            assert worker_instance.model_runner is model_runner
            assert steps == ["size", "fit", "runner"]
            assert worker_instance._num_tt_blocks == 7
        finally:
            # The test double cannot be passed to TT device cleanup.
            worker_instance.mesh_device = None

    def test_legacy_tt_dp_override_is_ignored_by_platform(
        self,
        monkeypatch: pytest.MonkeyPatch,
        vllm_config: SimpleNamespace,
        dummy_model_class: type,
    ) -> None:
        vllm_config.additional_config = {"tt": {"tt_data_parallel_size": 4}}
        vllm_config.parallel_config.data_parallel_size = 4

        self.register_dummy_model(monkeypatch, vllm_config, dummy_model_class)

        assert vllm_config.parallel_config.data_parallel_size == 4
        assert (
            vllm_config.parallel_config.dp_engine_core_proc_cls
            == "vllm.v1.engine.core.DPEngineCoreProc"
        )
        assert (
            vllm_config.scheduler_config.scheduler_cls
            == "vllm_tt_plugin.scheduler.TTScheduler"
        )

    def test_standard_dp_rejects_moe_models(
        self,
        monkeypatch: pytest.MonkeyPatch,
        vllm_config: SimpleNamespace,
        dummy_model_class: type,
    ) -> None:
        vllm_config.parallel_config.data_parallel_size = 4
        vllm_config.model_config.is_moe = True

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

            with pytest.raises(
                ValueError,
                match="TT standard DP does not support MoE models yet",
            ):
                TTPlatform.check_and_update_config(vllm_config)
