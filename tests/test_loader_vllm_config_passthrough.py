# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The loader must hand pooling models the resolved VllmConfig.

A pooling wrapper builds its Pooler from
``vllm_config.model_config.pooler_config`` -- that is where vLLM records the
pooling type and activation it derived from the checkpoint plus any
``--override-pooler-config``. The loader used to call ``initialize_vllm_model``
without it, so the wrapper had no config to build a Pooler from and serving died
during startup with

  Call to get_supported_tasks method failed: Qwen3EmbeddingForTTvLLM.pooler needs
  vllm_config.model_config.pooler_config

observed serving Qwen3-Embedding-0.6B on p150. Generative wrappers do not accept
the argument, so it is passed only to those whose signature takes it.
"""

from types import SimpleNamespace

from vllm_tt_plugin.loader import TTModelLoader


class _PoolingModel:
    """Wrapper that wants the VllmConfig (embedding / reranker style)."""

    seen = {}

    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, vllm_config=None, **kwargs
    ):
        cls.seen = {"vllm_config": vllm_config, "kwargs": kwargs}
        return "pooling-model"


class _GenerativeModel:
    """Wrapper with the fixed generative signature -- must not be handed one."""

    seen = {}

    @classmethod
    def initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size, **kwargs):
        cls.seen = {"kwargs": kwargs}
        return "generative-model"


def _load_with(monkeypatch, model_class):
    import vllm_tt_plugin.loader as loader_mod

    monkeypatch.setattr(
        loader_mod, "get_model_architecture", lambda model_config: (model_class, None)
    )
    monkeypatch.setattr(loader_mod, "get_tt_config", lambda vllm_config: {})
    monkeypatch.setattr(loader_mod, "get_tt_data_parallel_size", lambda vllm_config: 1)
    monkeypatch.setattr(loader_mod, "get_tt_max_batch_size", lambda vllm_config: 1)

    model_config = SimpleNamespace(hf_config=object(), max_model_len=8192)
    vllm_config = SimpleNamespace(
        device_config=SimpleNamespace(device="dev"), model_config=model_config
    )

    loader = TTModelLoader.__new__(TTModelLoader)
    return loader.load_model(vllm_config, model_config), vllm_config


def test_pooling_model_receives_the_vllm_config(monkeypatch):
    result, vllm_config = _load_with(monkeypatch, _PoolingModel)

    assert result == "pooling-model"
    assert _PoolingModel.seen["vllm_config"] is vllm_config, (
        "a pooling wrapper builds its Pooler from vllm_config.model_config."
        "pooler_config; without it startup fails in get_supported_tasks"
    )


def test_generative_model_is_not_handed_a_vllm_config(monkeypatch):
    result, _ = _load_with(monkeypatch, _GenerativeModel)

    assert result == "generative-model"
    assert "vllm_config" not in _GenerativeModel.seen["kwargs"], (
        "generative wrappers have a fixed signature; passing vllm_config would "
        "break them"
    )
