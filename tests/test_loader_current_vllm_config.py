# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The loader must construct the model inside the current-vLLM-config context.

Upstream's ``BaseModelLoader.load_model`` builds the model through
``initialize_model``, which wraps construction in ``set_current_vllm_config``.
vLLM layers rely on that: every ``CustomOp`` -- and the pooling methods under
``vllm.model_executor.layers.pooler`` -- call ``get_current_vllm_config()`` in
their ``__init__``. The TT loader bypasses ``initialize_model`` (TT models are
built by their own ``initialize_vllm_model``), so it has to establish the same
context itself. Without it, serving Qwen3-Embedding-0.6B on p150 died with

  AssertionError: Current vLLM config is not set.

the moment the embedding Pooler was constructed.
"""

from types import SimpleNamespace

import pytest
from vllm.config import get_current_vllm_config

from vllm_tt_plugin.loader import TTModelLoader


class _ConfigProbingModel:
    """Stands in for a wrapper that builds a config-reading layer (CustomOp,
    Pooler, ...) while it is being constructed."""

    seen = None

    @classmethod
    def initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size, **kwargs):
        cls.seen = get_current_vllm_config()
        return "model"


@pytest.fixture
def vllm_config():
    """A real VllmConfig, so set_current_vllm_config's own bookkeeping (it
    touches compilation_config) stays honest, with the two fields the loader
    reads pointed at test doubles."""
    from vllm.config import DeviceConfig, VllmConfig

    # An explicit DeviceConfig is required: the default one auto-detects the
    # platform, which fails in a plain test process with no accelerator.
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    config.model_config = SimpleNamespace(hf_config=object(), max_model_len=8192)
    return config


def _patch_loader_deps(monkeypatch):
    import vllm_tt_plugin.loader as loader_mod

    monkeypatch.setattr(
        loader_mod,
        "get_model_architecture",
        lambda model_config: (_ConfigProbingModel, None),
    )
    monkeypatch.setattr(loader_mod, "get_tt_config", lambda cfg: {})
    monkeypatch.setattr(loader_mod, "get_tt_data_parallel_size", lambda cfg: 1)
    monkeypatch.setattr(loader_mod, "get_tt_max_batch_size", lambda cfg: 1)


def test_model_is_built_under_set_current_vllm_config(monkeypatch, vllm_config):
    _patch_loader_deps(monkeypatch)

    loader = TTModelLoader.__new__(TTModelLoader)
    result = loader.load_model(vllm_config, vllm_config.model_config)

    assert result == "model"
    assert _ConfigProbingModel.seen is vllm_config, (
        "layers constructed during model init resolve the config through "
        "get_current_vllm_config(); the loader must set it"
    )


def test_context_does_not_leak_after_load(monkeypatch, vllm_config):
    _patch_loader_deps(monkeypatch)

    loader = TTModelLoader.__new__(TTModelLoader)
    loader.load_model(vllm_config, vllm_config.model_config)

    # The context is scoped to model construction; afterwards the previous
    # value (nothing, here) is restored and looking it up raises again.
    with pytest.raises(AssertionError, match="Current vLLM config is not set"):
        get_current_vllm_config()
