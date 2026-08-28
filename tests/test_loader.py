# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Device-free unit tests for the TT model loader.

Model code may read vLLM's ambient config while it is being constructed --
upstream's own layers do, e.g. ``pooler_for_classify`` calls
``get_current_vllm_config()`` to resolve a classification head's activation. So
the loader must build the model inside ``set_current_vllm_config``, the way
upstream's ``initialize_model`` does. It cannot pass the config explicitly
instead: ``initialize_vllm_model`` does not accept ``vllm_config`` on most TT
models.
"""
from types import SimpleNamespace

import pytest

from vllm.config import get_current_vllm_config_or_none

from vllm_tt_plugin import loader as loader_mod
from vllm_tt_plugin.loader import TTModelLoader


class _RecordingModel:
    """Stands in for a TT model class, recording whether the ambient vLLM config
    was visible while it was being constructed."""

    seen_ambient_config = None

    @classmethod
    def initialize_vllm_model(cls, hf_config, device, max_batch_size, **kwargs):
        cls.seen_ambient_config = get_current_vllm_config_or_none()
        return SimpleNamespace(kind="tt-model")


@pytest.fixture
def stub_loader(monkeypatch):
    monkeypatch.setattr(loader_mod, "get_model_architecture", lambda mc: (_RecordingModel, "arch"))
    monkeypatch.setattr(loader_mod, "get_tt_config", lambda vc: {})
    monkeypatch.setattr(loader_mod, "get_tt_data_parallel_size", lambda vc: 1)
    monkeypatch.setattr(loader_mod, "get_tt_max_batch_size", lambda vc: 8)
    _RecordingModel.seen_ambient_config = None
    return TTModelLoader.__new__(TTModelLoader)


def _vllm_config(monkeypatch):
    """A real VllmConfig, because ``set_current_vllm_config`` runs vLLM's own
    post-init over it, so a stub would not exercise the same code path.

    Constructing one runs the platform's ``check_and_update_config``, which reads
    a real ``ModelConfig``; building one of those needs a model on disk, which a
    device-free unit test must not require. Stub that hook out and attach the two
    model-config fields the loader itself reads.
    """
    from vllm.config import VllmConfig
    from vllm_tt_plugin.platform import TTPlatform

    monkeypatch.setattr(TTPlatform, "check_and_update_config", staticmethod(lambda config: None))
    config = VllmConfig()
    config.model_config = SimpleNamespace(hf_config=object(), max_model_len=8192)
    return config


def test_model_is_built_inside_the_vllm_config_context(stub_loader, monkeypatch):
    vllm_config = _vllm_config(monkeypatch)

    stub_loader.load_model(vllm_config=vllm_config, model_config=vllm_config.model_config)

    assert _RecordingModel.seen_ambient_config is vllm_config, (
        "the model was constructed outside set_current_vllm_config, so model code "
        "cannot read the ambient config the way upstream layers expect"
    )


def test_the_config_context_does_not_leak(stub_loader, monkeypatch):
    """The context must be scoped to the load, matching upstream."""
    vllm_config = _vllm_config(monkeypatch)
    before = get_current_vllm_config_or_none()
    stub_loader.load_model(vllm_config=vllm_config, model_config=vllm_config.model_config)
    assert get_current_vllm_config_or_none() is before
