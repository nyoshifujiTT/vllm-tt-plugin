# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Qwen3-Embedding must win the TTQwen3ForCausalLM slot on a pooling run.

Qwen3-Embedding's HF config declares ``architectures: ["Qwen3ForCausalLM"]`` --
byte-identical to the Qwen3 *text* model. check_and_update_config prepends "TT",
so both resolve to the same registry key, and whichever registration lands last
decides what gets loaded.

If the text generator wins on an embedding run, vLLM does not fail loudly at
registration: it sees a generative model where a pooling one is required and
wraps it with ``_create_pooling_model_cls``. The wrapper's ``__init__`` takes
only ``self``, so the TT loader's
``initialize_vllm_model(tt_model, model_args, mesh_device)`` dies with
"ModelForPooling.__init__() takes 1 positional argument but 4 were given" --
observed on p150 before this was fixed. These tests pin the disambiguation.
"""

from vllm_tt_plugin import platform as tt_platform

QWEN3_ARCH = "TTQwen3ForCausalLM"
EMBED_TARGET = "models.demos.qwen3_embedding.tt.generator_vllm:Qwen3EmbeddingForTTvLLM"


class _FakeModelRegistry:
    """Records register_model calls; mimics get_supported_archs()."""

    def __init__(self):
        self.registered: dict[str, str] = {}

    def get_supported_archs(self):
        return set(self.registered)

    def register_model(self, arch, path):
        self.registered[arch] = path


def _register(monkeypatch, runner_type):
    registry = _FakeModelRegistry()

    import vllm.model_executor.models.registry as vllm_registry

    monkeypatch.setattr(vllm_registry, "ModelRegistry", registry)
    tt_platform.register_tt_models(runner_type=runner_type)
    return registry


def test_pooling_run_registers_the_embedding_wrapper(monkeypatch):
    registry = _register(monkeypatch, "pooling")

    assert registry.registered[QWEN3_ARCH] == EMBED_TARGET, (
        "on a pooling run TTQwen3ForCausalLM must resolve to the embedding "
        "wrapper, not the text generator"
    )


def test_generate_run_keeps_the_text_generator(monkeypatch):
    registry = _register(monkeypatch, "generate")

    assert registry.registered[QWEN3_ARCH] != EMBED_TARGET
    assert "generator_vllm:QwenForCausalLM" in registry.registered[QWEN3_ARCH]


def test_unknown_runner_type_keeps_the_text_generator(monkeypatch):
    # Early registration (worker import, general_plugins entry point) has no
    # vllm_config yet and passes no runner_type; it must not steal the slot.
    registry = _register(monkeypatch, None)

    assert "generator_vllm:QwenForCausalLM" in registry.registered[QWEN3_ARCH]
