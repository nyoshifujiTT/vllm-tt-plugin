# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""Host-only test that the built-in TT model map registers the Qwen3-ASR
adapter under the plugin's ``TT``-prefixed convention.

Importing ``vllm_tt_plugin.platform`` pulls in ``torch``/``vllm``, so the
ttnn-enabled environment is required to run this test. A fake ModelRegistry is
injected so no real model classes are imported.
"""

import pytest


class _FakeModelRegistry:
    def __init__(self):
        self.registered: dict[str, str] = {}

    def get_supported_archs(self):
        return list(self.registered.keys())

    def register_model(self, arch, path):
        self.registered[arch] = path


def test_builtin_map_registers_qwen3_asr(monkeypatch):
    from vllm_tt_plugin import platform as plat

    fake = _FakeModelRegistry()
    # register_tt_models imports ModelRegistry lazily from vllm; patch the
    # symbol it resolves so no real vLLM registry / model classes are touched.
    monkeypatch.setattr(
        "vllm.model_executor.models.registry.ModelRegistry", fake, raising=False
    )
    # Skip the EXTRA_MODELS_DIR dynamic hook (no bundles in a unit test).
    monkeypatch.setattr(plat, "_register_models_from_extra_dir", lambda _reg: 0)

    plat.register_tt_models()

    assert (
        fake.registered.get("TTQwen3ASRForConditionalGeneration")
        == "models.demos.audio.qwen3_asr.tt.generator_vllm:TTQwen3ASRForConditionalGeneration"
    )
