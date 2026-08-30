# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Device-free tests for EXTRA_MODELS_DIR bundle registration.

A model is registered at startup by dropping a bundle folder (a
``vllm_metadata.json`` naming ``arch`` + ``main_class``) under
``EXTRA_MODELS_DIR``. These tests exercise the real plugin scan/registration
helpers (no vLLM engine, no device) and also validate the shipped
Qwen3-Embedding example bundle, so the adapter wiring stays in sync with the
registration convention.
"""

import json
import os

from vllm_tt_plugin import platform as tt_platform


class _FakeModelRegistry:
    """Records register_model calls; mimics get_supported_archs()."""

    def __init__(self):
        self.registered: dict[str, str] = {}

    def get_supported_archs(self):
        return set(self.registered)

    def register_model(self, arch, path):
        self.registered[arch] = path


def _write_bundle(base, name, data):
    folder = os.path.join(base, name)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "vllm_metadata.json"), "w") as fh:
        json.dump(data, fh)
    return folder


def test_bundle_registers_with_tt_prefix(tmp_path, monkeypatch):
    _write_bundle(
        str(tmp_path),
        "qwen3-embedding",
        {"arch": "Qwen3ForCausalLM", "main_class": "pkg.mod:Cls"},
    )
    monkeypatch.setenv("EXTRA_MODELS_DIR", str(tmp_path))
    reg = _FakeModelRegistry()

    count = tt_platform._register_models_from_extra_dir(reg)

    assert count == 1
    # The plugin prepends its TT convention to the HF arch.
    assert reg.registered == {"TTQwen3ForCausalLM": "pkg.mod:Cls"}


def test_bundle_does_not_double_prefix_tt(tmp_path, monkeypatch):
    _write_bundle(
        str(tmp_path), "already-tt", {"arch": "TTFooForCausalLM", "main_class": "m:C"}
    )
    monkeypatch.setenv("EXTRA_MODELS_DIR", str(tmp_path))
    reg = _FakeModelRegistry()

    tt_platform._register_models_from_extra_dir(reg)

    assert "TTFooForCausalLM" in reg.registered
    assert "TTTTFooForCausalLM" not in reg.registered


def test_malformed_bundle_is_skipped(tmp_path, monkeypatch):
    _write_bundle(str(tmp_path), "bad", {"arch": "OnlyArch"})  # missing main_class
    _write_bundle(str(tmp_path), "good", {"arch": "GoodArch", "main_class": "m:C"})
    monkeypatch.setenv("EXTRA_MODELS_DIR", str(tmp_path))
    reg = _FakeModelRegistry()

    count = tt_platform._register_models_from_extra_dir(reg)

    assert count == 1
    assert reg.registered == {"TTGoodArch": "m:C"}


def test_shipped_qwen3_embedding_example_bundle_is_valid():
    # The example bundle under examples/extra_models must stay a valid,
    # registrable bundle pointing at the real adapter class.
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    meta_path = os.path.join(
        here, "examples", "extra_models", "qwen3-embedding", "vllm_metadata.json"
    )
    with open(meta_path) as fh:
        data = json.load(fh)
    assert data["arch"] == "Qwen3ForCausalLM"
    assert data["main_class"] == (
        "models.demos.qwen3_embedding.tt.generator_vllm:Qwen3EmbeddingForTTvLLM"
    )
    # module:Class form the plugin resolves lazily.
    assert ":" in data["main_class"]
