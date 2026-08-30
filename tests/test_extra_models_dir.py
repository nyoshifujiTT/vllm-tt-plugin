# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""Device-free tests for EXTRA_MODELS_DIR bundle discovery and registration.

These cover the sanctioned way to add a model (e.g. the bge-reranker
cross-encoder) to the standalone plugin without editing plugin source: drop a
``<name>/vllm_metadata.json`` bundle under ``EXTRA_MODELS_DIR`` and let the
plugin register ``TT<arch> -> module:Class`` at startup.

``vllm_tt_plugin.platform`` pulls in ``vllm.platforms`` (the full platform
resolution stack), which is pinned to the canonical upstream vLLM (0.24).
Importing the module directly, before vLLM has activated the ``tt`` platform
plugin, trips vLLM's own lazy ``current_platform`` resolution, so we ``import
vllm`` first (that runs the plugin entry-points and registers the platform).
On a missing / mismatched vLLM the import still fails; the tests are then
skipped rather than reporting a false failure. The functions under test are
pure host-side helpers.
"""
import json
import os

import pytest

try:
    import vllm  # noqa: F401  -- activates the tt platform plugin (entry-points)
    from vllm_tt_plugin.platform import (
        _iter_extra_model_bundles,
        _register_models_from_extra_dir,
    )

    _PLATFORM_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - exercised only on vLLM skew
    _iter_extra_model_bundles = None
    _register_models_from_extra_dir = None
    _PLATFORM_IMPORT_ERROR = exc

pytestmark = pytest.mark.skipif(
    _iter_extra_model_bundles is None,
    reason=f"vllm_tt_plugin.platform unavailable on this vLLM: {_PLATFORM_IMPORT_ERROR}",
)


class _FakeRegistry:
    """Minimal stand-in for vLLM's ModelRegistry used by the plugin."""

    def __init__(self, already=()):
        self._archs = set(already)
        self.registered = {}

    def get_supported_archs(self):
        return set(self._archs) | set(self.registered)

    def register_model(self, arch, path):
        self.registered[arch] = path
        self._archs.add(arch)


def _write_bundle(base, name, meta):
    folder = os.path.join(base, name)
    os.makedirs(folder, exist_ok=True)
    if meta is not None:
        with open(os.path.join(folder, "vllm_metadata.json"), "w") as fh:
            json.dump(meta, fh)
    return folder


_RERANKER_META = {
    "arch": "XLMRobertaForSequenceClassification",
    "main_class": "models.demos.bge_reranker_v2_m3.demo.generator_vllm:BgeRerankerV2M3",
}


def test_iter_bundles_yields_reranker(tmp_path, monkeypatch):
    base = tmp_path / "extra"
    _write_bundle(str(base), "bge-reranker-v2-m3", _RERANKER_META)
    monkeypatch.setenv("EXTRA_MODELS_DIR", str(base))

    bundles = list(_iter_extra_model_bundles())

    assert len(bundles) == 1
    folder, arch, main_class = bundles[0]
    assert arch == _RERANKER_META["arch"]
    assert main_class == _RERANKER_META["main_class"]
    assert os.path.basename(folder) == "bge-reranker-v2-m3"


def test_register_prefixes_arch_with_tt(tmp_path, monkeypatch):
    base = tmp_path / "extra"
    _write_bundle(str(base), "bge-reranker-v2-m3", _RERANKER_META)
    monkeypatch.setenv("EXTRA_MODELS_DIR", str(base))
    registry = _FakeRegistry()

    count = _register_models_from_extra_dir(registry)

    assert count == 1
    assert registry.registered == {
        "TTXLMRobertaForSequenceClassification": _RERANKER_META["main_class"]
    }


def test_register_does_not_double_prefix_tt(tmp_path, monkeypatch):
    base = tmp_path / "extra"
    _write_bundle(
        str(base),
        "already-tt",
        {"arch": "TTXLMRobertaForSequenceClassification", "main_class": "m:C"},
    )
    monkeypatch.setenv("EXTRA_MODELS_DIR", str(base))
    registry = _FakeRegistry()

    _register_models_from_extra_dir(registry)

    assert "TTXLMRobertaForSequenceClassification" in registry.registered
    assert "TTTTXLMRobertaForSequenceClassification" not in registry.registered


def test_register_skips_when_arch_already_present(tmp_path, monkeypatch):
    base = tmp_path / "extra"
    _write_bundle(str(base), "bge-reranker-v2-m3", _RERANKER_META)
    monkeypatch.setenv("EXTRA_MODELS_DIR", str(base))
    registry = _FakeRegistry(already=["TTXLMRobertaForSequenceClassification"])

    _register_models_from_extra_dir(registry)

    # Already registered -> _register_model_if_missing must not overwrite.
    assert registry.registered == {}


def test_iter_skips_malformed_bundles(tmp_path, monkeypatch):
    base = tmp_path / "extra"
    _write_bundle(str(base), "no-meta", None)  # folder without metadata
    _write_bundle(str(base), "empty-meta", {})  # missing arch/main_class
    _write_bundle(str(base), "good", _RERANKER_META)
    monkeypatch.setenv("EXTRA_MODELS_DIR", str(base))

    names = {os.path.basename(f) for f, _, _ in _iter_extra_model_bundles()}

    assert names == {"good"}


def test_iter_empty_when_env_unset(monkeypatch):
    monkeypatch.delenv("EXTRA_MODELS_DIR", raising=False)
    assert list(_iter_extra_model_bundles()) == []
