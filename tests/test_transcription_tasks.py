# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""Host-only unit tests for the audio/transcription task wiring in
``TTModelRunner`` (Qwen3-ASR enablement).

These call the runner methods unbound with a fake ``self`` so no device or
model construction is required, but importing the plugin module still pulls in
``vllm``/``ttnn`` (the ttnn-enabled environment is required to run them).
"""

from types import SimpleNamespace

import pytest

from vllm_tt_plugin import model_runner as mr
from vllm_tt_plugin.model_runner import TTModelRunner


def test_validate_mm_feature_accepts_audio_and_image():
    # image and audio are accepted; other modalities raise.
    fake_self = SimpleNamespace()
    for modality in ("image", "audio"):
        TTModelRunner._validate_mm_feature(
            fake_self, SimpleNamespace(modality=modality)
        )
    with pytest.raises(NotImplementedError):
        TTModelRunner._validate_mm_feature(
            fake_self, SimpleNamespace(modality="video")
        )


def _call_gen_tasks(monkeypatch, model, supports):
    monkeypatch.setattr(mr, "supports_transcription", lambda m: supports)
    fake_self = SimpleNamespace(model=model)
    return TTModelRunner.get_supported_generation_tasks(fake_self)


def test_generation_tasks_text_only_when_model_absent(monkeypatch):
    assert _call_gen_tasks(monkeypatch, None, False) == ["generate"]


def test_generation_tasks_no_transcription_support(monkeypatch):
    model = SimpleNamespace()
    assert _call_gen_tasks(monkeypatch, model, False) == ["generate"]


def test_generation_tasks_appends_transcription(monkeypatch):
    model = SimpleNamespace(supports_transcription_only=False)
    assert _call_gen_tasks(monkeypatch, model, True) == ["generate", "transcription"]


def test_generation_tasks_transcription_only(monkeypatch):
    model = SimpleNamespace(supports_transcription_only=True)
    assert _call_gen_tasks(monkeypatch, model, True) == ["transcription"]
