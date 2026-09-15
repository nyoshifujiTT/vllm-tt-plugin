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
    # ``get_supported_generation_tasks`` imports ``supports_transcription``
    # lazily from ``vllm.model_executor.models`` inside the method (to defer the
    # vLLM submodule import), so patch it at that authoritative source rather
    # than on the plugin module.
    import vllm.model_executor.models as vllm_models

    monkeypatch.setattr(
        vllm_models, "supports_transcription", lambda m: supports, raising=False
    )
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


def _mm_item(**fields):
    """A mm_feature.data mapping whose values expose .data, as vLLM's do."""

    class _Item(dict):
        pass

    return _Item({k: SimpleNamespace(data=v) for k, v in fields.items()})


def _gather(requests):
    """Run _gather_multi_modal_inputs over a fake persistent batch.

    Each entry of ``requests`` is the mm_features list for one request (or
    None for a text-only one), in persistent-batch order.
    """
    req_ids = [f"r{i}" for i in range(len(requests))]
    states = {
        rid: SimpleNamespace(mm_features=features)
        for rid, features in zip(req_ids, requests)
    }
    fake_self = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=req_ids, num_reqs=len(req_ids)),
        requests=states,
        _validate_mm_feature=lambda feature: TTModelRunner._validate_mm_feature(
            SimpleNamespace(), feature
        ),
    )
    return TTModelRunner._gather_multi_modal_inputs(fake_self)


AUDIO_KEYS = ("input_audio_features", "audio_feature_lengths")
IMAGE_KEYS = ("pixel_values", "image_grid_thw")


def test_an_audio_request_carries_its_features_and_no_image_entry():
    """The ASR input path: features reach the adapter under the audio keys.

    Nothing tested this function at all, and it is what hands the encoder its
    mel features -- the transcription input itself.
    """
    feature = SimpleNamespace(
        modality="audio",
        data=_mm_item(input_audio_features="MEL", audio_feature_lengths="LEN"),
    )

    out = _gather([[feature]])

    assert out["input_audio_features"] == [["MEL"]]
    assert out["audio_feature_lengths"] == [["LEN"]]
    # the modality this request does not use must be absent, not empty
    for key in IMAGE_KEYS:
        assert out[key] == [None], f"{key} should be None for an audio request"


def test_an_absent_modality_is_none_rather_than_an_empty_list():
    """The docstring's contract, and the reason for `pv_array or None`.

    The adapter has to tell "this request had no image" from "it had an image
    entry with no data". An empty list is truthy-ambiguous at the call site;
    None is not.
    """
    audio = SimpleNamespace(
        modality="audio", data=_mm_item(input_audio_features="MEL")
    )
    image = SimpleNamespace(modality="image", data=_mm_item(pixel_values="PIX"))

    out = _gather([[audio], [image]])

    assert out["input_audio_features"] == [["MEL"], None]
    assert out["pixel_values"] == [None, ["PIX"]]


def test_a_text_only_request_keeps_every_list_aligned():
    """Length must equal the number of requests, on all four keys.

    A missing append does not raise: it shortens one list, so every later
    request is handed the previous one's features. That is a wrong
    transcription, not a crash.
    """
    audio = SimpleNamespace(
        modality="audio", data=_mm_item(input_audio_features="MEL")
    )

    out = _gather([None, [audio], None])

    for key in AUDIO_KEYS + IMAGE_KEYS:
        assert len(out[key]) == 3, f"{key} has {len(out[key])} entries for 3 requests"
    assert out["input_audio_features"] == [None, ["MEL"], None], (
        "the audio request's features must stay at its own index"
    )


def test_a_feature_with_no_data_holds_its_slot():
    """Alignment is per-feature too, not only per-request."""
    empty = SimpleNamespace(modality="audio", data=None)
    real = SimpleNamespace(
        modality="audio", data=_mm_item(input_audio_features="MEL")
    )

    out = _gather([[empty, real]])

    assert out["input_audio_features"] == [[None, "MEL"]]


def test_an_optional_audio_length_may_be_absent():
    """audio_feature_lengths is optional; its slot still has to be filled."""
    feature = SimpleNamespace(
        modality="audio", data=_mm_item(input_audio_features="MEL")
    )

    out = _gather([[feature]])

    assert out["input_audio_features"] == [["MEL"]]
    assert out["audio_feature_lengths"] == [[None]], (
        "a missing length must be None in place, not a dropped entry"
    )


def test_audio_features_do_not_land_under_the_image_keys():
    """The two branches must not be crossed.

    Both modalities go through one loop; swapping the branches would put mel
    features where the adapter looks for pixels, which fails far from here.
    """
    audio = SimpleNamespace(
        modality="audio", data=_mm_item(input_audio_features="MEL")
    )

    out = _gather([[audio]])

    for key in IMAGE_KEYS:
        assert out[key] == [None]
    assert "MEL" not in str(out["pixel_values"])
