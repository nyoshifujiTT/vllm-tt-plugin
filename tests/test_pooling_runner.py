# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Device-free unit tests for the TT pooling / embedding model runner.

``TTPoolingModelRunner`` owns only host-side batching and the vLLM output
contract: it pads the scheduled prompts, calls ``model.forward(input_ids,
attention_mask)`` and packs the returned per-request vectors into
``ModelRunnerOutput.pooler_output``. None of that needs a device, so these
tests substitute a fake model (a plain callable returning a known tensor) and
drive the runner with ``SimpleNamespace`` scheduler outputs.

Covered:
- embedding output ([B, hidden]) and cross-encoder / reranker output ([B, 1])
  both land, one host tensor per request, in ``pooler_output``;
- prompts are right-padded to the longest in the batch with a 0/1 mask;
- an empty schedule returns an empty output;
- the advertised task is ``["embed"]``;
- the worker selects the pooling runner for ``runner_type == "pooling"`` and
  short-circuits KV cache spec / available-memory / initialization for it.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_tt_plugin import pooling_runner as pooling_runner_mod
from vllm_tt_plugin.pooling_runner import TTPoolingModelRunner

# The worker module pulls in the full generative stack (scheduler, lane
# coordinator, ...), which is pinned to the canonical upstream vLLM (0.24). On
# an older/mismatched vLLM the import fails on an unrelated symbol; the two
# worker-level tests below are skipped there rather than reporting a false
# failure. The pooling-runner tests above need none of that and always run.
try:
    from vllm_tt_plugin import worker as worker_mod

    _WORKER_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - exercised only on vLLM skew
    worker_mod = None
    _WORKER_IMPORT_ERROR = exc

_requires_worker = pytest.mark.skipif(
    worker_mod is None,
    reason=f"vllm_tt_plugin.worker unavailable on this vLLM: {_WORKER_IMPORT_ERROR}",
)

# The standard-Pooler dispatch path builds a vllm.v1.pool.metadata.PoolingMetadata,
# which only exists on the canonical vLLM (v1) this plugin targets. On a missing
# or mismatched vLLM those tests are skipped; the pass-through path (no pooler)
# needs none of it and always runs.
try:
    from vllm.v1.pool.metadata import PoolingMetadata as _PoolingMetadata  # noqa: F401

    _POOL_METADATA_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - exercised only on vLLM skew
    _POOL_METADATA_IMPORT_ERROR = exc

_requires_pool_metadata = pytest.mark.skipif(
    _POOL_METADATA_IMPORT_ERROR is not None,
    reason=(
        f"vllm.v1.pool.metadata unavailable on this vLLM: {_POOL_METADATA_IMPORT_ERROR}"
    ),
)


def _bare_runner(max_num_seqs: int = 8) -> TTPoolingModelRunner:
    """A runner wired just enough for the host-side methods (no device).

    Pooling directives now live in the model's ``pooler`` (see the dispatch
    tests), so the runner itself needs no pooler-config state.
    """
    runner = TTPoolingModelRunner.__new__(TTPoolingModelRunner)
    runner.scheduler_config = SimpleNamespace(max_num_seqs=max_num_seqs)
    runner.model_config = SimpleNamespace(pooler_config=None)
    runner.max_batch_size = max_num_seqs
    runner.requests = {}
    runner.model = None
    return runner


def _req(req_id: str, prompt_token_ids, task="embed"):
    """A scheduled request. ``task`` seeds ``pooling_params.task`` (the vLLM
    per-request PoolingTask: ``"embed"`` for embeddings, ``"score"`` /
    ``"classify"`` for cross-encoder reranking). vLLM's PoolingMetadata requires
    every request to carry a task, so it defaults to ``"embed"`` here."""
    return SimpleNamespace(
        req_id=req_id,
        prompt_token_ids=list(prompt_token_ids),
        pooling_params=SimpleNamespace(task=task),
    )


def _scheduler_output(new_reqs, finished=()):
    return SimpleNamespace(
        scheduled_new_reqs=list(new_reqs),
        finished_req_ids=set(finished),
    )


class _FakeModel:
    """Returns a fixed-width vector per row so output shapes are checkable.

    Carries an identity ``pooler`` by default: every vLLM pooling model exposes
    a ``pooler`` (it is a required, non-Optional member), and the runner now
    always delegates to it, so a bare model without one is not a valid input.
    Tests that exercise a specific pooling policy override ``model.pooler``.
    """

    def __init__(self, width: int):
        self.width = width
        self.seen = None
        self.pooler = _StubIdentityPooler()

    def forward(self, input_ids, attention_mask, return_full_hidden_states=False):
        # The canonical runner always delegates to model.pooler, so it asks for
        # the un-pooled hidden with return_full_hidden_states=True. The fake
        # model returns the same fixed-width rows either way (the stub poolers
        # decide what to do with them); it records the flag so a test can assert
        # the runner requested full hidden states.
        self.seen = (input_ids, attention_mask)
        self.seen_return_full_hidden_states = return_full_hidden_states
        batch = input_ids.shape[0]
        # Row i -> vector filled with (i + 1), so per-request identity is checkable.
        out = torch.arange(1, batch + 1, dtype=torch.float32).reshape(batch, 1)
        return out.expand(batch, self.width).contiguous()


class _StubIdentityPooler:
    """Stub Pooler that returns the pooled rows unchanged (applies no directive).

    Represents the neutral case so batching / shape tests can run through the
    standard ``model.pooler`` path without asserting a particular policy.
    """

    def __init__(self):
        self.seen = None

    def __call__(self, hidden_states, pooling_metadata):
        self.seen = pooling_metadata
        return hidden_states


class _StubNormalizingPooler:
    """Stub embed Pooler: L2-normalizes each row. Records the metadata it saw.

    Stands in for vLLM's embed Pooler so tests can prove the runner delegates
    normalization to the Pooler instead of doing it itself. Returns a stacked
    [B, hidden] tensor (one valid PoolerOutput shape).
    """

    def __init__(self):
        self.seen = None

    def __call__(self, hidden_states, pooling_metadata):
        self.seen = pooling_metadata
        return torch.nn.functional.normalize(hidden_states, p=2, dim=-1)


class _StubClassifierPooler:
    """Stub ClassifierPooler: returns the raw [B, 1] logit unchanged (no
    normalization), as a cross-encoder / reranker scoring Pooler would."""

    def __init__(self):
        self.seen = None

    def __call__(self, hidden_states, pooling_metadata):
        self.seen = pooling_metadata
        return hidden_states


class _StubListPooler:
    """Stub Pooler returning a per-request list (the other valid PoolerOutput
    shape) rather than a stacked tensor."""

    def __init__(self):
        self.seen = None

    def __call__(self, hidden_states, pooling_metadata):
        self.seen = pooling_metadata
        return [row for row in hidden_states]


@_requires_pool_metadata
def test_embedding_output_lands_in_pooler_output():
    runner = _bare_runner()
    runner.model = _FakeModel(width=1024)
    sched = _scheduler_output([_req("a", [5, 6, 7]), _req("b", [8, 9])])

    out = runner.execute_model(sched)

    assert out.req_ids == ["a", "b"]
    assert out.req_id_to_index == {"a": 0, "b": 1}
    assert out.sampled_token_ids == [[], []]  # pooling emits no tokens
    assert len(out.pooler_output) == 2
    assert out.pooler_output[0].shape == (1024,)
    assert torch.allclose(out.pooler_output[0], torch.ones(1024))
    assert torch.allclose(out.pooler_output[1], torch.full((1024,), 2.0))


@_requires_pool_metadata
def test_reranker_single_logit_output():
    # Cross-encoder / reranker: forward returns [B, 1]; each request's vector is
    # a single relevance logit carried in pooler_output.
    runner = _bare_runner()
    runner.model = _FakeModel(width=1)
    sched = _scheduler_output([_req("q0", [1, 2, 3, 4]), _req("q1", [5])])

    out = runner.execute_model(sched)

    assert len(out.pooler_output) == 2
    assert out.pooler_output[0].shape == (1,)
    assert out.pooler_output[0].item() == 1.0
    assert out.pooler_output[1].item() == 2.0


@_requires_pool_metadata
def test_pooler_present_is_invoked_and_owns_normalization():
    # Standard path: when the model carries a Pooler, the runner delegates to
    # it (building a PoolingMetadata) instead of applying any directive itself.
    # The stub Pooler L2-normalizes, proving normalization is the Pooler's job.
    runner = _bare_runner()
    model = _FakeModel(width=4)
    model.pooler = _StubNormalizingPooler()
    runner.model = model
    out = runner.execute_model(_scheduler_output([_req("a", [1, 2, 3], task="embed")]))
    # Row 0 raw is ones(4) -> L2-normalized is 0.5 each; the Pooler ran.
    assert torch.allclose(out.pooler_output[0], torch.full((4,), 0.5))
    assert model.pooler.seen is not None  # pooler was actually called


@_requires_pool_metadata
def test_pooler_present_receives_correct_pooling_metadata():
    # The runner must hand the Pooler a PoolingMetadata whose prompt_lens and
    # per-request pooling_params match the batch (so the Pooler can dispatch on
    # task etc.). A classifier-style stub returns the raw hidden per request.
    runner = _bare_runner()
    model = _FakeModel(width=1)
    model.pooler = _StubClassifierPooler()
    runner.model = model
    out = runner.execute_model(
        _scheduler_output(
            [
                _req("q0", [1, 2, 3, 4], task="score"),
                _req("q1", [5], task="score"),
            ]
        )
    )
    meta = model.pooler.seen
    assert list(meta.prompt_lens) == [4, 1]
    assert [p.task for p in meta.pooling_params] == ["score", "score"]
    # Classifier stub passes the [B, 1] logit through untouched (no normalize).
    assert out.pooler_output[0].item() == 1.0
    assert out.pooler_output[1].item() == 2.0


@_requires_pool_metadata
def test_pooler_present_handles_list_pooler_output():
    # PoolerOutput may be a per-request list (not a stacked tensor); the runner
    # must accept both and emit one host tensor per request.
    runner = _bare_runner()
    model = _FakeModel(width=2)
    model.pooler = _StubListPooler()
    runner.model = model
    out = runner.execute_model(
        _scheduler_output(
            [_req("a", [1], task="embed"), _req("b", [2, 3], task="embed")]
        )
    )
    assert len(out.pooler_output) == 2
    assert torch.allclose(out.pooler_output[0], torch.ones(2))
    assert torch.allclose(out.pooler_output[1], torch.full((2,), 2.0))


@_requires_pool_metadata
def test_pool_via_model_pooler_drives_a_real_vllm_pooler():
    # Integration guard against the metadata/layout bug: feed a REAL vLLM
    # Pooler (LastPool) a flattened ``[total_tokens, hidden]`` with prompt_lens
    # != 1, and assert the runner's PoolingMetadata makes it pick each request's
    # true last token. A batched/pooled layout (or a wrong cursor) would misindex
    # here even though the stub-based tests still pass.
    from vllm.model_executor.layers.pooler.seqwise.methods import LastPool

    runner = _bare_runner()
    # Two requests, 3 and 2 real tokens, concatenated on the flat token axis.
    reqs = [_req("a", [10, 11, 12]), _req("b", [20, 21])]
    # Row i of the flat hidden is filled with i, so the last token of req a is
    # row 2 and the last token of req b is row 4.
    hidden = torch.arange(5, dtype=torch.float32).reshape(5, 1).repeat(1, 4)

    out = runner._pool_via_model_pooler(LastPool(), hidden, reqs)

    assert len(out) == 2
    assert torch.allclose(out[0], torch.full((4,), 2.0))  # req a last token
    assert torch.allclose(out[1], torch.full((4,), 4.0))  # req b last token


@_requires_pool_metadata
def test_runner_requests_full_hidden_states_from_forward():
    # The canonical runner delegates all pooling to model.pooler, so it must ask
    # forward for the un-pooled hidden with return_full_hidden_states=True (the
    # fork runner leaves it default-off and gets the pooled pass-through).
    runner = _bare_runner()
    model = _FakeModel(width=4)
    runner.model = model
    runner.execute_model(_scheduler_output([_req("a", [1, 2, 3])]))
    assert model.seen_return_full_hidden_states is True


class _FakeTTNNHidden:
    """Stand-in for a device (ttnn) hidden state whose ``.device`` is a method,
    not a ``torch.device`` (matches ttnn.Tensor). Used to prove the runner
    builds the pooling cursor without assuming a torch device."""

    def __init__(self, rows):
        self._rows = rows

    def device(self):  # ttnn.Tensor.device is a method returning a MeshDevice
        raise AssertionError("device() must not be called as a torch attribute")


class _DeviceNativePooler:
    """Stub TT-native pooler: indexes its own device hidden (ignores the torch
    cursor's device) and returns one host logit per request."""

    def __init__(self):
        self.seen = None

    def __call__(self, hidden_states, pooling_metadata):
        self.seen = pooling_metadata
        return [torch.tensor([float(r)]) for r in hidden_states._rows]


@_requires_pool_metadata
def test_pool_via_model_pooler_tolerates_non_torch_device_hidden():
    # A device-native (ttnn-like) hidden exposes no torch ``.device``; the runner
    # must still build a valid cursor (on CPU) and drive the pooler, which does
    # its own on-device gather. The embedding path (real torch tensor) is
    # exercised by test_pool_via_model_pooler_drives_a_real_vllm_pooler above.
    runner = _bare_runner()
    reqs = [_req("a", [10, 11, 12], task="score"), _req("b", [20, 21], task="score")]
    pooler = _DeviceNativePooler()
    hidden = _FakeTTNNHidden(rows=[2.0, 4.0])

    out = runner._pool_via_model_pooler(pooler, hidden, reqs)

    assert len(out) == 2
    assert out[0].item() == 2.0
    assert out[1].item() == 4.0
    # The cursor was built (prompt_lens match) despite the non-torch hidden.
    assert list(pooler.seen.prompt_lens) == [3, 2]


@_requires_pool_metadata
def test_prompts_are_right_padded_with_attention_mask():
    runner = _bare_runner()
    model = _FakeModel(width=4)
    runner.model = model
    sched = _scheduler_output([_req("a", [11, 12, 13]), _req("b", [21])])

    runner.execute_model(sched)
    input_ids, attention_mask = model.seen

    assert input_ids.shape == (2, 3)  # padded to longest prompt (3)
    assert input_ids.tolist() == [[11, 12, 13], [21, 0, 0]]
    assert attention_mask.tolist() == [[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]]


def test_empty_schedule_returns_empty_output():
    runner = _bare_runner()
    runner.model = _FakeModel(width=8)
    out = runner.execute_model(_scheduler_output([]))

    assert out.req_ids == []
    assert out.pooler_output == []
    assert out.sampled_token_ids == []


@_requires_pool_metadata
def test_finished_requests_are_evicted():
    runner = _bare_runner()
    runner.model = _FakeModel(width=2)
    runner.execute_model(_scheduler_output([_req("keep", [1]), _req("gone", [2])]))
    assert set(runner.requests) == {"keep", "gone"}

    runner.execute_model(_scheduler_output([], finished=["gone"]))
    assert "gone" not in runner.requests


def test_supported_pooling_tasks_delegate_to_model_pooler(monkeypatch):
    # Upstream contract: the supported pooling tasks come from the model's
    # Pooler, not a hard-coded runner list. ``is_pooling_model`` is patched to
    # isolate the delegation from vLLM's full duck-typed model predicate.
    monkeypatch.setattr(pooling_runner_mod, "is_pooling_model", lambda model: True)
    runner = _bare_runner()
    model = _FakeModel(width=8)
    model.pooler.get_supported_tasks = lambda: {"embed"}
    runner.model = model
    assert runner.get_supported_pooling_tasks() == ["embed"]
    assert runner.get_supported_tasks() == ("embed",)


def test_supported_pooling_tasks_report_reranker_tasks(monkeypatch):
    # A cross-encoder / reranker model advertises classify/score, proving the
    # runner no longer forces every pooling model onto the embed task.
    monkeypatch.setattr(pooling_runner_mod, "is_pooling_model", lambda model: True)
    runner = _bare_runner()
    model = _FakeModel(width=1)
    model.pooler.get_supported_tasks = lambda: {"classify", "score"}
    runner.model = model
    assert set(runner.get_supported_pooling_tasks()) == {"classify", "score"}


def test_supported_pooling_tasks_empty_for_non_pooling_model(monkeypatch):
    # A non-pooling model reports no pooling tasks (upstream returns []).
    monkeypatch.setattr(pooling_runner_mod, "is_pooling_model", lambda model: False)
    runner = _bare_runner()
    runner.model = _FakeModel(width=4)
    assert runner.get_supported_pooling_tasks() == []


def test_warmup_is_noop():
    runner = _bare_runner()
    assert runner.warmup_model() is None


def _pooling_vllm_config(runner_type: str):
    """Minimal VllmConfig stub carrying only what init_device's runner
    selection reads."""
    return SimpleNamespace(
        model_config=SimpleNamespace(runner_type=runner_type),
        lora_config=None,
        load_config=None,
        parallel_config=SimpleNamespace(),
        scheduler_config=SimpleNamespace(max_num_seqs=8),
        device_config=SimpleNamespace(device=None),
    )


@_requires_worker
def test_worker_selects_pooling_runner_for_pooling_models():
    # Exercise the same class-selection logic init_device uses, without opening
    # a device: pooling models must map to TTPoolingModelRunner and everything
    # else to the generative TTModelRunner.
    for runner_type, expected in (
        ("pooling", worker_mod.TTPoolingModelRunner),
        ("generate", worker_mod.TTModelRunner),
    ):
        cfg = _pooling_vllm_config(runner_type)
        chosen = (
            worker_mod.TTPoolingModelRunner
            if cfg.model_config.runner_type == "pooling"
            else worker_mod.TTModelRunner
        )
        assert chosen is expected


@_requires_worker
def test_worker_kv_methods_short_circuit_for_pooling():
    TTWorker = worker_mod.TTWorker

    worker = TTWorker.__new__(TTWorker)
    worker.model_runner = _bare_runner()  # a TTPoolingModelRunner instance

    assert TTWorker.get_kv_cache_spec(worker) == {}
    assert TTWorker.determine_available_memory(worker) == 0
    # initialize_from_config must not touch the (KV-less) pooling runner.
    worker.model_runner.initialize_kv_cache = MagicMock()
    TTWorker.initialize_from_config(worker, kv_cache_config=object())
    worker.model_runner.initialize_kv_cache.assert_not_called()


@_requires_worker
def test_worker_sample_tokens_rejected_for_pooling():
    TTWorker = worker_mod.TTWorker
    worker = TTWorker.__new__(TTWorker)
    worker.is_driver_worker = True
    worker.model_runner = _bare_runner()  # a TTPoolingModelRunner instance

    # Pooling completes in execute_model, so the engine never calls
    # sample_tokens; if something does, it fails clearly rather than as an
    # opaque AttributeError on the sampling-free pooling runner.
    with pytest.raises(RuntimeError, match="not applicable to pooling"):
        TTWorker.sample_tokens(worker, grammar_output=None)
