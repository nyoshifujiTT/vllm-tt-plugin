# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Model runner for TT pooling / embedding models.

Pooling models (text embedding and cross-encoder / reranker scoring) are far
simpler than generative models: they have no KV cache, no page tables, no
prefill/decode split and no sampling. Each request is a single forward pass
that turns tokenized input into a per-request vector (an embedding, or a
single relevance logit for a cross-encoder), returned to vLLM in the
``pooler_output`` field of :class:`ModelRunnerOutput`.

The generative :class:`~vllm_tt_plugin.model_runner.TTModelRunner` carries a
large amount of machinery (KV allocation, lane/DP orchestration, device
sampling, structured output) that pooling models neither need nor exercise.
Rather than thread ``runner_type == "pooling"`` branches through all of it,
the worker selects this dedicated runner for pooling models.

The TT model is a plain ``nn.Module`` whose ``forward(input_ids,
attention_mask)`` runs the encoder backbone on device and returns a host
tensor shaped ``[batch, hidden]`` (embedding) or ``[batch, 1]`` (reranker
logit); this runner only owns host-side batching and the vLLM output contract,
so it can be exercised without a device by substituting a fake model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import ttnn
from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces_base import is_pooling_model
from vllm.tasks import PoolingTask, SupportedTask
from vllm.v1.outputs import ModelRunnerOutput

from vllm_tt_plugin.loader import TTModelLoader
from vllm_tt_plugin.logger import init_tt_logger

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_tt_logger(__name__)


class TTPoolingModelRunner:
    """Simplified model runner for TT pooling / embedding models.

    Constructed with the same signature as
    :class:`~vllm_tt_plugin.model_runner.TTModelRunner` so the worker can pick
    either class without special-casing the call site. The generative-only
    arguments (``trace_mode``, ``enable_model_warmup``, ``num_devices``) are
    accepted and ignored: pooling runs a single un-traced forward per batch and
    does no device warmup.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        mesh_device: ttnn.MeshDevice,
        trace_mode: str,
        enable_model_warmup: bool,
        num_devices: int,
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.mesh_device = mesh_device
        self.num_devices = num_devices
        # Accepted for signature parity with TTModelRunner; unused for pooling.
        self.trace_mode = trace_mode
        self.enable_model_warmup = enable_model_warmup
        logger.info("TTPoolingModelRunner initialized (trace/warmup ignored)")
        # req_id -> per-request bookkeeping (kept minimal for pooling).
        self.requests: dict[str, dict] = {}
        self.max_batch_size = self.scheduler_config.max_num_seqs
        # Model is set by load_model().
        self.model: nn.Module | None = None

    def load_model(self) -> None:
        """Load the pooling model via the shared TT model loader."""
        if self.model is not None:
            logger.info("Pooling model already loaded, skipping")
            return
        logger.info("Loading TT pooling model...")
        loader = TTModelLoader(self.load_config)
        self.model = loader.load_model(
            vllm_config=self.vllm_config, model_config=self.model_config
        )

    def get_model(self) -> nn.Module:
        assert self.model is not None, "Model not loaded. Call load_model() first."
        return self.model

    def warmup_model(self) -> None:
        """Let the pooling model compile its device shapes before serving.

        Pooling runs a single un-traced forward, so there is no trace to capture
        here, but there is still per-shape kernel compilation: a shape first seen
        at request time compiles then, and the requesting client waits for it
        (measured tens of seconds on a TT encoder). Delegate to the model's
        ``warmup_model_prefill`` -- the same hook ``TTModelRunner`` uses for the
        decoder models -- so those compiles happen during startup instead.

        Models that do not implement the hook keep the previous no-op behaviour.
        """
        warmup_prefill = getattr(self.get_model(), "warmup_model_prefill", None)
        if warmup_prefill is None:
            logger.info("Pooling model has no warmup_model_prefill; skipping warmup")
            return
        logger.info("Warming up pooling model device shapes...")
        warmup_prefill(kv_cache=None, enable_trace=False)
        logger.info("Pooling model warmup complete")

    def _prepare_model_inputs(
        self, scheduler_output: SchedulerOutput
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, list]:
        """Build padded ``(tokens, attention_mask, req_data_list)`` for one batch.

        Pooling requests are single-shot prefills: every scheduled request is a
        new request whose full prompt is embedded in one pass. Sequences are
        right-padded to the batch's longest prompt with a 0/1 attention mask so
        the model can ignore pad positions.

        TODO(tt-quirk, upstream-conformance): This builds a *batched*
        ``[batch, max_seq_len]`` token tensor plus an explicit ``attention_mask``
        and right-pads every sequence. That is NOT how upstream vLLM feeds
        encoder / pooling models. Upstream flattens the batch into a single
        ``[total_tokens]`` ``input_ids`` (+ ``positions``), passes NO
        ``attention_mask``, and conveys per-request sequence boundaries through
        ``AttentionMetadata.seq_lens`` so the attention kernel does
        variable-length ("packed"/"varlen") attention -- each request attends
        only to its own tokens, with no padding. See the encoder-model forward
        signature ``forward(input_ids, positions, ...)`` (no mask) in vLLM
        v0.24.0:
        https://github.com/vllm-project/vllm/blob/v0.24.0/vllm/model_executor/models/bert.py

        The proper end state is: this runner flattens inputs upstream-style and
        the TT model consumes ``[total_tokens]`` + seq boundaries. We do NOT do
        that yet because the TT XLM-RoBERTa encoder
        (``models/demos/wormhole/bge_m3/tt/attention.py``) is built around a
        fixed ``[B, 1, S, D]`` dense SDPA with a rank-4 ``[B, 1, 1, S]`` mask and
        128-aligned seq_len; making it consume a flat/varlen layout is a full
        rewrite of the attention core, and that encoder is SHARED with the bge-m3
        embedding model, so the change would have to be co-designed with the
        embedding owner. Until then, this batched+mask input is a deliberate,
        localized TT special case kept here so the model runs; it is the only
        place the runner departs from upstream's flat-token contract.
        """
        scheduled_reqs = scheduler_output.scheduled_new_reqs
        if not scheduled_reqs:
            return None, None, []

        token_ids_list = []
        max_seq_len = 0
        req_data_list = []
        for req_data in scheduled_reqs:
            prompt_token_ids = req_data.prompt_token_ids
            self.requests[req_data.req_id] = {
                "prompt_token_ids": prompt_token_ids,
                "pooling_params": req_data.pooling_params,
            }
            max_seq_len = max(max_seq_len, len(prompt_token_ids))
            token_ids_list.append(prompt_token_ids)
            req_data_list.append(req_data)

        batch_size = len(token_ids_list)
        tokens = torch.zeros((batch_size, max_seq_len), dtype=torch.int64)
        attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.float32)
        for i, token_ids in enumerate(token_ids_list):
            seq_len = len(token_ids)
            tokens[i, :seq_len] = torch.tensor(token_ids, dtype=torch.int64)
            attention_mask[i, :seq_len] = 1.0
        return tokens, attention_mask, req_data_list

    @torch.no_grad()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
    ) -> ModelRunnerOutput:
        """Run one pooling forward and return embeddings in ``pooler_output``.

        Unlike the generative runner, this returns a completed
        :class:`ModelRunnerOutput` directly (there is no deferred
        ``sample_tokens`` step for pooling models).
        """
        tokens, attention_mask, req_data_list = self._prepare_model_inputs(
            scheduler_output
        )
        # Evict finished requests regardless of whether anything is scheduled
        # this step (a step can finish requests while scheduling no new ones).
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)

        if tokens is None:
            return self._empty_output()

        assert self.model is not None, "Model not loaded. Call load_model() first."
        # Two-track forward contract (shared with the fork's pooling runner):
        # a pooling model's ``forward`` defaults to returning its already-pooled
        # output (an embedding, or a reranker logit) so the fork runner -- which
        # never sets this flag -- keeps its byte-for-byte pass-through behaviour.
        # This canonical runner always delegates pooling to ``model.pooler``, so
        # it needs the *un-pooled* hidden states and asks for them explicitly
        # with ``return_full_hidden_states=True``. Every pooling model on this
        # runner accepts the flag (default off); a model that returns device
        # (ttnn) hidden here keeps pooling on device (see _pool_via_model_pooler).
        outputs = self.model.forward(
            input_ids=tokens,
            attention_mask=attention_mask,
            return_full_hidden_states=True,
        )
        # Pooling contract: pooling directives (normalize, activation, pooling
        # type, ...) are the responsibility of the model's ``pooler`` component,
        # exactly as in upstream vLLM. The runner emits hidden states and
        # delegates all pooling to ``model.pooler(hidden_states,
        # pooling_metadata)``; it must never re-implement those directives.
        #
        # ``pooler`` is a required (non-Optional) member of every vLLM pooling
        # model (``VllmModelForPooling.pooler: Pooler``), so upstream's pooling
        # runner calls it unconditionally and so do we -- no ``pooler is None``
        # fallback. An embed model carries a normalizing Pooler; a cross-encoder
        # / reranker carries a ClassifierPooler that keeps the raw logit.
        pooler = getattr(self.model, "pooler", None)
        assert pooler is not None, (
            "Pooling model exposes no pooler. Every vLLM pooling model must "
            "define `pooler` (VllmModelForPooling.pooler: Pooler); the runner "
            "delegates all pooling directives to it."
        )
        pooler_output = self._pool_via_model_pooler(pooler, outputs, req_data_list)

        req_ids = [req_data.req_id for req_data in req_data_list]
        req_id_to_index = {req_id: i for i, req_id in enumerate(req_ids)}
        sampled_token_ids: list[list[int]] = [[] for _ in req_data_list]

        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index=req_id_to_index,
            sampled_token_ids=sampled_token_ids,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=pooler_output,
        )

    @staticmethod
    def _empty_output() -> ModelRunnerOutput:
        return ModelRunnerOutput(
            req_ids=[],
            req_id_to_index={},
            sampled_token_ids=[],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        )

    def _pool_via_model_pooler(
        self, pooler, hidden_states: torch.Tensor, req_data_list: list
    ) -> list:
        """Delegate pooling to the model's ``pooler``, vLLM-standard style.

        Mirrors upstream ``GPUModelRunner._pool``: build a
        :class:`~vllm.v1.pool.metadata.PoolingMetadata` for the batch, call
        ``pooler(hidden_states=..., pooling_metadata=...)`` and normalize the
        :data:`~vllm.v1.outputs.PoolerOutput` (a tensor, or a per-request list)
        into ``pooler_output`` (one host tensor per request).

        The pooling task, normalize / activation policy and pooling type all
        live in the Pooler (an embed Pooler for embeddings, a ClassifierPooler
        for cross-encoder / reranker scoring), so the runner stays model- and
        directive-agnostic.

        Device tolerance: the standard cursor (``first/last_token_indices``) is a
        torch index tensor placed on ``hidden_states.device``, which assumes a
        torch tensor. A TT-native pooler may instead receive the model's device
        (ttnn) hidden and index it on device itself (via ``ttnn.slice``); such a
        tensor exposes no torch ``.device``. So the cursor device is taken from
        ``hidden_states.device`` only when it is a real ``torch.device`` (the
        embedding path, byte-for-byte unchanged) and falls back to CPU
        otherwise, where the small index tensors are harmless to a pooler that
        does its own on-device gather.

        Hidden-states layout contract (upstream-standard): ``hidden_states`` is
        the flattened, unpadded ``[total_tokens, hidden]`` tensor -- every
        scheduled request's real tokens concatenated in request order, with no
        batch dimension and no padding. The pooling cursor built here indexes
        that flat token axis (``LastPool`` -> last_token_indices, ``CLSPool`` ->
        first_token_indices, ``MeanPool`` -> per-request token spans), exactly
        as the standard vLLM Pooler expects. ``model.forward`` is therefore
        required to return this flat layout; a batched ``[B, S, D]`` (or already
        pooled ``[B, hidden]``) tensor would be misindexed as if the batch axis
        were the token axis.
        """
        import numpy as np

        # Imported lazily: this metadata module only exists on the canonical
        # vLLM (v1) this plugin targets; the runner's device-free pass-through
        # path must still import on older/mismatched vLLM.
        from vllm.v1.pool.metadata import PoolingMetadata, PoolingStates

        num_reqs = len(req_data_list)
        prompt_lens_list = [len(req.prompt_token_ids) for req in req_data_list]
        prompt_lens = torch.tensor(prompt_lens_list, dtype=torch.int64)

        # Single-shot prefill pooling: every scheduled request is fully
        # prefilled in one pass, so num_scheduled_tokens == seq_len == prompt
        # length for each request.
        pooling_metadata = PoolingMetadata(
            prompt_lens=prompt_lens,
            prompt_token_ids=None,
            prompt_token_ids_cpu=None,
            pooling_params=[req.pooling_params for req in req_data_list],
            pooling_states=[PoolingStates() for _ in range(num_reqs)],
        )
        # See "Device tolerance" above: use the tensor's torch device when it has
        # one (embedding path unchanged), else build the cursor on CPU for a
        # device-native (ttnn) hidden whose pooler indexes on device itself.
        hidden_device = getattr(hidden_states, "device", None)
        cursor_device = (
            hidden_device
            if isinstance(hidden_device, torch.device)
            else torch.device("cpu")
        )
        pooling_metadata.build_pooling_cursor(
            np.array(prompt_lens_list, dtype=np.int64),
            seq_lens_cpu=prompt_lens,
            device=cursor_device,
        )

        raw_pooler_output = pooler(
            hidden_states=hidden_states, pooling_metadata=pooling_metadata
        )
        # PoolerOutput = torch.Tensor | list[torch.Tensor | None]. Pooler output
        # is the small final result (embedding / logit) and is always host
        # torch; ``.cpu()`` is a no-op if it is already on host.
        if isinstance(raw_pooler_output, torch.Tensor):
            return [raw_pooler_output[i].cpu() for i in range(num_reqs)]
        return [out.cpu() if out is not None else None for out in raw_pooler_output]

    def get_supported_pooling_tasks(self) -> list[PoolingTask]:
        """Advertise the pooling tasks the loaded model actually supports.

        Mirrors upstream ``GPUModelRunner.get_supported_pooling_tasks``: defer to
        the model's ``pooler`` (``model.pooler.get_supported_tasks()``) instead
        of hard-coding a task list in the runner. The supported tasks are a
        property of the model's Pooler -- an embed Pooler reports ``embed``, a
        cross-encoder / reranker ClassifierPooler reports ``classify`` (the
        cross-encoder pooling task, which is what routes ``/score`` and
        ``/rerank``) -- so the runner must not second-guess it. A non-pooling
        model reports nothing.
        """
        model = self.get_model()
        if not is_pooling_model(model):
            return []
        return list(model.pooler.get_supported_tasks())

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return tuple(self.get_supported_pooling_tasks())
