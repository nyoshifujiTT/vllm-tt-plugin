# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

import inspect

from torch import nn
from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.model_loader import BaseModelLoader
from vllm.model_executor.model_loader.utils import get_model_architecture

from vllm_tt_plugin.config import (
    get_tt_config,
    get_tt_data_parallel_size,
    get_tt_max_batch_size,
)
from vllm_tt_plugin.logger import init_tt_logger

logger = init_tt_logger(__name__)


class TTModelLoader(BaseModelLoader):
    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig
    ) -> nn.Module:
        """Load a model with the given configurations."""

        device_config = vllm_config.device_config
        model_class, _ = get_model_architecture(model_config)
        optimizations = get_tt_config(vllm_config).get("optimizations", None)
        if optimizations is not None:
            assert optimizations in [
                "performance",
                "accuracy",
            ], f"""Invalid optimizations configuration `{optimizations}`,
            allowed values are 'performance' or 'accuracy'"""

        tt_data_parallel = get_tt_data_parallel_size(vllm_config)
        max_batch_size = get_tt_max_batch_size(vllm_config)

        # Some pooling wrappers also want the resolved VllmConfig handed to them
        # directly: their Pooler is built from vllm_config.model_config
        # .pooler_config, which carries the pooling type and activation vLLM
        # derived from the checkpoint plus any --override-pooler-config. Pass it
        # only to wrappers that accept it, so generative models -- whose
        # initialize_vllm_model has a fixed signature -- are unaffected.
        extra_kwargs = {}
        try:
            init_params = inspect.signature(
                model_class.initialize_vllm_model
            ).parameters
        except (TypeError, ValueError):
            init_params = {}
        if "vllm_config" in init_params:
            extra_kwargs["vllm_config"] = vllm_config

        # Build the model inside vLLM's config context, as upstream's loader
        # does (model_loader/utils.py::initialize_model wraps construction in
        # set_current_vllm_config). Model code is entitled to read the ambient
        # config while it is being constructed -- upstream's own layers do, e.g.
        # pooler_for_classify() calls get_current_vllm_config() to resolve a
        # classification head's activation instead of taking it as an argument.
        # Without this context such a lookup finds nothing, and the TT path could
        # not pass the config explicitly either: initialize_vllm_model does not
        # take vllm_config on most TT models.
        with set_current_vllm_config(vllm_config):
            return model_class.initialize_vllm_model(
                model_config.hf_config,
                device_config.device,
                max_batch_size,
                max_seq_len=model_config.max_model_len,
                tt_data_parallel=tt_data_parallel,
                optimizations=optimizations,
                **extra_kwargs,
            )

    def download_model(self, model_config: ModelConfig) -> None:
        """Download a model so that it can be immediately loaded."""
        raise NotImplementedError

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        """Load weights into a model. This standalone API allows
        inplace weights loading for an already-initialized model"""
        raise NotImplementedError
