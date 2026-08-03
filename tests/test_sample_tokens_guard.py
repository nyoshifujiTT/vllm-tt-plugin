# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""Host-only unit test for the empty-deque guard in ``sample_tokens``.

When a preceding ``execute_model`` raises before enqueuing a pending forward,
``sample_tokens`` must return ``None`` (so EngineCore re-raises the real
exception) instead of raising ``IndexError`` from ``popleft`` on an empty
deque. Mirrors tenstorrent/vllm PR #446.
"""

from collections import deque
from types import SimpleNamespace

from vllm_tt_plugin.model_runner import TTModelRunner


def test_sample_tokens_returns_none_on_empty_deque():
    fake_self = SimpleNamespace(_pending_samples=deque())
    assert TTModelRunner.sample_tokens(fake_self, None) is None


def test_sample_tokens_runs_pending_when_present():
    called = {}

    def finish(grammar_output):
        called["grammar"] = grammar_output
        return "runner_output"

    fake_self = SimpleNamespace(_pending_samples=deque([finish]))
    result = TTModelRunner.sample_tokens(fake_self, "grammar-obj")
    assert result == "runner_output"
    assert called["grammar"] == "grammar-obj"
    assert len(fake_self._pending_samples) == 0
