"""Single-forward-pass capture of per-token hidden states (all layers).

Returns a ``[n_layers+1, seq, hidden_dim]`` float32 cube (index 0 = embedding
layer). That is all four defenses need:
  * ours        — per-token projection onto the diff-means direction,
  * JBShield     — pooled projection onto toxic / jailbreak concept directions,
  * GUARD-SLM    — last-token activation fed to a per-layer SVM.

Mock models are routed to :mod:`core.mock` so the smoke test needs no weights.
"""

from __future__ import annotations

import numpy as np
import torch

from .model import LoadedModel, chat_prompt_ids


def capture_hidden_ids(lm: LoadedModel, input_ids: list[int]) -> tuple[list[int], np.ndarray]:
    """Forward pass over a prebuilt id sequence -> (ids, hidden[L+1, seq, dim])."""
    if lm.is_mock:
        from .mock import mock_capture_hidden
        return mock_capture_hidden(lm, [int(x) for x in input_ids])
    ids = [int(x) for x in input_ids]
    t = torch.tensor([ids], device=lm.device)
    with torch.no_grad():
        out = lm.model(input_ids=t, output_hidden_states=True,
                       use_cache=False, return_dict=True)
    hidden = np.stack([h[0].detach().cpu().float().numpy() for h in out.hidden_states], axis=0)
    return ids, hidden.astype(np.float32)


def capture_hidden(lm: LoadedModel, text: str) -> tuple[list[int], np.ndarray]:
    """Forward pass over a user-content string -> (ids, hidden[L+1, seq, dim])."""
    if lm.is_mock:
        from .mock import mock_capture_hidden_text
        return mock_capture_hidden_text(lm, text)
    return capture_hidden_ids(lm, chat_prompt_ids(lm, text))


def last_token(hidden: np.ndarray) -> np.ndarray:
    """[L+1, seq, dim] -> [L+1, dim] (final-position activation per layer)."""
    return hidden[:, -1, :]


def mean_pool(hidden: np.ndarray) -> np.ndarray:
    """[L+1, seq, dim] -> [L+1, dim] (mean over the sequence per layer)."""
    return hidden.mean(axis=1)
