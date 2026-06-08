"""Residual-stream activation steering via a post-forward hook on one decoder block.

Steering = add a fixed direction ``coef * unit_v`` to the residual stream that is
the OUTPUT of decoder block ``block_idx`` during EVERY forward pass that happens
while the context manager is active (the prompt pass AND every cached
single-token generation step). This is the causal counterpart to hc_4_claude's
``cos_to_attack`` read-out: the steering direction is that scalarizer's fitted
attack-centroid (see stages/00_build_vectors.py).

Layer convention (verified against core/capture.py): ``hidden_states[L]`` is the
output of ``model.model.layers[L-1]``. So to steer the residual stream that the
scalarizer measured at hidden-layer ``L`` (e.g. 32 for pos0, 6 for pos1) the hook
attaches to block ``block_idx = L - 1``.

transformers compatibility: in transformers >= 5 ``LlamaDecoderLayer.forward``
returns a BARE ``torch.Tensor``; in 4.x it returned a ``(hidden, ...)`` tuple.
The hook handles BOTH shapes (it never assumes ``out[0]``).

Mock models (``lm.is_mock``) have no real decoder blocks, so steering cannot
attach — callers must check :func:`can_steer` and treat mock runs as
``steering_observable=False`` (the smoke test only checks plumbing).
"""

from __future__ import annotations

from contextlib import contextmanager

import torch


def decoder_layers(lm):
    """The decoder-block ``ModuleList`` (``model.model.layers`` for Llama/Qwen)."""
    base = getattr(lm.model, "model", lm.model)
    layers = getattr(base, "layers", None)
    if layers is None:
        raise RuntimeError("Could not locate decoder layers for steering hook.")
    return layers


def can_steer(lm) -> bool:
    """True iff a real decoder-block stack is present (False for mock models)."""
    if getattr(lm, "is_mock", False):
        return False
    try:
        _ = decoder_layers(lm)
        return True
    except RuntimeError:
        return False


def as_unit(vec) -> torch.Tensor:
    """Return ``vec`` as a 1-D float32 unit vector (raises on a zero vector)."""
    t = torch.as_tensor(vec, dtype=torch.float32).flatten()
    n = torch.linalg.vector_norm(t)
    if float(n) == 0.0:
        raise ValueError("Cannot unit-normalize a zero steering vector.")
    return t / n


@contextmanager
def steer(lm, block_idx: int, unit_v, coef: float, token_mask=None):
    """Add ``coef * unit_v`` to block ``block_idx``'s residual-stream output.

    ``unit_v``      1-D direction (should already be unit norm; not re-normalized).
    ``coef``        signed scale. Negative = steer AWAY from attack (defense);
                    positive = amplify. Typically ``alpha * rho[L]`` so ``coef`` is
                    a fraction of the layer's typical residual norm.
    ``token_mask``  optional bool tensor ``[seq]``; when given, only those prompt
                    positions are nudged (the surgical "attack_slot" arm). During
                    cached generation each step has seq==1, so a prompt-length mask
                    naturally applies only on the prompt pass. ``None`` steers every
                    position of every pass (the "all" arm).
    """
    layer = decoder_layers(lm)[block_idx]
    v = torch.as_tensor(unit_v, dtype=torch.float32).flatten()
    mask = None if token_mask is None else torch.as_tensor(token_mask, dtype=torch.bool).flatten()

    def hook(_module, _inp, out):
        is_tuple = isinstance(out, tuple)
        h = out[0] if is_tuple else out                      # [B, T, D]
        add = (coef * v).to(device=h.device, dtype=h.dtype)  # [D]
        if mask is None or h.shape[1] != mask.shape[0]:
            # No mask, or shape mismatch (e.g. a 1-token generation step under a
            # prompt-length mask) -> steer all positions of this pass. For the
            # surgical arm the mismatch path means generation steps are NOT nudged,
            # only the prompt pass (where lengths match) is.
            if mask is None:
                h = h + add
            # else: prompt-length mask but T!=len(mask) -> skip (generation step).
        else:
            m = mask.to(h.device)
            delta = torch.zeros_like(h)
            delta[:, m, :] = add
            h = h + delta
        return (h,) + tuple(out[1:]) if is_tuple else h

    handle = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
