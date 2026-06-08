from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from .model import LoadedModel


def _stable_seed(ids: list[int]) -> int:
    raw = ",".join(str(int(x)) for x in ids).encode("utf-8")
    return int(hashlib.md5(raw).hexdigest()[:8], 16)


def _mock_log_probs(ids: list[int]) -> np.ndarray:
    rng = np.random.default_rng(_stable_seed(ids))
    logits = rng.normal(0.0, 0.2, size=64)
    attackish = any(int(x) in {128009, 128006, 128007, 100489, 5809, 5810} for x in ids)
    logits[1 if attackish else 0] += 4.0
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    return np.log(probs + 1e-12)


def first_token_log_probs(lm: LoadedModel, input_ids: list[int]) -> np.ndarray:
    if lm.is_mock:
        return _mock_log_probs(input_ids)
    import torch

    ids = torch.tensor([list(int(x) for x in input_ids)], device=lm.device)
    with torch.no_grad():
        out = lm.model(input_ids=ids, use_cache=False, return_dict=True)
        lp = torch.log_softmax(out.logits[0, -1].detach().float().cpu(), dim=-1)
    return lp.numpy()


def first_token_metrics(lm: LoadedModel, before_ids: list[int], after_ids: list[int]) -> dict:
    p = first_token_log_probs(lm, before_ids)
    q = first_token_log_probs(lm, after_ids)
    prob = np.exp(p)
    kl = float(np.sum(prob * (p - q)))
    return {
        "top1_match": bool(int(np.argmax(p)) == int(np.argmax(q))),
        "kl": round(kl, 6),
        "before_top1": int(np.argmax(p)),
        "after_top1": int(np.argmax(q)),
    }


def decoder_layers(model: Any) -> list[Any]:
    base = getattr(model, "model", model)
    layers = getattr(base, "layers", None) or getattr(model, "layers", None)
    return list(layers or [])


def generate_with_steering(
    lm: LoadedModel,
    input_ids: list[int],
    flagged_positions: set[int],
    vectors: dict[str, np.ndarray],
    steer_layers: list[int],
    mode: str,
    alpha: float,
    max_new_tokens: int,
) -> str:
    if lm.is_mock:
        if flagged_positions and alpha > 0.0 and mode in {"add", "project_out", "pull_to_benign"}:
            return "I cannot assist with that request."
        return lm.generate_from_ids(input_ids, max_new_tokens=max_new_tokens)

    import torch

    layers = decoder_layers(lm.model)
    prompt_len = len(input_ids)
    positions = sorted(int(p) for p in flagged_positions if 0 <= int(p) < prompt_len)
    if not positions or not steer_layers or alpha == 0.0:
        return lm.generate_from_ids(input_ids, max_new_tokens=max_new_tokens)

    directions = torch.tensor(vectors["directions"], device=lm.device)
    benign_mu = torch.tensor(vectors["benign_mu"], device=lm.device)
    gaps = torch.tensor(vectors["gaps"], device=lm.device)
    handles = []

    def make_hook(hidden_layer: int):
        def hook(_module, _inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            if hs.ndim != 3 or hs.shape[0] != 1 or hs.shape[1] != prompt_len:
                return out
            valid = [p for p in positions if p < hs.shape[1]]
            if not valid:
                return out
            new_hs = hs.clone()
            d = directions[hidden_layer].to(dtype=new_hs.dtype)
            mu = benign_mu[hidden_layer].to(dtype=new_hs.dtype)
            gap = gaps[hidden_layer].to(dtype=new_hs.dtype)
            cur = new_hs[0, valid, :]
            if mode == "add":
                new_hs[0, valid, :] = cur + float(alpha) * gap * d
            elif mode == "project_out":
                centered = cur - mu
                proj = centered @ d
                new_hs[0, valid, :] = cur - float(alpha) * proj.unsqueeze(-1) * d
            elif mode == "pull_to_benign":
                new_hs[0, valid, :] = (1.0 - float(alpha)) * cur + float(alpha) * mu
            else:
                return out
            if isinstance(out, tuple):
                return (new_hs,) + out[1:]
            return new_hs

        return hook

    for hidden_layer in steer_layers:
        # hidden layer 0 is the embedding output; decoder layer i writes hidden i+1.
        if hidden_layer <= 0:
            continue
        idx = hidden_layer - 1
        if idx < len(layers):
            handles.append(layers[idx].register_forward_hook(make_hook(hidden_layer)))

    try:
        ids = torch.tensor([list(int(x) for x in input_ids)], device=lm.device)
        eos = getattr(lm.tokenizer, "eos_token_id", None)
        with torch.no_grad():
            out = lm.model.generate(
                ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=eos,
            )
        return lm.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    finally:
        for handle in handles:
            handle.remove()
