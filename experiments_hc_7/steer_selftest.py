"""Model-free unit test for core/steer.py — validates the residual-stream hook
math AND the transformers-5 bare-tensor vs 4.x tuple return handling, without
loading a real model.

Run:  python -m experiments_hc_7.steer_selftest
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch import nn

from experiments_hc_7.core.steer import steer, can_steer


class _IdentityBlock(nn.Module):
    """Returns a BARE tensor (transformers >= 5 LlamaDecoderLayer behaviour)."""
    def forward(self, x):
        return x


class _TupleBlock(nn.Module):
    """Returns a (hidden, aux) TUPLE (transformers 4.x behaviour)."""
    def forward(self, x):
        return (x, "aux")


def _toy_lm(block):
    base = SimpleNamespace(layers=nn.ModuleList([_IdentityBlock(), block]))
    top = SimpleNamespace(model=base)
    return SimpleNamespace(model=top, is_mock=False)


def _check(block, is_tuple: bool):
    lm = _toy_lm(block)
    assert can_steer(lm), "can_steer should be True for a real-shaped stack"
    D, T = 8, 5
    x = torch.zeros(1, T, D)
    unit_v = torch.zeros(D); unit_v[0] = 1.0     # already unit
    coef = 3.0
    mask = torch.zeros(T, dtype=torch.bool); mask[2] = True

    layer = lm.model.model.layers[1]
    with steer(lm, 1, unit_v, coef, token_mask=mask):
        out = layer(x)
        h = out[0] if is_tuple else out
        # masked position shifted by coef on dim 0; others untouched.
        assert abs(float(h[0, 2, 0]) - 3.0) < 1e-6, f"masked shift wrong: {float(h[0,2,0])}"
        assert abs(float(h[0, 1, 0]) - 0.0) < 1e-6, "unmasked position must be untouched"
        if is_tuple:
            assert out[1] == "aux", "tuple tail must be preserved"

    # handle removed -> identity again
    out2 = layer(x)
    h2 = out2[0] if is_tuple else out2
    assert float(h2[0, 2, 0]) == 0.0, "hook not removed on context exit"

    # no-mask path steers ALL positions
    with steer(lm, 1, unit_v, coef, token_mask=None):
        out3 = layer(x)
        h3 = out3[0] if is_tuple else out3
        assert abs(float(h3[0, 0, 0]) - 3.0) < 1e-6 and abs(float(h3[0, 4, 0]) - 3.0) < 1e-6, \
            "no-mask must steer every position"


def main() -> int:
    _check(_IdentityBlock(), is_tuple=False)   # transformers >= 5 bare tensor
    _check(_TupleBlock(), is_tuple=True)        # transformers 4.x tuple
    # mock model cannot steer
    mock = SimpleNamespace(model=SimpleNamespace(), is_mock=True)
    assert not can_steer(mock), "mock must not be steerable"
    print("[steer_selftest] OK -- bare-tensor + tuple paths, mask, removal, mock guard.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
