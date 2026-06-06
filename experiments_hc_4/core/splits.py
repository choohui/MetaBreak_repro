from __future__ import annotations

import numpy as np


def _sizes(n: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    if n >= 3:
        val = max(1, int(round(n * ratios[1])))
        test = max(1, int(round(n * ratios[2])))
        train = max(1, n - val - test)
        while train + val + test > n:
            train = max(1, train - 1)
        return train, val, test
    return n, 0, 0


def stratified_group_split(group_to_letter: dict[int, str], ratios=(0.6, 0.2, 0.2),
                           seed: int = 0) -> dict[int, str]:
    rng = np.random.default_rng(seed)
    out: dict[int, str] = {}
    by_letter: dict[str, list[int]] = {}
    for gid, letter in group_to_letter.items():
        by_letter.setdefault(letter, []).append(int(gid))
    for letter, gids in by_letter.items():
        gids = list(gids)
        rng.shuffle(gids)
        tr, va, te = _sizes(len(gids), ratios)
        for gid in gids[:tr]:
            out[gid] = "train"
        for gid in gids[tr:tr + va]:
            out[gid] = "val"
        for gid in gids[tr + va:tr + va + te]:
            out[gid] = "test"
    return out

