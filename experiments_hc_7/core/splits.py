"""Prompt-level splits used by hc_3."""

from __future__ import annotations

import numpy as np


def holdout_mask(rows: list[dict], seed: int = 0, test_frac: float = 1 / 3) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic prompt-level train/test split."""
    samples = np.array(sorted({int(r["sample_index"]) for r in rows}), dtype=int)
    if len(samples) < 6:
        mask = np.ones(len(rows), dtype=bool)
        return mask, mask
    rng = np.random.default_rng(seed)
    shuffled = samples.copy()
    rng.shuffle(shuffled)
    n_test = max(1, int(round(len(shuffled) * test_frac)))
    test_samples = set(int(x) for x in shuffled[:n_test])
    is_test = np.array([int(r["sample_index"]) in test_samples for r in rows], dtype=bool)
    is_train = ~is_test
    return is_train, is_test


def prompt_groups(rows: list[dict]) -> np.ndarray:
    return np.array([int(r["sample_index"]) for r in rows], dtype=int)

