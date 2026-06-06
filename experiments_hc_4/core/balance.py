from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from .labels import ALL_LETTERS, VARIANT_TO_LETTER
from .splits import stratified_group_split


def primary_letter_for_prompt(rows: list[dict]) -> dict[int, str]:
    by_group: dict[int, list[str]] = defaultdict(list)
    for r in rows:
        by_group[int(r["sample_index"])].append(str(r["letter"]))
    out = {}
    for gid, letters in by_group.items():
        for letter in "BCDEFG":
            if letter in letters:
                out[gid] = letter
                break
        else:
            out[gid] = "A"
    return out


def balanced_rows_by_split(rows: list[dict], seed: int, ratios=(0.6, 0.2, 0.2)) -> tuple[list[dict], dict]:
    group_letters = primary_letter_for_prompt(rows)
    split_by_group = stratified_group_split(group_letters, ratios=ratios, seed=seed)
    split_rows: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for r in rows:
        split = split_by_group.get(int(r["sample_index"]))
        if split:
            rr = dict(r)
            rr["split"] = split
            split_rows[split].append(rr)

    selected: list[dict] = []
    manifest = {
        "balance_mode": "letter_pos_split",
        "split_group_key": "sample_index",
        "cap_by_split": {},
        "counts_before": {},
        "counts_after": {},
    }
    rng = np.random.default_rng(seed)
    for split, rs in split_rows.items():
        before = Counter((r["letter"], int(r["pos_offset"])) for r in rs)
        manifest["counts_before"][split] = {f"{k[0]}_pos{k[1]}": int(v) for k, v in sorted(before.items())}
        cells = [(letter, pos) for letter in ALL_LETTERS for pos in sorted({int(r["pos_offset"]) for r in rows})]
        missing = [cell for cell in cells if before.get(cell, 0) == 0]
        if missing:
            raise RuntimeError(f"cannot balance split={split}; missing cells={missing}")
        cap = min(before[cell] for cell in cells)
        manifest["cap_by_split"][split] = int(cap)
        after_counter = Counter()
        for cell in cells:
            pool = [r for r in rs if (r["letter"], int(r["pos_offset"])) == cell]
            idx = np.arange(len(pool))
            rng.shuffle(idx)
            for i in sorted(idx[:cap].tolist()):
                selected.append(pool[i])
                after_counter[cell] += 1
        manifest["counts_after"][split] = {f"{k[0]}_pos{k[1]}": int(v) for k, v in sorted(after_counter.items())}
    selected.sort(key=lambda r: (r["split"], r["letter"], int(r["pos_offset"]), int(r["row_id"])))
    manifest["n_balanced_rows"] = len(selected)
    manifest["balanced_row_ids"] = [int(r["row_id"]) for r in selected]
    return selected, manifest


def assert_balanced(manifest: dict) -> None:
    for split, counts in manifest["counts_after"].items():
        vals = list(counts.values())
        if vals and len(set(vals)) != 1:
            raise AssertionError(f"unbalanced counts in split={split}: {counts}")

