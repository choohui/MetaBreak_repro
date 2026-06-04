"""Stage 03c — capture ONLY category C (benign mimicry) and append to artifacts.

Recovers the missing C control (Main.md §7.1) by token-level injection of the
exact attack replacement id into benign carriers — the decoded string does not
re-tokenize to the attack id in natural context, which is why the full run
produced C=0. Needs the model, but only for the ~``n_benign`` C prompts; the
existing A/B/D/E/F/G artifacts are left untouched.

Appends new rows to ``tokens.jsonl`` and matching cubes to ``features.npz``
(contiguous ``row_id``), and updates the census in ``extract_summary.json``.

Run:
    python experiments_hc_1/03c_append_benign_mimicry.py --model <Llama-3.1-8B path>
    # then re-run the model-free analyses (stages 04/05/06) to surface C.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
for _p in (str(REPO_ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments_hc_1.config import ExpConfig, config_from_args, get_model, make_stage_parser  # noqa: E402
from experiments_hc_1.core import io  # noqa: E402
from experiments_hc_1.core.benign_inject import build_c_prompts, capture_c_rows  # noqa: E402
from experiments_hc_1.core.labels import CAT_C  # noqa: E402


def run(cfg: ExpConfig, lm=None, force: bool = False) -> dict:
    out_dir = cfg.out_dir
    rows = io.read_jsonl(out_dir / "tokens.jsonl")
    summary = io.read_json(out_dir / "extract_summary.json")
    census = dict(summary.get("census", {}))
    npz = out_dir / "features.npz"
    hidden = np.load(npz)["hidden"]

    has_c = census.get(CAT_C, 0) > 0 or any(r["category"] == CAT_C for r in rows)
    if has_c and not force:
        raise SystemExit(
            f"[03c] {CAT_C} already present ({census.get(CAT_C, 0)} rows). "
            f"Use --force to drop and re-append.")
    if has_c and force:
        kept = [r for r in rows if r["category"] != CAT_C]
        keep_idx = [int(r["row_id"]) for r in kept]
        hidden = hidden[keep_idx] if hidden.size else hidden
        for new_id, r in enumerate(kept):
            r["row_id"] = new_id
        rows = kept
        print(f"[03c] --force: dropped existing C; {len(rows)} rows remain")

    lm = get_model(cfg, lm)
    repl = io.load_replacement(cfg.replacement)
    c_prompts = build_c_prompts(cfg, repl)
    if not c_prompts:
        print("[03c] no C prompts built (check benign_mimicry seeds / replacement); nothing to do.")
        return {}

    start_row_id = len(rows)
    start_sample_index = max((int(r["sample_index"]) for r in rows), default=-1) + 1
    new_rows, new_cubes = capture_c_rows(lm, c_prompts, cfg, start_row_id, start_sample_index)
    if not new_rows:
        print("[03c] no C positions captured (injected ids fell outside content?); nothing appended.")
        return {}

    # --- append tokens.jsonl ---
    io.write_jsonl(out_dir / "tokens.jsonl", rows + new_rows)

    # --- append features.npz hidden cube ---
    new_hidden = np.stack(new_cubes, axis=0).astype(
        hidden.dtype if hidden.size else np.float16)
    if hidden.size:
        if hidden.shape[1:] != new_hidden.shape[1:]:
            raise SystemExit(f"[03c] hidden shape mismatch {hidden.shape} vs {new_hidden.shape}")
        all_hidden = np.concatenate([hidden, new_hidden], axis=0)
    else:
        all_hidden = new_hidden
    np.savez_compressed(npz, hidden=all_hidden)

    # --- update census + n_rows ---
    all_rows = rows + new_rows
    census = {}
    for r in all_rows:
        census[r["category"]] = census.get(r["category"], 0) + 1
    summary["census"] = census
    summary["n_rows"] = len(all_rows)
    io.write_json(out_dir / "extract_summary.json", summary)

    print(f"[03c] appended {len(new_rows)} C rows from {len(c_prompts)} prompts; "
          f"n_rows -> {len(all_rows)}; hidden {all_hidden.shape}; "
          f"C census = {census.get(CAT_C, 0)}")
    return {"n_c_rows": len(new_rows), "n_rows": len(all_rows), "census": census}


def main() -> None:
    p = make_stage_parser(__doc__)
    p.add_argument("--force", action="store_true",
                   help="Drop any existing C rows and re-append (otherwise refuse).")
    args = p.parse_args()
    run(config_from_args(args), force=args.force)


if __name__ == "__main__":
    main()
