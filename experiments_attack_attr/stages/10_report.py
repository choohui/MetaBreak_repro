"""Stage 10 (choan.md §3.4) — consolidated conclusion report.

Gathers the artifacts of every section into one human-readable summary that maps
1:1 to choan.md §1→§3.4:

    §1   embedding separability AUC (≈0.5 expected)            [stage 00]
    §2.1 internal-rep logistic-probe AUC                       [stage 04]
    §2.2 cos_to_attack (clean headline) + diff_means held-out  [stage 06]
    §3.1 mask ASR     §3.2 steer ASR     §3.3 drop±1 ASR       [stages 07/08/09]
    §3.4 conclusion: diff_means detect -> drop flagged ±1

Model-free; reads what the earlier stages wrote.

Outputs (under ``out_dir``): final_report.json + final_report.md
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
for _p in (str(REPO_ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments_attack_attr.config import ExpConfig, config_from_args, make_stage_parser  # noqa: E402
from experiments_attack_attr.core import io  # noqa: E402


def _maybe(path: Path):
    return io.read_json(path) if path.exists() else None


def _defense_row(cfg, off, action):
    rep = _maybe(cfg.pos_dir(off) / f"defense_{action}.json")
    if not rep:
        return None
    return {"asr_before": rep.get("asr_before"),
            "asr_after_proxy": rep.get("asr_after_proxy"),
            "asr_after_real": (rep.get("real") or {}).get("asr_after"),
            "block_rate_among_successful": rep.get("block_rate_among_successful")}


def run(cfg: ExpConfig, lm=None) -> dict:
    emb = _maybe(cfg.out_dir / "embedding_analysis.json")
    report = {"out_dir": str(cfg.out_dir), "pos_offsets": cfg.pos_offsets,
              "defense_family": cfg.defense_family, "mask_mode": cfg.mask_mode,
              "s1_embedding_separability_auc": (emb or {}).get("separability_auc"),
              "per_pos": {}}
    for off in cfg.pos_offsets:
        sep = _maybe(cfg.pos_dir(off) / "separability.json")
        det = _maybe(cfg.pos_dir(off) / "detect_summary.json")
        steer = _maybe(cfg.pos_dir(off) / "defense_steer.json")
        report["per_pos"][f"pos{off}"] = {
            "s21_probe_best_layer": (sep or {}).get("best_layer"),
            "s21_probe_best_auc": (sep or {}).get("best_auc"),
            # choan §2.2 named signals (verbatim) + the auto-selected best-of-family
            "s22_cos_to_attack": (det or {}).get("cos_to_attack"),
            "s22_diff_means": (det or {}).get("diff_means"),
            "s22_clean_headline": (det or {}).get("clean_headline"),
            "s22_borderline_detector": (det or {}).get("borderline_detector"),
            "s31_mask": _defense_row(cfg, off, "mask"),
            "s32_steer": {"asr_before": (steer or {}).get("asr_before"),
                          "asr_after": (steer or {}).get("asr_after"),
                          "flag_coverage": (steer or {}).get("flag_coverage")},
            "s33_drop_pm1": _defense_row(cfg, off, "drop_token_pm1"),
        }
    io.write_json(cfg.out_dir / "final_report.json", report)
    io.write_text(cfg.out_dir / "final_report.md", _md(report))
    print(f"[10] final report -> {cfg.out_dir/'final_report.md'}")
    return report


def _fmt(x):
    return "n/a" if x is None else x


def _md(r: dict) -> str:
    L = ["# Attack-attribution defense — final report (choan.md §0–§3.4)", "",
         f"- out_dir: `{r['out_dir']}`",
         f"- detector family: **{r['defense_family']}** (diff_means = borderline), "
         f"mask_mode: {r['mask_mode']}", ""]
    L.append("## §1 — token embedding alone does NOT separate special vs regular")
    L.append(f"- separability AUC (≈0.5 expected): {_fmt(r['s1_embedding_separability_auc'])}")
    L.append("")
    for pos, d in r["per_pos"].items():
        L.append(f"## {pos}")
        L.append(f"- **§2.1** internal-rep logistic probe: best layer "
                 f"{_fmt(d['s21_probe_best_layer'])}, AUC **{_fmt(d['s21_probe_best_auc'])}**")
        ca = d.get("s22_cos_to_attack") or {}
        dm = d.get("s22_diff_means") or {}
        L.append(f"- **§2.2 cos_to_attack (clean headline)** @L{_fmt(ca.get('layer'))}: "
                 f"held-out AUC **{_fmt(ca.get('test_auc'))}**, TPR {_fmt(ca.get('test_tpr'))}, "
                 f"benign-FPR {_fmt(ca.get('test_benign_fpr'))}")
        L.append(f"- **§2.2 diff_means (token detector)** @L{_fmt(dm.get('layer'))}: "
                 f"held-out AUC **{_fmt(dm.get('test_auc'))}**, TPR {_fmt(dm.get('test_tpr'))}, "
                 f"benign-FPR {_fmt(dm.get('test_benign_fpr'))}")
        m, st, dr = d["s31_mask"] or {}, d["s32_steer"] or {}, d["s33_drop_pm1"] or {}
        L.append(f"- **§3.1 mask**: ASR {_fmt(m.get('asr_before'))} → proxy "
                 f"{_fmt(m.get('asr_after_proxy'))} / real {_fmt(m.get('asr_after_real'))}")
        L.append(f"- **§3.2 steer**: ASR {_fmt(st.get('asr_before'))} → "
                 f"{_fmt(st.get('asr_after'))} (partial; flag-coverage {_fmt(st.get('flag_coverage'))})")
        L.append(f"- **§3.3 drop±1 (HEADLINE)**: ASR {_fmt(dr.get('asr_before'))} → proxy "
                 f"{_fmt(dr.get('asr_after_proxy'))} / real {_fmt(dr.get('asr_after_real'))} "
                 f"(block-rate-among-successful {_fmt(dr.get('block_rate_among_successful'))})")
        L.append("")
    L.append("## §3.4 — conclusion")
    L.append("> Detect malicious tokens from the internal representation (diff_means) "
             "and drop the flagged token ±1. This sanitizes the prompt token-by-token "
             "(utility preserved vs prompt-level refusal) with no classifier / 2nd "
             "inference. Masking partially helps (neutral) or backfires (unk/eos); "
             "steering only partially helps — so drop±1 is the headline defense.")
    return "\n".join(L) + "\n"


def main() -> None:
    run(config_from_args(make_stage_parser(__doc__).parse_args()))


if __name__ == "__main__":
    main()
