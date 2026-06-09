"""Stage 04 — aggregate all models' metrics into REPORT.md (model-free).

Builds the choan.md §4 comparison: models × defenses tables for attack ASR,
block-rate, GSM8k+header utility, and benign false-positive rate, so the
token-level ``ours`` (utility-preserving) vs prompt-level baselines contrast is
visible at a glance.
"""

from __future__ import annotations

from core import io
from config import ExpConfig


def _cell(v) -> str:
    return "—" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def _table(title: str, note: str, models: list[str], defenses: list[str],
           get, extra_cols: list[tuple[str, object]] | None = None) -> str:
    cols = [c for c, _ in (extra_cols or [])] + defenses
    head = "| model | " + " | ".join(cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    lines = [f"### {title}", "", note, "", head, sep]
    for m in models:
        cells = [_cell(fn(m)) for _, fn in (extra_cols or [])]
        cells += [_cell(get(m, d)) for d in defenses]
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def run(cfg: ExpConfig, lm=None) -> dict:
    models, defenses = cfg.models, cfg.defenses
    M: dict[str, dict] = {}
    for m in models:
        p = cfg.result_dir / m / "metrics.json"
        if p.exists():
            M[m] = io.read_json(p)
    avail = [m for m in models if m in M]

    def base(m, key):
        return M[m]["baseline"].get(key) if m in M else None

    def dfn(m, d, *path):
        cur = M.get(m, {}).get("defenses", {}).get(d, {})
        for k in path:
            cur = cur.get(k, {}) if isinstance(cur, dict) else None
            if cur is None:
                return None
        return cur if not isinstance(cur, dict) else None

    parts = [f"# experiments_defense — §4 report ({cfg.slug()})", "",
             "4 defenses × {} model(s) on MetaBreak TM-1 attacks + GSM8k(+mimicry header)."
             .format(len(avail)), "",
             "**ours** = diff-means token detection + ±1 drop (sanitize). "
             "**llama_guard / jbshield / guard_slm** = prompt-level detect→refuse.", ""]

    parts.append(_table(
        "Attack success rate after defense  (lower = better)",
        f"baseline `asr_before` is the no-defense ASR on the held-out attack set.",
        avail, defenses,
        lambda m, d: dfn(m, d, "attack", "asr_after"),
        extra_cols=[("asr_before", lambda m: base(m, "asr_before"))]))

    parts.append(_table(
        "Block rate among originally-successful attacks  (higher = better)",
        "fraction of baseline-successful attacks neutralised by the defense.",
        avail, defenses,
        lambda m, d: dfn(m, d, "attack", "block_rate_among_successful")))

    parts.append(_table(
        "GSM8k (+mimicry header) accuracy  (higher = better; utility)",
        "`plain` / `hdr_nodef` = no-defense accuracy without / with the header. "
        "Prompt-level refusers should drop here; `ours` should not.",
        avail, defenses,
        lambda m, d: dfn(m, d, "gsm8k_header", "acc_after"),
        extra_cols=[("plain", lambda m: base(m, "gsm8k_acc_plain")),
                    ("hdr_nodef", lambda m: base(m, "gsm8k_acc_header_nodef"))]))

    parts.append(_table(
        "Benign false-positive rate  (lower = better)",
        "benign prompts refused (prompt-level) — `ours` reports flag-rate instead.",
        avail, defenses,
        lambda m, d: (dfn(m, d, "benign", "refuse_rate")
                      if d != "ours" else dfn(m, d, "benign", "flag_rate"))))

    report = "\n".join(parts)
    io.write_text(cfg.result_dir / "REPORT.md", report)
    io.write_json(cfg.result_dir / "summary.json",
                  {"run": cfg.slug(), "models": avail, "defenses": defenses,
                   "metrics": {m: M[m] for m in avail}})
    return {"models": avail, "report_path": str(cfg.result_dir / "REPORT.md")}
