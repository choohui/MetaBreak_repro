"""Stage 04 - render a concise Markdown report."""

from __future__ import annotations

import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
REPO_ROOT = PKG.parent
for p in (str(REPO_ROOT), str(PKG.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from experiments_hc_4.config import ExpConfig, config_from_args, make_parser  # noqa: E402
from experiments_hc_4.core import io  # noqa: E402


def _md(report: dict) -> str:
    lines = [
        "# experiments_hc_4 - Active Sink % Sweep",
        "",
        f"- score: `{report['score']}`",
        f"- selection: {report['selection']}",
        f"- split: {report['split']}",
        f"- FPR target: {report['fpr_target']}",
        f"- monotonic kept tokens: {report['monotonic_kept']}",
        "",
        "| keep % | kept | ratio | B rec | D rec | BD rec | C FPR | E FPR | F FPR | G FPR | prompt block | prompt FPR | ASR after |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["sweep"]:
        ev = row["evaluation"]
        token = ev["token"]
        per = token["per_letter"]
        prompt = ev["prompt"]
        asr = ev.get("asr") or {}
        lines.append(
            f"| {row['keep_pct']} | {row['n_kept']} | {row['reduction_ratio']} | "
            f"{per.get('B', {}).get('rate')} | {per.get('D', {}).get('rate')} | "
            f"{token.get('recall')} | {per.get('C', {}).get('rate')} | "
            f"{per.get('E', {}).get('rate')} | {per.get('F', {}).get('rate')} | "
            f"{per.get('G', {}).get('rate')} | {prompt.get('block_rate')} | "
            f"{prompt.get('prompt_fpr')} | {asr.get('asr_after')} |"
        )
    lines.extend([
        "",
        "## Artifacts",
        "",
        "- `replacement.json`",
        "- `prompts.jsonl`",
        "- `active_value_rows.jsonl`",
        "- `responses.jsonl` when generation is enabled",
        "- `pct_threshold_report.json`",
        "- `sweep_summary.csv`",
    ])
    return "\n".join(lines) + "\n"


def run(cfg: ExpConfig) -> dict:
    report = io.read_json(cfg.report_json)
    io.write_text(cfg.report_md, _md(report))
    print(f"[04] wrote markdown -> {cfg.report_md}")
    return {"path": str(cfg.report_md)}


def main() -> None:
    run(config_from_args(make_parser(__doc__).parse_args()))


if __name__ == "__main__":
    main()

