from __future__ import annotations

from experiments_hc_5.config import ExpConfig
from experiments_hc_5.core import io
from experiments_hc_5.core.reporting import render_final_report


def run(cfg: ExpConfig, lm=None) -> dict:
    report = render_final_report(cfg.out_dir)
    io.write_text(cfg.out_dir / "report.md", report)
    io.write_text(cfg.out_dir / "final_report.md", report)
    print(f"[09] wrote report -> {cfg.out_dir / 'report.md'}")
    return {
        "report": str(cfg.out_dir / "report.md"),
        "metrics": str(cfg.out_dir / "metrics.json"),
    }
