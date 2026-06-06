from __future__ import annotations

from experiments_hc_4.config import ExpConfig
from experiments_hc_4.core import io
from experiments_hc_4.core.reporting import render_final_report


def run(cfg: ExpConfig, lm=None) -> dict:
    report = render_final_report(cfg.out_dir)
    io.write_text(cfg.out_dir / "final_report.md", report)
    print(f"[09] wrote final report -> {cfg.out_dir / 'final_report.md'}")
    return {"path": str(cfg.out_dir / "final_report.md")}

