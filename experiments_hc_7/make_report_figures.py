"""Text-free figures for the hc_7 steering report (matplotlib Agg, dpi 140).

Reads each ``pos{off}/steer_analysis.json`` and renders:
  fig1_dose_response   ASR vs alpha (defense headline arm) + over-refusal overlay
  fig2_pareto          ASR-reduction vs over-refusal-increase (defense side)
  fig3_controls        ASR at alpha* for attack / random / control-layer arms
  fig4_amplification   rescue rate vs +alpha (causal up-test)
  fig5_head_to_head    baseline vs steering(alpha*) vs hc_4 token-exclusion

All figures go to ``out_dir/report_figures/``. Safe to call without a model.
"""

from __future__ import annotations

from pathlib import Path


def make_figures(cfg) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiments_hc_7.core import io

    figdir = Path(cfg.out_dir) / "report_figures"
    figdir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    for off in cfg.pos_offsets:
        ap = cfg.pos_dir(off) / "steer_analysis.json"
        if not ap.exists():
            continue
        a = io.read_json(ap)
        tag = f"pos{off}"

        # fig1 dose-response
        dr = [d for d in a.get("dose_response", []) if d.get("alpha") is not None]
        if dr:
            xs = [d["alpha"] for d in dr]
            asr = [d["asr"] for d in dr]
            orr = [d.get("over_refusal_rate") for d in dr]
            fig, ax1 = plt.subplots(figsize=(6, 4))
            ax1.plot(xs, asr, "o-", color="C3", label="ASR")
            ax1.set_xlabel("alpha (units of rho)"); ax1.set_ylabel("ASR", color="C3")
            ax1.axvline(0, color="gray", lw=0.8, ls=":")
            astar = a.get("alpha_star")
            if astar is not None:
                ax1.axvline(astar, color="C0", lw=1.2, ls="--")
            if any(v is not None for v in orr):
                ax2 = ax1.twinx()
                ax2.plot(xs, [o if o is not None else float("nan") for o in orr], "s--",
                         color="C2", label="over-refusal")
                ax2.set_ylabel("over-refusal", color="C2")
            fig.tight_layout(); p = figdir / f"fig1_dose_response_{tag}.png"
            fig.savefig(p, dpi=140); plt.close(fig); written.append(str(p))

        # fig2 pareto
        pf = a.get("pareto_frontier", [])
        pts = [(d["over_refusal_increase"], d["asr_reduction"]) for d in pf
               if d.get("over_refusal_increase") is not None and d.get("asr_reduction") is not None]
        if pts:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter([x for x, _ in pts], [y for _, y in pts], c="C0")
            ax.axvline(cfg.over_refusal_budget, color="C3", ls="--", lw=1)
            ax.set_xlabel("over-refusal increase"); ax.set_ylabel("ASR reduction")
            fig.tight_layout(); p = figdir / f"fig2_pareto_{tag}.png"
            fig.savefig(p, dpi=140); plt.close(fig); written.append(str(p))

        # fig3 controls
        ctrl = a.get("controls", {})
        if ctrl:
            names = list(ctrl.keys()); vals = [ctrl[n].get("asr") for n in names]
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.bar(names, [v if v is not None else 0 for v in vals], color="C0")
            if a.get("baseline_asr") is not None:
                ax.axhline(a["baseline_asr"], color="gray", ls="--", label="baseline")
                ax.legend()
            ax.set_ylabel("ASR at alpha*"); ax.tick_params(axis="x", rotation=30)
            fig.tight_layout(); p = figdir / f"fig3_controls_{tag}.png"
            fig.savefig(p, dpi=140); plt.close(fig); written.append(str(p))

        # fig4 amplification
        amp = a.get("amplification", [])
        if amp:
            fig, ax = plt.subplots(figsize=(5, 4))
            for vt in sorted({d["vector_type"] for d in amp}):
                pts2 = [(d["alpha"], d["rescue_rate"]) for d in amp
                        if d["vector_type"] == vt and d["rescue_rate"] is not None]
                if pts2:
                    ax.plot([x for x, _ in pts2], [y for _, y in pts2], "o-", label=vt)
            ax.set_xlabel("+alpha"); ax.set_ylabel("rescue rate"); ax.legend()
            fig.tight_layout(); p = figdir / f"fig4_amplification_{tag}.png"
            fig.savefig(p, dpi=140); plt.close(fig); written.append(str(p))

        # fig5 head-to-head
        h = a.get("head_to_head", {})
        bars = [("baseline", h.get("baseline_asr")),
                ("steering(a*)", h.get("steering_asr_at_alpha_star")),
                ("hc4 token-excl", h.get("hc4_asr_after"))]
        bars = [(n, v) for n, v in bars if v is not None]
        if bars:
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.bar([n for n, _ in bars], [v for _, v in bars], color=["gray", "C0", "C2"][:len(bars)])
            ax.set_ylabel("ASR")
            fig.tight_layout(); p = figdir / f"fig5_head_to_head_{tag}.png"
            fig.savefig(p, dpi=140); plt.close(fig); written.append(str(p))

    return written
