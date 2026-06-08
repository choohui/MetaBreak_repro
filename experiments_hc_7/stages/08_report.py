"""Stage 08 (model-free) — write summary.md + render report figures.

Summarizes, per pos_offset: baseline ASR, alpha*, steered ASR (with bootstrap CI
and the paired permutation p-value for the reduction), the over-refusal cost, the
controls/layer-specificity verdicts, the amplification trend, and the head-to-head
against hc_4's token-exclusion proxy.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
for _p in (str(REPO_ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments_hc_7.config import ExpConfig, config_from_args, make_stage_parser  # noqa: E402
from experiments_hc_7.core import io                                                 # noqa: E402


def _fmt(x, pct=False):
    if x is None:
        return "—"
    return f"{100*x:.1f}%" if pct else f"{x:.4g}"


def _section(cfg, off) -> str:
    p = cfg.pos_dir(off) / "steer_analysis.json"
    if not p.exists():
        return f"### pos{off}\n\n(no analysis found)\n"
    a = io.read_json(p)
    ci = a.get("bootstrap_ci", {})
    perm = a.get("permutation_reduction", {})
    h = a.get("head_to_head", {})
    cc = a.get("controls_check") or {}
    lines = [f"### pos{off} — hidden-layer {a.get('layer')} (block {a.get('block_idx')}), rho={_fmt(a.get('rho'))}", ""]
    lines.append(f"- baseline ASR: **{_fmt(a.get('baseline_asr'), pct=True)}**")
    lines.append(f"- alpha* (within over-refusal budget {cfg.over_refusal_budget}): "
                 f"**{_fmt(a.get('alpha_star'))}** → steered ASR **{_fmt(a.get('asr_at_alpha_star'), pct=True)}** "
                 f"(over-refusal {_fmt(a.get('over_refusal_at_alpha_star'), pct=True)})")
    if ci:
        b, s = ci.get("baseline", {}), ci.get("alpha_star", {})
        lines.append(f"- ASR 95% CI: baseline [{_fmt(b.get('lo'),1)}, {_fmt(b.get('hi'),1)}] "
                     f"vs steered [{_fmt(s.get('lo'),1)}, {_fmt(s.get('hi'),1)}]")
    if perm:
        lines.append(f"- paired permutation reduction Δ={_fmt(perm.get('delta'))}, p={_fmt(perm.get('p_value'))}")
    if cc:
        lines.append(f"- control check: attack Δ={_fmt(cc.get('attack_delta'))}, "
                     f"random Δ={_fmt(cc.get('random_delta'))} "
                     f"(random inert={cc.get('random_is_inert')})")
    if a.get("layer_specificity"):
        for k, v in a["layer_specificity"].items():
            lines.append(f"- {k}: ASR {_fmt(v.get('asr'), pct=True)} (Δ {_fmt(v.get('delta_vs_baseline'))})")
    amp = a.get("amplification", [])
    if amp:
        top = max((d for d in amp if d.get("rescue_rate") is not None),
                  key=lambda d: d["rescue_rate"], default=None)
        if top:
            lines.append(f"- amplification: rescue rate up to {_fmt(top['rescue_rate'], pct=True)} "
                         f"at +alpha={_fmt(top['alpha'])} ({top['vector_type']})")
    lines.append(f"- head-to-head: baseline {_fmt(h.get('baseline_asr'), pct=True)} → "
                 f"steering(alpha*) {_fmt(h.get('steering_asr_at_alpha_star'), pct=True)} "
                 f"| hc_4 token-exclusion {_fmt(h.get('hc4_asr_after'), pct=True)}")
    lines.append("")
    return "\n".join(lines)


def run(cfg: ExpConfig, lm=None) -> dict:
    body = ["# experiments_hc_7 — cos_to_attack activation steering\n",
            "Causal test of hc_4_claude's `cos_to_attack` detection direction: does "
            "steering the residual stream along/against it change ASR (held-out), and "
            "at what utility cost? Steering vector = the fitted attack centroid; "
            "evaluation reuses hc_4's saved held-out split.\n"]
    for off in cfg.pos_offsets:
        body.append(_section(cfg, off))
    summary = "\n".join(body)
    io.write_text(cfg.out_dir / "summary.md", summary)

    figs = []
    try:
        from experiments_hc_7.make_report_figures import make_figures
        figs = make_figures(cfg)
    except Exception as e:  # matplotlib missing or headless issue — report stays useful
        print(f"[08] figures skipped: {e}")
    print(f"[08] wrote {cfg.out_dir/'summary.md'} (+{len(figs)} figures)")
    return {"summary": str(cfg.out_dir / "summary.md"), "n_figures": len(figs)}


def main() -> None:
    run(config_from_args(make_stage_parser(__doc__).parse_args()))


if __name__ == "__main__":
    main()
