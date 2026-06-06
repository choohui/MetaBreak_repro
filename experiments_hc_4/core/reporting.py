from __future__ import annotations

from pathlib import Path

from . import io


def render_final_report(out_dir: Path) -> str:
    parts = ["# experiments_hc_4 final report", ""]
    for name in [
        "balanced_manifest.json",
        "scalar_discovery_summary.json",
        "threshold_rules.json",
        "defense_eval.json",
        "counterfactual_report.json",
        "stress_report.json",
    ]:
        path = out_dir / name
        if not path.exists():
            parts.append(f"## {name}\n\nmissing\n")
            continue
        data = io.read_json(path)
        parts.append(f"## {name}\n")
        parts.append("```json")
        parts.append(_short_json(data))
        parts.append("```\n")
    return "\n".join(parts)


def _short_json(data) -> str:
    import json

    text = json.dumps(data, ensure_ascii=False, indent=2)
    lines = text.splitlines()
    if len(lines) <= 120:
        return text
    return "\n".join(lines[:120] + ["  ... truncated ...", "}"])

