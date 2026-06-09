"""Stage 02 — calibrate each defense for one model (needs the loaded model).

Reads the TRAIN splits from stage 01, builds the per-defense calibration set, and
calls ``defense.prepare``. Returns the in-memory prepared defense objects so
stage 03 can use them in the same process (the SVM / direction state is not
round-tripped through disk). Writes ``<model>/prepare.json`` summaries.
"""

from __future__ import annotations

from core import io
from config import ExpConfig
from defenses import build_defense


def run(cfg: ExpConfig, lm, model: str) -> tuple[dict, dict]:
    ddir = cfg.result_dir / "data" / model
    calib = {
        "attack_train": io.read_jsonl(ddir / "attack_train.jsonl"),
        "benign_train": io.read_jsonl(ddir / "benign_train.jsonl"),
    }
    summaries: dict[str, dict] = {}
    defenses: dict[str, object] = {}
    for name in cfg.defenses:
        kwargs = {"smoke": cfg.smoke}
        if name == "llama_guard":
            kwargs.update(guard_model=cfg.guard_model, dtype=cfg.dtype, device=cfg.device)
        d = build_defense(name, **kwargs)
        summaries[name] = d.prepare(lm, calib)
        defenses[name] = d
    io.write_json(cfg.result_dir / model / "prepare.json", summaries)
    return summaries, defenses
