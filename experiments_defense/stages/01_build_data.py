"""Stage 01 — build datasets + train/test splits (model-free).

Per model family writes under ``results/<run>/data/<model>/``:
  attack_train.jsonl / attack_test.jsonl   (MetaBreak TM-1 mimicry prompts)
  benign_train.jsonl / benign_test.jsonl   (utility + FPR controls)
  gsm8k.jsonl                              (math questions + gold; header built at eval)
"""

from __future__ import annotations

from tqdm import tqdm

from core import data, io
from config import ExpConfig


def run(cfg: ExpConfig, lm=None) -> dict:
    summary = {"run": cfg.slug(), "models": {}}
    for model in tqdm(cfg.models, desc="[01] build data", unit="model", dynamic_ncols=True):
        ddir = cfg.result_dir / "data" / model

        attacks = data.load_attack_prompts(model, cfg.n_attack,
                                            mimicry=cfg.mimicry, smoke=cfg.smoke)
        atk_train, atk_test = data.split_train_test(attacks, cfg.frac_train)

        benign = data.load_benign_prompts(cfg.n_benign, smoke=cfg.smoke)
        ben_train, ben_test = data.split_train_test(benign, cfg.frac_train)

        gsm8k = data.load_gsm8k(cfg.n_gsm8k, smoke=cfg.smoke)

        io.write_jsonl(ddir / "attack_train.jsonl", atk_train)
        io.write_jsonl(ddir / "attack_test.jsonl", atk_test)
        io.write_jsonl(ddir / "benign_train.jsonl", ben_train)
        io.write_jsonl(ddir / "benign_test.jsonl", ben_test)
        io.write_jsonl(ddir / "gsm8k.jsonl", gsm8k)

        summary["models"][model] = {
            "attack_train": len(atk_train), "attack_test": len(atk_test),
            "benign_train": len(ben_train), "benign_test": len(ben_test),
            "gsm8k": len(gsm8k),
            "mimicked": bool(attacks and attacks[0].get("mimicked")),
        }
    io.write_json(cfg.result_dir / "data" / "manifest.json", summary)
    return summary
