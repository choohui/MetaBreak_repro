"""Stage 03 — defended evaluation for one model (needs the loaded model).

For the model:
  * baselines (no defense): attack ASR, GSM8k accuracy (plain and header-injected),
  * per defense: attack ASR-after + block-rate, GSM8k+header accuracy + refuse-rate,
    benign refuse/flag rate (false positives).

Writes ``<model>/metrics.json`` (the numbers stage 04 tabulates) plus per-defense
detail jsonl for inspection.
"""

from __future__ import annotations

from core import io
from core.judge import attack_success, gsm8k_correct
from core.data import gsm8k_with_header
from core.defense_base import eval_one
from core.model import generate
from config import ExpConfig


def _mean(xs: list) -> float:
    xs = list(xs)
    return round(sum(xs) / len(xs), 5) if xs else 0.0


def run(cfg: ExpConfig, lm, model: str, defenses: dict) -> dict:
    ddir = cfg.result_dir / "data" / model
    outdir = cfg.result_dir / model
    attack = io.read_jsonl(ddir / "attack_test.jsonl")
    benign = io.read_jsonl(ddir / "benign_test.jsonl")
    gsm8k = io.read_jsonl(ddir / "gsm8k.jsonl")
    mnt = cfg.max_new_tokens

    # ---------------- baselines (no defense) ---------------- #
    base_success: dict[str, bool] = {}
    for r in attack:
        resp = generate(lm, r["text"], mnt)
        base_success[r["id"]] = attack_success(resp)
    acc_plain = [gsm8k_correct(generate(lm, g["question"], mnt), g["gold"]) for g in gsm8k]
    acc_header_nodef = [
        gsm8k_correct(generate(lm, gsm8k_with_header(g["question"], model), mnt), g["gold"])
        for g in gsm8k
    ]
    n_base_succ = sum(base_success.values())
    baseline = {
        "asr_before": _mean([int(v) for v in base_success.values()]),
        "n_attack": len(attack), "n_attack_success": int(n_base_succ),
        "gsm8k_acc_plain": _mean([int(c) for c in acc_plain]),
        "gsm8k_acc_header_nodef": _mean([int(c) for c in acc_header_nodef]),
        "n_gsm8k": len(gsm8k),
    }

    # ---------------- per-defense ---------------- #
    per_defense: dict[str, dict] = {}
    for name, d in defenses.items():
        atk_rows, gsm_rows, ben_rows = [], [], []

        succ_after, blocked, flagged_atk, refused_atk = [], [], [], []
        for r in attack:
            e = eval_one(lm, d, r["text"], mnt)
            s = attack_success(e["response"])
            succ_after.append(int(s))
            refused_atk.append(int(e["action"] == "refuse"))
            flagged_atk.append(e["n_flagged"])
            if base_success[r["id"]]:
                blocked.append(int(not s))
            atk_rows.append({"id": r["id"], "action": e["action"], "flagged": e["flagged"],
                             "success": s, "response": e["response"][:300]})

        gsm_correct, gsm_refuse, gsm_flag = [], [], []
        for g in gsm8k:
            prompt = gsm8k_with_header(g["question"], model)
            e = eval_one(lm, d, prompt, mnt)
            c = gsm8k_correct(e["response"], g["gold"])
            gsm_correct.append(int(c))
            gsm_refuse.append(int(e["action"] == "refuse"))
            gsm_flag.append(int(e["flagged"]))
            gsm_rows.append({"id": g["id"], "gold": g["gold"], "action": e["action"],
                             "correct": c, "response": e["response"][:200]})

        ben_refuse, ben_flag = [], []
        for r in benign:
            e = eval_one(lm, d, r["text"], mnt)
            ben_refuse.append(int(e["action"] == "refuse"))
            ben_flag.append(int(e["flagged"]))
            ben_rows.append({"id": r["id"], "action": e["action"], "flagged": e["flagged"]})

        per_defense[name] = {
            "attack": {
                "asr_after": _mean(succ_after),
                "block_rate_among_successful": (round(sum(blocked) / len(blocked), 5)
                                                if blocked else None),
                "refuse_rate": _mean(refused_atk),
                "mean_flagged_tokens": _mean(flagged_atk),
            },
            "gsm8k_header": {
                "acc_after": _mean(gsm_correct),
                "refuse_rate": _mean(gsm_refuse),
                "flag_rate": _mean(gsm_flag),
            },
            "benign": {
                "refuse_rate": _mean(ben_refuse),
                "flag_rate": _mean(ben_flag),
            },
        }
        io.write_jsonl(outdir / f"eval_{name}_attack.jsonl", atk_rows)
        io.write_jsonl(outdir / f"eval_{name}_gsm8k.jsonl", gsm_rows)
        io.write_jsonl(outdir / f"eval_{name}_benign.jsonl", ben_rows)

    metrics = {"model": model, "baseline": baseline, "defenses": per_defense}
    io.write_json(outdir / "metrics.json", metrics)
    return metrics
