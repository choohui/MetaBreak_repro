"""Model-free smoke test — runs the whole pipeline on the mock model.

Exercises stages 01–04 and every defense's prepare()/guard() on synthetic hidden
states, then asserts all artifacts exist with the right schema. No real weights,
no GPU, no network. Exit 0 == OK.

    python experiments_defense/smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from config import ExpConfig, ALL_DEFENSES  # noqa: E402
from core import io  # noqa: E402

FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  ok  - " if cond else "FAIL  - ") + msg)
    if not cond:
        FAILS.append(msg)


def main() -> int:
    import run_all
    from core import data
    from core.mock import build_mock_loaded_model
    from core.defense_base import eval_one
    from defenses import build_defense

    cfg = ExpConfig(models=["llama"], defenses=list(ALL_DEFENSES),
                    n_benign=8, n_gsm8k=6, smoke=True, max_new_tokens=8)

    # ---- unit: each defense prepare()/guard() returns a valid GuardResult ---- #
    lm = build_mock_loaded_model()
    calib = {
        "attack_train": data.load_attack_prompts("llama", smoke=True),
        "benign_train": data.load_benign_prompts(smoke=True),
    }
    for name in ALL_DEFENSES:
        d = build_defense(name, smoke=True)
        summ = d.prepare(lm, calib)
        check(isinstance(summ, dict) and summ.get("defense") == name,
              f"{name}.prepare returns summary")
        r = d.guard(lm, calib["attack_train"][0]["text"])
        check(r.action in ("pass", "refuse", "sanitize"), f"{name}.guard action valid")
        if name == "ours":
            check(r.action == "sanitize" and r.sanitized_ids is not None,
                  "ours.guard sanitizes with ids")
        e = eval_one(lm, d, calib["attack_train"][0]["text"], max_new_tokens=8)
        check(isinstance(e.get("response"), str), f"{name} eval_one produces response")

    # ---- full pipeline ---- #
    run_all.run(cfg)
    rd = cfg.result_dir

    check((rd / "data" / "manifest.json").exists(), "stage01 manifest written")
    man = io.read_json(rd / "data" / "manifest.json")
    check(man["models"]["llama"]["attack_train"] > 0, "attack_train non-empty")

    check((rd / "llama" / "prepare.json").exists(), "stage02 prepare.json written")
    prep = io.read_json(rd / "llama" / "prepare.json")
    check(set(prep) == set(ALL_DEFENSES), "prepare.json has all 4 defenses")

    check((rd / "llama" / "metrics.json").exists(), "stage03 metrics.json written")
    met = io.read_json(rd / "llama" / "metrics.json")
    check("asr_before" in met["baseline"], "baseline has asr_before")
    check("gsm8k_acc_plain" in met["baseline"], "baseline has gsm8k accuracy")
    for name in ALL_DEFENSES:
        dd = met["defenses"].get(name, {})
        check("asr_after" in dd.get("attack", {}), f"{name} has attack.asr_after")
        check("acc_after" in dd.get("gsm8k_header", {}), f"{name} has gsm8k acc_after")
        check("flag_rate" in dd.get("benign", {}), f"{name} has benign flag_rate")
        check((rd / "llama" / f"eval_{name}_attack.jsonl").exists(),
              f"{name} attack detail jsonl written")

    check((rd / "REPORT.md").exists(), "stage04 REPORT.md written")
    check((rd / "summary.json").exists(), "stage04 summary.json written")
    report = (rd / "REPORT.md").read_text(encoding="utf-8")
    check("Attack success rate" in report and "GSM8k" in report, "REPORT has expected sections")

    print()
    if FAILS:
        print(f"SMOKE TEST FAILED: {len(FAILS)} check(s) failed")
        return 1
    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
