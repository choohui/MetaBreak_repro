"""Stage 09 (choan.md §3.3, HEADLINE) — drop±1 sanitizing defense.

Delete each flagged attack token AND its ±1 neighbours from the prompt, then
re-generate. choan's conclusion (§3.4): detecting malicious tokens with diff_means
and dropping the special token ±1 is the defense that works. The model-free
block-rate proxy is the held-out drop-semantics lower bound; with a real
``--model`` and ``--real_intervention`` the sanitized prompts are RE-GENERATED and
re-judged for the real ASR drop.

Outputs (per ``pos{off}/``): defense_drop_token_pm1.json (+ real_asr_drop_token_pm1.json).
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
for _p in (str(REPO_ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments_attack_attr.config import ExpConfig, config_from_args, get_model, make_stage_parser  # noqa: E402
from experiments_attack_attr.core import defense  # noqa: E402


def run(cfg: ExpConfig, lm=None) -> dict:
    if cfg.real_intervention and lm is None:
        lm = get_model(cfg, None)
    return {f"pos{off}": defense.run_offset(cfg, off, lm, "drop_token_pm1")
            for off in cfg.pos_offsets}


def main() -> None:
    run(config_from_args(make_stage_parser(__doc__).parse_args()))


if __name__ == "__main__":
    main()
