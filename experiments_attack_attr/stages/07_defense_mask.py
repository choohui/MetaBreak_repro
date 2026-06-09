"""Stage 07 (choan.md §3.1) — masking defense.

Replace every flagged attack token (detector from stage 06, diff_means by default)
with a NEUTRAL word (``--mask_mode neutral``; or ``unk`` / ``eos`` for the
ASR-increasing controls choan reports). The model-free block-rate proxy is always
written; with a real ``--model`` and ``--real_intervention`` the masked prompts
are RE-GENERATED and re-judged for the honest ASR (the only way masking's effect
— which can INCREASE ASR for unk/eos — is actually measured).

Outputs (per ``pos{off}/``): defense_mask.json (+ real_asr_mask.json if real).
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
    return {f"pos{off}": defense.run_offset(cfg, off, lm, "mask")
            for off in cfg.pos_offsets}


def main() -> None:
    run(config_from_args(make_stage_parser(__doc__).parse_args()))


if __name__ == "__main__":
    main()
