"""ASR-reduction (deployed-defense) study for the experiments_hwichan signals.

The sibling modules (`defense_thresholds`, `cascade_defense`) report token-level
TPR/FPR/AUC — *how separable* attack tokens are. This package answers the
downstream question instead: **when each detector is wired up as a block-on-detect
guard, how much does it lower the attack success rate (ASR)?**

See `README.md` for the full writeup and `run_defense_asr.py` for the entry point.
"""
