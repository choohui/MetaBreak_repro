"""experiments_defense — multi-model defense comparison for choan.md §4.

Self-contained package: it depends ONLY on ``repro_mb/src`` (the shared attack
pipeline: ``src.model_configs``, ``src.evaluate``) and never imports from any
other ``experiments_*`` folder. The few pieces re-used from experiments_hc_4 /
experiments_hc_4_claude (diff-means direction fit, ±1 token drop, span
detection, mock model) are *copied* in here, per the repo convention that each
experiment folder is independently runnable.
"""
