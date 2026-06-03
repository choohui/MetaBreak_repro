"""Self-contained core utilities for the ``experiments_hc_1`` study.

This package deliberately has **no dependency on any other ``experiments_*``
folder**. It may use ``repro_mb/src/*`` (model configs, mimicry, attack,
evaluate) and the repo-level ``results/llama/replacement.json`` only.

The model/forward-capture/sink-score/labeling logic is re-implemented here
(based on the proven design in ``experiments_hwichan/common.py``) so the new
experiment is fully independent.

Modules:
    labels    - the 7 token-type definitions (A..G) + defense label algebra
    io        - small JSON / JSONL / CSV helpers
    metrics   - numpy-only ROC-AUC / threshold / cosine / probe helpers
    model     - victim-model loading (eager attention)
    capture   - single-forward-pass internal-signal capture + sink scores
    template  - chat-template metadata, span finders, header builders
    labeling  - 7-type position labeling
    features  - per-token signal extraction (the 5 measurement signals)
    mock      - fake model/tokenizer for the model-free smoke test
"""
