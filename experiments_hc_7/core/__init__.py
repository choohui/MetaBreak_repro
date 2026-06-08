"""Self-contained core utilities for the ``experiments_hc_4_claude`` study.

This package deliberately has **no dependency on any other ``experiments_*``
folder**. It may use ``repro_mb/src/*`` (model configs, mimicry, attack,
evaluate) and the repo-level ``results/llama/replacement.json`` only.

The model/forward-capture/sink-score/labeling logic is copied in here so the
experiment is fully independent of the other ``experiments_*`` folders.

Modules:
    labels      - the 7 token-type definitions (A..G) + defense label algebra
    io          - small JSON / JSONL / CSV helpers
    metrics     - numpy-only ROC-AUC / threshold / cosine helpers
    model       - victim-model loading (eager attention)
    capture     - single-forward-pass internal-signal capture + sink scores
    template    - chat-template metadata, span finders, header builders
    labeling    - 7-type position labeling
    features    - per-token measurement-signal extraction
    mock        - fake model/tokenizer for the model-free smoke test
    splits      - prompt-level train/held-out split
    scalarizers - reduce internal reps to ONE scalar/token (clean + borderline)
    thresholds  - TRAIN-fit threshold selectors + threshold stability
    stats       - prompt-grouped bootstrap CIs + permutation tests
    curves      - ROC / DET / PR / calibration point generators
    intervene   - optional real token-exclusion re-generation (needs model)
"""
