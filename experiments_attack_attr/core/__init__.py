"""Self-contained core utilities for the ``experiments_attack_attr`` study.

This package deliberately has **no dependency on any other ``experiments_*``
folder**. It may use ``repro_mb/src/*`` (model configs, mimicry, attack, evaluate)
and the data vendored in this folder's ``data/`` only — never a sibling result
directory. The model / forward-capture / sink-score / labeling logic is copied in
here so the experiment is fully independent.

Modules:
    labels       - the 7 token-type definitions (A..G) + defense label algebra
    io           - small JSON / JSONL / CSV helpers
    metrics      - numpy-only ROC-AUC / threshold / cosine + logistic probe
    model        - victim-model loading (eager attention)
    capture      - single-forward-pass internal-signal capture + sink scores
    template     - chat-template metadata, span finders, header builders
    labeling     - 7-type position labeling
    features     - per-token measurement-signal extraction (§2.2 raw signals)
    mock         - fake model/tokenizer for the model-free smoke test
    splits       - prompt-level train/held-out split
    scalarizers  - reduce internal reps to ONE scalar/token (clean① + borderline②)
    separability - §2.1 per-layer logistic-probe AUC over the full hidden vector
    thresholds   - TRAIN-fit threshold selectors + threshold stability
    cascade      - binary labels + per-type rates + prompt block / ASR proxy
    stats        - prompt-grouped bootstrap CIs + permutation tests
    curves       - ROC / DET / PR / calibration point generators
    defense      - §3.1/§3.3 mask + drop±1 token surgery + ASR (proxy / real)
    steer        - §3.2 lightweight activation steering along -diff_means
    benign_gen / benign_inject - category C/E benign-context prompt builders
"""
