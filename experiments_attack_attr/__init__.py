"""experiments_attack_attr — MetaBreak semantic-mimicry attack attribution + defense.

A single, structurally-clean reproduction of ``choan.md`` §0–§3.4: detect the
attack tokens from the victim model's internal representation (diff_means /
cos_to_attack) and sanitize by dropping the flagged token ±1.

Self-contained: depends only on repo-level ``src/*`` and ``prompts/*`` — never on
any other ``experiments_*`` folder. Datasets + the mimicry ``replacement.json`` are
vendored in ``data/`` (stage 00 regenerates the latter for the actual model).
"""
