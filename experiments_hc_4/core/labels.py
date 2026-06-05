"""A-G token categories and defense labels."""

from __future__ import annotations

CAT_A = "A_system_special"
CAT_B = "B_malicious_mimicry"
CAT_C = "C_benign_mimicry"
CAT_D = "D_malicious_special"
CAT_E = "E_benign_special"
CAT_F = "F_positioned_regular"
CAT_G = "G_ordinary_regular"

ALL_CATEGORIES = [CAT_A, CAT_B, CAT_C, CAT_D, CAT_E, CAT_F, CAT_G]
LETTER_TO_CAT = {c.split("_", 1)[0]: c for c in ALL_CATEGORIES}
CAT_TO_LETTER = {c: c.split("_", 1)[0] for c in ALL_CATEGORIES}

POSITIVE_CATS = {CAT_B, CAT_D}
NEGATIVE_CATS = {CAT_C, CAT_E, CAT_F, CAT_G}
REFERENCE_CATS = {CAT_A}

VARIANT_TO_PRIMARY_CAT = {
    "malicious_special": CAT_D,
    "malicious_mimicry": CAT_B,
    "positioned_regular": CAT_F,
    "benign_mimicry": CAT_C,
    "benign_special": CAT_E,
    "ordinary": CAT_G,
}

ASR_VARIANTS = ["malicious_mimicry", "malicious_special", "positioned_regular"]


def defense_label(category: str) -> int | None:
    if category in POSITIVE_CATS:
        return 1
    if category in NEGATIVE_CATS:
        return 0
    return None

