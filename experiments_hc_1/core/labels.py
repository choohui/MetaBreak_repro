"""The 7 token types (Main.md §2.1) and the defense-label algebra.

Each analyzed token position is assigned exactly one letter A..G:

    A : system special  - chat-template special token        [reference]
    B : malicious mimicry regular - L2-substitution attack token   [POSITIVE / attack]
    C : benign mimicry regular - substitution token in benign context [negative / token-identity control]
    D : malicious special - literal special token in the attack slot  [POSITIVE / attack]
    E : benign special - special token in a benign context     [negative]
    F : positioned regular - plain benign word in the attack slot [negative / position control]
    G : ordinary regular - ordinary body token                 [negative / baseline]

Defense labels:  positive = B u D , negative = C u E u F u G , A = reference.
"""

from __future__ import annotations

# Canonical letter -> human-readable name.
CAT_A = "A_system_special"
CAT_B = "B_malicious_mimicry"
CAT_C = "C_benign_mimicry"
CAT_D = "D_malicious_special"
CAT_E = "E_benign_special"
CAT_F = "F_positioned_regular"
CAT_G = "G_ordinary_regular"

# Ordered list of all categories (stable ordering for reports/CSV columns).
ALL_CATEGORIES = [CAT_A, CAT_B, CAT_C, CAT_D, CAT_E, CAT_F, CAT_G]

# Letter <-> full-name maps.
LETTER_TO_CAT = {c.split("_", 1)[0]: c for c in ALL_CATEGORIES}
CAT_TO_LETTER = {c: c.split("_", 1)[0] for c in ALL_CATEGORIES}

# Defense-label sets.
POSITIVE_CATS = {CAT_B, CAT_D}            # attack
NEGATIVE_CATS = {CAT_C, CAT_E, CAT_F, CAT_G}  # benign / control
REFERENCE_CATS = {CAT_A}                  # reference baseline (cos_to_ref centroid)

# The prompt "variant" string written by stage 01 -> the category its
# attack/control slot positions receive at labeling time.
VARIANT_TO_PRIMARY_CAT = {
    "malicious_special": CAT_D,
    "malicious_mimicry": CAT_B,
    "positioned_regular": CAT_F,
    "benign_mimicry": CAT_C,
    "benign_special": CAT_E,
    "ordinary": CAT_G,
}

# Variants that carry an actual attack payload and are worth generating for ASR
# (Main.md decision: ASR generation runs only on B / D / F).
ASR_VARIANTS = ["malicious_mimicry", "malicious_special", "positioned_regular"]


def defense_label(category: str) -> int | None:
    """1 = attack (positive), 0 = benign (negative), None = reference (A)."""
    if category in POSITIVE_CATS:
        return 1
    if category in NEGATIVE_CATS:
        return 0
    return None  # reference / unknown -> excluded from binary defense metrics
