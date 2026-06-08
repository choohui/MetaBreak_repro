from __future__ import annotations

CAT_A = "A_system_special"
CAT_B = "B_malicious_mimicry"
CAT_C = "C_benign_mimicry"
CAT_D = "D_malicious_special"
CAT_E = "E_benign_special"
CAT_F = "F_positioned_regular"
CAT_G = "G_ordinary_regular"

LETTER_TO_CAT = {
    "A": CAT_A,
    "B": CAT_B,
    "C": CAT_C,
    "D": CAT_D,
    "E": CAT_E,
    "F": CAT_F,
    "G": CAT_G,
}
CAT_TO_LETTER = {v: k for k, v in LETTER_TO_CAT.items()}

ATTACK_LETTERS = {"B", "D"}
BENIGN_LETTERS = {"C", "E", "F", "G"}
REFERENCE_LETTERS = {"A"}
DEFENSE_LETTERS = tuple("BCDEFG")
ALL_LETTERS = tuple("ABCDEFG")

VARIANT_TO_LETTER = {
    "malicious_mimicry": "B",
    "benign_mimicry": "C",
    "malicious_special": "D",
    "benign_special": "E",
    "positioned_regular": "F",
    "ordinary": "G",
}


def binary_label(letter: str) -> int:
    if letter in ATTACK_LETTERS:
        return 1
    if letter in BENIGN_LETTERS:
        return 0
    return -1


