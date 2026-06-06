from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScalarSpec:
    name: str
    source: str
    layer: int | None = None
    direction: str = "auto"
    normalizer: str = "none"


BASE_SOURCES = ("sink", "value_norm", "output_norm", "active_value", "active_output", "hidden_norm")

