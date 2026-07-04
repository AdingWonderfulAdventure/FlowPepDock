from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class FeatureLayout:
    node_slices: Dict[str, slice]
    global_slices: Dict[str, slice]


FEATURE_LAYOUT = FeatureLayout(
    node_slices={
        "residue_identity": slice(0, 52),
        "geometry": slice(52, 59),
        "physchem": slice(59, 64),
        "clash": slice(64, 67),
        "direction": slice(67, 70),
    },
    global_slices={
        "basic": slice(0, 4),
        "clash": slice(4, 12),
    },
)

