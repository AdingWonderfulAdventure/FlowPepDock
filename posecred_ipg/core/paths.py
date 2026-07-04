from __future__ import annotations

from pathlib import Path
from typing import Union


REPO_ROOT = Path(__file__).resolve().parents[2]
POSECRED_ROOT = REPO_ROOT / "posecred_ipg"
DATA_ROOT = REPO_ROOT / "data"
TMP_ROOT = REPO_ROOT / "tmp"
RUNTIME_TABLES_ROOT = DATA_ROOT / "runtime_tables"
FLOW_TRAIN_REL_CSV = RUNTIME_TABLES_ROOT / "flow_train_rel.csv"
FLOW_VAL_REL_CSV = RUNTIME_TABLES_ROOT / "flow_val_rel.csv"
FLOW_INFER_TEST536_REL_CSV = RUNTIME_TABLES_ROOT / "flow_infer_test536_rel.csv"
POSECRED_TRAIN_DOCKED_REL_TABLE = RUNTIME_TABLES_ROOT / "posecred_ipg_train_docked_rel.csv"
POSECRED_VAL_DOCKED_REL_TABLE = RUNTIME_TABLES_ROOT / "posecred_ipg_val_docked_rel.csv"
LEGACY_REPO_ROOTS = (
    Path("/root/FlowPepDock"),
    Path("/root/archive_mydocking_module"),
)


def remap_legacy_repo_path(path: Union[str, Path]) -> Path:
    candidate = Path(path)
    candidate_str = str(candidate)
    for legacy_root in LEGACY_REPO_ROOTS:
        legacy_prefix = str(legacy_root)
        if not candidate_str.startswith(legacy_prefix):
            continue
        relative = Path(candidate_str[len(legacy_prefix) :].lstrip("/"))
        remapped = REPO_ROOT / relative
        if remapped.exists():
            return remapped
    return candidate
