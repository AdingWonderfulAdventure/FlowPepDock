from __future__ import annotations

from pathlib import Path

from ..paths import POSECRED_ROOT

DEFAULT_SNAPSHOT_ROOT = (
    POSECRED_ROOT
    / "record_snapshots"
    / "cocrystal_positive_only_hybrid_npair32_v1"
    / "hybrid_npair32"
)
DEFAULT_TRAIN_RECORDS_INDEX = DEFAULT_SNAPSHOT_ROOT / "train" / "records_index.csv"
DEFAULT_VAL_RECORDS_INDEX = DEFAULT_SNAPSHOT_ROOT / "val" / "records_index.csv"
DEFAULT_TRAIN_SHARD_INDEX = DEFAULT_SNAPSHOT_ROOT / "train_shards_v1" / "shard_index.csv"
DEFAULT_VAL_SHARD_INDEX = DEFAULT_SNAPSHOT_ROOT / "val_shards_v1" / "shard_index.csv"


def apply_default_shard_snapshot_args(args) -> None:
    if not getattr(args, "use_default_shard_snapshot", False):
        return
    if getattr(args, "train_records_shard_index", None) is None:
        args.train_records_shard_index = DEFAULT_TRAIN_SHARD_INDEX
    if getattr(args, "val_records_shard_index", None) is None:
        args.val_records_shard_index = DEFAULT_VAL_SHARD_INDEX
