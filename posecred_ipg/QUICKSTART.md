# PoseCred-IPG Quickstart

Run commands from the repository root. Install the FlowPepDock environment first
as described in `docs/INSTALL.md`.

## Required Assets

For full PoseCred-IPG inference or evaluation, place the checkpoint at:

```text
posecred_ipg/final_exports/graph_main_best.pt
```

This checkpoint is distributed separately and is not committed to Git.

## Train

```bash
python -m posecred_ipg.train --help
```

Use the help output to inspect the available dataset, snapshot, and optimization
arguments for the installed version.

## Evaluate a Checkpoint

```bash
python -m posecred_ipg.evaluate_checkpoint --help
python -m posecred_ipg.evaluate_pooled_global --help
```

These entry points report checkpoint-level and pooled/global ranking metrics.

## Score Cross-Pose Tables

```bash
python -m posecred_ipg.score_cross_pose_table --help
```

Use this command to score candidate pose tables after preparing the required
input CSV and structure assets.

## Build Records and Shards

```bash
python -m posecred_ipg.build_records --help
python -m posecred_ipg.build_record_snapshot --help
python -m posecred_ipg.build_record_shards --help
```

These commands construct PoseRecord datasets and shard indexes for training or
evaluation workflows.

## Notes

- Generated record snapshots, shard files, and scoring outputs can be large and
  should stay outside normal Git history.
- Lightweight exported CSV/JSON summaries under `posecred_ipg/final_exports/`
  are included when they are useful for public documentation.
