# PoseCred-IPG

PoseCred-IPG is the optional pose rescoring and ranking module distributed with
FlowPepDock. It scores candidate protein-peptide docking poses and can be used
for checkpoint evaluation, pooled/global evaluation, and cross-pose scoring.

## Public Entry Points

- `posecred_ipg/QUICKSTART.md`: command examples for training, evaluation, and
  scoring.
- `posecred_ipg/docs/00_INDEX.md`: topic-oriented documentation index.
- `posecred_ipg/final_exports/README.md`: exported artifact notes.
- `posecred_ipg/docs/11_SCORE_SEMANTICS.md`: score interpretation notes.

## Included Files

The public repository includes source code, lightweight result summaries, and
CSV index snapshots under `posecred_ipg/final_exports/`. Large model checkpoints
are external release assets.

Expected optional checkpoint path:

```text
posecred_ipg/final_exports/graph_main_best.pt
```

The checkpoint is not tracked by Git. Its size and checksum are listed in
`docs/RELEASE_ASSETS.md`.

## Package Layout

- `posecred_ipg/core/`: configuration, constants, paths, and record utilities.
- `posecred_ipg/data/`: PDB parsing, feature construction, datasets, and IO.
- `posecred_ipg/models/`: graph and baseline models.
- `posecred_ipg/engine/`: training, losses, metrics, and runtime helpers.
- `posecred_ipg/evaluation/`: checkpoint, pooled/global, and cross-pose scoring.
- `posecred_ipg/pipelines/`: record, snapshot, shard, and validation pipelines.
- `posecred_ipg/experiments/`: smoke tests and benchmark helpers.
- `posecred_ipg/final_exports/`: lightweight exported summaries and indexes.

Root-level `posecred_ipg/*.py` files are retained as compatibility entry points
for older commands.
