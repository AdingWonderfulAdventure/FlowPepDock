# Dataset and Asset Summary

This public note summarizes the dataset entry points and external assets needed
to run FlowPepDock. It avoids manuscript-specific working paths and focuses on
the files a reader needs for installation, smoke tests, and reproducible runs.

## Included in Git

The repository includes code, configuration files, small examples, and lightweight
CSV tables:

- `examples/csv/inference_smoke_2_cases.csv`: two-case CPU smoke-test input.
- `examples/pdb/`: small receptor and peptide PDB files used by the smoke test.
- `data/runtime_tables/flow_train_rel.csv`: training table with relative paths.
- `data/runtime_tables/flow_val_rel.csv`: validation table with relative paths.
- `data/runtime_tables/flow_infer_test536_rel.csv`: inference/evaluation table
  with relative receptor and peptide paths.

These CSV tables are retained as lightweight run manifests. The full structure
assets they reference may need to be supplied separately, depending on the run.

## External Assets

Full training or benchmark runs require assets that are intentionally excluded
from normal Git history:

- Flow checkpoint:
  `train_models/CGTensorProductEquivariantModel/flowpepdock_best.pt`
- Optional PoseCred-IPG checkpoint:
  `posecred_ipg/final_exports/graph_main_best.pt`
- Full receptor and peptide structure collections for non-smoke-test runs.

The expected release-asset layout and checksum policy are documented in
`docs/RELEASE_ASSETS.md`.

SO(3) sampling cache files are local runtime caches. If absent, FlowPepDock
computes and writes them automatically during the first run.

## Smoke-Test Input

The public smoke test is designed to run against files included in this
repository once the required checkpoint is available:

```bash
PYTHONPATH=$(pwd) python inference.py \
  --config default_inference_args.yaml \
  --protein_peptide_csv examples/csv/inference_smoke_2_cases.csv \
  --output_dir results/diagnostics/readme_smoke_cpu \
  --model_dir train_models/CGTensorProductEquivariantModel \
  --ckpt flowpepdock_best.pt \
  --device cpu \
  --batch_size 2 \
  --N 1 \
  --cpu 5
```

## CSV Expectations

FlowPepDock uses CSV inputs to locate structures and label complexes:

- Training and validation tables use `complex_name,pdb_dir`.
- Inference tables use `complex_name,receptor_pdb,peptide_pdb`.
- Paths in public CSVs are relative to the repository root unless documented
  otherwise.

When preparing new data, keep generated outputs, raw downloads, and large
structure collections outside Git. Commit only small example files and
reproducible manifests that are useful to readers.
