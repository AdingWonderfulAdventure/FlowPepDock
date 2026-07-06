# FlowPepDock

FlowPepDock is a research codebase for protein-peptide docking.  It contains a
flow-based generative docking model for peptide pose sampling and an optional
PoseCred-IPG module for pose rescoring and ranking.

This repository is organized as a reproducible research artifact: the main
entry points, environment files, example inputs, dataset contracts, and result
documentation are kept in the repository, while large checkpoints and structural
datasets should be distributed separately.

## Highlights

- Flow-based protein-peptide complex generation and docking.
- ESM-based sequence/structure feature support.
- Optional PyRosetta `ref2015` interface rescoring.
- PoseCred-IPG rescoring module for docking-pose ranking.
- CPU smoke-test input for quick installation checks.
- Dataset and result contracts for reproducible training, inference, and
  evaluation.

## Repository Layout

```text
FlowPepDock/
|-- inference.py                         # FlowPepDock inference entry point
|-- train_flow.py                        # Flow model training entry point
|-- scoreing.py                          # Optional PyRosetta rescoring script
|-- default_inference_args.yaml          # Default inference configuration
|-- flowpepdock_env.yaml                 # Conda environment specification
|-- requirement.txt                      # Pip-only Python dependencies
|-- models/                              # Model definitions
|-- dataset/                             # Protein/peptide feature builders
|-- utils/                               # Geometry, sampling, flow, and parsing utilities
|-- posecred_ipg/                        # PoseCred-IPG rescoring subproject
|-- train_models/                        # Model config directory; checkpoints are external
|-- examples/csv/                        # Small example CSV inputs
|-- docs/                                # Runtime, dataset, and project documentation
|-- notes/                               # Maintainer notes and historical context
`-- data/runtime_tables/                 # Versioned CSV entry tables
```

Large generated outputs, checkpoints, PDB/CIF files, caches, and local datasets
are intentionally excluded from Git.  See `.gitignore`, `.dockerignore`, and
`docs/GITHUB_RELEASE_CHECKLIST.md` before publishing a release.

## Installation

FlowPepDock targets Python 3.9.  The reference environment uses PyTorch 1.11,
CUDA 11.5, PyG, RDKit, MDAnalysis, and FAIR-ESM.

```bash
conda env create -f flowpepdock_env.yaml -n FlowPepDock
conda activate FlowPepDock
pip install -r requirement.txt
```

For a detailed setup guide, dependency notes, and optional PyRosetta setup, see
`docs/INSTALL.md`.

## Required Assets

The repository contains code, configuration files, and small CSV examples.  Full
training/inference assets are not expected to be committed to GitHub.

Before running full experiments, prepare the following assets:

- Flow checkpoint:
  `train_models/CGTensorProductEquivariantModel/flowpepdock_best.pt`
- PoseCred-IPG checkpoint, if the rescoring module is used:
  `posecred_ipg/final_exports/graph_main_best.pt`
- Flow runtime tables:
  `data/runtime_tables/flow_train_rel.csv`,
  `data/runtime_tables/flow_val_rel.csv`,
  `data/runtime_tables/flow_infer_test536_rel.csv`
- Full benchmark structural assets, if running beyond the included smoke-test
  examples.
- PoseCred-IPG assets and snapshot files under `posecred_ipg/`, if the rescoring
  module is used.

The current dataset and checkpoint policy is defined in
`FLOW_DATASET_CONTRACT.md`.  The release-asset placement and checksum policy is
listed in `docs/RELEASE_ASSETS.md`.  Read both before running training,
inference, or evaluation jobs.

SO(3) sampling cache files are generated automatically on first run when absent;
they do not need to be downloaded as release assets.

For a GitHub-style source release, the two required checkpoint files should be
distributed outside Git as `FlowPepDock_external_assets.tar.gz`.  Extract that
archive from the repository root before running the smoke test:

```bash
tar -xzf FlowPepDock_external_assets.tar.gz -C .
sha256sum -c SHA256SUMS.txt
```

## Quick Start

All commands below assume they are executed from the repository root.

### CPU Inference Smoke Test

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

The smoke test verifies that parsing, feature construction, model loading, and
pose writing are wired correctly.  It requires the default checkpoint to be
available in `train_models/CGTensorProductEquivariantModel/`.  Its PDB inputs
are tracked under `examples/pdb/`, so it does not depend on private local data
directories.

### Default Flow Inference

```bash
PYTHONPATH=$(pwd) python inference.py \
  --config default_inference_args.yaml \
  --protein_peptide_csv data/runtime_tables/flow_infer_test536_rel.csv \
  --output_dir results/docking_module_compare/FlowPep_Strict_536 \
  --model_dir train_models/CGTensorProductEquivariantModel \
  --ckpt flowpepdock_best.pt \
  --cpu 4
```

Default values in `default_inference_args.yaml`:

- `scoring_function: none`
- `batch_size: 16`
- `N: 10`
- `flow_num_steps: 10`
- `flow_solver: euler`
- `amp: false`
- `prealign_to_native_center: true`

Alternative checkpoints may require different inference settings. Record the
checkpoint path, checksum, and effective `flow_num_steps` when reporting custom
or reproduced benchmark runs.

### Flow Training Smoke Test

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python train_flow.py \
  --config train_models/CGTensorProductEquivariantModel/model_parameters.yml \
  --train_csv data/runtime_tables/flow_train_rel.csv \
  --val_csv data/runtime_tables/flow_val_rel.csv \
  --log_dir logs/flow_runtime_rel \
  --batch_size 1 \
  --epochs 1 \
  --num_workers 0 \
  --embedding esm \
  --eval_every 0
```

### Optional PyRosetta Rescoring

```bash
python scoreing.py \
  --pdb results/default/7BBG/rank1_ref2015.pdb \
  --interface AC
```

PyRosetta is optional and may require separate licensing/installation.

### PoseCred-IPG Smoke Test

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd) python -m posecred_ipg.overfit_smoke \
  --num_poses 8 \
  --epochs 1 \
  --gpu_ids 0 \
  --target_top1_success 0 \
  --target_mrr 0 \
  --target_spearman 0 \
  --target_ndcg 0 \
  --output_dir tmp/posecred_ipg_runtime_rel
```

For more PoseCred-IPG details, see `posecred_ipg/README.md` and
`posecred_ipg/QUICKSTART.md`.

## Docker

```bash
docker build --no-cache -t flowpepdock:latest .
docker run --gpus all -it --rm flowpepdock:latest /bin/bash
```

The Docker image installs the Conda environment from `flowpepdock_env.yaml`.
Large data, checkpoints, and result folders are excluded from the build context
by `.dockerignore`.

## Reproducibility Contracts

FlowPepDock keeps dataset and result assumptions in explicit contract files:

- `FLOW_DATASET_CONTRACT.md` defines official Flow/PoseCred-IPG data entry
  points and checkpoint assumptions.
- `FLOW_RESULT_CONTRACT.md` summarizes runtime outputs, result reporting, and
  artifact publication conventions.
- `docs/FILE_STRUCTURE.md` provides a public overview of the repository layout.
- `docs/RELEASE_ASSETS.md` lists external checkpoints, optional caches, target
  paths, and checksum expectations.

If dataset paths, checkpoints, or key hyperparameters change, update the contract
files and the README in the same pull request.

## Outputs

Inference outputs are written below the selected `--output_dir`.  With
`--scoring_function none`, generated structures are named:

```text
pose1.pdb
pose2.pdb
...
poseN.pdb
```

If visualization is enabled, reverse-process structures are written as
`pose*_reverseprocess.pdb`.  Generated PDB files, result tables, logs, and caches
are ignored by Git by default.

## Development Notes

- Use `rg` for repository search.
- Keep CLI scripts thin and move reusable logic into `utils/`, `dataset/`, or
  model modules.
- Prefer typed helper functions and explicit configuration over hidden global
  state.
- Do not commit checkpoints, generated structures, raw datasets, local caches, or
  personal analysis artifacts.
- Before release, run the checklist in `docs/GITHUB_RELEASE_CHECKLIST.md`.

## Citation

If this repository is useful for your research, please cite it using the metadata
in `CITATION.cff`.  Update the citation entry with the final paper title, author
list, venue, DOI, and release tag when the manuscript or preprint is available.

## License

This project is distributed under the license in `LICENSE`.  Third-party
packages, pretrained models, datasets, and PyRosetta may be governed by their own
licenses.  Verify those terms before redistribution.
