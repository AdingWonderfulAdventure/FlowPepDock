# Installation

This document describes the recommended setup for running FlowPepDock from a
fresh clone.

## System Requirements

- Linux x86_64 is the reference platform.
- Python 3.9.
- CUDA-capable GPU for practical training/inference.
- Conda or Mamba.
- Git LFS or an external file host for checkpoints and large assets.

The reference Conda environment in this repository uses PyTorch 1.11 and CUDA
11.5 packages.  A CPU-only smoke test is supported, but full-scale docking and
training are expected to run on GPU.

## Conda Environment

```bash
conda env create -f flowpepdock_env.yaml -n FlowPepDock
conda activate FlowPepDock
pip install -r requirement.txt
```

Verify core imports:

```bash
python - <<'PY'
import torch
import MDAnalysis
import esm
import rdkit

print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("MDAnalysis", MDAnalysis.__version__)
print("environment ok")
PY
```

## Checkpoints

For the normal source release, download the external asset archive:

```text
FlowPepDock_external_assets.tar.gz
```

Extract it from the repository root:

```bash
tar -xzf FlowPepDock_external_assets.tar.gz -C .
sha256sum -c SHA256SUMS.txt
```

This installs the default Flow checkpoint at:

```text
train_models/CGTensorProductEquivariantModel/flowpepdock_best.pt
```

and the default PoseCred-IPG checkpoint at:

```text
posecred_ipg/final_exports/graph_main_best.pt
```

Large checkpoint files should not be committed to Git.  Publish them through a
release asset, institutional storage, Zenodo, Hugging Face, or another external
artifact store.

The exact release-asset paths, reference sizes, and SHA256 values are listed in
`docs/RELEASE_ASSETS.md`.  If you publish the archive through GitHub Releases or
another artifact host, update that document with the final download URL.

## Data

The official Flow runtime CSV files are:

```text
data/runtime_tables/flow_train_rel.csv
data/runtime_tables/flow_val_rel.csv
data/runtime_tables/flow_infer_test536_rel.csv
```

The strict-536 inference CSV expects canonical structural assets under:

```text
data/processed_test30/
```

Read `FLOW_DATASET_CONTRACT.md` before changing dataset paths, split files, or
checkpoint assumptions.

## Optional PyRosetta

`scoreing.py` requires PyRosetta.  PyRosetta may require a separate license and
its own installer flow.  If you do not need `ref2015` rescoring, you can skip
PyRosetta and run Flow inference with:

```yaml
scoring_function: none
```

## Docker

Build the image:

```bash
docker build --no-cache -t flowpepdock:latest .
```

Run with GPU access:

```bash
docker run --gpus all -it --rm flowpepdock:latest /bin/bash
```

The Docker build context excludes large local assets through `.dockerignore`.
Mount checkpoints and datasets at runtime when possible.

## Smoke Test

After installing dependencies and placing the default checkpoint, run:

```bash
PYTHONPATH=$(pwd) python inference.py \
  --config default_inference_args.yaml \
  --protein_peptide_csv examples/csv/inference_smoke_2_cases.csv \
  --output_dir results/diagnostics/install_smoke_cpu \
  --model_dir train_models/CGTensorProductEquivariantModel \
  --ckpt flowpepdock_best.pt \
  --device cpu \
  --batch_size 2 \
  --N 1 \
  --cpu 5
```

Successful completion should create one output directory per example complex
under `results/diagnostics/install_smoke_cpu/`.

The smoke CSV uses small tracked PDB examples under `examples/pdb/`.  If these
files are missing, the source checkout or release archive is incomplete.
Release archives should also include the three legacy `2kid / 2rui` smoke-test
PDB files listed in `docs/RELEASE_ASSETS.md` for compatibility with older CSVs.
