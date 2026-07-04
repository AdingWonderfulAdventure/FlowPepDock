# GitHub Release Checklist

Use this checklist before pushing a public repository or cutting a release.

## Required Files

- `README.md` explains the project, features, install steps, examples, assets,
  citation, and license.
- `docs/INSTALL.md` contains environment and smoke-test instructions.
- `docs/RELEASE_ASSETS.md` lists external checkpoints, optional SO(3) caches,
  target paths, file sizes, and SHA256 hashes.
- `FLOW_DATASET_CONTRACT.md` reflects the official dataset and checkpoint
  contract.
- `FLOW_RESULT_CONTRACT.md` reflects the official result/reporting contract.
- `CITATION.cff` contains citation metadata.
- `LICENSE` and `NOTICE.md` are present and consistent.
- `.gitignore` and `.dockerignore` exclude generated and local-only assets.

## Do Not Commit

- Model checkpoints: `*.pt`, `*.pth`, `*.ckpt`.
- Raw or generated structures: `*.pdb`, `*.cif`, except the tiny tracked
  `examples/pdb/` smoke-test inputs.
- Large arrays/caches: `*.npy`, `*.npz`, `*.pkl`.
- Runtime outputs: `logs/`, `results/`, `tmp/`, `swanlog/`, `outputs/`.
- Local data mirrors: `data/processed_test30/`, `data/diagnostics/`,
  `data/rebuild_isolated/`, benchmark corpora, and raw dataset dumps.
- Personal thesis/defense artifacts, temporary review bundles, and scratch files.
- Secrets, private paths, credentials, tokens, and machine-specific config.

## Paper Artifacts

- Keep manuscript sources, generated figures, timing artifacts, and full result
  archives out of the normal code repository unless they are intentionally
  published as a separate supplement or release asset.
- Keep public-facing documentation focused on code usage, installation,
  external assets, and reproducibility inputs.
## Recommended Validation

```bash
git status --short
git ls-files -z | xargs -0 du -b | sort -n | tail -40
python -m py_compile inference.py scoreing.py
```

If dependencies and checkpoints are available, also run:

```bash
PYTHONPATH=$(pwd) python inference.py \
  --config default_inference_args.yaml \
  --protein_peptide_csv examples/csv/inference_smoke_2_cases.csv \
  --output_dir results/diagnostics/release_smoke_cpu \
  --model_dir train_models/CGTensorProductEquivariantModel \
  --ckpt flowpepdock_best.pt \
  --device cpu \
  --batch_size 2 \
  --N 1 \
  --cpu 5
```

## Artifact Policy

For public release, publish large files outside the Git repository and document
where they should be placed locally:

- Checkpoints: release asset, Zenodo, Hugging Face, or institutional storage.
- Datasets: repository-agnostic archive with checksum and license information.
- Benchmark results: compressed archive or DOI-backed supplement.
- Smoke-test PDB examples: keep the small `examples/pdb/` inputs in Git so
  `examples/csv/inference_smoke_2_cases.csv` never points to private local data.
- Legacy smoke-test PDB compatibility files: keep the three `2kid / 2rui`
  files listed in `docs/RELEASE_ASSETS.md` so older smoke CSVs do not fail with
  `FileNotFoundError`.

Record checksums for all externally hosted artifacts.
