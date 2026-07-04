# Repository Structure

This document gives a public, reader-facing overview of the FlowPepDock source
tree. It is intended to help users install the project, locate the main entry
points, and understand which files are included directly in GitHub.

## Top-Level Files

- `README.md`: project overview, installation, quick-start commands, required
  assets, citation, and license notes.
- `inference.py`: main FlowPepDock inference entry point.
- `train_flow.py`: Flow model training entry point.
- `scoreing.py`: optional PyRosetta rescoring script.
- `default_inference_args.yaml`: default inference configuration.
- `flowpepdock_env.yaml`: reference Conda environment.
- `requirement.txt`: pip dependency list.
- `CITATION.cff`: software citation metadata.
- `LICENSE` and `NOTICE.md`: license and attribution information.

## Source Packages

- `models/`: model definitions and assembly code.
- `dataset/`: protein and peptide feature builders.
- `utils/`: geometry, sampling, flow matching, inference, and utility modules.
- `posecred_ipg/`: optional PoseCred-IPG pose rescoring and ranking module.
- `scripts/`: data preparation, quality-control, evaluation, post-processing,
  and benchmark helper scripts.

## Configuration and Examples

- `train_models/CGTensorProductEquivariantModel/model_parameters.yml`: default
  model configuration.
- `train_models/CGTensorProductEquivariantModel/README.md`: checkpoint placement
  and checksum notes.
- `examples/csv/`: small CSV inputs for smoke tests and examples.
- `examples/pdb/`: small PDB structures used by the public smoke-test inputs.
- `data/runtime_tables/`: versioned relative-path CSV tables included with the
  public repository.

## Public Documentation

- `docs/INSTALL.md`: setup guide and smoke-test instructions.
- `docs/RELEASE_ASSETS.md`: external asset paths, expected files, and checksum
  policy.
- `docs/GITHUB_RELEASE_CHECKLIST.md`: release hygiene checklist.
- `docs/csv_pipeline.md`: CSV preparation overview.
- `docs/MODEL_BLUEPRINT.md`: short model blueprint.
- `docs/project_overview.md`: compact project summary.
- `FLOW_DATASET_CONTRACT.md`: public dataset and asset entry-point summary.
- `FLOW_RESULT_CONTRACT.md`: public result output and reporting summary.
- `RESULT.md`: public headline result summary.

## External Assets

Large model checkpoints, generated outputs, full structural datasets, and
manuscript-specific artifacts are not stored directly in Git. They should be
provided separately as release assets, DOI-backed archives, or institutional
storage, then placed at the paths documented in `README.md` and
`docs/RELEASE_ASSETS.md`.
