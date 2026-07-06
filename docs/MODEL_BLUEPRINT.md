# Model Blueprint

FlowPepDock combines a flow-based generative docking model with optional
post-hoc pose rescoring.

## FlowPepDock

- `train_flow.py`: training entry point.
- `inference.py`: inference entry point.
- `models/`: model definitions.
- `dataset/`: protein and peptide feature builders.
- `utils/`: geometry, SO(3), sampling, and flow-matching utilities.

## PoseCred-IPG

The optional `posecred_ipg/` module provides graph-based pose scoring and
ranking utilities. Its public entry points are documented in
`posecred_ipg/README.md` and `posecred_ipg/QUICKSTART.md`.

Model checkpoints are external release assets and are not stored in normal Git
history.
