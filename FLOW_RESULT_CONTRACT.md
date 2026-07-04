# Result Reporting Summary

This public note describes how FlowPepDock outputs and reported results are
organized for readers of the repository.

## Runtime Outputs

Inference and evaluation commands write generated files under user-selected
output directories, typically below `results/`. These directories are excluded
from Git because they can contain large pose sets, logs, intermediate files, and
benchmark artifacts.

Recommended local layout:

- `results/diagnostics/`: smoke tests and installation checks.
- `results/inference/`: user inference runs.
- `results/benchmarks/`: reproduced benchmark outputs.

Users may choose different output directories with the relevant command-line
arguments.

## Included Result Summary

`RESULT.md` provides a compact public summary of headline results and
reproducibility notes. Full benchmark tables, generated figures, timing logs,
and manuscript-specific evidence files should be distributed separately when
needed for publication review or archival release.

## Reproducibility Notes

For reproducible reporting:

- Record the Git commit hash.
- Record the checkpoint filename and checksum.
- Record the input CSV path.
- Record major inference settings such as `N`, `batch_size`, `flow_num_steps`,
  solver, device, and scoring options.
- Keep generated result directories outside Git unless publishing a small,
  deliberate example.

## External Result Artifacts

Large benchmark outputs should be published as release assets, supplementary
archives, Zenodo records, or other DOI-backed storage. The code repository should
remain focused on source code, installation instructions, examples, and
lightweight reproducibility manifests.
