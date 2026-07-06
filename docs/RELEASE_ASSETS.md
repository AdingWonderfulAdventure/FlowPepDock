# Release Assets

This repository keeps source code, configuration files, documentation, and small
smoke-test examples in Git. Large checkpoints, full datasets, generated outputs,
and manuscript-specific artifacts should be distributed separately.

## Required Checkpoints

The following checkpoint files are required for full inference or rescoring
runs. Do not commit them as normal Git files.

| Purpose | Target path | Reference size | SHA256 |
| --- | --- | ---: | --- |
| FlowPepDock inference checkpoint | `train_models/CGTensorProductEquivariantModel/flowpepdock_best.pt` | 91,757,338 bytes | `d7dfdb0e5189d498c1fa2845924fee62a1703faf0a47953506909e713db0ebfd` |
| PoseCred-IPG checkpoint | `posecred_ipg/final_exports/graph_main_best.pt` | 2,039,230 bytes | `e70211f6f31990cf89712fe33ac2b509f974ad66174874767520bed69177028c` |

Recommended distribution options include GitHub Release assets, Zenodo,
Hugging Face, or institutional storage.

## Optional Runtime Caches

FlowPepDock may create SO(3) sampling cache files in the repository root on the
first run. These files are local runtime caches, not required release assets:

- `.so3_omegas_array2.npy`
- `.so3_cdf_vals2.npy`
- `.so3_score_norms2.npy`
- `.so3_exp_score_norms2.npy`

If these files are missing, `utils/so3.py` computes them automatically. They are
ignored by Git and do not need to be downloaded by users.

## Checksum Verification

After downloading checkpoint assets, verify the files from the repository root:

```bash
sha256sum train_models/CGTensorProductEquivariantModel/flowpepdock_best.pt
sha256sum posecred_ipg/final_exports/graph_main_best.pt
```

The output should match the checksums above. If a checksum does not match,
download the asset again before running full inference or benchmark jobs.

## Smoke-Test Inputs

The public smoke-test CSV references small example structures included in Git:

```text
examples/pdb/7bbg/receptor.pdb
examples/pdb/7bbg/peptide.pdb
examples/pdb/7al2/receptor.pdb
examples/pdb/7al2/peptide.pdb
```

These files are intentionally small and are safe to keep in the source
repository. Full structural datasets should remain external.
