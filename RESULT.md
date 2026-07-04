# FlowPepDock Public Result Summary

This public result summary follows the current small-paper reporting口径. It does not use the older default-checkpoint strict-536 table as the headline result.

## Stage I: FlowPepDock Docking

Small-paper Stage I reports the tclip-step3 full strict-536 evaluation:

| Method | N | Peptide CA RMSD <= 2 A | Complex RMSD <= 2 A | Median peptide CA RMSD | Median complex RMSD | Mean DockQ | Median DockQ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FlowPepDock tclip-step3 | 536 | 0.7481 | 0.8209 | 1.3775 | 1.0554 | 0.5432 | 0.5554 |
| RAPiDock | 536 | 0.4795 | 0.7799 | 2.1174 | 0.8511 | 0.6360 | 0.6554 |
| ADCP | 536 | 0.1063 | 0.4552 | 3.6152 | 2.0901 | 0.2553 | 0.2180 |
| Protenix | 536 | 0.3619 | 0.0000 | 2.4544 | 17.6708 | 0.0270 | 0.0154 |
| AlphaFold3 | 536 | 0.2575 | 0.2705 | 6.0914 | 7.1227 | 0.4584 | 0.4584 |

The headline interpretation is that FlowPepDock improves peptide and complex hit rates, while RAPiDock has a higher DockQ mean/median in this table.

## Stage I: Runtime

| Benchmark | RAPiDock | FlowPepDock | Unit |
| --- | ---: | ---: | --- |
| 32-complex fair timing | 3.54 | 1.24 | s/complex |
| Full strict-536 segmented timing | 1.97 | 0.95 | s/complex |

## Stage II: PoseCred-IPG Scoring

Held-out ranking results:

| Split | Top-1 | Top-5 | MRR | NDCG |
| --- | ---: | ---: | ---: | ---: |
| Held-out test | 0.4763 | 0.6852 | 0.5655 | 0.9232 |
| Multi-receptor subset | 0.5211 | 0.7077 | 0.6008 | 0.9167 |

Cross-docking sample-04 headline values used in the small paper:

| Method | Receptor Top-1 | Receptor Top-10 | Receptor Top-20% | Pose Top-1 | Pose Top-25 | Pose Top-50 | Pose Top-20% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PoseCred-IPG | 0.10 | 0.55 | 0.55 | 0.10 | 0.60 | 0.70 | 0.90 |
| GraphPep | 0.20 | 0.35 | 0.35 | 0.20 | 0.45 | 0.55 | 0.85 |
| InterPepScore | 0.05 | 0.30 | 0.30 | 0.05 | 0.30 | 0.40 | 0.60 |

PoseCred-IPG runtime values reported for this setting are 8.29 s for 1,000 poses and 39.95 s for 10,000 poses.

## Reproducibility Notes

- Flow official CSV entries in this public folder are `data/runtime_tables/flow_train_rel.csv`, `data/runtime_tables/flow_val_rel.csv`, and `data/runtime_tables/flow_infer_test536_rel.csv`.
- Model checkpoints and structure assets are not included in Git; obtain them separately as described in `README.md` and `docs/INSTALL.md`.
- Detailed paper-to-repository alignment is summarized in `docs/PAPER_REPOSITORY_ALIGNMENT.md`.
