# Paper and Repository Alignment Audit

This note records how the current small-paper manuscript maps back to the
repository assets.  It is intended to prevent the paper, README, and result
ledger from drifting into different numerical stories.

## Manuscript Checked

This public audit records the numerical values used by the current small-paper manuscript. The manuscript source files and private evidence handoff documents are not included in the GitHub code release.

## Stage I Main Accuracy Table

The small paper uses the tclip-step3 full strict-536 evaluation, not the older
`flowpepdock_best.pt` strict-536 table.

| Method | Cases | Peptide CA RMSD <= 2 A | Complex RMSD <= 2 A | Median peptide CA RMSD | Median complex RMSD | Mean DockQ | Median DockQ | Evidence note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FlowPepDock | 536 | 0.7481 | 0.8209 | 1.3775 | 1.0554 | 0.5432 | 0.5554 | internal paper evidence table |
| RAPiDock | 536 | 0.4795 | 0.7799 | 2.1174 | 0.8511 | 0.6360 | 0.6554 | internal paper evidence table |
| ADCP | 536 | 0.1063 | 0.4552 | 3.6152 | 2.0901 | 0.2553 | 0.2180 | internal benchmark evidence table |
| Protenix | 536 | 0.3619 | 0.0000 | 2.4544 | 17.6708 | 0.0270 | 0.0154 | internal benchmark evidence table |
| AlphaFold3 | 536 | 0.2575 | 0.2705 | 6.0914 | 7.1227 | 0.4584 | 0.4584 | internal benchmark evidence table |
| AlphaFold3 (BCS) | 290 | 0.2759 | 0.3276 | 3.8508 | 5.2487 | 0.5662 | 0.6625 | internal paper evidence handoff |
| FlexPepDock (BCS) | 290 | 0.0000 | 0.1483 | 19.4735 | 4.5433 | 0.1149 | 0.0771 | internal benchmark evidence table |
| HADDOCK3 (RSS) | 96 | 0.0000 | 0.1290 | 16.8507 | 3.8740 | 0.3091 | 0.3110 | internal benchmark evidence table |

Important distinction:

- internal paper evidence table
  is the small-paper Stage I main source.
- older internal default-checkpoint table
  is an older `flowpepdock_best.pt` strict-536 table and should be treated as a
  historical/default-checkpoint record, not the small-paper headline result.

## Stage I Runtime Table

The small paper reports:

| Timing setting | RAPiDock | FlowPepDock | Unit | Evidence note |
| --- | ---: | ---: | --- | --- |
| 32-complex fair timing benchmark | 3.54 | 1.24 | s/complex | internal paper timing table |
| Full strict-536 segmented timing task | 1.97 | 0.95 | s/complex | internal paper timing table plus appendix timing JSON |

The full strict-536 table in `full536_speed_compare.csv` stores Flow as
`0.946826 s/complex`.  The small paper rounds this to `0.95`.  RAPiDock's
`1.97 s/complex` is the same-caliber half536/single-process timing interpretation
documented in internal paper evidence handoff.

## Stage II Held-Out Reranking Table

The small paper reports PoseCred-IPG held-out metrics from internal held-out evaluation reports.

Headline values:

- Held-out test: 7,180 poses, 718 receptor groups, 502 peptides.
- Groupwise Top-1 / Top-5 / MRR / NDCG: `0.4763 / 0.6852 / 0.5655 / 0.9232`.
- Global Top-1 / Top-5 / Top-10% enrichment / best-of-group retrieval:
  `0.4701 / 0.6892 / 2.6356 / 0.9880`.
- Multi-receptor subset global values:
  `0.6176 / 0.8088 / 2.9460 / 0.9118`.

## Stage II Cross-Docking and Speed Table

The small paper Table 4 uses the sample-04 bootstrap display values and selected
speed sources:

The small paper Table 4 values come from internal cross-docking and scoring-speed evidence tables.

Small-paper Table 4 headline values:

| Method | Receptor Top-1 | Receptor Top-10 | Receptor Top-20% | Pose Top-1 | Pose Top-25 | Pose Top-50 | Pose Top-20% | Runtime 1,000 poses | Runtime 10,000 poses |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PoseCred-IPG | 0.10 | 0.55 | 0.55 | 0.10 | 0.60 | 0.70 | 0.90 | 8.29 | 39.95 |
| GraphPep | 0.20 | 0.35 | 0.35 | 0.20 | 0.45 | 0.55 | 0.85 | 1411.31 | - |
| InterPepScore | 0.05 | 0.30 | 0.30 | 0.05 | 0.30 | 0.40 | 0.60 | 127.60 | - |

Do not replace these display values with the older
older internal cross-docking summary
numbers unless the manuscript is explicitly revised.

## Case Tables

The current small paper contains two case tables:

- 9RVT / GR1.4: top-ranked `pose14`, unified score `1.1720`, DockQ `0.8400`.
- HKILHRLLQE receptor-panel case:
  - rank 1: `1ZBK`, non-cognate receptor, score `1.944771`
  - rank 8: `6IJR`, cognate receptor, score `1.557410`, `pose1`
  - rank 14: `4JYI`, cognate receptor, score `1.418758`, `pose4`

The detailed case source is documented in internal paper evidence handoff and
under `results/paper_cases_hybrid_N50_pose_replacements_strict_structure_20260525/`.

## GitHub Release Implications

For a public code release:

- Keep manuscript sources, generated figures, timing artifacts, and generated result assets out of the normal GitHub upload unless preparing a separate artifact/supplement repository.
- Keep source scripts that reproduce paper figures only if the required input
  tables are also available or the scripts are clearly documented as optional.
- In public-facing README text, avoid hard-coding all paper result tables unless
  the corresponding result artifacts are included or externally hosted with
  checksums.
- Use `RESULT.md` and this audit note to distinguish paper headline values from non-headline exploratory records.
