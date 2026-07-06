# CSV Pipeline

FlowPepDock uses CSV files to define training, validation, inference, and
smoke-test inputs. Public CSV examples are intentionally small and use paths
relative to the repository root.

## Public Examples

- `examples/csv/inference_smoke_2_cases.csv`: minimal inference smoke-test
  manifest.
- `examples/csv/prepare_training_data_example.csv`: example input for the data
  preparation helper.
- `data/runtime_tables/flow_train_rel.csv`: lightweight training manifest.
- `data/runtime_tables/flow_val_rel.csv`: lightweight validation manifest.
- `data/runtime_tables/flow_infer_test536_rel.csv`: lightweight inference
  manifest.

## Expected Columns

- Training and validation manifests use `complex_name,pdb_dir`.
- Inference manifests use `complex_name,receptor_pdb,peptide_pdb`.

Large structure collections referenced by full benchmark manifests should be
distributed as external assets rather than committed directly to Git.
