# Contributing

Thanks for improving FlowPepDock.  Contributions should keep the repository
usable as a reproducible research artifact.

## Development Rules

- Keep changes focused and avoid unrelated cleanup in the same pull request.
- Do not commit checkpoints, generated structures, raw datasets, result folders,
  cache files, or personal analysis artifacts.
- Preserve the official dataset/checkpoint contract in
  `FLOW_DATASET_CONTRACT.md`.
- Update `README.md`, `docs/QUICKSTART_RUNTIME.md`, and relevant contract files
  when commands, paths, defaults, or key hyperparameters change.
- Prefer small, typed helper functions over hidden side effects in CLI scripts.

## Code Style

- Python 3.9.
- Four-space indentation.
- `snake_case` for functions and variables.
- `CamelCase` for classes.
- Standard-library imports first, then third-party imports, then local imports.
- Use `logging` or structured status output for new code where practical.

## Validation

At minimum, run syntax checks before opening a pull request:

```bash
python -m py_compile inference.py scoreing.py
```

When dependencies and checkpoints are available, run the CPU smoke test from
`README.md` or `docs/INSTALL.md`.

## Pull Request Notes

Include:

- Motivation for the change.
- Dataset/checkpoint assumptions.
- Commands used for validation.
- Any numerical changes in output metrics, score tables, or generated poses.
