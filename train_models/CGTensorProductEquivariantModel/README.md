# FlowPepDock checkpoint

This directory stores the FlowPepDock model configuration and the expected
location for the external Flow checkpoint.

Required file for the default inference commands:

```text
train_models/CGTensorProductEquivariantModel/flowpepdock_best.pt
```

The checkpoint is intentionally not tracked by Git because it is a large binary
artifact. Download it from the external release location described in
`docs/RELEASE_ASSETS.md`, then place it at the exact path above.

After copying the file, verify the local layout from the repository root:

```bash
test -s train_models/CGTensorProductEquivariantModel/flowpepdock_best.pt
test -f train_models/CGTensorProductEquivariantModel/model_parameters.yml
```

The default checkpoint filename is `flowpepdock_best.pt`; place it in this
directory before running the default inference commands documented in
`default_inference_args.yaml` and `README.md`.
