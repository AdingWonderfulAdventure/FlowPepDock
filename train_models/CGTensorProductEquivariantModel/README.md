# FlowPepDock checkpoint

This directory stores the FlowPepDock model configuration and the expected
location for the external Flow checkpoint.

Required file for the default inference commands:

```text
train_models/CGTensorProductEquivariantModel/flowpepdock_best.pt
```

The checkpoint is intentionally not tracked by Git because it is a large binary
artifact. For the normal source release, download and extract
`FlowPepDock_external_assets.tar.gz` from the repository root.  The archive
places this checkpoint at the exact path above and also installs the default
PoseCred-IPG checkpoint.  Reference file sizes and SHA256 hashes are listed in
`docs/RELEASE_ASSETS.md`.

After copying the file, verify the local layout from the repository root:

```bash
test -s train_models/CGTensorProductEquivariantModel/flowpepdock_best.pt
test -f train_models/CGTensorProductEquivariantModel/model_parameters.yml
```

The default `flowpepdock_best.pt` checkpoint uses the default inference settings
documented in `default_inference_args.yaml` and `README.md`.
