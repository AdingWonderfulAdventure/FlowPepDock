# Git Release Commands

> This checklist is scoped to `/root/FlowPepDock_github`.
> Do not run these commands from `/root` or another parent directory.

## 1. Enter the Repository

```bash
cd /root/FlowPepDock_github
git rev-parse --show-toplevel
```

Expected output:

```text
/root/FlowPepDock_github
```

## 2. Configure Identity

```bash
git config user.name "linf35927-cmd"
git config user.email "linf35927@gmail.com"
```

## 3. Stage Source Files

```bash
git add \
  .dockerignore .gitignore \
  CITATION.cff CONTRIBUTING.md Dockerfile LICENSE NOTICE.md \
  README.md RESULT.md FLOW_DATASET_CONTRACT.md FLOW_RESULT_CONTRACT.md \
  default_inference_args.yaml flowpepdock_env.yaml requirement.txt \
  inference.py train_flow.py scoreing.py \
  docs data/runtime_tables dataset examples models scripts utils posecred_ipg \
  train_models/CGTensorProductEquivariantModel/model_parameters.yml \
  train_models/CGTensorProductEquivariantModel/README.md
```

## 4. Stage Legacy Smoke-Test PDBs

```bash
git add \
  data/rebuild_isolated/rebuild_20251221_163301/processed/2kid/receptor.pdb \
  data/rebuild_isolated/rebuild_20251221_163301/processed/2kid/peptide.pdb \
  data/rebuild_isolated/rebuild_20251221_163301/processed/2rui/receptor.pdb
```

## 5. Inspect Staged Files

```bash
git status --short
```

Stop if any of these appear as staged files:

```text
train_models/CGTensorProductEquivariantModel/flowpepdock_best.pt
posecred_ipg/final_exports/graph_main_best.pt
FlowPepDock_external_assets.tar.gz
小论文.docx
.so3_*.npy
results/
__pycache__/
```

If a checkpoint was staged by mistake, unstage it while keeping the local file:

```bash
git rm --cached "train_models/CGTensorProductEquivariantModel/flowpepdock_best.pt"
git rm --cached "posecred_ipg/final_exports/graph_main_best.pt"
```

If the manuscript was staged by mistake:

```bash
git rm --cached "小论文.docx"
```

## 6. Commit

```bash
git commit -m "Prepare FlowPepDock GitHub release"
```

## 7. Use the Main Branch

```bash
git branch -M main
```

## 8. Configure Remote

If `origin` does not exist:

```bash
git remote add origin git@github.com:linf35927-cmd/FlowPepdock.git
```

If `origin` already exists but points somewhere else:

```bash
git remote set-url origin git@github.com:linf35927-cmd/FlowPepdock.git
```

Check it:

```bash
git remote -v
```

## 9. Push

```bash
git push -u origin main
```

If SSH still fails with `Permission denied (publickey)`, fix the GitHub SSH key
first or switch the remote to HTTPS.

## 10. Publish External Assets Separately

Do not commit checkpoint files or the external asset archive to normal Git.

Upload this file to GitHub Release assets:

```text
/root/FlowPepDock_external_assets.tar.gz
```

It contains:

```text
EXTERNAL_ASSETS_README.md
SHA256SUMS.txt
train_models/CGTensorProductEquivariantModel/flowpepdock_best.pt
posecred_ipg/final_exports/graph_main_best.pt
```

Users install it from the repository root with:

```bash
tar -xzf FlowPepDock_external_assets.tar.gz -C .
sha256sum -c SHA256SUMS.txt
```
