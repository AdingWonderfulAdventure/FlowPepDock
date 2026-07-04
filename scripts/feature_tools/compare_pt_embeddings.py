#!/usr/bin/env python
"""
对比同一 pdbid 的 onehot 和 esm PT，验证特征前缀是否一致，仅尾部嵌入不同。

用法示例：
  PYTHONPATH=$(pwd) python scripts/feature_tools/compare_pt_embeddings.py \
    --onehot data/processed/1a0n/features_onehot.pt \
    --esm data/processed/1a0n/features_esm.pt \
    --tail_onehot 104 \
    --tail_esm 1280
"""

import argparse
import torch


def split_feat(x, tail_dim):
    """将节点特征拆成前缀 + 尾部"""
    if x.shape[1] < tail_dim:
        raise ValueError(f"特征维度 {x.shape[1]} 小于尾部 {tail_dim}")
    prefix = x[:, :-tail_dim]
    tail = x[:, -tail_dim:]
    return prefix, tail


def compare_pair(onehot_path, esm_path, tail_onehot, tail_esm):
    onehot = torch.load(onehot_path, map_location="cpu")
    esm = torch.load(esm_path, map_location="cpu")

    report = []
    for node_type in ["receptor", "pep"]:
        xo = onehot[node_type].x
        xe = esm[node_type].x
        po, to = split_feat(xo, tail_onehot)
        pe, te = split_feat(xe, tail_esm)
        prefix_equal = torch.allclose(po, pe, atol=1e-6)
        report.append(
            {
                "node": node_type,
                "prefix_shape": po.shape,
                "tail_onehot_shape": to.shape,
                "tail_esm_shape": te.shape,
                "prefix_equal": bool(prefix_equal),
            }
        )
    return report


def main():
    ap = argparse.ArgumentParser(description="对比 onehot / esm PT 的前缀是否一致")
    ap.add_argument("--onehot", required=True, help="features_onehot.pt 路径")
    ap.add_argument("--esm", required=True, help="features_esm.pt 路径")
    ap.add_argument("--tail_onehot", type=int, default=104, help="onehot 尾部维度")
    ap.add_argument("--tail_esm", type=int, default=1280, help="esm 尾部维度")
    args = ap.parse_args()

    rep = compare_pair(args.onehot, args.esm, args.tail_onehot, args.tail_esm)
    for r in rep:
        print(
            f"[{r['node']}] prefix {r['prefix_shape']}, tail_onehot {r['tail_onehot_shape']}, "
            f"tail_esm {r['tail_esm_shape']}, prefix_equal={r['prefix_equal']}"
        )


if __name__ == "__main__":
    main()
