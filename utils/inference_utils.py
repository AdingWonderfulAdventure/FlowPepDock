
import copy
import os
import shutil
import multiprocessing
import hashlib
import esm
import MDAnalysis
import torch
import torch.nn.functional as F
from dataset.protein_feature import three2idx as rec_three2idx
from dataset.peptide_feature import three2idx as pep_three2idx
import Bio.PDB
from torch_geometric.data import Dataset, HeteroData
import numpy as np
from esm import FastaBatchedDataset, pretrained
from utils.dataset_utils import three_to_one, standard_residue_sort, get_sequences, get_sequences_from_pdbfile
from dataset.protein_feature import get_protein_feature_mda
from dataset.peptide_feature import get_ori_peptide_feature_mda, get_updated_peptide_feature

_RECEPTOR_CACHE_SCHEMA_VERSION = "flow_receptor_graph_v1"
_COMPLEX_GRAPH_CACHE_SCHEMA_VERSION = "flow_complex_graph_v1"

def set_nones(l):
    """把字符串'nan'统一转成None，方便后续判空"""
    return [s if str(s) != 'nan' else None for s in l]


def _normalize_local_path(path):
    return os.path.realpath(os.path.abspath(os.path.expanduser(str(path))))


def _build_receptor_cache_meta(protein_file, embedding_mode, has_lm_embedding):
    source_path = _normalize_local_path(protein_file)
    stat = os.stat(source_path)
    return {
        "schema_version": _RECEPTOR_CACHE_SCHEMA_VERSION,
        "source_path": source_path,
        "source_name": os.path.basename(source_path),
        "source_stem": os.path.splitext(os.path.basename(source_path))[0],
        "source_size": int(stat.st_size),
        "source_mtime_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))),
        "embedding_mode": str(embedding_mode),
        "has_lm_embedding": bool(has_lm_embedding),
    }


def _receptor_cache_key(meta):
    raw = "|".join(
        [
            str(meta["schema_version"]),
            str(meta["source_path"]),
            str(meta["source_size"]),
            str(meta["source_mtime_ns"]),
            str(meta["embedding_mode"]),
            str(int(bool(meta["has_lm_embedding"]))),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _receptor_cache_path(cache_dir, meta):
    cache_key = _receptor_cache_key(meta)
    subdir = os.path.join(_normalize_local_path(cache_dir), _RECEPTOR_CACHE_SCHEMA_VERSION, cache_key[:2])
    filename = f"{meta['source_stem']}.{cache_key}.pt"
    return os.path.join(subdir, filename)


def _validate_receptor_graph_data(rec_data):
    if not isinstance(rec_data, HeteroData):
        raise TypeError(f"receptor cache payload 必须是 HeteroData，收到 {type(rec_data)}")
    required = [
        ("receptor", "pos"),
        ("receptor", "tips"),
        ("receptor", "node_v"),
        ("receptor", "x"),
        (("receptor", "rec_contact", "receptor"), "edge_index"),
        (("receptor", "rec_contact", "receptor"), "edge_s"),
        (("receptor", "rec_contact", "receptor"), "edge_v"),
    ]
    for node_key, attr in required:
        if not hasattr(rec_data[node_key], attr):
            raise KeyError(f"receptor cache 缺少字段: {node_key}.{attr}")
    if rec_data["receptor"].pos.ndim != 2 or rec_data["receptor"].pos.shape[0] <= 1:
        raise ValueError("receptor cache pos 形状非法")
    return True


def _load_receptor_graph_payload(pt_path, expected_meta=None):
    payload = torch.load(pt_path, map_location="cpu")
    meta = payload.get("meta") if isinstance(payload, dict) else None
    rec_data = payload.get("data") if isinstance(payload, dict) and "data" in payload else payload
    _validate_receptor_graph_data(rec_data)
    if expected_meta is not None:
        if meta is None:
            raise ValueError("receptor cache 缺少 meta，无法校验")
        for key in ["schema_version", "source_path", "source_size", "source_mtime_ns", "embedding_mode", "has_lm_embedding"]:
            if meta.get(key) != expected_meta.get(key):
                raise ValueError(f"receptor cache meta 不匹配: key={key} got={meta.get(key)} expected={expected_meta.get(key)}")
    return rec_data, meta


def _clone_receptor_graph_data(rec_data):
    cloned = HeteroData()
    cloned["receptor"].pos = rec_data["receptor"].pos.clone()
    cloned["receptor"].tips = rec_data["receptor"].tips.clone()
    cloned["receptor"].node_v = rec_data["receptor"].node_v.clone()
    cloned["receptor"].x = rec_data["receptor"].x.clone()
    cloned["receptor", "rec_contact", "receptor"].edge_index = rec_data["receptor", "rec_contact", "receptor"].edge_index.clone()
    cloned["receptor", "rec_contact", "receptor"].edge_s = rec_data["receptor", "rec_contact", "receptor"].edge_s.clone()
    cloned["receptor", "rec_contact", "receptor"].edge_v = rec_data["receptor", "rec_contact", "receptor"].edge_v.clone()
    if hasattr(rec_data, "original_center"):
        cloned.original_center = rec_data.original_center.clone()
    return cloned


def _clone_cached_complex_graph_for_inference(complex_graph):
    """缓存命中时做轻量副本，避免整图 clone 的大开销。"""
    cloned = copy.copy(complex_graph)
    try:
        pep_a = complex_graph["pep_a"]
    except Exception:
        pep_a = None
    if pep_a is not None and hasattr(pep_a, "pos"):
        cloned["pep_a"].pos = pep_a.pos.clone()
    return cloned


def _build_receptor_graph_data(
    c_alpha_coords_rec,
    tip_coords_rec,
    node_v_rec,
    rec_feat,
    edge_index_rec,
    edge_s_rec,
    edge_v_rec,
):
    rec_data = HeteroData()
    protein_center = torch.mean(c_alpha_coords_rec, dim=0, keepdim=True).to(dtype=torch.float32)
    rec_data["receptor"].x = rec_feat.to(dtype=torch.float32)
    rec_data["receptor"].pos = c_alpha_coords_rec.to(dtype=torch.float32) - protein_center
    rec_data["receptor"].tips = tip_coords_rec.to(dtype=torch.float32) - protein_center
    rec_data["receptor"].node_v = node_v_rec.to(dtype=torch.float32)
    rec_data["receptor", "rec_contact", "receptor"].edge_index = edge_index_rec
    rec_data["receptor", "rec_contact", "receptor"].edge_s = edge_s_rec.to(dtype=torch.float32)
    rec_data["receptor", "rec_contact", "receptor"].edge_v = edge_v_rec.to(dtype=torch.float32)
    rec_data.original_center = protein_center
    return rec_data


def _build_receptor_feature_tensor(
    seq_rec,
    node_s_rec,
    lm_embeddings_rec,
    embedding_mode,
):
    if embedding_mode == "esm" and lm_embeddings_rec is not None:
        lm_rec = torch.tensor(lm_embeddings_rec, dtype=torch.float32)
        return torch.cat(
            [
                seq_rec.reshape([-1, 1]).to(dtype=torch.float32),
                node_s_rec.to(dtype=torch.float32),
                lm_rec,
            ],
            dim=1,
        )
    if embedding_mode == "onehot":
        onehot = F.one_hot(seq_rec.long(), num_classes=len(pep_three2idx)).float()
        return torch.cat([seq_rec.reshape([-1, 1]), node_s_rec, onehot], dim=1)
    return torch.cat([seq_rec.reshape([-1, 1]), node_s_rec], dim=1)


def _save_receptor_graph_payload_atomic(cache_path, meta, rec_data):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp_path = f"{cache_path}.tmp.{os.getpid()}"
    payload = {
        "meta": dict(meta),
        "data": rec_data,
    }
    torch.save(payload, tmp_path)
    os.replace(tmp_path, cache_path)


def _build_complex_graph_cache_meta(
    complex_name,
    protein_file,
    peptide_description,
    receptor_pt,
    embedding_mode,
    has_rec_embedding,
    has_pep_embedding,
):
    protein_path = _normalize_local_path(protein_file) if protein_file else ""
    peptide_path = _normalize_local_path(peptide_description) if peptide_description else ""
    receptor_pt_path = _normalize_local_path(receptor_pt) if receptor_pt else ""
    protein_stat = os.stat(protein_path) if protein_path and os.path.isfile(protein_path) else None
    peptide_stat = os.stat(peptide_path) if peptide_path and os.path.isfile(peptide_path) else None
    receptor_pt_stat = os.stat(receptor_pt_path) if receptor_pt_path and os.path.isfile(receptor_pt_path) else None
    return {
        "schema_version": _COMPLEX_GRAPH_CACHE_SCHEMA_VERSION,
        "complex_name": str(complex_name),
        "protein_path": protein_path,
        "protein_size": int(protein_stat.st_size) if protein_stat else 0,
        "protein_mtime_ns": int(getattr(protein_stat, "st_mtime_ns", int(protein_stat.st_mtime * 1e9))) if protein_stat else 0,
        "peptide_path": peptide_path,
        "peptide_size": int(peptide_stat.st_size) if peptide_stat else 0,
        "peptide_mtime_ns": int(getattr(peptide_stat, "st_mtime_ns", int(peptide_stat.st_mtime * 1e9))) if peptide_stat else 0,
        "receptor_pt_path": receptor_pt_path,
        "receptor_pt_size": int(receptor_pt_stat.st_size) if receptor_pt_stat else 0,
        "receptor_pt_mtime_ns": int(getattr(receptor_pt_stat, "st_mtime_ns", int(receptor_pt_stat.st_mtime * 1e9))) if receptor_pt_stat else 0,
        "embedding_mode": str(embedding_mode),
        "has_rec_embedding": bool(has_rec_embedding),
        "has_pep_embedding": bool(has_pep_embedding),
    }


def _complex_graph_cache_key(meta):
    raw = "|".join(
        [
            str(meta["schema_version"]),
            str(meta["complex_name"]),
            str(meta["protein_path"]),
            str(meta["protein_size"]),
            str(meta["protein_mtime_ns"]),
            str(meta["peptide_path"]),
            str(meta["peptide_size"]),
            str(meta["peptide_mtime_ns"]),
            str(meta["receptor_pt_path"]),
            str(meta["receptor_pt_size"]),
            str(meta["receptor_pt_mtime_ns"]),
            str(meta["embedding_mode"]),
            str(int(bool(meta["has_rec_embedding"]))),
            str(int(bool(meta["has_pep_embedding"]))),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _complex_graph_cache_path(cache_dir, meta):
    cache_key = _complex_graph_cache_key(meta)
    subdir = os.path.join(
        _normalize_local_path(cache_dir),
        _COMPLEX_GRAPH_CACHE_SCHEMA_VERSION,
        cache_key[:2],
    )
    filename = f"{meta['complex_name']}.{cache_key}.pt"
    return os.path.join(subdir, filename)


def _save_complex_graph_payload_atomic(cache_path, meta, complex_graph):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp_path = f"{cache_path}.tmp.{os.getpid()}"
    payload = {
        "meta": dict(meta),
        "graph": complex_graph,
    }
    torch.save(payload, tmp_path)
    os.replace(tmp_path, cache_path)


def _load_complex_graph_payload(cache_path, expected_meta=None):
    payload = torch.load(cache_path, map_location="cpu")
    if isinstance(payload, dict):
        meta = payload.get("meta")
        graph = payload.get("graph")
    else:
        meta = None
        graph = payload
    if not isinstance(graph, HeteroData):
        raise TypeError(f"complex graph cache payload 必须是 HeteroData，收到 {type(graph)}")
    if expected_meta is not None:
        if meta is None:
            raise ValueError("complex graph cache 缺少 meta，无法校验")
        for key in expected_meta.keys():
            if meta.get(key) != expected_meta.get(key):
                raise ValueError(
                    f"complex graph cache meta 不匹配: key={key} got={meta.get(key)} expected={expected_meta.get(key)}"
                )
    return graph, meta


def _resolve_receptor_build_workers(requested_workers, cpu_budget, task_count):
    if task_count <= 1:
        return 0
    if requested_workers is not None:
        return max(0, min(int(requested_workers), int(task_count)))
    cpu_budget = max(1, int(cpu_budget or 1))
    auto_workers = min(max(cpu_budget - 1, 0), int(task_count), 4)
    return max(0, auto_workers)


def _prebuild_single_receptor_cache(task):
    protein_file = task["protein_file"]
    lm_embedding = task["lm_embedding"]
    embedding_mode = task["embedding_mode"]
    receptor_meta = task["receptor_meta"]
    save_path = task["save_path"]
    if os.path.isfile(save_path):
        return {"status": "exists", "path": save_path}
    c_alpha_coords_rec, tip_coords_rec, lm_embeddings_rec, seq_rec, node_s_rec, node_v_rec, edge_index_rec, edge_s_rec, edge_v_rec = get_protein_feature_mda(
        protein_file, lm_embedding_chains=lm_embedding
    )
    rec_feat = _build_receptor_feature_tensor(
        seq_rec=seq_rec,
        node_s_rec=node_s_rec,
        lm_embeddings_rec=lm_embeddings_rec,
        embedding_mode=embedding_mode,
    )
    rec_data = _build_receptor_graph_data(
        c_alpha_coords_rec,
        tip_coords_rec,
        node_v_rec,
        rec_feat,
        edge_index_rec,
        edge_s_rec,
        edge_v_rec,
    )
    if not os.path.isfile(save_path):
        _save_receptor_graph_payload_atomic(
            save_path,
            receptor_meta,
            _clone_receptor_graph_data(rec_data),
        )
        return {"status": "saved", "path": save_path}
    return {"status": "exists", "path": save_path}


def _edge_index_to_numpy(edge_index):
    if torch.is_tensor(edge_index):
        arr = edge_index.detach().cpu().numpy()
    else:
        arr = np.asarray(edge_index)
    if arr.ndim != 2:
        raise ValueError(f"edge_index 维度不对：shape={arr.shape}")
    if arr.shape[0] == 2:
        arr = arr.T
    elif arr.shape[1] != 2:
        raise ValueError(f"edge_index 形状不对：shape={arr.shape}")
    return arr.astype(np.int64, copy=False)


def _compute_bridge_small_components(all_edge_index, num_nodes):
    edges = _edge_index_to_numpy(all_edge_index)
    adjacency = [[] for _ in range(num_nodes)]
    undirected_edges = {}
    for src, dst in edges:
        src_i, dst_i = int(src), int(dst)
        if src_i == dst_i:
            continue
        key = (src_i, dst_i) if src_i < dst_i else (dst_i, src_i)
        if key in undirected_edges:
            continue
        undirected_edges[key] = True
        adjacency[src_i].append(dst_i)
        adjacency[dst_i].append(src_i)

    tin = [-1] * num_nodes
    low = [-1] * num_nodes
    component_of = [-1] * num_nodes
    component_nodes = {}
    bridge_child_nodes = {}
    timer = 0
    component_id = 0

    def dfs(node, parent, comp_idx):
        nonlocal timer
        tin[node] = low[node] = timer
        timer += 1
        component_of[node] = comp_idx
        subtree_nodes = [node]
        for nxt in adjacency[node]:
            if nxt == parent:
                continue
            if tin[nxt] != -1:
                low[node] = min(low[node], tin[nxt])
                continue
            child_nodes = dfs(nxt, node, comp_idx)
            low[node] = min(low[node], low[nxt])
            if low[nxt] > tin[node]:
                key = (node, nxt) if node < nxt else (nxt, node)
                bridge_child_nodes[key] = child_nodes
            subtree_nodes.extend(child_nodes)
        return subtree_nodes

    for node in range(num_nodes):
        if tin[node] != -1:
            continue
        if not adjacency[node]:
            tin[node] = low[node] = timer
            timer += 1
            component_of[node] = component_id
            component_nodes[component_id] = [node]
            component_id += 1
            continue
        nodes = dfs(node, -1, component_id)
        component_nodes[component_id] = nodes
        component_id += 1

    bridge_small_components = {}
    for key, child_nodes in bridge_child_nodes.items():
        comp_idx = component_of[child_nodes[0]]
        comp_nodes = component_nodes[comp_idx]
        child_sorted = sorted(child_nodes)
        child_set = set(child_sorted)
        other_nodes = [node for node in comp_nodes if node not in child_set]
        other_sorted = sorted(other_nodes)

        if len(child_sorted) < len(other_sorted):
            smaller = child_sorted
        elif len(child_sorted) > len(other_sorted):
            smaller = other_sorted
        else:
            child_min = child_sorted[0] if child_sorted else num_nodes
            other_min = other_sorted[0] if other_sorted else num_nodes
            smaller = child_sorted if child_min <= other_min else other_sorted

        bridge_small_components[key] = smaller if len(smaller) > 1 else []

    return bridge_small_components


def _build_rotation_masks(all_edge_index, candidate_edge_index, num_nodes, bridge_small_components):
    edges = _edge_index_to_numpy(all_edge_index)
    candidate_edges = _edge_index_to_numpy(candidate_edge_index)
    candidate_set = {tuple(edge.tolist()) for edge in candidate_edges}
    mask_edges = np.zeros(len(edges), dtype=bool)
    rotate_lists = []

    for idx, edge in enumerate(edges):
        src, dst = int(edge[0]), int(edge[1])
        if (src, dst) not in candidate_set:
            continue
        key = (src, dst) if src < dst else (dst, src)
        smaller = bridge_small_components.get(key, [])
        if not smaller:
            continue
        smaller_set = set(smaller)
        if dst not in smaller_set or src in smaller_set:
            continue
        mask_edges[idx] = True
        rotate_lists.append(smaller)

    mask_rotate = np.zeros((len(rotate_lists), num_nodes), dtype=bool)
    for idx, nodes in enumerate(rotate_lists):
        mask_rotate[idx, np.asarray(nodes, dtype=int)] = True
    return mask_edges, mask_rotate


def _description_to_esm_embedding_path(description):
    """从输入的 PDB 路径推断同目录 sidecar `esm_embedding.pt`。"""
    if not description:
        return None
    path = os.path.expanduser(str(description).strip())
    if not path.lower().endswith(".pdb"):
        return None
    if not os.path.isfile(path):
        return None
    esm_path = os.path.join(os.path.dirname(path), "esm_embedding.pt")
    return esm_path if os.path.isfile(esm_path) else None


def _chain_lengths_from_pdb(description):
    """
    按 `extract_esm_embedding.py` / 特征构图一致的口径统计每条链残基数：
    - 只看 ATOM
    - 过滤氢和非 A altloc
    - 只保留同时有 N/CA/C/O 的残基
    - 不按残基编号空洞补 '-'
    """
    universe = MDAnalysis.Universe(str(description))
    chain_lengths = []
    for seg in universe.segments:
        count = 0
        for residue in seg.residues:
            res_atoms = residue.atoms.select_atoms("not type H")
            if len(res_atoms) > 0:
                mask = (res_atoms.altLocs == "") | (res_atoms.altLocs == "A")
                res_atoms = res_atoms[mask]
            c_alpha = res_atoms.select_atoms("name CA and record_type ATOM")
            n = res_atoms.select_atoms("name N and record_type ATOM")
            c = res_atoms.select_atoms("name C and record_type ATOM")
            o = res_atoms.select_atoms("name O and record_type ATOM")
            if len(c_alpha) == 1 and len(n) == 1 and len(c) == 1 and len(o) == 1:
                count += 1
        if count > 0:
            chain_lengths.append(count)
    if not chain_lengths:
        raise ValueError(f"无法从 PDB 提取有效链长度: {description}")
    return chain_lengths


def _collect_chain_keys(description, require_atom_record: bool = True):
    """收集每条链的有效残基 key 及其 embedding_idx。"""
    universe = MDAnalysis.Universe(str(description))
    chain_items = []
    for seg in universe.segments:
        trans = {}
        current_idx = -1
        for residue in seg.residues:
            res_atoms = residue.atoms.select_atoms("not type H")
            if len(res_atoms) > 0:
                mask = (res_atoms.altLocs == "") | (res_atoms.altLocs == "A")
                res_atoms = res_atoms[mask]
            selector_suffix = " and record_type ATOM" if require_atom_record else ""
            c_alpha = res_atoms.select_atoms(f"name CA{selector_suffix}")
            n = res_atoms.select_atoms(f"name N{selector_suffix}")
            c = res_atoms.select_atoms(f"name C{selector_suffix}")
            o = res_atoms.select_atoms(f"name O{selector_suffix}")
            if len(c_alpha) == 1 and len(n) == 1 and len(c) == 1 and len(o) == 1:
                current_idx += 1
                key = int(residue.resid) if residue.icode == "" else f"{residue.resid}{residue.icode.strip()}"
                trans[key] = current_idx
        if not trans:
            continue
        int_keys = [item for item in trans.keys() if isinstance(item, int)]
        embedding_idx = sorted(
            set(trans.keys())
            | (
                set(range(min(int_keys), max(int_keys) + 1))
                if int_keys else set()
            ),
            key=standard_residue_sort,
        )
        chain_items.append(
            {
                "trans": trans,
                "ordered_keys": sorted(set(trans.keys()), key=standard_residue_sort),
                "embedding_idx": embedding_idx,
            }
        )
    if not chain_items:
        raise ValueError(f"无法从 PDB 收集有效链信息: {description}")
    return chain_items


def _build_padded_chain_embeddings(description, embedding, require_atom_record: bool = True):
    """
    把 sidecar 的压缩 embedding 恢复成 `get_protein_feature_mda` / `get_ori_peptide_feature_mda`
    可直接吃的“按 embedding_idx 补链内缺口”格式。
    """
    if not torch.is_tensor(embedding):
        embedding = torch.as_tensor(embedding)
    chain_items = _collect_chain_keys(description, require_atom_record=require_atom_record)
    total_valid = sum(len(item["ordered_keys"]) for item in chain_items)
    if total_valid != int(embedding.shape[0]):
        raise ValueError(
            f"ESM 有效残基数与 PDB 不匹配: desc={description} valid_len={total_valid} emb_len={int(embedding.shape[0])}"
        )
    emb_dim = int(embedding.shape[1])
    offset = 0
    chain_embeddings = []
    for item in chain_items:
        valid_count = len(item["ordered_keys"])
        chain_valid = embedding[offset: offset + valid_count]
        offset += valid_count
        key_to_row = {
            key: chain_valid[row_idx].clone()
            for row_idx, key in enumerate(item["ordered_keys"])
        }
        padded = [
            key_to_row.get(key, torch.zeros(emb_dim, dtype=embedding.dtype))
            for key in item["embedding_idx"]
        ]
        chain_embeddings.append(padded)
    return chain_embeddings


def _split_flat_embedding_by_description(description, embedding):
    """把扁平 [N,1280] embedding 按 PDB 链长度切回 list[tensor]。"""
    if not torch.is_tensor(embedding):
        embedding = torch.as_tensor(embedding)
    chain_lengths = _chain_lengths_from_pdb(description)
    total_len = sum(chain_lengths)
    if total_len != int(embedding.shape[0]):
        raise ValueError(
            f"ESM 长度与 PDB 不匹配: desc={description} pdb_len={total_len} emb_len={int(embedding.shape[0])}"
        )
    chain_embeddings = []
    offset = 0
    for chain_len in chain_lengths:
        chain_embeddings.append(embedding[offset: offset + chain_len].clone())
        offset += chain_len
    return chain_embeddings


def load_sidecar_esm_embeddings(protein_description_list, peptide_description_list):
    """
    从输入 PDB 同目录复用 `esm_embedding.pt`。

    - receptor 读 `rec_emb`
    - peptide 读 `pep_emb`
    - 若某一侧没有 sidecar 或对不上，就返回 None，后续走在线生成兜底
    """
    cache = {}
    rec_embeddings = [None] * len(protein_description_list)
    pep_embeddings = [None] * len(peptide_description_list)

    def _load_payload(path):
        if path in cache:
            return cache[path]
        payload = torch.load(path, map_location="cpu")
        cache[path] = payload if isinstance(payload, dict) else {}
        return cache[path]

    for idx, description in enumerate(protein_description_list):
        esm_path = _description_to_esm_embedding_path(description)
        if esm_path is None:
            continue
        try:
            payload = _load_payload(esm_path)
            rec_emb = payload.get("rec_emb", None)
            if rec_emb is None:
                continue
            rec_embeddings[idx] = _build_padded_chain_embeddings(
                description,
                rec_emb,
                require_atom_record=True,
            )
            print(f"[info] reuse receptor esm sidecar: {esm_path}")
        except Exception as e:
            print(f"[warn] receptor esm sidecar invalid, fallback to online ESM: path={esm_path} err={e}")

    for idx, description in enumerate(peptide_description_list):
        esm_path = _description_to_esm_embedding_path(description)
        if esm_path is None:
            continue
        try:
            payload = _load_payload(esm_path)
            pep_emb = payload.get("pep_emb", None)
            if pep_emb is None:
                continue
            pep_embeddings[idx] = _build_padded_chain_embeddings(
                description,
                pep_emb,
                require_atom_record=False,
            )
            print(f"[info] reuse peptide esm sidecar: {esm_path}")
        except Exception as e:
            print(f"[warn] peptide esm sidecar invalid, fallback to online ESM: path={esm_path} err={e}")

    return rec_embeddings, pep_embeddings

# 推理阶段：检测肽主链是否缺原子（N/CA/C/O）
def _detect_missing_backbone_atoms(peptide_path=None, universe=None):
    u = universe
    if u is None:
        try:
            u = MDAnalysis.Universe(peptide_path)
        except Exception as e:
            return [f"load_failed:{e}"]
    missing = []
    for residue in u.residues:
        res_atoms = residue.atoms.select_atoms('not type H')
        if len(res_atoms) > 0:
            mask = (res_atoms.altLocs == "") | (res_atoms.altLocs == "A")
            res_atoms = res_atoms[mask]
        ca = res_atoms.select_atoms("name CA")
        n = res_atoms.select_atoms("name N")
        c = res_atoms.select_atoms("name C")
        o = res_atoms.select_atoms("name O")
        if not (len(ca) == 1 and len(n) == 1 and len(c) == 1 and len(o) == 1):
            resid = int(residue.resid) if residue.icode == "" else f"{residue.resid}{residue.icode.strip()}"
            missing.append(f"{resid}:{residue.resname}(N{len(n)} CA{len(ca)} C{len(c)} O{len(o)})")
    return missing


def _peptide_needs_renumber(u):
    res_keys = []
    for residue in u.residues:
        key = int(residue.resid) if residue.icode == "" else f"{residue.resid}{residue.icode.strip()}"
        res_keys.append(key)
    if not res_keys:
        return True
    if any(not isinstance(k, int) for k in res_keys):
        return True
    res_keys_sorted = sorted(res_keys)
    return res_keys_sorted != list(range(min(res_keys_sorted), max(res_keys_sorted) + 1))


def _peptide_has_altloc(u):
    try:
        altlocs = u.atoms.altLocs
    except Exception:
        return False
    if len(altlocs) == 0:
        return False
    return any(a not in ("", "A") for a in altlocs)


def _peptide_has_hydrogen(u):
    if len(u.atoms) == 0:
        return False
    try:
        elements = u.atoms.elements
        if len(elements) > 0 and any(str(e).upper() == "H" for e in elements):
            return True
    except Exception:
        pass
    try:
        names = u.atoms.names
        if len(names) > 0 and any(str(n).upper().startswith("H") for n in names):
            return True
    except Exception:
        pass
    return False


def _peptide_precheck(peptide_path):
    info = {
        "load_failed": None,
        "missing_backbone": [],
        "has_h": False,
        "has_altloc": False,
        "needs_renumber": False,
        "multi_chain": False,
    }
    try:
        u = MDAnalysis.Universe(peptide_path)
    except Exception as e:
        info["load_failed"] = str(e)
        return info
    info["missing_backbone"] = _detect_missing_backbone_atoms(universe=u)
    info["has_h"] = _peptide_has_hydrogen(u)
    info["has_altloc"] = _peptide_has_altloc(u)
    info["needs_renumber"] = _peptide_needs_renumber(u)
    info["multi_chain"] = len(u.segments) > 1
    return info


def _sanitize_peptide_pdb(peptide_path, output_path):
    parser = Bio.PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("pep", peptide_path)
    new_structure = Bio.PDB.Structure.Structure("pep")
    new_model = Bio.PDB.Model.Model(0)
    new_chain = Bio.PDB.Chain.Chain("A")
    new_model.add(new_chain)
    new_structure.add(new_model)
    next_resid = 1
    for model in structure:
        for chain in model:
            for residue in chain:
                new_residue = Bio.PDB.Residue.Residue((" ", next_resid, " "), residue.resname, residue.segid)
                for atom in residue:
                    altloc = atom.get_altloc()
                    if altloc not in (" ", "A", ""):
                        continue
                    element = atom.element.strip().upper() if atom.element else ""
                    name = atom.get_name().strip().upper()
                    if element == "H" or name.startswith("H"):
                        continue
                    atom_copy = atom.copy()
                    atom_copy.set_altloc(" ")
                    new_residue.add(atom_copy)
                if len(new_residue):
                    new_chain.add(new_residue)
                    next_resid += 1
    io = Bio.PDB.PDBIO()
    io.set_structure(new_structure)
    io.save(output_path)
    return output_path


def _expected_esm_length_from_pdb(peptide_path):
    seq = get_sequences_from_pdbfile(peptide_path, suppress_missed_aa=True)
    return len(seq.replace(":", "")) if seq else 0


def _log_peptide_repair(output_dir, name, message):
    try:
        os.makedirs(os.path.join(output_dir, name), exist_ok=True)
        log_path = os.path.join(output_dir, name, "peptide_repair.log")
        with open(log_path, "a") as f:
            f.write(message.rstrip() + "\n")
    except Exception:
        pass


def _clear_peptide_repair_log(output_dir, name):
    try:
        log_path = os.path.join(output_dir, name, "peptide_repair.log")
        if os.path.isfile(log_path):
            os.remove(log_path)
    except Exception:
        pass


def _record_bad_inference_sample(output_dir, name, reason):
    """记录推理期坏样本，避免同一坨烂数据反复炸进程。"""
    try:
        bad_csv = os.path.join(output_dir, "bad_inference_samples.csv")
        need_header = not os.path.isfile(bad_csv)
        with open(bad_csv, "a") as f:
            if need_header:
                f.write("complex_name,reason\n")
            safe_reason = str(reason).replace("\n", " ").replace(",", ";").strip()
            f.write(f"{name},{safe_reason}\n")
    except Exception:
        pass


def _make_failed_complex_graph(name, reason):
    graph = HeteroData()
    graph["name"] = name
    graph["success"] = False
    graph["failure_reason"] = str(reason)
    return graph


def _validate_peptide_topology(
    coords_pep,
    all_edge_index_pep,
    atom2res_index,
    pep_feat,
    mask_rotate_backbone,
    mask_rotate_sidechain,
):
    num_atoms = int(coords_pep.shape[0])
    num_res = int(pep_feat.shape[0])
    if num_atoms <= 0 or num_res <= 0:
        raise RuntimeError(f"empty peptide graph atoms={num_atoms} residues={num_res}")
    if all_edge_index_pep.numel() > 0:
        edge_max = int(all_edge_index_pep.max().item())
        if edge_max >= num_atoms:
            raise RuntimeError(f"edge_index_out_of_range edge_max={edge_max} atoms={num_atoms}")
    atom2res_index_t = torch.as_tensor(atom2res_index)
    if atom2res_index_t.numel() != num_atoms:
        raise RuntimeError(
            f"atom2res_length_mismatch atom2res={atom2res_index_t.numel()} atoms={num_atoms}"
        )
    if int(atom2res_index_t.max().item()) >= num_res:
        raise RuntimeError(
            f"atom2res_out_of_range atom2res_max={int(atom2res_index_t.max().item())} residues={num_res}"
        )
    mask_rotate_backbone = np.asarray(mask_rotate_backbone)
    mask_rotate_sidechain = np.asarray(mask_rotate_sidechain)
    if mask_rotate_backbone.ndim == 2 and mask_rotate_backbone.shape[1] != num_atoms:
        raise RuntimeError(
            f"mask_backbone_width_mismatch width={mask_rotate_backbone.shape[1]} atoms={num_atoms}"
        )
    if mask_rotate_sidechain.ndim == 2 and mask_rotate_sidechain.shape[1] != num_atoms:
        raise RuntimeError(
            f"mask_sidechain_width_mismatch width={mask_rotate_sidechain.shape[1]} atoms={num_atoms}"
        )


def _prepare_single_peptide_description(task):
    output_dir, complex_name, pep_desc = task
    result = {
        "complex_name": complex_name,
        "input_path": pep_desc,
        "repaired_path": pep_desc,
        "skip": False,
        "log_messages": [],
        "console_message": None,
        "precheck_clean": False,
    }
    if not pep_desc or "pdb" not in str(pep_desc).lower() or not os.path.isfile(pep_desc):
        return result

    os.makedirs(os.path.join(output_dir, complex_name), exist_ok=True)
    pre = _peptide_precheck(pep_desc)
    actions = []
    repaired_path = pep_desc
    if pre["load_failed"]:
        result["skip"] = True
        result["repaired_path"] = None
        result["log_messages"].append(
            f"[precheck] load_failed={pre['load_failed']} -> skip native peptide"
        )
        return result

    needs_sanitize = any(
        [
            pre["has_h"],
            pre["has_altloc"],
            pre["needs_renumber"],
            pre["multi_chain"],
        ]
    )
    if needs_sanitize:
        fixed_path = os.path.join(
            output_dir,
            complex_name,
            f"{complex_name}_peptide_fixed.pdb",
        )
        try:
            _sanitize_peptide_pdb(repaired_path, fixed_path)
            repaired_path = fixed_path
            actions.append("sanitize")
        except Exception as e:
            result["skip"] = True
            result["repaired_path"] = None
            result["log_messages"].append(
                f"[repair] sanitize_failed error={e} -> skip native peptide"
            )
            return result

    after_missing = _detect_missing_backbone_atoms(repaired_path)
    if after_missing:
        result["skip"] = True
        result["repaired_path"] = None
        result["log_messages"].append(
            f"[precheck] missing_backbone={after_missing} -> skip native peptide"
        )
        return result

    result["repaired_path"] = repaired_path
    if actions:
        msg = (
            f"[repair] {pep_desc} actions={actions} "
            f"altloc={pre['has_altloc']} h={pre['has_h']} "
            f"renumber={pre['needs_renumber']} multichain={pre['multi_chain']}"
        )
        result["console_message"] = msg
        result["log_messages"].append(msg)
    else:
        result["precheck_clean"] = True
    return result


def compute_ESM_embeddings(model, alphabet, labels, sequences):
    """批量跑ESM2得到序列级表示，缓存到label字典"""
    # settings used
    toks_per_batch = 4096
    repr_layers = [33]
    truncation_seq_length = 4096

    dataset = FastaBatchedDataset(labels, sequences)
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(truncation_seq_length), batch_sampler=batches
    )

    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in repr_layers)
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in repr_layers]
    embeddings = {}

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)")
            if torch.cuda.is_available():
                toks = toks.to(device="cuda", non_blocking=True)

            out = model(toks, repr_layers=repr_layers, return_contacts=False)
            representations = {layer: t.to(device="cpu") for layer, t in out["representations"].items()}

            for i, label in enumerate(labels):
                truncate_len = min(truncation_seq_length, len(strs[i]))
                embeddings[label] = representations[33][i, 1: truncate_len + 1].clone()
    return embeddings

def generate_ESM_structure(model, filename, sequence):
    """只给序列时用ESMFold补结构，OOM会自动降chunk"""
    model.set_chunk_size(256)
    chunk_size = 256
    output = None

    while output is None:
        try:
            with torch.no_grad():
                output = model.infer_pdb(sequence)

            with open(filename, "w") as f:
                f.write(output)
                print("saved", filename)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory on chunk_size', chunk_size)
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                chunk_size = chunk_size // 2
                if chunk_size > 2:
                    model.set_chunk_size(chunk_size)
                else:
                    print("Not enough memory for ESMFold")
                    break
            else:
                raise e
    return output is not None

class InferenceDataset(Dataset):
    """推理阶段把输入描述转换成图结构，包括蛋白/肽特征和语言模型嵌入"""
    def __init__(
        self,
        output_dir,
        complex_name_list,
        protein_description_list,
        peptide_description_list,
        lm_embeddings,
        lm_embeddings_pep,
        precomputed_lm_embeddings=None,
        precomputed_lm_embeddings_pep=None,
        embedding_mode: str = "onehot",
        use_native_peptide_pose: bool = True,
        receptor_pt_list=None,
        peptide_esm_path=None,
        receptor_cache_dir=None,
        graph_cache_dir=None,
        save_receptor_cache_dir=None,
        save_graph_cache_dir=None,
        receptor_build_workers=None,
        cache_peptide_graph: bool = True,
        precheck_workers: int = 0,
    ):

        super(InferenceDataset, self).__init__()

        self.output_dir = output_dir
        self.embedding_mode = embedding_mode
        self.complex_names = complex_name_list
        self.original_protein_descriptions = list(protein_description_list)
        self.original_peptide_descriptions = list(peptide_description_list)
        self.protein_descriptions = protein_description_list
        self.peptide_descriptions = peptide_description_list
        # 当前仓库只保留 native peptide 路线，不再提供序列生成器兜底。
        self.use_native_peptide_pose = bool(use_native_peptide_pose)
        self.receptor_pt_list = receptor_pt_list or [None] * len(self.complex_names)
        self.receptor_cache_dir = (
            _normalize_local_path(receptor_cache_dir) if receptor_cache_dir else None
        )
        self.graph_cache_dir = (
            _normalize_local_path(graph_cache_dir) if graph_cache_dir else None
        )
        self.save_receptor_cache_dir = (
            _normalize_local_path(save_receptor_cache_dir) if save_receptor_cache_dir else None
        )
        self.save_graph_cache_dir = (
            _normalize_local_path(save_graph_cache_dir) if save_graph_cache_dir else None
        )
        self.receptor_cache_dirs = []
        if self.receptor_cache_dir:
            self.receptor_cache_dirs.append(self.receptor_cache_dir)
        if self.save_receptor_cache_dir and self.save_receptor_cache_dir not in self.receptor_cache_dirs:
            self.receptor_cache_dirs.append(self.save_receptor_cache_dir)
        self.graph_cache_dirs = []
        if self.graph_cache_dir:
            self.graph_cache_dirs.append(self.graph_cache_dir)
        if self.save_graph_cache_dir and self.save_graph_cache_dir not in self.graph_cache_dirs:
            self.graph_cache_dirs.append(self.save_graph_cache_dir)
        self.receptor_build_workers = receptor_build_workers
        self.cache_peptide_graph = bool(cache_peptide_graph)
        self._peptide_cache = {}
        self._receptor_graph_cache = {}
        self._complex_graph_cache = {}
        self._receptor_cache_stats = {
            "explicit_pt_hit": 0,
            "cache_dir_hit": 0,
            "memory_hit": 0,
            "built": 0,
            "saved": 0,
            "cache_invalid": 0,
        }
        self._graph_cache_stats = {
            "cache_dir_hit": 0,
            "memory_hit": 0,
            "saved": 0,
            "cache_invalid": 0,
        }
        self._peptide_esm_override = None
        self.precheck_workers = max(0, int(precheck_workers or 0))
        if self.save_receptor_cache_dir:
            os.makedirs(self.save_receptor_cache_dir, exist_ok=True)
        if self.save_graph_cache_dir:
            os.makedirs(self.save_graph_cache_dir, exist_ok=True)
        self._graph_cache_candidate_paths = {}
        self._graph_cache_has_rec_embedding = bool(lm_embeddings)
        self._graph_cache_has_pep_embedding = bool(lm_embeddings_pep or peptide_esm_path)
        for idx, name in enumerate(self.complex_names):
            receptor_pt = self.receptor_pt_list[idx] if idx < len(self.receptor_pt_list) else None
            complex_meta = _build_complex_graph_cache_meta(
                complex_name=name,
                protein_file=self.original_protein_descriptions[idx],
                peptide_description=self.original_peptide_descriptions[idx],
                receptor_pt=receptor_pt,
                embedding_mode=self.embedding_mode,
                has_rec_embedding=self._graph_cache_has_rec_embedding,
                has_pep_embedding=self._graph_cache_has_pep_embedding,
            )
            cache_path = self._lookup_graph_cache_path(complex_meta)
            self._graph_cache_candidate_paths[idx] = cache_path

        prep_tasks = [
            (self.output_dir, self.complex_names[i], pep_desc)
            for i, pep_desc in enumerate(self.peptide_descriptions)
        ]
        if self.precheck_workers > 1 and len(prep_tasks) > 1:
            with multiprocessing.Pool(self.precheck_workers) as pool:
                prep_results = pool.map(_prepare_single_peptide_description, prep_tasks)
        else:
            prep_results = [_prepare_single_peptide_description(task) for task in prep_tasks]

        repaired = []
        for result in prep_results:
            complex_name = result["complex_name"]
            if result.get("precheck_clean", False):
                _clear_peptide_repair_log(self.output_dir, complex_name)
            for msg in result.get("log_messages", []):
                _log_peptide_repair(self.output_dir, complex_name, msg)
            console_message = result.get("console_message", None)
            if console_message:
                print(console_message)
            repaired.append(result["repaired_path"] if not result["skip"] else None)
        self.peptide_descriptions = repaired

        if peptide_esm_path:
            pep_payload = torch.load(peptide_esm_path, map_location="cpu")
            if isinstance(pep_payload, dict) and "pep_emb" in pep_payload:
                pep_payload = pep_payload["pep_emb"]
            if not torch.is_tensor(pep_payload):
                pep_payload = torch.as_tensor(pep_payload)
            self._peptide_esm_override = pep_payload
        
        model = None
        # generate LM embeddings for protein
        # 先处理蛋白LM嵌入：优先复用缓存，没有就现场跑ESM2
        if precomputed_lm_embeddings is None:
            self.lm_embeddings = [None] * len(self.complex_names)
        else:
            self.lm_embeddings = list(precomputed_lm_embeddings)
        if lm_embeddings:
            esm_indices = [
                i
                for i, pt in enumerate(self.receptor_pt_list)
                if pt is None
                and protein_description_list[i] is not None
                and self.lm_embeddings[i] is None
                and self._graph_cache_candidate_paths.get(i) is None
            ]
            if esm_indices:
                print("Generating ESM language model embeddings for protein")
                model_location = "esm2_t33_650M_UR50D"
                model, alphabet = pretrained.load_model_and_alphabet(model_location)
                model.eval()
                if torch.cuda.is_available():
                    model = model.cuda()

                protein_sequences = get_sequences(
                    [protein_description_list[i] for i in esm_indices], suppress_missed_aa=True
                )
                labels, sequences = [], []
                chain_counts = []
                for local_idx, seq in enumerate(protein_sequences):
                    s = seq.split(':')
                    chain_counts.append(len(s))
                    sequences.extend(s)
                    labels.extend(
                        [
                            str(complex_name_list[esm_indices[local_idx]]) + "_chain_" + str(j)
                            for j in range(len(s))
                        ]
                    )

                lm_embeddings = compute_ESM_embeddings(model, alphabet, labels, sequences)
                for local_idx, count in enumerate(chain_counts):
                    global_idx = esm_indices[local_idx]
                    self.lm_embeddings[global_idx] = [
                        lm_embeddings[f"{complex_name_list[global_idx]}_chain_{j}"] for j in range(count)
                    ]
        elif not lm_embeddings:
            self.lm_embeddings = [None] * len(self.complex_names)
        
        # generate LM embeddings for peptide
        if self._peptide_esm_override is not None:
            self.lm_embeddings_pep = [[self._peptide_esm_override]] * len(self.complex_names)
        elif lm_embeddings_pep:
            if precomputed_lm_embeddings_pep is None:
                self.lm_embeddings_pep = [None] * len(self.complex_names)
            else:
                self.lm_embeddings_pep = list(precomputed_lm_embeddings_pep)
            esm_indices = [i for i, emb in enumerate(self.lm_embeddings_pep) if emb is None]
            esm_indices = [
                i
                for i in esm_indices
                if self._graph_cache_candidate_paths.get(i) is None
                and self.peptide_descriptions[i] is not None
            ]
            if esm_indices:
                print("Generating ESM language model embeddings for peptide")
                if model is None:
                    model_location = "esm2_t33_650M_UR50D"
                    model, alphabet = pretrained.load_model_and_alphabet(model_location)
                    model.eval()
                    if torch.cuda.is_available():
                        model = model.cuda()

                peptide_sequences = get_sequences(
                    [self.peptide_descriptions[i] for i in esm_indices], suppress_missed_aa=True
                )
                labels, sequences = [], []
                chain_counts = []
                for local_idx, seq in enumerate(peptide_sequences):
                    s = seq.split(':')
                    chain_counts.append(len(s))
                    sequences.extend(s)
                    labels.extend([str(complex_name_list[esm_indices[local_idx]]) + '_chain_' + str(j) for j in range(len(s))])
                
                lm_embeddings_pep = compute_ESM_embeddings(model, alphabet, labels, sequences)
                for local_idx, count in enumerate(chain_counts):
                    global_idx = esm_indices[local_idx]
                    self.lm_embeddings_pep[global_idx] = [
                        lm_embeddings_pep[f'{complex_name_list[global_idx]}_chain_{j}'] for j in range(count)
                    ]
        else:
            self.lm_embeddings_pep = [None] * len(self.complex_names)
            
        # generate protein structures with ESMFold if only protein sequences are provided
        # 输入只给了序列就临时用ESMFold补模型
        protein_structure_missing = len(
            [
                protein_description
                for idx, protein_description in enumerate(protein_description_list)
                if 'pdb' not in protein_description and self._graph_cache_candidate_paths.get(idx) is None
            ]
        ) > 0
        if protein_structure_missing:
            print("generating missing protein structures with ESMFold")
            model = esm.pretrained.esmfold_v1()
            model = model.eval().cuda()
            
            for i in range(len(protein_description_list)):
                if 'pdb' not in protein_description_list[i] and self._graph_cache_candidate_paths.get(i) is None:
                    self.protein_descriptions[i] = f"{output_dir}/{complex_name_list[i]}/{complex_name_list[i]}_esmfold.pdb"
                    if not os.path.exists(self.protein_descriptions[i]):
                        print("generating", self.protein_descriptions[i])
                        generate_ESM_structure(model, self.protein_descriptions[i], protein_description_list[i])
        self._prebuild_missing_receptor_caches()
    
    def len(self):
        """Pytorch Geometric 需要覆写len/get"""
        return len(self.complex_names)

    def receptor_cache_stats(self):
        return dict(self._receptor_cache_stats)

    def graph_cache_stats(self):
        return dict(self._graph_cache_stats)

    def _lookup_receptor_cache_path(self, receptor_meta):
        for cache_dir in self.receptor_cache_dirs:
            cache_path = _receptor_cache_path(cache_dir, receptor_meta)
            if os.path.isfile(cache_path):
                return cache_path
        return None

    def _lookup_graph_cache_path(self, complex_meta):
        for cache_dir in self.graph_cache_dirs:
            cache_path = _complex_graph_cache_path(cache_dir, complex_meta)
            if os.path.isfile(cache_path):
                return cache_path
        return None

    def _prebuild_missing_receptor_caches(self):
        if not self.save_receptor_cache_dir:
            return
        tasks = []
        seen = set()
        for idx, protein_file in enumerate(self.protein_descriptions):
            if not protein_file or "pdb" not in str(protein_file).lower():
                continue
            receptor_pt = self.receptor_pt_list[idx] if idx < len(self.receptor_pt_list) else None
            if receptor_pt:
                continue
            receptor_meta = _build_receptor_cache_meta(
                protein_file=protein_file,
                embedding_mode=self.embedding_mode,
                has_lm_embedding=self.lm_embeddings[idx] is not None,
            )
            cache_key = _receptor_cache_key(receptor_meta)
            if cache_key in seen:
                continue
            seen.add(cache_key)
            if self._lookup_receptor_cache_path(receptor_meta) is not None:
                continue
            tasks.append(
                {
                    "protein_file": protein_file,
                    "lm_embedding": self.lm_embeddings[idx],
                    "embedding_mode": self.embedding_mode,
                    "receptor_meta": receptor_meta,
                    "save_path": _receptor_cache_path(self.save_receptor_cache_dir, receptor_meta),
                }
            )
        worker_count = _resolve_receptor_build_workers(
            self.receptor_build_workers,
            max(1, self.precheck_workers + 1),
            len(tasks),
        )
        if not tasks:
            return
        print(
            "[receptor-cache] prebuild "
            f"tasks={len(tasks)} workers={worker_count} save_dir={self.save_receptor_cache_dir}"
        )
        results = []
        if worker_count > 1:
            with multiprocessing.Pool(worker_count) as pool:
                results = pool.map(_prebuild_single_receptor_cache, tasks)
        else:
            results = [_prebuild_single_receptor_cache(task) for task in tasks]
        saved = sum(1 for item in results if item.get("status") == "saved")
        exists = sum(1 for item in results if item.get("status") == "exists")
        failed = len(results) - saved - exists
        self._receptor_cache_stats["saved"] += saved
        print(
            "[receptor-cache] prebuild_done "
            f"saved={saved} exists={exists} failed={failed}"
        )
    
    def _get_impl(self, idx):
        """加载单个复合物，生成异质图并落盘初始文件"""
        name, protein_file, peptide_description, lm_embedding, lm_embedding_pep = self.complex_names[idx], self.protein_descriptions[idx], self.peptide_descriptions[idx], self.lm_embeddings[idx], self.lm_embeddings_pep[idx]
        receptor_pt = self.receptor_pt_list[idx] if idx < len(self.receptor_pt_list) else None
        complex_meta = _build_complex_graph_cache_meta(
            complex_name=name,
            protein_file=self.original_protein_descriptions[idx],
            peptide_description=self.original_peptide_descriptions[idx],
            receptor_pt=receptor_pt,
            embedding_mode=self.embedding_mode,
            has_rec_embedding=self._graph_cache_has_rec_embedding,
            has_pep_embedding=self._graph_cache_has_pep_embedding,
        )
        if protein_file and str(protein_file).lower().endswith(".pdb"):
            try:
                protein_raw_path = f"{self.output_dir}/{name}/{name}_protein_raw.pdb"
                if not os.path.isfile(protein_raw_path):
                    shutil.copyfile(protein_file, protein_raw_path)
            except Exception:
                pass
        if peptide_description and str(peptide_description).lower().endswith(".pdb"):
            try:
                peptide_raw_path = f"{self.output_dir}/{name}/{name}_peptide_raw.pdb"
                if not os.path.isfile(peptide_raw_path):
                    shutil.copyfile(peptide_description, peptide_raw_path)
            except Exception:
                pass

        graph_mem_key = ("complex_graph", _complex_graph_cache_key(complex_meta))
        cached_graph = self._complex_graph_cache.get(graph_mem_key)
        if cached_graph is not None:
            self._graph_cache_stats["memory_hit"] += 1
            return _clone_cached_complex_graph_for_inference(cached_graph)
        cache_path = self._lookup_graph_cache_path(complex_meta)
        if cache_path:
            try:
                cached_graph, _graph_meta = _load_complex_graph_payload(
                    cache_path,
                    expected_meta=complex_meta,
                )
                self._complex_graph_cache[graph_mem_key] = cached_graph
                self._graph_cache_stats["cache_dir_hit"] += 1
                return _clone_cached_complex_graph_for_inference(cached_graph)
            except Exception as cache_err:
                print(f"[graph-cache] invalid cache ignored: path={cache_path} err={cache_err}")
                self._graph_cache_stats["cache_invalid"] += 1

        # 受体特征：优先用预制 PT
        rec_data = None
        protein_center = None
        if receptor_pt:
            receptor_pt_key = ("explicit_pt", _normalize_local_path(receptor_pt))
            cached_rec = self._receptor_graph_cache.get(receptor_pt_key)
            if cached_rec is None:
                rec_data, _rec_meta = _load_receptor_graph_payload(receptor_pt, expected_meta=None)
                self._receptor_graph_cache[receptor_pt_key] = _clone_receptor_graph_data(rec_data)
            else:
                rec_data = _clone_receptor_graph_data(cached_rec)
                self._receptor_cache_stats["memory_hit"] += 1
            self._receptor_cache_stats["explicit_pt_hit"] += 1
        else:
            receptor_meta = _build_receptor_cache_meta(
                protein_file=protein_file,
                embedding_mode=self.embedding_mode,
                has_lm_embedding=lm_embedding is not None,
            )
            receptor_mem_key = ("receptor_pdb", _receptor_cache_key(receptor_meta))
            cached_rec = self._receptor_graph_cache.get(receptor_mem_key)
            if cached_rec is not None:
                rec_data = _clone_receptor_graph_data(cached_rec)
                self._receptor_cache_stats["memory_hit"] += 1
            else:
                cache_path = self._lookup_receptor_cache_path(receptor_meta)
                if cache_path:
                    try:
                        rec_data, _rec_meta = _load_receptor_graph_payload(
                            cache_path,
                            expected_meta=receptor_meta,
                        )
                        self._receptor_cache_stats["cache_dir_hit"] += 1
                    except Exception as cache_err:
                        print(f"[receptor-cache] invalid cache ignored: path={cache_path} err={cache_err}")
                        self._receptor_cache_stats["cache_invalid"] += 1
                        rec_data = None
                if rec_data is None:
                    c_alpha_coords_rec, tip_coords_rec, lm_embeddings_rec, seq_rec, node_s_rec, node_v_rec, edge_index_rec, edge_s_rec, edge_v_rec = get_protein_feature_mda(
                        protein_file, lm_embedding_chains=lm_embedding
                    )
                    rec_feat = _build_receptor_feature_tensor(
                        seq_rec=seq_rec,
                        node_s_rec=node_s_rec,
                        lm_embeddings_rec=lm_embeddings_rec,
                        embedding_mode=self.embedding_mode,
                    )
                    rec_data = _build_receptor_graph_data(
                        c_alpha_coords_rec,
                        tip_coords_rec,
                        node_v_rec,
                        rec_feat,
                        edge_index_rec,
                        edge_s_rec,
                        edge_v_rec,
                    )
                    self._receptor_cache_stats["built"] += 1
                    if self.save_receptor_cache_dir:
                        save_path = _receptor_cache_path(self.save_receptor_cache_dir, receptor_meta)
                        if not os.path.isfile(save_path):
                            try:
                                _save_receptor_graph_payload_atomic(
                                    save_path,
                                    receptor_meta,
                                    _clone_receptor_graph_data(rec_data),
                                )
                                self._receptor_cache_stats["saved"] += 1
                                print(f"[receptor-cache] saved {save_path}")
                            except Exception as save_err:
                                print(f"[receptor-cache] save failed: path={save_path} err={save_err}")
                self._receptor_graph_cache[receptor_mem_key] = _clone_receptor_graph_data(rec_data)

        if not isinstance(rec_data, HeteroData):
            raise TypeError(f"受体图加载失败，收到 {type(rec_data)}")
        c_alpha_coords_rec = rec_data["receptor"].pos.clone()
        tip_coords_rec = rec_data["receptor"].tips.clone()
        node_v_rec = rec_data["receptor"].node_v.clone()
        rec_feat = rec_data["receptor"].x.clone()
        edge_index_rec = rec_data["receptor", "rec_contact", "receptor"].edge_index.clone()
        edge_s_rec = rec_data["receptor", "rec_contact", "receptor"].edge_s.clone()
        edge_v_rec = rec_data["receptor", "rec_contact", "receptor"].edge_v.clone()
        if hasattr(rec_data, "original_center"):
            protein_center = rec_data.original_center.clone()

        # build the initial peptide, either from file or seq
        # 肽可以来自PDB也可以直接给序列
        cache_key = (
            peptide_description,
            self.use_native_peptide_pose,
            self.embedding_mode,
            self._peptide_esm_override is not None,
        )
        native_mode = False
        cached = self._peptide_cache.get(cache_key) if self.cache_peptide_graph else None
        if cached is not None:
            partials = list(cached["partials"])
            peptide_inits = list(cached["peptide_inits"])
            noh_mda_pep = cached["noh_mda_pep"]
            ori_coords_pep = cached["ori_coords_pep"].clone()
            coords_pep = cached["coords_pep"].clone()
            all_edge_index_pep = cached["all_edge_index_pep"]
            backbone_edge_index_pep = cached["backbone_edge_index_pep"]
            sidechain_edge_index_pep = cached["sidechain_edge_index_pep"]
            atom2res_index = cached["atom2res_index"]
            atom2resid_index = cached["atom2resid_index"]
            atom2atomid_index = cached["atom2atomid_index"]
            pep_a_s = cached["pep_a_s"]
            mask_edges_sidechain = cached["mask_edges_sidechain"]
            mask_rotate_sidechain = cached["mask_rotate_sidechain"]
            mask_edges_backbone = cached["mask_edges_backbone"]
            mask_rotate_backbone = cached["mask_rotate_backbone"]
            mask_edges_sidechain_t = mask_edges_sidechain
            mask_edges_backbone_t = mask_edges_backbone
            pep_feat = cached["pep_feat"]
        else:
            if not peptide_description or 'pdb' not in str(peptide_description) or not os.path.isfile(peptide_description):
                msg = "peptide_generator_removed: 推理与预处理现在只接受现成 peptide.pdb"
                _log_peptide_repair(self.output_dir, name, msg)
                return _make_failed_complex_graph(name, msg)

            peptide_raw_path = f"{self.output_dir}/{name}/{name}_peptide_raw.pdb"
            try:
                if not os.path.isfile(peptide_raw_path):
                    shutil.copyfile(peptide_description, peptide_raw_path)
            except Exception:
                pass
            partials = [1]
            peptide_inits = [peptide_raw_path]
            native_mode = True
            native_target = peptide_raw_path
            if lm_embedding_pep is not None:
                emb_len = sum(len(x) for x in lm_embedding_pep) if isinstance(lm_embedding_pep, (list, tuple)) else len(lm_embedding_pep)
                expected_len = _expected_esm_length_from_pdb(native_target)
                if expected_len != emb_len:
                    msg = (
                        f"peptide_esm_length_mismatch: expected={expected_len} emb={emb_len} "
                        f"path={native_target}"
                    )
                    _log_peptide_repair(self.output_dir, name, msg)
                    return _make_failed_complex_graph(name, msg)

            match_path = os.path.join(os.path.dirname(peptide_inits[0]), "peptide_match.pdb")
            if os.path.exists(match_path):
                os.remove(match_path)
            
            # 构建肽原子级特征，并返回映射关系供扭转时用
            try:
                noh_mda_pep, ori_coords_pep, coords_pep, lm_embeddings_pep, seq_pep, all_edge_index_pep, backbone_edge_index_pep,sidechain_edge_index_pep, atom2res_index, atom2resid_index, atom2atomid_index, pep_a_s = get_ori_peptide_feature_mda(
                    peptide_inits[0], match=False, lm_embedding_chains=lm_embedding_pep
                )
            except Exception as e:
                if lm_embedding_pep is not None:
                    print(f"[repair] peptide feature build failed for {peptide_inits[0]}: {e}; retry without peptide ESM embeddings.")
                    try:
                        noh_mda_pep, ori_coords_pep, coords_pep, lm_embeddings_pep, seq_pep, all_edge_index_pep, backbone_edge_index_pep,sidechain_edge_index_pep, atom2res_index, atom2resid_index, atom2atomid_index, pep_a_s = get_ori_peptide_feature_mda(
                            peptide_inits[0], match=False, lm_embedding_chains=None
                        )
                    except Exception as retry_e:
                        msg = (
                            f"bad_peptide_sample: feature_build_failed "
                            f"path={peptide_inits[0]} first_err={e} retry_err={retry_e}"
                        )
                        print(f"[repair] {msg}")
                        _log_peptide_repair(self.output_dir, name, msg)
                        _record_bad_inference_sample(self.output_dir, name, msg)
                        return _make_failed_complex_graph(name, msg)
                else:
                    msg = f"bad_peptide_sample: feature_build_failed path={peptide_inits[0]} err={e}"
                    print(f"[repair] {msg}")
                    _log_peptide_repair(self.output_dir, name, msg)
                    _record_bad_inference_sample(self.output_dir, name, msg)
                    return _make_failed_complex_graph(name, msg)
            try:
                os.remove(os.path.join(f"{self.output_dir}/{name}",'peptide_noh.pdb'))
            except:pass

            num_nodes = int(coords_pep.shape[0])
            bridge_small_components = _compute_bridge_small_components(
                all_edge_index_pep, num_nodes
            )
            mask_edges_sidechain, mask_rotate_sidechain = _build_rotation_masks(
                all_edge_index_pep,
                sidechain_edge_index_pep,
                num_nodes,
                bridge_small_components,
            )
            mask_edges_backbone, mask_rotate_backbone = _build_rotation_masks(
                all_edge_index_pep,
                backbone_edge_index_pep,
                num_nodes,
                bridge_small_components,
            )
            if mask_edges_sidechain.shape[0] != all_edge_index_pep.shape[1] or mask_edges_backbone.shape[0] != all_edge_index_pep.shape[1]:
                raise RuntimeError("peptide mask length mismatch with edge list")

            mask_edges_sidechain_t = torch.from_numpy(mask_edges_sidechain).bool()
            mask_edges_backbone_t = torch.from_numpy(mask_edges_backbone).bool()

            num_pep_classes = len(pep_three2idx)
            if self.embedding_mode == "esm" and lm_embeddings_pep is not None:
                lm_pep = torch.tensor(lm_embeddings_pep, dtype=torch.float32)
                pep_feat = torch.cat(
                    [seq_pep.reshape([-1, 1]).to(dtype=torch.float32), lm_pep],
                    dim=1,
                )
            elif self.embedding_mode == "onehot":
                onehot = F.one_hot(seq_pep.long(), num_classes=num_pep_classes).float()
                pep_feat = torch.cat([seq_pep.reshape([-1,1]), onehot], dim=1)
            else:
                pep_feat = seq_pep.reshape([-1,1])

            atom2res_index = torch.tensor(atom2res_index)
            atom2resid_index = torch.tensor(atom2resid_index)
            atom2atomid_index = torch.tensor(atom2atomid_index)

            if self.cache_peptide_graph:
                self._peptide_cache[cache_key] = {
                    "partials": list(partials),
                    "peptide_inits": list(peptide_inits),
                    "noh_mda_pep": noh_mda_pep,
                    "ori_coords_pep": ori_coords_pep.detach().cpu().clone(),
                    "coords_pep": coords_pep.detach().cpu().clone(),
                    "all_edge_index_pep": all_edge_index_pep.clone(),
                    "backbone_edge_index_pep": backbone_edge_index_pep.clone(),
                    "sidechain_edge_index_pep": sidechain_edge_index_pep.clone(),
                    "atom2res_index": atom2res_index.clone(),
                    "atom2resid_index": atom2resid_index.clone(),
                    "atom2atomid_index": atom2atomid_index.clone(),
                    "pep_a_s": pep_a_s.clone(),
                    "mask_edges_sidechain": mask_edges_sidechain_t.clone(),
                    "mask_rotate_sidechain": mask_rotate_sidechain,
                    "mask_edges_backbone": mask_edges_backbone_t.clone(),
                    "mask_rotate_backbone": mask_rotate_backbone,
                    "pep_feat": pep_feat.clone(),
                }

        # build the pytorch geometric heterogeneous graph
        complex_graph = HeteroData()
        complex_graph['name'] = name
        complex_graph['pep_a'].pos = coords_pep.to(dtype=torch.float)
        complex_graph['pep_a'].orig_pos = ori_coords_pep.to(dtype=torch.float)
        complex_graph['pep_a'].atom2res_index = atom2res_index
        complex_graph['pep_a'].atom2resid_index = atom2resid_index
        complex_graph['pep_a'].atom2atomid_index = atom2atomid_index
        complex_graph['pep_a'].x = pep_a_s.to(torch.int64)
        complex_graph['pep_a','pep_a'].edge_index = all_edge_index_pep
        complex_graph['pep_a','pep_a'].backbone_edge_index = backbone_edge_index_pep
        complex_graph['pep_a','pep_a'].sidechain_edge_index = sidechain_edge_index_pep
        complex_graph['pep_a'].mask_edges_sidechain = mask_edges_sidechain_t
        complex_graph['pep_a'].mask_rotate_sidechain = mask_rotate_sidechain
        complex_graph['pep_a'].mask_edges_backbone = mask_edges_backbone_t
        complex_graph['pep_a'].mask_rotate_backbone = mask_rotate_backbone

        num_residues = len(c_alpha_coords_rec)
        if num_residues <= 1:
            raise ValueError(f"rec contains only 1 residue!")

        # 统一前缀：索引 + 几何(node_s)，尾部根据模式切换
        if rec_data is None:
            rec_num_classes = len(pep_three2idx)  # 受体也用肽的onehot维度，便于与ESM版本互换尾巴
            if self.embedding_mode == "esm" and lm_embeddings_rec is not None:
                lm_rec = torch.tensor(lm_embeddings_rec, dtype=torch.float32)
                rec_feat = torch.cat(
                    [seq_rec.reshape([-1, 1]).to(dtype=torch.float32), node_s_rec.to(dtype=torch.float32), lm_rec],
                    dim=1,
                )
            elif self.embedding_mode == "onehot":
                onehot = F.one_hot(seq_rec.long(), num_classes=rec_num_classes).float()
                rec_feat = torch.cat([seq_rec.reshape([-1,1]), node_s_rec, onehot], dim=1)
            else:
                rec_feat = torch.cat([seq_rec.reshape([-1,1]), node_s_rec], axis=1)
        complex_graph['receptor'].x = rec_feat # [num_res, feat]
        complex_graph['receptor'].pos = c_alpha_coords_rec.to(dtype=torch.float)
        complex_graph['receptor'].tips = tip_coords_rec.to(dtype=torch.float)
        complex_graph['receptor'].node_v = node_v_rec.to(dtype=torch.float)
        complex_graph['receptor', 'rec_contact', 'receptor'].edge_index = edge_index_rec
        complex_graph['receptor', 'rec_contact', 'receptor'].edge_s = edge_s_rec.to(dtype=torch.float)
        complex_graph['receptor', 'rec_contact', 'receptor'].edge_v = edge_v_rec.to(dtype=torch.float)

        complex_graph['pep'].x = pep_feat
        complex_graph['pep'].noh_mda = noh_mda_pep

        if protein_center is None:
            protein_center = torch.mean(complex_graph['receptor'].pos, dim=0, keepdim=True).to(dtype=torch.float)
            complex_graph['receptor'].pos -= protein_center
            complex_graph['receptor'].tips -= protein_center
        complex_graph['pep_a'].pos -= protein_center
        complex_graph.original_center = protein_center
        complex_graph['success'] = True
        complex_graph['partials'] = partials
        complex_graph['peptide_inits'] = peptide_inits
        complex_graph.repair_backbone_used = False

        try:
            _validate_peptide_topology(
                coords_pep=complex_graph['pep_a'].pos,
                all_edge_index_pep=complex_graph['pep_a', 'pep_a'].edge_index,
                atom2res_index=complex_graph['pep_a'].atom2res_index,
                pep_feat=complex_graph['pep'].x,
                mask_rotate_backbone=complex_graph['pep_a'].mask_rotate_backbone,
                mask_rotate_sidechain=complex_graph['pep_a'].mask_rotate_sidechain,
            )
        except Exception as e:
            msg = f"[repair] graph validation failed: {e} -> skip complex"
            print(msg)
            _log_peptide_repair(self.output_dir, name, msg)
            return _make_failed_complex_graph(name, msg)

        if self.save_graph_cache_dir:
            save_path = _complex_graph_cache_path(self.save_graph_cache_dir, complex_meta)
            if not os.path.isfile(save_path):
                try:
                    _save_complex_graph_payload_atomic(save_path, complex_meta, complex_graph)
                    self._graph_cache_stats["saved"] += 1
                except Exception as save_err:
                    print(f"[graph-cache] save failed: path={save_path} err={save_err}")
            self._complex_graph_cache[graph_mem_key] = complex_graph.clone()

        return complex_graph

    def get(self, idx):
        """总兜底：坏样本一律记日志并返回失败图，避免 DataLoader/主进程被单个样本打断。"""
        try:
            return self._get_impl(idx)
        except Exception as e:
            try:
                name = self.complex_names[idx]
            except Exception:
                name = f"complex_{idx}"
            msg = f"bad_inference_sample: unexpected_get_failure idx={idx} err={e}"
            print(f"[repair] {msg}")
            _log_peptide_repair(self.output_dir, name, msg)
            _record_bad_inference_sample(self.output_dir, name, msg)
            return _make_failed_complex_graph(name, msg)
