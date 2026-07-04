
import os
import sys
import shutil
import json
import yaml
import torch
import MDAnalysis
import pandas as pd
import numpy as np
import warnings
import time
import subprocess
from tqdm import tqdm
from io import StringIO
from argparse import Namespace
import traceback
from pathlib import Path
from typing import Optional
from MDAnalysis.coordinates.memory import MemoryReader
from torch_geometric.loader import DataListLoader
from utils.inference_parsing import get_parser
from utils.utils import get_model, ExponentialMovingAverage
from utils.inference_utils import InferenceDataset, set_nones, load_sidecar_esm_embeddings
from utils.peptide_updater import peptide_updater, _get_torsion_edge_counts
from utils.so3 import sample_vec, sample
from utils.flow_matching import wrap_to_pi
from utils.sampling import sampling
import multiprocessing

warnings.filterwarnings("ignore")

_CKPT_PAYLOAD_CACHE = {}
_STRICT536_CANONICAL_CACHE = None


def _normalize_path_like(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return Path(text.replace("\\", "/")).as_posix()


def _load_strict536_canonical_map() -> dict[str, dict[str, str]]:
    """加载正式 strict-536 的 canonical 资产映射。"""
    global _STRICT536_CANONICAL_CACHE
    if _STRICT536_CANONICAL_CACHE is not None:
        return _STRICT536_CANONICAL_CACHE

    canonical_csv = Path("data/runtime_tables/flow_infer_test536_rel.csv")
    mapping: dict[str, dict[str, str]] = {}
    if not canonical_csv.is_file():
        print(f"[strict536] canonical csv missing: {canonical_csv}")
        _STRICT536_CANONICAL_CACHE = mapping
        return mapping

    try:
        df = pd.read_csv(canonical_csv)
    except Exception as e:
        print(f"[strict536] canonical csv load failed: path={canonical_csv} err={e}")
        _STRICT536_CANONICAL_CACHE = mapping
        return mapping

    required_cols = {"complex_name", "receptor_pdb", "peptide_pdb"}
    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        print(
            f"[strict536] canonical csv missing required columns: {missing_cols} "
            f"path={canonical_csv}"
        )
        _STRICT536_CANONICAL_CACHE = mapping
        return mapping

    for row in df.itertuples(index=False):
        complex_name = str(getattr(row, "complex_name", "") or "").strip().lower()
        receptor_pdb = str(getattr(row, "receptor_pdb", "") or "").strip()
        peptide_pdb = str(getattr(row, "peptide_pdb", "") or "").strip()
        if not complex_name or not receptor_pdb or not peptide_pdb:
            continue
        mapping[complex_name] = {
            "receptor_pdb": receptor_pdb,
            "peptide_pdb": peptide_pdb,
        }

    _STRICT536_CANONICAL_CACHE = mapping
    return mapping


def _canonicalize_strict536_input_df(df: pd.DataFrame, source_csv: Optional[str] = None) -> pd.DataFrame:
    """
    对命中正式 strict-536 名单的样本，强制收口到 processed_test30 的 canonical 资产。

    这样即便外部 CSV 误把 536 行改回 raw source 路径，也不会再把推理带回坏资产。
    """
    if "complex_name" not in df.columns:
        return df

    canonical_map = _load_strict536_canonical_map()
    if not canonical_map:
        return df

    df = df.copy()
    changed_rows: list[tuple[str, list[str]]] = []
    receptor_fields = ("receptor_pdb", "protein_description")
    peptide_fields = ("peptide_pdb", "peptide_description")

    for row_idx, raw_name in df["complex_name"].items():
        complex_name = str(raw_name or "").strip().lower()
        canonical = canonical_map.get(complex_name)
        if canonical is None:
            continue

        changed_fields: list[str] = []
        canonical_receptor = canonical["receptor_pdb"]
        canonical_peptide = canonical["peptide_pdb"]

        for field_name in receptor_fields:
            if field_name not in df.columns:
                continue
            current_value = _normalize_path_like(df.at[row_idx, field_name])
            if current_value != _normalize_path_like(canonical_receptor):
                df.at[row_idx, field_name] = canonical_receptor
                changed_fields.append(field_name)

        for field_name in peptide_fields:
            if field_name not in df.columns:
                continue
            current_value = _normalize_path_like(df.at[row_idx, field_name])
            if current_value != _normalize_path_like(canonical_peptide):
                df.at[row_idx, field_name] = canonical_peptide
                changed_fields.append(field_name)

        if changed_fields:
            changed_rows.append((complex_name, changed_fields))

    if changed_rows:
        src = source_csv or "<in-memory>"
        print(
            "[strict536] "
            f"canonicalized_rows={len(changed_rows)} source_csv={src} "
            "target_root=data/processed_test30"
        )
        for complex_name, changed_fields in changed_rows[:10]:
            print(
                "[strict536] "
                f"{complex_name}: remapped_fields={','.join(changed_fields)}"
            )
        remaining = len(changed_rows) - 10
        if remaining > 0:
            print(f"[strict536] ... {remaining} more row(s) canonicalized")

    return df


def _parse_gpu_list(gpus_arg) -> list[str]:
    if gpus_arg is None:
        return []
    raw = str(gpus_arg).replace(";", ",").replace(" ", ",").strip()
    if not raw:
        return []
    gpu_ids = []
    seen = set()
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        if token in seen:
            continue
        seen.add(token)
        gpu_ids.append(token)
    return gpu_ids


def _shard_bounds(total_rows: int, num_shards: int, shard_index: int) -> tuple[int, int]:
    if num_shards <= 0:
        raise ValueError(f"num_shards 必须 > 0，当前={num_shards}")
    if shard_index < 1 or shard_index > num_shards:
        raise ValueError(
            f"shard_index 越界，要求 1..{num_shards}，当前={shard_index}"
        )
    base = total_rows // num_shards
    remainder = total_rows % num_shards
    start = (shard_index - 1) * base + min(shard_index - 1, remainder)
    end = start + base + (1 if shard_index <= remainder else 0)
    return start, end


def _strip_cli_arg(argv: list[str], flag: str, takes_value: bool) -> list[str]:
    result = []
    skip = 0
    prefix = f"{flag}="
    for token in argv:
        if skip > 0:
            skip -= 1
            continue
        if token == flag:
            if takes_value:
                skip = 1
            continue
        if takes_value and token.startswith(prefix):
            continue
        result.append(token)
    return result


def _launch_sharded_inference(args) -> int:
    gpu_ids = _parse_gpu_list(getattr(args, "gpus", None))
    if not gpu_ids:
        return 0
    if len(gpu_ids) > 1 and not getattr(args, "protein_peptide_csv", None):
        raise ValueError("多卡自动分片模式要求提供 --protein_peptide_csv")

    output_dir = Path(str(args.output_dir)).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    launch_log_dir = output_dir / "launch_logs"
    launch_log_dir.mkdir(parents=True, exist_ok=True)
    launch_tsv = output_dir / "launch_pids.tsv"

    base_argv = list(sys.argv[1:])
    for flag, takes_value in [
        ("--gpus", True),
        ("--num_shards", True),
        ("--shard_index", True),
    ]:
        base_argv = _strip_cli_arg(base_argv, flag, takes_value=takes_value)

    num_shards = len(gpu_ids)
    rows = ["shard\tgpu\tpid\tlog"]
    print(
        "[launcher] "
        f"gpus={','.join(gpu_ids)} "
        f"num_shards={num_shards} "
        f"output_dir={output_dir}"
    )

    for shard_index, gpu_id in enumerate(gpu_ids, start=1):
        log_file = launch_log_dir / f"shard{shard_index}.gpu{gpu_id}.log"
        child_cmd = [
            sys.executable,
            "-u",
            "inference.py",
            *base_argv,
            "--num_shards",
            str(num_shards),
            "--shard_index",
            str(shard_index),
        ]
        child_env = os.environ.copy()
        child_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        child_env["PYTHONPATH"] = os.getcwd()
        with log_file.open("w") as log_handle:
            proc = subprocess.Popen(
                child_cmd,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                cwd=os.getcwd(),
                env=child_env,
            )
        rows.append(f"{shard_index}\t{gpu_id}\t{proc.pid}\t{log_file}")
        pid_file = output_dir / f"shard{shard_index}.gpu{gpu_id}.pid"
        pid_file.write_text(f"{proc.pid}\n")
        print(
            "[launcher] started "
            f"shard={shard_index}/{num_shards} gpu={gpu_id} pid={proc.pid} log={log_file}"
        )

    launch_tsv.write_text("\n".join(rows) + "\n")
    print(f"[launcher] pid table written to {launch_tsv}")
    return 1


def _resolve_device(args) -> torch.device:
    choice = str(getattr(args, "device", "auto") or "auto").lower()
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "gpu":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[warn] --device gpu 但检测不到CUDA，回退到CPU")
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _format_seconds(seconds: float) -> str:
    if seconds < 1.0:
        return f"{seconds * 1000:.1f}ms"
    return f"{seconds:.3f}s"


def _empty_batch_timing_stats():
    return {
        "build_seconds": 0.0,
        "init_seconds": 0.0,
        "predict_seconds": 0.0,
        "visualize_seconds": 0.0,
        "save_seconds": 0.0,
        "sampling_forward_seconds": 0.0,
        "sampling_update_step_seconds": 0.0,
        "sampling_final_refine_seconds": 0.0,
        "sampling_update_seconds": 0.0,
        "sample_seconds": 0.0,
        "batch_total_seconds": 0.0,
    }


def _accumulate_timing_stats(total_stats, batch_stats):
    if not batch_stats:
        return total_stats
    for key in total_stats.keys():
        total_stats[key] += float(batch_stats.get(key, 0.0) or 0.0)
    return total_stats


def _build_timing_output_payload(args, runtime_stats):
    build_seconds = float(runtime_stats["batch_timing"]["build_seconds"])
    init_seconds = float(runtime_stats["batch_timing"]["init_seconds"])
    predict_seconds = float(runtime_stats["batch_timing"]["predict_seconds"])
    visualize_seconds = float(runtime_stats["batch_timing"]["visualize_seconds"])
    save_seconds = float(runtime_stats["batch_timing"]["save_seconds"])
    forward_seconds = float(runtime_stats["batch_timing"]["sampling_forward_seconds"])
    update_seconds = float(runtime_stats["batch_timing"]["sampling_update_seconds"])
    final_refine_seconds = float(
        runtime_stats["batch_timing"]["sampling_final_refine_seconds"]
    )
    loop_fetch_seconds = float(runtime_stats["loop_fetch_seconds"])
    loop_process_seconds = float(runtime_stats["loop_process_seconds"])
    loop_total_seconds = loop_fetch_seconds + loop_process_seconds

    fetch_preprocess_seconds = loop_fetch_seconds + build_seconds + init_seconds
    save_postprocess_seconds = predict_seconds + visualize_seconds + save_seconds
    other_overhead_seconds = loop_process_seconds - (
        build_seconds
        + init_seconds
        + forward_seconds
        + update_seconds
        + predict_seconds
        + visualize_seconds
        + save_seconds
    )
    five_segment_sum_seconds = (
        fetch_preprocess_seconds
        + forward_seconds
        + update_seconds
        + save_postprocess_seconds
        + other_overhead_seconds
    )
    success_complexes = max(
        0,
        int(runtime_stats["total_complexes"])
        - int(runtime_stats["failed_complexes"])
        - int(runtime_stats["skipped_complexes"]),
    )
    requested_n = int(getattr(args, "N", 1) or 1)
    return {
        "timing_schema_version": "flow_5segment_v1",
        "output_dir": str(Path(str(args.output_dir)).resolve()),
        "protein_peptide_csv": str(
            Path(str(args.protein_peptide_csv)).resolve()
        )
        if getattr(args, "protein_peptide_csv", None)
        else None,
        "model_dir": str(Path(str(args.model_dir)).resolve()),
        "ckpt": str(getattr(args, "ckpt", None)),
        "timing_force_cuda_sync": bool(
            getattr(args, "timing_force_cuda_sync", False)
        ),
        "total_complexes": int(runtime_stats["total_complexes"]),
        "success_complexes": success_complexes,
        "failed_complexes": int(runtime_stats["failed_complexes"]),
        "skipped_complexes": int(runtime_stats["skipped_complexes"]),
        "fetch_preprocess_seconds": fetch_preprocess_seconds,
        "forward_seconds": forward_seconds,
        "update_seconds": update_seconds,
        "save_postprocess_seconds": save_postprocess_seconds,
        "other_overhead_seconds": other_overhead_seconds,
        "five_segment_sum_seconds": five_segment_sum_seconds,
        "loop_fetch_seconds": loop_fetch_seconds,
        "loop_process_seconds": loop_process_seconds,
        "loop_total_seconds": loop_total_seconds,
        "batch_timing": {
            **runtime_stats["batch_timing"],
            "sampling_final_refine_merge_rule": (
                "sampling_update_seconds = "
                "sampling_update_step_seconds + sampling_final_refine_seconds"
            ),
        },
        "run_timing": {
            "data_seconds": float(runtime_stats["data_seconds"]),
            "model_seconds": float(runtime_stats["model_seconds"]),
            "confidence_seconds": float(runtime_stats["confidence_seconds"]),
            "loop_fetch_seconds": loop_fetch_seconds,
            "loop_process_seconds": loop_process_seconds,
            "loop_total_seconds": loop_total_seconds,
            "loop_wall_clock_observed_seconds": float(
                runtime_stats["loop_wall_clock_observed_seconds"]
            ),
        },
        "throughput": {
            "sec_per_complex": (
                float(loop_total_seconds / success_complexes)
                if success_complexes > 0
                else None
            ),
            "complexes_per_second": (
                float(success_complexes / loop_total_seconds)
                if success_complexes > 0 and loop_total_seconds > 0
                else 0.0
            ),
            "poses_per_second": (
                float(success_complexes * requested_n / loop_total_seconds)
                if success_complexes > 0 and loop_total_seconds > 0
                else 0.0
            ),
        },
        "five_segments": {
            "fetch_preprocess_seconds": fetch_preprocess_seconds,
            "forward_seconds": forward_seconds,
            "update_seconds": update_seconds,
            "save_postprocess_seconds": save_postprocess_seconds,
            "other_overhead_seconds": other_overhead_seconds,
            "five_segment_sum_seconds": five_segment_sum_seconds,
        },
        "notes": {
            "loop_total_definition": "loop_total_seconds = loop_fetch_seconds + loop_process_seconds",
            "flow_five_segment_definition": {
                "fetch_preprocess_seconds": "loop_fetch + build + init",
                "forward_seconds": "sampling 内所有 step 的 forward 累加",
                "update_seconds": "sampling 内所有 step 的 update 累加 + final_refine",
                "save_postprocess_seconds": "predict + visualize + save",
                "other_overhead_seconds": "loop_process - (build + init + forward + update + predict + visualize + save)",
            },
            "final_refine_merge_rule": "final_refine 未单拆 forward/update，整段并入 update_seconds。",
        },
    }


def _write_timing_output_json(path_str, payload):
    if not path_str:
        return
    output_path = Path(str(path_str)).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _read_pdb_center(pdb_path: Path) -> Optional[np.ndarray]:
    if not pdb_path.is_file():
        return None
    coords = []
    with pdb_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except Exception:
                continue
            coords.append((x, y, z))
    if not coords:
        return None
    return np.asarray(coords, dtype=np.float32).mean(axis=0)


def _prealign_peptide_to_native_center(graph, receptor_pdb: str) -> bool:
    """将当前图中的肽中心平移到 receptor.pdb 同目录 peptide.pdb 中心（仅平移，不旋转）。"""
    try:
        pep_pos = graph["pep_a"].pos
        original_center = graph.original_center
    except Exception:
        return False
    if pep_pos is None or pep_pos.numel() == 0:
        return False
    if not receptor_pdb:
        return False
    receptor_path = Path(str(receptor_pdb))
    native_peptide_path = receptor_path.parent / "peptide.pdb"
    native_center_abs = _read_pdb_center(native_peptide_path)
    if native_center_abs is None:
        return False
    if torch.is_tensor(original_center):
        center_vec = original_center.to(device=pep_pos.device, dtype=pep_pos.dtype).reshape(-1)
        if center_vec.numel() >= 3:
            center_vec = center_vec[:3]
        else:
            center_vec = torch.zeros(3, device=pep_pos.device, dtype=pep_pos.dtype)
    else:
        center_vec = torch.zeros(3, device=pep_pos.device, dtype=pep_pos.dtype)
    native_center_rel = torch.tensor(native_center_abs, device=pep_pos.device, dtype=pep_pos.dtype) - center_vec
    pep_center = pep_pos.mean(dim=0)
    shift = native_center_rel - pep_center
    graph["pep_a"].pos = pep_pos + shift
    return True


def _load_ckpt_payload(ckpt_path):
    if not ckpt_path or not os.path.exists(ckpt_path):
        return None
    cache_key = os.path.realpath(ckpt_path)
    if cache_key not in _CKPT_PAYLOAD_CACHE:
        _CKPT_PAYLOAD_CACHE[cache_key] = torch.load(
            ckpt_path, map_location=torch.device("cpu")
        )
    return _CKPT_PAYLOAD_CACHE[cache_key]


def load_model(score_model_args, ckpt_path, device, state_dict=None, move_to_device=True):
    model = get_model(score_model_args, no_parallel=True)
    if ckpt_path:
        try:
            if state_dict is None:
                state_dict = _load_ckpt_payload(ckpt_path)
            if state_dict is None:
                raise FileNotFoundError(ckpt_path)
            # 若权重包含 ESM 层但当前配置未开启，强制开启避免加载失败
            model_keys = state_dict.get("model", {}).keys()
            if any("lm_embedding_layer" in k for k in model_keys):
                if getattr(score_model_args, "esm_embeddings_path_train", None) is None:
                    score_model_args.esm_embeddings_path_train = "inline"
                if getattr(score_model_args, "esm_embeddings_peptide_train", None) is None:
                    score_model_args.esm_embeddings_peptide_train = "inline"
                model = get_model(score_model_args, no_parallel=True)
            # 若权重里包含 tail_proj，先显式创建对应层以便严格加载
            if any("encoder.rec_node_embedding.tail_proj" in k for k in model_keys):
                emb_dim = model.encoder.rec_node_embedding.amino_ebd.embedding_dim
                if not hasattr(model.encoder.rec_node_embedding, "tail_proj"):
                    model.encoder.rec_node_embedding.tail_proj = torch.nn.Linear(104, emb_dim)
            if hasattr(model.encoder, "pep_node_embedding") and any("encoder.pep_node_embedding.tail_proj" in k for k in model_keys):
                emb_dim = model.encoder.pep_node_embedding.amino_ebd.embedding_dim
                if not hasattr(model.encoder.pep_node_embedding, "tail_proj"):
                    model.encoder.pep_node_embedding.tail_proj = torch.nn.Linear(104, emb_dim)
            try:
                model.load_state_dict(state_dict["model"], strict=True)
            except RuntimeError as strict_err:
                # 兼容历史 ckpt：新加的轻量模块（如 self_cond）在旧权重里不存在时，放宽到 non-strict
                model.load_state_dict(state_dict["model"], strict=False)
                print(f"[warn] strict load failed, fallback to strict=False: {strict_err}")
            if "ema_weights" in state_dict:
                ema_weights = ExponentialMovingAverage(
                    model.parameters(), decay=score_model_args.ema_rate
                )
                ema_weights.load_state_dict(state_dict["ema_weights"], device=device)
                ema_weights.copy_to(model.parameters())
            print(f"Loaded pretrained weights from {ckpt_path}")
        except FileNotFoundError:
            print(f"⚠️ Checkpoint not found at {ckpt_path}; using random init.")
        except Exception as e:
            print(f"⚠️ Failed to load checkpoint {ckpt_path}: {e}; using random init.")
    else:
        print("⚠️ No checkpoint provided/found; using randomly initialized weights for inference.")
    if move_to_device:
        model = model.to(device)
    return model

def _load_ckpt_config(ckpt_path):
    try:
        state_dict = _load_ckpt_payload(ckpt_path)
    except Exception:
        return None
    if state_dict is None:
        return None
    cfg = state_dict.get("config")
    if not isinstance(cfg, dict):
        return None
    return cfg

def _merge_cfg_into_namespace(ns, cfg_dict, allow_keys=None):
    if not cfg_dict:
        return
    if allow_keys is None:
        allow_keys = set(cfg_dict.keys())
    for key, value in cfg_dict.items():
        if key not in allow_keys:
            continue
        if isinstance(value, dict):
            cur = getattr(ns, key, {}) or {}
            if not isinstance(cur, dict):
                cur = {}
            merged = dict(cur)
            merged.update(value)
            setattr(ns, key, merged)
        else:
            setattr(ns, key, value)


def _resolve_peptide_source(args, ckpt_cfg, score_model_args):
    peptide_source = str(getattr(args, "peptide_source", "pdb") or "pdb").lower()
    if peptide_source != "pdb":
        raise ValueError(
            f"peptide_source={peptide_source} 已废弃；当前仓库只支持现成 peptide.pdb"
        )
    args.peptide_source = "pdb"
    print("[info] peptide_source fixed: pdb")
    return "pdb"


def _validate_inference_args(args):
    """尽早拦截废弃语义和明显缺参，别让脚本跑到半路才炸。"""
    if not getattr(args, "model_dir", None):
        raise ValueError("缺少 --model_dir；请提供模型目录。")

    csv_mode = getattr(args, "protein_peptide_csv", None) is not None
    if not csv_mode:
        missing = []
        if not getattr(args, "complex_name", None):
            missing.append("--complex_name")
        if not getattr(args, "protein_description", None):
            missing.append("--protein_description")
        if not getattr(args, "peptide_description", None):
            missing.append("--peptide_description")
        if missing:
            raise ValueError(
                f"单样本模式缺少必要参数：{', '.join(missing)}；"
                "或者改用 --protein_peptide_csv 批量输入。"
            )
        peptide_description = str(getattr(args, "peptide_description", "") or "")
        if peptide_description and "pdb" not in peptide_description.lower():
            raise ValueError(
                "当前仓库已废弃‘直接给肽序列’入口；"
                "--peptide_description 现在必须是现成 peptide.pdb 路径。"
            )

    if getattr(args, "scoring_function", "none") == "confidence":
        missing = []
        if not getattr(args, "confidence_model_dir", None):
            missing.append("--confidence_model_dir")
        if not getattr(args, "confidence_ckpt", None):
            missing.append("--confidence_ckpt")
        if missing:
            raise ValueError(
                f"scoring_function=confidence 缺少必要参数：{', '.join(missing)}"
            )

def load_config(args):
    if args.config:
        # config 只覆盖“未显式提供”的参数；parser 默认值也视为未显式提供。
        config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
        arg_dict = args.__dict__
        default_args = vars(get_parser().parse_args([]))
        for key, value in config_dict.items():
            cur = arg_dict.get(key, None)
            default = default_args.get(key, None)
            # 仅在当前参数为空或仍等于 parser 默认值时，才用 config 回填。
            if (
                cur in (None, "", [])
                or (isinstance(cur, list) and len(cur) == 0)
                or cur == default
            ):
                if isinstance(value, list):
                    arg_dict[key] = list(value)
                else:
                    arg_dict[key] = value


def _run_auto_eval_if_enabled(args):
    """可选：推理结束后自动计算 RMSD/DockQ。"""
    if not bool(getattr(args, "auto_eval_metrics", False)):
        return

    pred_root = Path(str(args.output_dir)).resolve()
    csv_arg = getattr(args, "auto_eval_csv", None) or getattr(args, "protein_peptide_csv", None)
    if not csv_arg:
        print("[auto_eval] skipped: 缺少 --auto_eval_csv 且未提供 --protein_peptide_csv")
        return

    csv_path = Path(str(csv_arg)).expanduser().resolve()
    if not csv_path.is_file():
        print(f"[auto_eval] skipped: CSV 不存在 -> {csv_path}")
        return

    output_arg = str(getattr(args, "auto_eval_output", "metrics_rmsd_dockq.csv") or "metrics_rmsd_dockq.csv")
    out_path = Path(output_arg).expanduser()
    if not out_path.is_absolute():
        out_path = pred_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dockq_cmd = getattr(args, "auto_eval_dockq_cmd", None)
    if not dockq_cmd:
        which_dockq = shutil.which("DockQ")
        if which_dockq:
            dockq_cmd = which_dockq
        else:
            env_dockq = Path(sys.prefix) / "bin" / "DockQ"
            if env_dockq.is_file():
                dockq_cmd = str(env_dockq)

    eval_script = Path(__file__).resolve().parent / "scripts" / "eval_rmsd_from_preds.py"
    if not eval_script.is_file():
        print(f"[auto_eval] skipped: 评测脚本不存在 -> {eval_script}")
        return

    cmd = [
        sys.executable,
        str(eval_script),
        "--pred_root",
        str(pred_root),
        "--csv",
        str(csv_path),
        "--output",
        str(out_path),
    ]
    if dockq_cmd:
        cmd += ["--dockq_cmd", str(dockq_cmd)]

    print("[auto_eval] start:", " ".join(cmd))
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    if proc.stdout:
        print(proc.stdout.strip())
    if proc.stderr:
        print(proc.stderr.strip())
    if proc.returncode != 0:
        print(f"[auto_eval] failed (exit={proc.returncode}) elapsed={_format_seconds(elapsed)}")
        return
    print(f"[auto_eval] done elapsed={_format_seconds(elapsed)} output={out_path}")

def _enable_esm_from_ckpt(score_model_args, ckpt_path):
    """若 ckpt 含 ESM 权重，提前开启 ESM 开关，保证数据与模型一致。"""
    try:
        state_dict = _load_ckpt_payload(ckpt_path)
    except Exception:
        return
    if state_dict is None:
        return
    model_keys = state_dict.get("model", {}).keys()
    if any("lm_embedding_layer" in k for k in model_keys):
        if getattr(score_model_args, "esm_embeddings_path_train", None) is None:
            score_model_args.esm_embeddings_path_train = "inline"
        if getattr(score_model_args, "esm_embeddings_peptide_train", None) is None:
            score_model_args.esm_embeddings_peptide_train = "inline"

def prepare_data(args, score_model_args):
    if args.protein_peptide_csv is not None:
        df = pd.read_csv(args.protein_peptide_csv)
        df = df.fillna("")
        num_shards = getattr(args, "num_shards", None)
        shard_index = getattr(args, "shard_index", None)
        if num_shards is not None or shard_index is not None:
            num_shards = int(num_shards or 1)
            shard_index = int(shard_index or 1)
            start, end = _shard_bounds(len(df), num_shards, shard_index)
            print(
                "[shard] "
                f"index={shard_index}/{num_shards} "
                f"rows={start}:{end} "
                f"count={end - start}"
            )
            df = df.iloc[start:end].reset_index(drop=True)
        df = _canonicalize_strict536_input_df(
            df,
            source_csv=getattr(args, "protein_peptide_csv", None),
        )
        receptor_pt_list = None
        if "receptor_pt" in df.columns:
            receptor_pt_list = set_nones(df["receptor_pt"].tolist())
        elif "receptor_pt_path" in df.columns:
            receptor_pt_list = set_nones(df["receptor_pt_path"].tolist())
        if "complex_name" not in df.columns and "pdb_id" in df.columns:
            df["complex_name"] = df["pdb_id"]
        # 如果没有描述列，尝试用 pdb 路径填充，防止 KeyError
        if "protein_description" not in df.columns:
            if "receptor_pdb" in df.columns:
                df["protein_description"] = df["receptor_pdb"]
            else:
                df["protein_description"] = ""
        if "peptide_description" not in df.columns:
            if "peptide_pdb" in df.columns:
                df["peptide_description"] = df["peptide_pdb"]
            else:
                df["peptide_description"] = ""
        complex_name_list = set_nones(df["complex_name"].tolist())
        protein_description_list = set_nones(df["protein_description"].tolist())
        peptide_description_list = set_nones(df["peptide_description"].tolist())
    else:
        complex_name_list = [args.complex_name]
        protein_description_list = [args.protein_description]
        peptide_description_list = [args.peptide_description]
        receptor_pt_list = None
        if (
            args.protein_description is not None
            and str(args.protein_description).lower().endswith(".pt")
        ):
            receptor_pt_list = [args.protein_description]
    
    precomputed_rec_lm_embeddings = None
    precomputed_pep_lm_embeddings = None
    if getattr(score_model_args, "esm_embeddings_path_train", None) is not None:
        precomputed_rec_lm_embeddings, precomputed_pep_lm_embeddings = load_sidecar_esm_embeddings(
            protein_description_list,
            peptide_description_list,
        )
    
    complex_name_list = [
        name if name is not None else f"complex_{i}"
        for i, name in enumerate(complex_name_list)
    ]
    for name in complex_name_list:
        write_dir = f"{args.output_dir}/{name}"
        os.makedirs(write_dir, exist_ok=True)
    
    # preprocessing of initial proteins and peptides into geometric graphs
    embedding_mode = "esm" if score_model_args.esm_embeddings_path_train is not None else "onehot"
    use_native = True
    precheck_workers = min(max(int(getattr(args, "cpu", 1) or 1) - 1, 0), 4)
    return InferenceDataset(
        output_dir=args.output_dir,
        complex_name_list=complex_name_list,
        protein_description_list=protein_description_list,
        peptide_description_list=peptide_description_list,
        lm_embeddings=score_model_args.esm_embeddings_path_train is not None,
        lm_embeddings_pep=score_model_args.esm_embeddings_peptide_train is not None,
        precomputed_lm_embeddings=precomputed_rec_lm_embeddings,
        precomputed_lm_embeddings_pep=precomputed_pep_lm_embeddings,
        receptor_pt_list=receptor_pt_list,
        peptide_esm_path=getattr(args, "peptide_esm_path", None),
        cache_peptide_graph=bool(getattr(args, "cache_peptide_graph", True)),
        embedding_mode=embedding_mode,
        use_native_peptide_pose=use_native,
        precheck_workers=precheck_workers,
    )

def prepare_data_list(original_complex_graph, N):
    def _clone_complex_graph(pep_pos_override=None):
        cloned = original_complex_graph.clone()
        # 这些全局 list 在推理里只读，但这里复制一份可以避免后续有人原地改列表时串样本。
        if "partials" in original_complex_graph:
            cloned["partials"] = list(original_complex_graph["partials"])
        if "peptide_inits" in original_complex_graph:
            cloned["peptide_inits"] = list(original_complex_graph["peptide_inits"])
        if pep_pos_override is not None:
            cloned["pep_a"].pos = pep_pos_override.to(dtype=cloned["pep_a"].pos.dtype)
        return cloned

    data_list = []
    nums = []
    if len(original_complex_graph["peptide_inits"]) == 1:
        data_list = [_clone_complex_graph() for _ in range(N)]
    elif len(original_complex_graph["peptide_inits"]) > 1:
         for i, peptide_init in enumerate(
                    original_complex_graph["peptide_inits"]
                ):
            pep_pos_override = None
            if i != 0:
                pep_pos_override = (
                    torch.from_numpy(
                        MDAnalysis.Universe(peptide_init).atoms.positions
                    )
                    - original_complex_graph.original_center
                )
            num = N - sum(nums) if i == len(original_complex_graph["peptide_inits"]) - 1 else round(
                original_complex_graph["partials"][i] / sum(original_complex_graph["partials"]) * N
            )
            nums.append(num)
            data_list.extend([_clone_complex_graph(pep_pos_override) for _ in range(num)])
    return data_list


def _sample_flow_time(flow_cfg):
    cfg = flow_cfg or {}
    t_min = float(cfg.get("t_min", 0.0) or 0.0)
    t_max = float(cfg.get("t_max", 1.0) or 1.0)
    if not (0.0 <= t_min <= t_max <= 1.0):
        raise ValueError(
            f"[flow] invalid t_min/t_max: t_min={t_min} t_max={t_max}, expected 0<=t_min<=t_max<=1"
        )
    mode = str(cfg.get("time_sampling", "uniform") or "uniform").lower()
    already_scaled = False
    if mode == "uniform":
        t = float(np.random.rand())
    elif mode in {"sqrt", "sqrt_uniform"}:
        t = float(np.sqrt(np.random.rand()))
    elif mode == "beta":
        alpha = float(cfg.get("beta_alpha", 2.0) or 2.0)
        beta = float(cfg.get("beta_beta", 1.0) or 1.0)
        if alpha <= 0 or beta <= 0:
            raise ValueError(f"[flow] invalid beta params: alpha={alpha} beta={beta}")
        t = float(np.random.beta(alpha, beta))
    elif mode == "mixed":
        def _rand_uniform(lo, hi):
            return float(lo + (hi - lo) * np.random.rand())

        mix_fixed_prob = float(cfg.get("mix_fixed_prob", 0.0) or 0.0)
        mix_fixed_t = float(cfg.get("mix_fixed_t", 0.6) or 0.6)
        mix_beta_prob = float(cfg.get("mix_beta_prob", 0.0) or 0.0)
        mix_beta_alpha = float(cfg.get("mix_beta_alpha", 2.0) or 2.0)
        mix_beta_beta = float(cfg.get("mix_beta_beta", 1.0) or 1.0)
        mix_beta_min = float(cfg.get("mix_beta_min", t_min) or t_min)
        mix_beta_max = float(cfg.get("mix_beta_max", t_max) or t_max)
        mix_small_prob = float(cfg.get("mix_small_prob", 0.0) or 0.0)
        mix_small_min = float(cfg.get("mix_small_min", t_min) or t_min)
        mix_small_max = float(cfg.get("mix_small_max", t_max) or t_max)

        if not (0.0 <= mix_fixed_t <= 1.0):
            raise ValueError(f"[flow] invalid mix_fixed_t={mix_fixed_t} (expected 0~1)")
        if not (t_min <= mix_fixed_t <= t_max):
            raise ValueError(
                f"[flow] mix_fixed_t={mix_fixed_t} must be within t_min/t_max ({t_min}~{t_max})"
            )
        def _check_range(name, lo, hi):
            if not (0.0 <= lo <= hi <= 1.0):
                raise ValueError(f"[flow] invalid {name} range: {lo}~{hi} (expected 0~1)")
            if lo < t_min or hi > t_max:
                raise ValueError(
                    f"[flow] {name} range must be within t_min/t_max: {lo}~{hi} vs {t_min}~{t_max}"
                )

        _check_range("mix_beta", mix_beta_min, mix_beta_max)
        _check_range("mix_small", mix_small_min, mix_small_max)
        if mix_beta_alpha <= 0 or mix_beta_beta <= 0:
            raise ValueError(
                f"[flow] invalid mix_beta params: alpha={mix_beta_alpha} beta={mix_beta_beta} (expected >0)"
            )
        total = mix_fixed_prob + mix_beta_prob + mix_small_prob
        if total <= 0:
            raise ValueError("[flow] mixed sampling weights must sum to >0")
        r = np.random.rand() * total
        if r < mix_fixed_prob:
            t = mix_fixed_t
        elif r < mix_fixed_prob + mix_beta_prob:
            u = np.random.beta(mix_beta_alpha, mix_beta_beta)
            t = mix_beta_min + (mix_beta_max - mix_beta_min) * float(u)
        else:
            t = _rand_uniform(mix_small_min, mix_small_max)
        already_scaled = True
    elif mode in {"clipped", "clamp"}:
        t = float(np.random.rand())
    else:
        raise ValueError(f"[flow] unsupported time_sampling={mode}")
    if not already_scaled:
        t = t_min + (t_max - t_min) * t
    eps = float(cfg.get("t_eps", 1e-6) or 1e-6)
    if t <= eps:
        t = eps
    return t


def _resolve_flow_sigma_max(score_model_args):
    flow_cfg = getattr(score_model_args, "flow", {}) or {}
    sigma = {
        "tr": flow_cfg.get("sigma_tr_max"),
        "rot": flow_cfg.get("sigma_rot_max"),
        "tor_bb": flow_cfg.get("sigma_tor_bb_max"),
        "tor_sc": flow_cfg.get("sigma_tor_sc_max"),
    }
    missing = [name for name, value in sigma.items() if value is None]
    if missing:
        raise ValueError(f"[flow] 缺少 sigma_max 配置: {', '.join(missing)}")
    return {name: float(value) for name, value in sigma.items()}


def _get_graph_center(graph, flow_cfg):
    rec_pos = graph["receptor"].pos if "receptor" in graph else None
    pep_a_pos = graph["pep_a"].pos if "pep_a" in graph else None
    if rec_pos is None or not torch.is_tensor(rec_pos) or rec_pos.numel() == 0:
        rec_com = None
    else:
        rec_com = rec_pos.mean(dim=0)
    if pep_a_pos is None or not torch.is_tensor(pep_a_pos) or pep_a_pos.numel() == 0:
        pep_com = None
    else:
        pep_com = pep_a_pos.mean(dim=0)
    center_mode = str(flow_cfg.get("tr_center_mode", "receptor_com") or "receptor_com").lower()
    if center_mode == "receptor_com":
        center = rec_com or pep_com
    elif center_mode == "pep_com":
        center = pep_com or rec_com
    else:
        raise ValueError(f"[flow] unsupported tr_center_mode={center_mode}")
    return center, pep_com


def _sample_tr_update_shell(graph, flow_cfg):
    cfg = flow_cfg or {}
    r_min = float(cfg.get("tr_r_min", 0.0) or 0.0)
    r_max = cfg.get("tr_r_max", None)
    if r_max is None:
        r_max = r_min
    r_max = float(r_max)
    if r_min < 0 or r_max <= 0 or r_max < r_min:
        raise ValueError(f"[flow] invalid tr_r_min/tr_r_max: r_min={r_min} r_max={r_max}")
    r_mu = cfg.get("tr_r_mu", None)
    r_sigma = cfg.get("tr_r_sigma", None)
    if r_mu is None:
        r_mu = 0.5 * (r_min + r_max)
    if r_sigma is None:
        r_sigma = max((r_max - r_min) / 4.0, 1e-3)
    r_mu = float(r_mu)
    r_sigma = float(r_sigma)
    if r_sigma <= 0:
        raise ValueError(f"[flow] invalid tr_r_sigma={r_sigma}")

    center, pep_com = _get_graph_center(graph, cfg)
    rec_pos = graph["receptor"].pos if "receptor" in graph else None
    pep_a_pos = graph["pep_a"].pos if "pep_a" in graph else None
    if center is None or pep_com is None:
        return None

    min_dist = float(cfg.get("tr_min_dist", 0.0) or 0.0)
    max_tries = int(cfg.get("tr_reject_max_tries", 30) or 30)
    max_tries = max(1, max_tries)

    last_update = None
    for _ in range(max_tries):
        direction = torch.randn(3)
        norm = direction.norm()
        if norm < 1e-6:
            direction = torch.tensor([1.0, 0.0, 0.0])
            norm = 1.0
        direction = direction / norm

        radius = None
        for _ in range(32):
            cand = float(torch.normal(mean=r_mu, std=r_sigma, size=(1,)).item())
            if r_min <= cand <= r_max:
                radius = cand
                break
        if radius is None:
            radius = float(torch.empty(1).uniform_(r_min, r_max).item())

        target_com = center + direction * radius
        tr_update = (target_com - pep_com).unsqueeze(0)
        last_update = tr_update

        if min_dist <= 0 or rec_pos is None or pep_a_pos is None:
            return tr_update
        moved_pep = pep_a_pos + tr_update
        try:
            dist = torch.cdist(moved_pep, rec_pos).min().item()
        except Exception:
            return tr_update
        if dist >= min_dist:
            return tr_update

    return last_update


def _apply_inference_init_noise(data_list, score_model_args):
    flow_cfg = getattr(score_model_args, "flow", {}) or {}
    sigma_max = _resolve_flow_sigma_max(score_model_args)
    tr_sampling = str(flow_cfg.get("tr_sampling", "gaussian") or "gaussian").lower()
    target_mode = str(flow_cfg.get("target_mode", "velocity") or "velocity").lower()
    rot_noise_mode = str(flow_cfg.get("rot_noise_mode", "isotropic") or "isotropic").lower()
    if rot_noise_mode not in {"isotropic", "anchor"}:
        raise ValueError(f"[flow] 不支持的 rot_noise_mode={rot_noise_mode}（支持 isotropic/anchor）")
    t_eps = float(flow_cfg.get("t_eps", 1e-6) or 1e-6)

    def _safe_normalize(vec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        norm = vec.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
        return vec / norm

    def _frame_from_rec_anchor(rec_pos: Optional[torch.Tensor], pep_center: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if rec_pos is None or rec_pos.numel() == 0 or pep_center is None:
            return None
        if rec_pos.shape[0] < 3:
            return None
        dists = (rec_pos - pep_center).norm(dim=1)
        idx = torch.argsort(dists)
        p0 = rec_pos[idx[0]]
        p1 = rec_pos[idx[1]]
        p2 = rec_pos[idx[2]]
        x = _safe_normalize(p1 - p0)
        y_raw = _safe_normalize(p2 - p0)
        z = torch.cross(x, y_raw)
        if z.norm().item() < 1e-6:
            centered = rec_pos - rec_pos.mean(dim=0, keepdim=True)
            cov = centered.T @ centered
            _evals, evecs = torch.linalg.eigh(cov)
            x = _safe_normalize(evecs[:, 2])
            y = _safe_normalize(evecs[:, 1])
            z = torch.cross(x, y)
            if z.norm().item() < 1e-6:
                return None
        z = _safe_normalize(z)
        y = _safe_normalize(torch.cross(z, x))
        rec_com = rec_pos.mean(dim=0)
        if torch.dot(z, pep_center - rec_com) < 0:
            z = -z
            y = -y
        return torch.stack([x, y, z], dim=1)

    def _sample_rot_target(graph) -> torch.Tensor:
        if rot_noise_mode == "anchor":
            rec_pos = graph["receptor"].pos if "receptor" in graph else None
            pep_pos = graph["pep_a"].pos if "pep_a" in graph else None
            if rec_pos is not None and pep_pos is not None:
                pep_center = pep_pos.mean(dim=0)
                rec_frame = _frame_from_rec_anchor(rec_pos, pep_center)
                if rec_frame is not None:
                    axis = _safe_normalize(rec_frame[:, 2])
                    omega = float(sample(sigma_max["rot"]))
                    return axis * omega
        return torch.from_numpy(sample_vec(sigma_max["rot"])).float()

    for graph in data_list:
        t = _sample_flow_time(flow_cfg)
        time_scale = t if target_mode == "velocity" else 1.0 - t
        if time_scale < t_eps:
            time_scale = t_eps

        tr_update = None
        if tr_sampling in {"shell", "ring", "outside", "gaussian_shell"}:
            tr_update = _sample_tr_update_shell(graph, flow_cfg)
        if tr_update is None:
            center_update = torch.zeros((1, 3))
            center, pep_com = _get_graph_center(graph, flow_cfg)
            if center is not None and pep_com is not None:
                center_update = (center - pep_com).unsqueeze(0)
            tr_noise = torch.normal(0.0, sigma_max["tr"], size=(1, 3))
            tr_update = center_update + tr_noise * float(time_scale)

        rot_target = _sample_rot_target(graph)
        rot_update = rot_target * float(time_scale)

        bb_count, sc_count = _get_torsion_edge_counts(graph)
        if bb_count > 0:
            tor_bb_target = torch.from_numpy(
                np.random.uniform(-sigma_max["tor_bb"], sigma_max["tor_bb"], size=bb_count)
            ).float()
            tor_bb_target = wrap_to_pi(tor_bb_target)
            tor_bb_update = wrap_to_pi(tor_bb_target * float(time_scale)).cpu().numpy()
        else:
            tor_bb_update = None

        if sc_count > 0:
            tor_sc_target = torch.from_numpy(
                np.random.uniform(-sigma_max["tor_sc"], sigma_max["tor_sc"], size=sc_count)
            ).float()
            tor_sc_target = wrap_to_pi(tor_sc_target)
            tor_sc_update = wrap_to_pi(tor_sc_target * float(time_scale)).cpu().numpy()
        else:
            tor_sc_update = None

        peptide_updater(
            graph,
            tr_update,
            rot_update,
            tor_bb_update,
            tor_sc_update,
        )

def _coerce_graph_float32(graph):
    if hasattr(graph, "success") and not graph.success:
        return
    if "pep_a" in graph and hasattr(graph["pep_a"], "pos"):
        graph["pep_a"].pos = graph["pep_a"].pos.to(dtype=torch.float32)
    if "receptor" in graph and hasattr(graph["receptor"], "pos"):
        graph["receptor"].pos = graph["receptor"].pos.to(dtype=torch.float32)
    if "receptor" in graph and hasattr(graph["receptor"], "tips"):
        graph["receptor"].tips = graph["receptor"].tips.to(dtype=torch.float32)
    if "receptor" in graph and hasattr(graph["receptor"], "node_v"):
        graph["receptor"].node_v = graph["receptor"].node_v.to(dtype=torch.float32)
    if "receptor" in graph and hasattr(graph["receptor"], "x"):
        graph["receptor"].x = graph["receptor"].x.to(dtype=torch.float32)
    if "pep" in graph and hasattr(graph["pep"], "x"):
        graph["pep"].x = graph["pep"].x.to(dtype=torch.float32)
    if ("receptor", "rec_contact", "receptor") in graph:
        if hasattr(graph["receptor", "rec_contact", "receptor"], "edge_s"):
            graph["receptor", "rec_contact", "receptor"].edge_s = graph[
                "receptor", "rec_contact", "receptor"
            ].edge_s.to(dtype=torch.float32)
        if hasattr(graph["receptor", "rec_contact", "receptor"], "edge_v"):
            graph["receptor", "rec_contact", "receptor"].edge_v = graph[
                "receptor", "rec_contact", "receptor"
            ].edge_v.to(dtype=torch.float32)


def _build_data_list(original_complex_graph, num_samples):
    data_list = prepare_data_list(original_complex_graph, num_samples)
    for graph in data_list:
        _coerce_graph_float32(graph)
    return data_list


def _build_eager_inference_batches(dataset, batch_size):
    total = len(dataset)
    if total <= 0:
        return []
    batches = []
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batches.append([dataset[i] for i in range(start, end)])
    return batches


def _consume_dataloader_base_seed():
    """对齐 PyTorch DataLoader 建 iterator 时的默认 RNG 消耗。"""
    _ = torch.empty((), dtype=torch.int64).random_().item()


def _save_visualization(write_dir, visualization_list, original_complex_graph, args, re_order):
    raw_pdb = MDAnalysis.Universe(
        StringIO(original_complex_graph["pep"].noh_mda), format="pdb"
    )
    if args.scoring_function in ["confidence", "ref2015"]:
        for rank, batch_idx in enumerate(re_order):
            raw_pdb.load_new(visualization_list[batch_idx], format=MemoryReader)
            with MDAnalysis.Writer(
                os.path.join(write_dir, f"rank{rank+1}_reverseprocess.pdb"),
                multiframe=True,
                bonds=None,
                n_atoms=raw_pdb.atoms.n_atoms,
            ) as pdb_writer:
                for ts in raw_pdb.trajectory:
                    pdb_writer.write(raw_pdb)
    else:
        for rank in range(len(visualization_list)):
            raw_pdb.load_new(visualization_list[rank], format=MemoryReader)
            with MDAnalysis.Writer(
                os.path.join(write_dir, f"pose{rank+1}_reverseprocess.pdb"),
                multiframe=True,
                bonds=None,
                n_atoms=raw_pdb.atoms.n_atoms,
            ) as pdb_writer:
                for ts in raw_pdb.trajectory:
                    pdb_writer.write(raw_pdb)

def save_predictions(write_dir, predict_pos, original_complex_graph, args, confidence):
    raw_pdb = MDAnalysis.Universe(StringIO(original_complex_graph["pep"].noh_mda), format="pdb")
    peptide_unrelaxed_files = []
    
    re_order = None
    # reorder predictions based on confidence output
    if confidence is not None:
        confidence = confidence.cpu().numpy()
        re_order = np.argsort(confidence)[::-1]
        confidence = confidence[re_order]
        if isinstance(predict_pos, np.ndarray):
            predict_pos = predict_pos[re_order]
        else:
            predict_pos = [predict_pos[i] for i in re_order]

    for rank, pos in enumerate(predict_pos):
        raw_pdb.atoms.positions = pos
        file_name = f"rank{rank+1}_{args.scoring_function}.pdb" if confidence is not None else f"pose{rank+1}.pdb"
        peptide_unrelaxed_file = os.path.join(write_dir, file_name)
        peptide_unrelaxed_files.append(peptide_unrelaxed_file)
        raw_pdb.atoms.write(peptide_unrelaxed_file)

    if args.scoring_function == "ref2015" or args.fastrelax:
        from utils.pyrosetta_utils import relax_score
        relaxed_poses = [peptide.replace(".pdb", "_relaxed.pdb") for peptide in peptide_unrelaxed_files]
        protein_raw_file = f"{write_dir}/{os.path.basename(write_dir)}_protein_raw.pdb"

        with multiprocessing.Pool(args.cpu) as pool:
            ref2015_scores = pool.map(
                relax_score,
                zip(
                    [protein_raw_file] * len(peptide_unrelaxed_files),
                    peptide_unrelaxed_files,
                    relaxed_poses,
                    [args.scoring_function == "ref2015"] * len(peptide_unrelaxed_files),
                ),
            )
        if ref2015_scores and ref2015_scores[0] is not None:
            re_order = np.argsort(ref2015_scores)
            score_results = [['file','ref2015score']]
            for rank, order in enumerate(re_order):
                os.rename(relaxed_poses[order], os.path.join(write_dir, f"rank{rank+1}_{args.scoring_function}.pdb"))
                score_results.append([f"rank{rank+1}_{args.scoring_function}", f"{ref2015_scores[order]:.2f}"])
            open(os.path.join(write_dir, "ref2015_score.csv"),'w').write('\n'.join([','.join(i) for i in score_results]))
    
    if re_order is not None:
        return re_order
    else:
        return 0


def _resolve_sampling_overrides(run_args, model_args):
    """合并CLI和模型配置，决定 flow 的实际参数。"""
    model_sampling_cfg = getattr(model_args, "sampling", {}) or {}
    flow_num_steps = run_args.flow_num_steps or model_sampling_cfg.get("num_steps_flow", 50)
    flow_solver = run_args.flow_solver or model_sampling_cfg.get("solver_flow", "euler")
    return flow_num_steps, flow_solver


def _apply_flow_infer_overrides(run_args, model_args):
    """把推理阶段的 flow 开关写回 model_args，保证 sampling/model forward 都能读到。"""
    flow_cfg = getattr(model_args, "flow", {}) or {}
    if not isinstance(flow_cfg, dict):
        flow_cfg = {}

    if run_args.flow_sparse_interface is not None:
        flow_cfg["sparse_interface"] = bool(run_args.flow_sparse_interface)
    if run_args.flow_sparse_interface_topk is not None:
        flow_cfg["sparse_interface_topk"] = int(run_args.flow_sparse_interface_topk)
    if run_args.flow_self_condition_infer is not None:
        flow_cfg["self_condition"] = bool(run_args.flow_self_condition_infer)
    if run_args.flow_final_refine is not None:
        flow_cfg["final_refine"] = bool(run_args.flow_final_refine)
    if run_args.flow_final_refine_scale is not None:
        flow_cfg["final_refine_scale"] = float(run_args.flow_final_refine_scale)
    if run_args.flow_final_refine_tr_scale is not None:
        flow_cfg["final_refine_tr_scale"] = float(run_args.flow_final_refine_tr_scale)
    if run_args.flow_final_refine_rot_scale is not None:
        flow_cfg["final_refine_rot_scale"] = float(run_args.flow_final_refine_rot_scale)
    if run_args.flow_final_refine_tor_scale is not None:
        flow_cfg["final_refine_tor_scale"] = float(run_args.flow_final_refine_tor_scale)
    if run_args.flow_steric_guidance is not None:
        flow_cfg["steric_guidance"] = bool(run_args.flow_steric_guidance)
    if run_args.flow_steric_guidance_scale is not None:
        flow_cfg["steric_guidance_scale"] = float(run_args.flow_steric_guidance_scale)
    if run_args.flow_steric_guidance_cutoff is not None:
        flow_cfg["steric_guidance_cutoff"] = float(run_args.flow_steric_guidance_cutoff)
    if run_args.flow_steric_guidance_temperature is not None:
        flow_cfg["steric_guidance_temperature"] = float(run_args.flow_steric_guidance_temperature)
    if run_args.flow_steric_guidance_torque_scale is not None:
        flow_cfg["steric_guidance_torque_scale"] = float(run_args.flow_steric_guidance_torque_scale)
    if run_args.flow_steric_guidance_max_tr is not None:
        flow_cfg["steric_guidance_max_tr"] = float(run_args.flow_steric_guidance_max_tr)
    if run_args.flow_steric_guidance_max_rot is not None:
        flow_cfg["steric_guidance_max_rot"] = float(run_args.flow_steric_guidance_max_rot)
    if run_args.flow_hard_overlap_guard is not None:
        flow_cfg["hard_overlap_guard"] = bool(run_args.flow_hard_overlap_guard)
    if run_args.flow_hard_overlap_guard_min_dist is not None:
        flow_cfg["hard_overlap_guard_min_dist"] = float(run_args.flow_hard_overlap_guard_min_dist)
    if run_args.flow_hard_overlap_guard_backoff is not None:
        flow_cfg["hard_overlap_guard_backoff"] = float(run_args.flow_hard_overlap_guard_backoff)
    if run_args.flow_hard_overlap_guard_max_backtracks is not None:
        flow_cfg["hard_overlap_guard_max_backtracks"] = int(run_args.flow_hard_overlap_guard_max_backtracks)
    if run_args.flow_hard_overlap_guard_last_steps is not None:
        flow_cfg["hard_overlap_guard_last_steps"] = int(run_args.flow_hard_overlap_guard_last_steps)
    model_args.flow = flow_cfg

    setattr(
        model_args,
        "flow_self_condition_infer",
        bool(flow_cfg.get("self_condition", False)),
    )
    setattr(
        model_args,
        "flow_final_refine",
        bool(flow_cfg.get("final_refine", False)),
    )
    setattr(
        model_args,
        "flow_final_refine_scale",
        float(flow_cfg.get("final_refine_scale", 0.35) or 0.35),
    )
    setattr(
        model_args,
        "flow_final_refine_tr_scale",
        float(flow_cfg.get("final_refine_tr_scale", 0.0) or 0.0),
    )
    setattr(
        model_args,
        "flow_final_refine_rot_scale",
        float(flow_cfg.get("final_refine_rot_scale", 0.35) or 0.35),
    )
    setattr(
        model_args,
        "flow_final_refine_tor_scale",
        float(flow_cfg.get("final_refine_tor_scale", 0.35) or 0.35),
    )
    setattr(
        model_args,
        "flow_steric_guidance",
        bool(flow_cfg.get("steric_guidance", False)),
    )
    setattr(
        model_args,
        "flow_steric_guidance_scale",
        float(flow_cfg.get("steric_guidance_scale", 0.15) or 0.0),
    )
    setattr(
        model_args,
        "flow_steric_guidance_cutoff",
        float(flow_cfg.get("steric_guidance_cutoff", 3.6) or 3.6),
    )
    setattr(
        model_args,
        "flow_steric_guidance_temperature",
        float(flow_cfg.get("steric_guidance_temperature", 0.35) or 0.35),
    )
    setattr(
        model_args,
        "flow_steric_guidance_torque_scale",
        float(flow_cfg.get("steric_guidance_torque_scale", 0.35) or 0.0),
    )
    setattr(
        model_args,
        "flow_steric_guidance_max_tr",
        float(flow_cfg.get("steric_guidance_max_tr", 2.0) or 0.0),
    )
    setattr(
        model_args,
        "flow_steric_guidance_max_rot",
        float(flow_cfg.get("steric_guidance_max_rot", 0.5) or 0.0),
    )
    setattr(
        model_args,
        "flow_hard_overlap_guard",
        bool(flow_cfg.get("hard_overlap_guard", False)),
    )
    setattr(
        model_args,
        "flow_hard_overlap_guard_min_dist",
        float(flow_cfg.get("hard_overlap_guard_min_dist", 1.6) or 0.0),
    )
    setattr(
        model_args,
        "flow_hard_overlap_guard_backoff",
        float(flow_cfg.get("hard_overlap_guard_backoff", 0.5) or 0.5),
    )
    setattr(
        model_args,
        "flow_hard_overlap_guard_max_backtracks",
        int(flow_cfg.get("hard_overlap_guard_max_backtracks", 4) or 0),
    )
    setattr(
        model_args,
        "flow_hard_overlap_guard_last_steps",
        int(flow_cfg.get("hard_overlap_guard_last_steps", 0) or 0),
    )
    
def process_complex(model, confidence_model, score_model_args, args, original_complex_graph, write_dir):
    return process_complex_batch(
        model,
        confidence_model,
        score_model_args,
        args,
        [original_complex_graph],
        [write_dir],
        [getattr(original_complex_graph, "name", "unknown")],
    )


def process_complex_batch(
    model,
    confidence_model,
    score_model_args,
    args,
    original_complex_graphs,
    write_dirs,
    complex_names,
):
    num_samples = args.N
    t_batch_start = time.perf_counter()
    batch_timing = _empty_batch_timing_stats()
    setattr(args, "_last_batch_timing", batch_timing)
    data_list = []
    counts = []
    t_build_start = time.perf_counter()
    for graph in original_complex_graphs:
        graph_list = _build_data_list(graph, num_samples)
        counts.append(len(graph_list))
        data_list.extend(graph_list)
    t_build = time.perf_counter() - t_build_start
    batch_timing["build_seconds"] = float(t_build)

    t_init_start = time.perf_counter()
    _apply_inference_init_noise(data_list, score_model_args)
    t_init = time.perf_counter() - t_init_start
    batch_timing["init_seconds"] = float(t_init)

    visualization_list = None
    if args.save_visualisation:
        visualization_list = [
            np.asarray(
                [
                    g["pep_a"].pos.cpu().numpy()
                    + g.original_center.cpu().numpy()
                    for g in data_list
                ]
            )
        ]

    (flow_num_steps, flow_solver) = _resolve_sampling_overrides(args, score_model_args)

    try:
        sample_batch_size = max(1, int(getattr(args, "batch_size", 1)))
        sample_batch_size = min(sample_batch_size, len(data_list))
        setattr(score_model_args, "_last_sampling_timing", None)
        t_sample_start = time.perf_counter()
        data_list, confidence, visualization_list = sampling(
            data_list=data_list,
            model=model,
            args=score_model_args,
            batch_size=sample_batch_size,
            visualization_list=visualization_list,
            confidence_model=confidence_model,
            flow_num_steps=flow_num_steps,
            flow_solver=flow_solver,
        )
        t_sample = time.perf_counter() - t_sample_start
        batch_timing["sample_seconds"] = float(t_sample)
        sampling_timing = getattr(score_model_args, "_last_sampling_timing", None) or {}
        batch_timing["sampling_forward_seconds"] = float(
            sampling_timing.get("forward_seconds", 0.0) or 0.0
        )
        batch_timing["sampling_update_step_seconds"] = float(
            sampling_timing.get("update_seconds", 0.0) or 0.0
        )
        batch_timing["sampling_final_refine_seconds"] = float(
            sampling_timing.get("final_refine_seconds", 0.0) or 0.0
        )
        batch_timing["sampling_update_seconds"] = (
            batch_timing["sampling_update_step_seconds"]
            + batch_timing["sampling_final_refine_seconds"]
        )
    except Exception as e:
        fail_log = os.path.join(args.output_dir, "failures_traceback.log")
        for name in complex_names:
            print("Failed on", name, e)
            try:
                with open(fail_log, "a") as f:
                    f.write(f"{name}: {e}\n")
                    f.write(traceback.format_exc())
                    f.write("\n")
            except Exception:
                pass
        batch_timing["batch_total_seconds"] = float(time.perf_counter() - t_batch_start)
        setattr(args, "_last_batch_timing", batch_timing)
        return len(complex_names)

    t_pred_start = time.perf_counter()
    predict_pos = [
        complex_graph["pep_a"].pos.cpu().numpy()
        + complex_graph.original_center.cpu().numpy()
        for complex_graph in data_list
    ]
    t_pred = time.perf_counter() - t_pred_start
    batch_timing["predict_seconds"] = float(t_pred)

    if visualization_list is not None:
        t_vis_start = time.perf_counter()
        visualization_list = list(
            np.transpose(np.array(visualization_list), (1, 0, 2, 3))
        )
        t_vis = time.perf_counter() - t_vis_start
    else:
        t_vis = 0.0
    batch_timing["visualize_seconds"] = float(t_vis)

    failures = 0
    offset = 0
    t_save_total = 0.0
    for graph, write_dir, name, count in zip(
        original_complex_graphs, write_dirs, complex_names, counts
    ):
        start = offset
        end = offset + count
        offset = end
        try:
            t_save_start = time.perf_counter()
            predict_slice = predict_pos[start:end]
            confidence_slice = None
            if confidence is not None:
                confidence_slice = confidence[start:end]
            re_order = save_predictions(
                write_dir, predict_slice, graph, args, confidence_slice
            )
            if args.save_visualisation:
                vis_slice = visualization_list[start:end]
                _save_visualization(write_dir, vis_slice, graph, args, re_order)
            t_save_total += time.perf_counter() - t_save_start
        except Exception as e:
            print("Failed on", name, e)
            try:
                fail_log = os.path.join(args.output_dir, "failures_traceback.log")
                with open(fail_log, "a") as f:
                    f.write(f"{name}: {e}\n")
                    f.write(traceback.format_exc())
                    f.write("\n")
            except Exception:
                pass
            failures += 1
    batch_timing["save_seconds"] = float(t_save_total)
    t_batch = time.perf_counter() - t_batch_start
    batch_timing["batch_total_seconds"] = float(t_batch)
    setattr(args, "_last_batch_timing", batch_timing)
    print(
        "[timing] batch_summary "
        f"build={_format_seconds(t_build)} "
        f"init={_format_seconds(t_init)} "
        f"sample={_format_seconds(t_sample)} "
        f"predict={_format_seconds(t_pred)} "
        f"visualize={_format_seconds(t_vis)} "
        f"save={_format_seconds(t_save_total)} "
        f"total={_format_seconds(t_batch)}"
    )
    return failures

def main(args):
    # Input parameters by config file
    load_config(args)
    _validate_inference_args(args)
    if getattr(args, "shard_index", None) is None:
        launched = _launch_sharded_inference(args)
        if launched:
            return
    os.makedirs(args.output_dir, exist_ok=True)
    device = _resolve_device(args)

    model_cfg_path = Path(args.model_dir) / "model_parameters.yml"
    if not model_cfg_path.is_file():
        fallback_cfg = Path("train_models/CGTensorProductEquivariantModel/model_parameters.yml")
        if not fallback_cfg.is_file():
            raise FileNotFoundError(
                f"找不到模型配置：{model_cfg_path}；也找不到默认配置：{fallback_cfg}"
            )
        if device.type == "cuda":
            print(f"[warn] {model_cfg_path} 不存在，回退使用默认配置：{fallback_cfg}")
        with fallback_cfg.open() as f:
            score_model_args = Namespace(**yaml.full_load(f))
    else:
        with model_cfg_path.open() as f:
            score_model_args = Namespace(**yaml.full_load(f))

    ckpt_path = None
    if args.ckpt not in (None, "", "none", "None"):
        ckpt_path = os.path.join(args.model_dir, args.ckpt)

    ckpt_cfg = _load_ckpt_config(ckpt_path)
    if ckpt_cfg:
        _merge_cfg_into_namespace(score_model_args, ckpt_cfg, allow_keys=None)
        print("[info] merged ckpt config into inference args (all keys; CLI args still take precedence)")

    _enable_esm_from_ckpt(score_model_args, ckpt_path)
    use_amp = bool(getattr(args, "amp", True)) and device.type == "cuda"
    setattr(score_model_args, "inference_amp", use_amp)
    setattr(score_model_args, "inference_mode", True)
    setattr(score_model_args, "inference_timing", bool(getattr(args, "timing", False)))
    setattr(score_model_args, "inference_device", device)
    estimated_graphs = max(1, int(getattr(args, "batch_size", 1) or 1)) * max(
        1, int(getattr(args, "N", 1) or 1)
    )
    auto_fastpath = device.type == "cuda" and estimated_graphs <= 256
    if getattr(args, "gpu_update_fastpath", None) is None:
        setattr(args, "gpu_update_fastpath", auto_fastpath)
    if getattr(args, "torsion_device", None) is None:
        setattr(args, "torsion_device", "gpu" if auto_fastpath else "cpu")
    setattr(score_model_args, "gpu_update_fastpath", bool(getattr(args, "gpu_update_fastpath", False)))
    setattr(score_model_args, "torsion_device", getattr(args, "torsion_device", "cpu"))
    setattr(score_model_args, "torsion_debug", bool(getattr(args, "torsion_debug", False)))
    _apply_flow_infer_overrides(args, score_model_args)
    print(
        "[flow-infer] "
        f"self_cond={getattr(score_model_args, 'flow_self_condition_infer', False)} "
        f"sparse_interface={getattr(score_model_args, 'flow', {}).get('sparse_interface', False)} "
        f"sparse_topk={getattr(score_model_args, 'flow', {}).get('sparse_interface_topk', 0)} "
        f"final_refine={getattr(score_model_args, 'flow_final_refine', False)}"
    )

    _resolve_peptide_source(args, ckpt_cfg, score_model_args)
    cached_ckpt_payload = _load_ckpt_payload(ckpt_path) if ckpt_path else None
    t_data_start = time.perf_counter()
    inference_dataset = prepare_data(args, score_model_args)
    t_data = time.perf_counter() - t_data_start
    # batch_size 只控制每次进 GPU 的 pdbid 数
    args.batch_size = max(1, int(getattr(args, "batch_size", 1)))
    cpu_budget = max(1, int(getattr(args, "cpu", 1) or 1))
    loader_workers = getattr(args, "loader_workers", None)
    if loader_workers is None:
        auto_workers = min(max(cpu_budget - 1, 0), args.batch_size, 4)
        if getattr(args, "protein_peptide_csv", None) is None:
            auto_workers = min(auto_workers, 2)
        loader_workers = auto_workers
    else:
        loader_workers = max(0, int(loader_workers))
    loader_kwargs = {
        "dataset": inference_dataset,
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": loader_workers,
        "pin_memory": device.type == "cuda",
    }
    if loader_workers > 0:
        loader_kwargs["persistent_workers"] = bool(
            getattr(args, "loader_persistent_workers", True)
        )
        prefetch_factor = max(1, int(getattr(args, "loader_prefetch_factor", 2) or 2))
        loader_kwargs["prefetch_factor"] = prefetch_factor
    eager_graph_loading = bool(getattr(args, "eager_graph_loading", False))
    setattr(args, "eager_graph_loading", eager_graph_loading)
    loader_iter = None
    if not eager_graph_loading:
        inference_loader = DataListLoader(**loader_kwargs)
        inference_batches = inference_loader
        # 让 worker 预取与模型加载重叠，减少首批 fetch 空转。
        loader_iter = iter(inference_loader)
    else:
        inference_loader = None
        inference_batches = None
        # eager 仍保留实验态；保持与 DataLoader 一致的 base_seed 消耗时机。
        _consume_dataloader_base_seed()
    print(
        "[loader] "
        f"mode={'eager' if eager_graph_loading else 'dataloader'} "
        f"batch_size={args.batch_size} "
        f"workers={loader_workers} "
        f"pin_memory={loader_kwargs['pin_memory']} "
        f"persistent={loader_kwargs.get('persistent_workers', False)} "
        f"prefetch={loader_kwargs.get('prefetch_factor', 0)}"
    )

    t_model_start = time.perf_counter()
    model = load_model(
        score_model_args,
        ckpt_path,
        device,
        state_dict=cached_ckpt_payload,
    )
    t_model = time.perf_counter() - t_model_start

    # load confidence model
    confidence_model = None
    confidence_args = None
    t_conf = 0.0
    if args.scoring_function == "confidence":
        t_conf_start = time.perf_counter()
        with open(f"{args.confidence_model_dir}/model_parameters.yml") as f:
            confidence_args = Namespace(**yaml.full_load(f))

        confidence_model = get_model(
            confidence_args, no_parallel=True, confidence_mode=True
        )
        state_dict = torch.load(
            f"{args.confidence_model_dir}/{args.confidence_ckpt}",
            map_location=torch.device("cpu"),
        )
        confidence_model.load_state_dict(state_dict["model"], strict=True)
        confidence_model = confidence_model.to(device)
        confidence_model.eval()
        t_conf = time.perf_counter() - t_conf_start

    failures, skipped = 0, 0
    print("Size of test dataset: ", len(inference_dataset))

    global_idx = 0
    t_loop_start = time.perf_counter()
    t_fetch_total = 0.0
    t_process_total = 0.0
    timing_totals = _empty_batch_timing_stats()
    if eager_graph_loading:
        eager_fetch_start = time.perf_counter()
        inference_batches = _build_eager_inference_batches(
            inference_dataset,
            args.batch_size,
        )
        eager_fetch_total = time.perf_counter() - eager_fetch_start
        t_fetch_total += eager_fetch_total
        print(
            "[loader] "
            f"eager_materialize={_format_seconds(eager_fetch_total)} "
            f"batches={len(inference_batches)}"
        )
    if loader_iter is None:
        loader_iter = iter(inference_batches)
    pbar = tqdm(total=len(inference_batches))
    while True:
        t_fetch_start = time.perf_counter()
        try:
            batch = next(loader_iter)
        except StopIteration:
            break
        t_fetch = 0.0 if eager_graph_loading else (time.perf_counter() - t_fetch_start)
        t_fetch_total += t_fetch
        t_process_start = time.perf_counter()
        # DataListLoader 返回 list
        batch_graphs = []
        batch_write_dirs = []
        batch_names = []
        for original_complex_graph in batch:
            if not original_complex_graph.success:
                skipped += 1
                try:
                    cname = inference_dataset.complex_names[global_idx]
                except Exception:
                    cname = getattr(original_complex_graph, "name", "unknown")
                try:
                    pep_desc = inference_dataset.peptide_descriptions[global_idx]
                    prot_desc = inference_dataset.protein_descriptions[global_idx]
                except Exception:
                    pep_desc = "unknown"
                    prot_desc = "unknown"
                print(
                    f"HAPPENING | The inference dataset did not contain {cname} for {pep_desc} and {prot_desc}. We are skipping this complex."
                )
                global_idx += 1
                continue
            try:
                cname = inference_dataset.complex_names[global_idx]
            except Exception:
                cname = getattr(original_complex_graph, "name", "unknown")
            write_dir = f"{args.output_dir}/{cname}"
            if bool(getattr(args, "prealign_to_native_center", True)):
                try:
                    receptor_pdb = inference_dataset.protein_descriptions[global_idx]
                except Exception:
                    receptor_pdb = ""
                aligned = _prealign_peptide_to_native_center(original_complex_graph, receptor_pdb)
                if not aligned:
                    print(
                        f"[prealign] skip {cname}: receptor_dir/peptide.pdb missing/invalid "
                        f"(receptor_pdb={receptor_pdb})"
                    )
            batch_graphs.append(original_complex_graph)
            batch_write_dirs.append(write_dir)
            batch_names.append(cname)
            global_idx += 1

        if batch_graphs:
            failures += process_complex_batch(
                model,
                confidence_model,
                score_model_args,
                args,
                batch_graphs,
                batch_write_dirs,
                batch_names,
            )
            timing_totals = _accumulate_timing_stats(
                timing_totals, getattr(args, "_last_batch_timing", None)
            )
        t_process = time.perf_counter() - t_process_start
        t_process_total += t_process
        print(
            "[timing] batch_loop "
            f"fetch={_format_seconds(t_fetch)} "
            f"process={_format_seconds(t_process)}"
        )
        pbar.update(1)
    pbar.close()
    t_loop = time.perf_counter() - t_loop_start

    print(f"Failed for {failures} complexes")
    print(f"Skipped {skipped} complexes")
    print(f"Results are in {args.output_dir}")
    print(
        "[timing] run_summary "
        f"data={_format_seconds(t_data)} "
        f"model={_format_seconds(t_model)} "
        f"confidence={_format_seconds(t_conf)} "
        f"loop_fetch={_format_seconds(t_fetch_total)} "
        f"loop_process={_format_seconds(t_process_total)} "
        f"loop_total={_format_seconds(t_loop)}"
    )
    timing_output_payload = _build_timing_output_payload(
        args,
        {
            "data_seconds": t_data,
            "model_seconds": t_model,
            "confidence_seconds": t_conf,
            "loop_fetch_seconds": t_fetch_total,
            "loop_process_seconds": t_process_total,
            "loop_wall_clock_observed_seconds": t_loop,
            "batch_timing": timing_totals,
            "total_complexes": len(inference_dataset),
            "failed_complexes": failures,
            "skipped_complexes": skipped,
        },
    )
    _write_timing_output_json(getattr(args, "timing_output", None), timing_output_payload)
    _run_auto_eval_if_enabled(args)


if __name__ == "__main__":
    _args = get_parser().parse_args()
    main(_args)
