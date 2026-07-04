from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
from torch import nn

from ..models import StatsMLPRanker


DEPRECATED_STATE_DICT_PREFIXES = ("bad_head.",)


def parse_gpu_ids(gpu_ids: str | None) -> List[int]:
    if gpu_ids is None:
        return []
    items = [item.strip() for item in gpu_ids.split(",") if item.strip()]
    return [int(item) for item in items]


def resolve_device_and_gpu_ids(device: str, gpu_ids: str | None) -> Tuple[torch.device, List[int]]:
    parsed_gpu_ids = parse_gpu_ids(gpu_ids)
    if device == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu"), []
    if parsed_gpu_ids:
        return torch.device(f"cuda:{parsed_gpu_ids[0]}"), parsed_gpu_ids
    if device.startswith("cuda:"):
        return torch.device(device), [int(device.split(":", 1)[1])]
    if device == "cuda":
        return torch.device("cuda:0"), [0]
    return torch.device(device), parsed_gpu_ids


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def is_stats_model(model: nn.Module) -> bool:
    return isinstance(unwrap_model(model), StatsMLPRanker)


def maybe_wrap_data_parallel(model: nn.Module, gpu_ids: Sequence[int]) -> nn.Module:
    if len(gpu_ids) <= 1:
        return model
    if not is_stats_model(model):
        # Graph inputs are not batch-first tensors, so vanilla DataParallel will
        # split edge_index along the wrong dimension and corrupt graph structure.
        return model
    return nn.DataParallel(model, device_ids=list(gpu_ids), output_device=int(gpu_ids[0]))


def strip_deprecated_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key: value
        for key, value in state_dict.items()
        if not key.startswith(DEPRECATED_STATE_DICT_PREFIXES)
    }


def load_model_state_dict_allowing_deprecated_heads(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    cleaned_state_dict = strip_deprecated_state_dict_keys(state_dict)
    unwrap_model(model).load_state_dict(cleaned_state_dict, strict=True)
