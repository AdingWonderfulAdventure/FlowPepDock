#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
notes 同步守卫（防止“改了代码但忘了更新 notes”）

用途
----
当你修改了仓库代码/脚本/配置，但没有同步更新 `notes/` 时，本脚本直接报错退出。
它不会自动改文档，只负责“发现你没写 notes”并把你拦住。

典型用法
--------
1) 检查工作区（含未暂存+已暂存）
   python scripts/ops/note_guard.py

2) 只检查已暂存（准备 commit 前）
   python scripts/ops/note_guard.py --staged

返回码
------
0: 通过
1: 检测到“代码改动但 notes 未改”
2: 非 git 仓库或 git 不可用（当前实现已降级为警告并返回 0）
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]

# “触发 notes 必须同步”的文件类型（粗暴但实用）
TRACK_EXTS = {
    ".py",
    ".sh",
    ".bash",
    ".yaml",
    ".yml",
    ".toml",
    ".json",
    ".md",
}

# 明确不纳入 guard 的大体积/产物目录（避免误报）
IGNORE_PREFIXES = (
    "data/",
    "logs/",
    "results/",
    "outputs/",
    "swanlog/",
    "__pycache__/",
)


@dataclass(frozen=True)
class GitChanges:
    changed_paths: List[str]


def _run(cmd: Sequence[str]) -> str:
    return subprocess.check_output(cmd, cwd=str(REPO_ROOT), text=True, stderr=subprocess.STDOUT)


def _is_git_repo() -> bool:
    try:
        return _run(["git", "rev-parse", "--is-inside-work-tree"]).strip() == "true"
    except Exception:
        return False


def _list_changed_paths(staged_only: bool) -> GitChanges:
    if staged_only:
        out = _run(["git", "diff", "--name-only", "--cached"])
        paths = [ln.strip().replace("\\", "/") for ln in out.splitlines() if ln.strip()]
        return GitChanges(changed_paths=paths)

    out = _run(["git", "status", "--porcelain"])
    paths: List[str] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        tail = line[3:]
        if " -> " in tail:
            tail = tail.split(" -> ", 1)[1]
        paths.append(tail.strip().replace("\\", "/"))
    return GitChanges(changed_paths=paths)


def _is_ignored(path: str) -> bool:
    return any(path.startswith(prefix) for prefix in IGNORE_PREFIXES)


def _is_notes(path: str) -> bool:
    return path.startswith("notes/")


def _is_tracked_code(path: str) -> bool:
    if _is_ignored(path) or _is_notes(path):
        return False
    return Path(path).suffix.lower() in TRACK_EXTS


def _filter(paths: Iterable[str], pred) -> List[str]:
    return [p for p in paths if pred(p)]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="notes 同步守卫：代码改了就必须同步 notes/。")
    parser.add_argument("--staged", action="store_true", help="只检查已暂存改动（git diff --cached）")
    args = parser.parse_args(argv)

    if not _is_git_repo():
        print("[note_guard] WARN: 当前目录不是 git 仓库，跳过基于 git 的改动检查。", file=sys.stderr)
        return 0

    changes = _list_changed_paths(staged_only=bool(args.staged))
    changed = changes.changed_paths
    if not changed:
        print("[note_guard] OK: 没有检测到改动。")
        return 0

    notes_changed = _filter(changed, _is_notes)
    code_changed = _filter(changed, _is_tracked_code)

    if code_changed and not notes_changed:
        print("[note_guard] ERROR: 检测到代码/配置/文档改动，但 notes/ 没有同步更新。", file=sys.stderr)
        print("  最小同步集合：notes/STATE.md + notes/ISSUES.md（必要时再改 CODEMAP/ARCHIVE）", file=sys.stderr)
        print("  触发改动文件：", file=sys.stderr)
        for p in code_changed[:50]:
            print(f"  - {p}", file=sys.stderr)
        if len(code_changed) > 50:
            print(f"  ... 还有 {len(code_changed) - 50} 个", file=sys.stderr)
        return 1

    print("[note_guard] OK")
    if code_changed:
        print(f"  code_changed={len(code_changed)} notes_changed={len(notes_changed)}")
    else:
        print(f"  notes_only_changed={len(notes_changed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
