#!/usr/bin/env python3
"""Plot all-pose rank curves from FlowPepDock/PoseCred score tables.

This script reads a pose-level CSV/TSV table and writes editable vector figures
for Illustrator, plus a high-resolution PNG preview when requested.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.ticker import MaxNLocator


RANK_CANDIDATES = (
    "rank_in_peptide",
    "rank_in_group",
    "pose_rank",
    "rank",
    "f_pose_rank",
)
SCORE_CANDIDATES = (
    "score",
    "final_score",
    "pred_score",
    "ipg_score",
    "top_ranked_score",
)
GROUP_CANDIDATES = (
    "case",
    "peptide_id",
    "group_id",
    "complex_name",
)
POSE_CANDIDATES = (
    "pose_name",
    "pose_id",
    "pose",
    "copied_pdb_path",
    "source_pose_path",
)
PALETTE = (
    "#D55E00",
    "#0072B2",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#000000",
    "#7F7F7F",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw all-pose rank-vs-score curves as PDF/SVG vector figures.",
    )
    parser.add_argument("--input", required=True, type=Path, help="Pose-level CSV/TSV score table.")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="Output path prefix without extension. Defaults to <input>_pose_rank.",
    )
    parser.add_argument("--rank-col", default="auto", help="Rank column, or auto.")
    parser.add_argument("--score-col", default="auto", help="Score column, or auto.")
    parser.add_argument("--group-col", default="auto", help="Facet/group column, none, or auto.")
    parser.add_argument("--pose-col", default="auto", help="Pose label column, none, or auto.")
    parser.add_argument(
        "--groups",
        default="",
        help="Comma-separated group values to plot. Defaults to all groups up to --max-groups.",
    )
    parser.add_argument(
        "--highlight-col",
        default=None,
        help="Column used for highlighted points, e.g. receptor_id.",
    )
    parser.add_argument(
        "--highlight-values",
        default="",
        help="Comma-separated values in --highlight-col to emphasize.",
    )
    parser.add_argument(
        "--score-direction",
        choices=("desc", "asc"),
        default="desc",
        help="Used only when rank column is absent. desc means larger score ranks first.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Annotate top-k poses per group. Use 0 to disable annotations.",
    )
    parser.add_argument(
        "--max-groups",
        type=int,
        default=8,
        help="Safety cap for automatic faceting.",
    )
    parser.add_argument(
        "--rasterize-points-above",
        type=int,
        default=10000,
        help="Rasterize scatter points above this row count while keeping axes/text vector.",
    )
    parser.add_argument(
        "--formats",
        default="pdf,svg,png",
        help="Comma-separated output formats. Recommended: pdf,svg,png.",
    )
    parser.add_argument("--dpi", type=int, default=600, help="DPI for PNG and rasterized points.")
    parser.add_argument("--title", default="", help="Figure title.")
    parser.add_argument("--xlabel", default="Pose rank", help="X-axis label.")
    parser.add_argument("--ylabel", default="Ranking score", help="Y-axis label.")
    return parser.parse_args()


def split_csv(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def resolve_column(
    df: pd.DataFrame,
    requested: str | None,
    candidates: Sequence[str],
    role: str,
    required: bool = True,
) -> str | None:
    if requested is None or requested.lower() == "none":
        if required:
            raise ValueError(f"{role} column is required.")
        return None
    if requested != "auto":
        if requested not in df.columns:
            raise ValueError(f"{role} column {requested!r} not found. Available: {list(df.columns)}")
        return requested
    for column in candidates:
        if column in df.columns:
            return column
    if required:
        raise ValueError(f"Could not auto-detect {role} column. Available: {list(df.columns)}")
    return None


def selected_groups(df: pd.DataFrame, group_col: str | None, requested: Iterable[str], max_groups: int) -> list[str | None]:
    if group_col is None:
        return [None]

    groups = [str(value) for value in pd.unique(df[group_col].dropna())]
    wanted = list(requested)
    if wanted:
        missing = sorted(set(wanted) - set(groups))
        if missing:
            raise ValueError(f"Requested groups not found in {group_col!r}: {missing}")
        return wanted

    if len(groups) > max_groups:
        shown = ", ".join(groups[:max_groups])
        raise ValueError(
            f"{group_col!r} has {len(groups)} groups. Use --groups to choose specific values "
            f"or increase --max-groups. First groups: {shown}"
        )
    return groups


def prepare_ranked_table(
    df: pd.DataFrame,
    rank_col: str | None,
    score_col: str,
    group_col: str | None,
    score_direction: str,
) -> pd.DataFrame:
    out = df.copy()
    out[score_col] = pd.to_numeric(out[score_col], errors="coerce")
    out = out.dropna(subset=[score_col])

    if group_col is not None:
        out[group_col] = out[group_col].astype(str)

    if rank_col is None:
        sort_cols = [score_col]
        if group_col is not None:
            sort_cols = [group_col, score_col]
        ascending = [True, score_direction == "asc"] if group_col is not None else [score_direction == "asc"]
        out = out.sort_values(sort_cols, ascending=ascending, kind="mergesort").copy()
        if group_col is None:
            out["plot_rank"] = np.arange(1, len(out) + 1)
        else:
            out["plot_rank"] = out.groupby(group_col).cumcount() + 1
        return out

    out[rank_col] = pd.to_numeric(out[rank_col], errors="coerce")
    out = out.dropna(subset=[rank_col]).copy()
    out["plot_rank"] = out[rank_col].astype(float)
    return out


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )


def value_color_map(values: Sequence[str]) -> dict[str, str]:
    return {value: PALETTE[idx % len(PALETTE)] for idx, value in enumerate(values)}


def annotate_top_poses(
    ax: Axes,
    data: pd.DataFrame,
    score_col: str,
    pose_col: str | None,
    highlight_col: str | None,
    top_k: int,
) -> None:
    if top_k <= 0:
        return
    top = data.sort_values("plot_rank", ascending=True).head(top_k)
    y_span = max(float(data[score_col].max() - data[score_col].min()), 1e-9)
    for idx, row in enumerate(top.itertuples(index=False)):
        x = float(getattr(row, "plot_rank"))
        y = float(getattr(row, score_col))
        label_parts = []
        if highlight_col is not None and highlight_col in data.columns:
            label_parts.append(str(getattr(row, highlight_col)))
        if pose_col is not None and pose_col in data.columns:
            label_parts.append(Path(str(getattr(row, pose_col))).name)
        label = " / ".join(label_parts) if label_parts else f"rank {int(x)}"
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(4, 8 + idx % 3 * 7),
            textcoords="offset points",
            fontsize=7,
            ha="left",
            va="bottom",
            arrowprops={"arrowstyle": "-", "lw": 0.4, "color": "#444444"},
        )
        ax.set_ylim(top=max(ax.get_ylim()[1], y + 0.08 * y_span))


def plot_panel(
    ax: Axes,
    data: pd.DataFrame,
    group_label: str | None,
    score_col: str,
    pose_col: str | None,
    highlight_col: str | None,
    highlight_values: Sequence[str],
    top_k: int,
    rasterized: bool,
) -> list[tuple[str, str]]:
    data = data.sort_values("plot_rank", ascending=True)
    ax.plot(
        data["plot_rank"],
        data[score_col],
        color="#30343B",
        linewidth=0.9,
        alpha=0.8,
        zorder=1,
    )
    ax.scatter(
        data["plot_rank"],
        data[score_col],
        s=10,
        color="#AAB2BA",
        edgecolors="none",
        alpha=0.70,
        rasterized=rasterized,
        zorder=2,
    )

    legend_items: list[tuple[str, str]] = []
    if highlight_col is not None:
        if highlight_col not in data.columns:
            raise ValueError(f"Highlight column {highlight_col!r} not found. Available: {list(data.columns)}")
        data[highlight_col] = data[highlight_col].astype(str)
        values = list(highlight_values)
        if not values:
            values = [str(v) for v in pd.unique(data[highlight_col].dropna())]
            if len(values) > 12:
                raise ValueError(
                    f"--highlight-col {highlight_col!r} has {len(values)} values. "
                    "Use --highlight-values to avoid an unreadable legend."
                )
        colors = value_color_map(values)
        for value in values:
            mask = data[highlight_col] == value
            if not bool(mask.any()):
                continue
            color = colors[value]
            ax.scatter(
                data.loc[mask, "plot_rank"],
                data.loc[mask, score_col],
                s=20,
                color=color,
                edgecolors="white",
                linewidths=0.35,
                alpha=0.95,
                rasterized=rasterized,
                zorder=3,
            )
            legend_items.append((value, color))

    annotate_top_poses(ax, data, score_col, pose_col, highlight_col, top_k)
    ax.set_title(str(group_label) if group_label is not None else "All poses", fontsize=10, pad=8)
    ax.grid(axis="y", color="#D6DADF", linewidth=0.45, alpha=0.85)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.tick_params(labelsize=8, length=3, width=0.7)
    return legend_items


def build_figure(
    df: pd.DataFrame,
    groups: Sequence[str | None],
    group_col: str | None,
    score_col: str,
    pose_col: str | None,
    highlight_col: str | None,
    highlight_values: Sequence[str],
    top_k: int,
    title: str,
    xlabel: str,
    ylabel: str,
    rasterize_points: bool,
) -> plt.Figure:
    configure_matplotlib()
    n_panels = len(groups)
    ncols = 1 if n_panels <= 2 else 2
    nrows = math.ceil(n_panels / ncols)
    fig_width = 7.2
    fig_height = max(3.2, 2.65 * nrows + (0.35 if title else 0.0))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height), squeeze=False)

    all_legend_items: list[tuple[str, str]] = []
    for ax, group in zip(axes.ravel(), groups):
        if group_col is None:
            part = df
        else:
            part = df[df[group_col].astype(str) == str(group)]
        legend_items = plot_panel(
            ax,
            part,
            group,
            score_col,
            pose_col,
            highlight_col,
            highlight_values,
            top_k,
            rasterize_points,
        )
        all_legend_items.extend(legend_items)

    for ax in axes.ravel()[n_panels:]:
        ax.set_visible(False)

    unique_legend = list(dict(all_legend_items).items())
    if unique_legend:
        handles = [
            plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor="white", markersize=6)
            for label, color in unique_legend
        ]
        labels = [label for label, color in unique_legend]
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0 if not title else 0.94),
            ncol=min(len(labels), 4),
            frameon=False,
            fontsize=8,
        )

    if title:
        fig.suptitle(title, fontsize=12, y=0.995)
    fig.supxlabel(xlabel, fontsize=10)
    fig.supylabel(ylabel, fontsize=10)
    fig.tight_layout(rect=(0.035, 0.04, 0.995, 0.90 if unique_legend or title else 0.98))
    return fig


def output_paths(prefix: Path, formats: Sequence[str]) -> list[Path]:
    return [prefix.with_suffix(f".{fmt.lower().lstrip('.')}") for fmt in formats]


def main() -> None:
    args = parse_args()
    df = load_table(args.input)
    rank_requested = None if args.rank_col.lower() == "none" else args.rank_col
    group_requested = None if args.group_col.lower() == "none" else args.group_col
    pose_requested = None if args.pose_col.lower() == "none" else args.pose_col
    rank_col = resolve_column(df, rank_requested, RANK_CANDIDATES, "rank", required=False)
    score_col = resolve_column(df, args.score_col, SCORE_CANDIDATES, "score", required=True)
    group_col = resolve_column(df, group_requested, GROUP_CANDIDATES, "group", required=False)
    pose_col = resolve_column(df, pose_requested, POSE_CANDIDATES, "pose", required=False)

    ranked = prepare_ranked_table(df, rank_col, score_col, group_col, args.score_direction)
    groups = selected_groups(ranked, group_col, split_csv(args.groups), args.max_groups)
    highlight_values = split_csv(args.highlight_values)
    formats = split_csv(args.formats)
    if not formats:
        raise ValueError("No output formats requested.")

    prefix = args.output_prefix
    if prefix is None:
        prefix = args.input.with_name(f"{args.input.stem}_pose_rank")
    prefix.parent.mkdir(parents=True, exist_ok=True)

    fig = build_figure(
        ranked,
        groups,
        group_col,
        score_col,
        pose_col,
        args.highlight_col,
        highlight_values,
        args.top_k,
        args.title,
        args.xlabel,
        args.ylabel,
        rasterize_points=len(ranked) > args.rasterize_points_above,
    )

    written = []
    for path in output_paths(prefix, formats):
        save_kwargs = {"bbox_inches": "tight"}
        if path.suffix.lower() in {".png", ".tif", ".tiff", ".jpg", ".jpeg"}:
            save_kwargs["dpi"] = args.dpi
        fig.savefig(path, **save_kwargs)
        written.append(path)
    plt.close(fig)

    print(f"Input rows: {len(df)}")
    print(f"Plotted rows: {len(ranked)}")
    print(f"Rank column: {rank_col or 'computed from score'}")
    print(f"Score column: {score_col}")
    print(f"Group column: {group_col or 'none'}")
    print("Wrote:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
