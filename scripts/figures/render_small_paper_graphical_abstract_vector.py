#!/usr/bin/env python3
"""Render an Illustrator-editable vector graphical abstract.

This is a vector redrawing of the small-paper model workflow. It avoids embedded
raster images so Illustrator can edit boxes, arrows, nodes, and labels.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SVG = ROOT / "outputs/figures/flow_ipg_graphical_abstract_vector.svg"

W = 3600
H = 1250


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render vector graphical abstract SVG.")
    parser.add_argument("--output", type=Path, default=DEFAULT_SVG)
    return parser.parse_args()


class SVG:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.items: list[str] = []

    def add(self, text: str) -> None:
        self.items.append(text)

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str = "none",
        stroke: str = "#111827",
        sw: float = 3,
        rx: float = 0,
        opacity: float | None = None,
    ) -> None:
        op = f' opacity="{opacity}"' if opacity is not None else ""
        self.add(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
            f'rx="{rx:.2f}" fill="{fill}" stroke="{stroke}" stroke-width="{sw:.2f}"{op}/>'
        )

    def line(self, x1: float, y1: float, x2: float, y2: float, stroke: str = "#111827", sw: float = 3, dash: str | None = None) -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{stroke}" stroke-width="{sw:.2f}" stroke-linecap="round"{dash_attr}/>'
        )

    def circle(self, cx: float, cy: float, r: float, fill: str, stroke: str = "#111827", sw: float = 2) -> None:
        self.add(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw:.2f}"/>'
        )

    def ellipse(self, cx: float, cy: float, rx: float, ry: float, fill: str, stroke: str = "#111827", sw: float = 2, opacity: float | None = None) -> None:
        op = f' opacity="{opacity}"' if opacity is not None else ""
        self.add(
            f'<ellipse cx="{cx:.2f}" cy="{cy:.2f}" rx="{rx:.2f}" ry="{ry:.2f}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw:.2f}"{op}/>'
        )

    def path(self, d: str, fill: str = "none", stroke: str = "#111827", sw: float = 3, opacity: float | None = None) -> None:
        op = f' opacity="{opacity}"' if opacity is not None else ""
        self.add(
            f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw:.2f}" '
            f'stroke-linecap="round" stroke-linejoin="round"{op}/>'
        )

    def polyline(self, points: list[tuple[float, float]], fill: str = "none", stroke: str = "#111827", sw: float = 3) -> None:
        pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        self.add(f'<polyline points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="{sw:.2f}" stroke-linecap="round" stroke-linejoin="round"/>')

    def polygon(self, points: list[tuple[float, float]], fill: str, stroke: str = "#111827", sw: float = 2) -> None:
        pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        self.add(f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="{sw:.2f}" stroke-linejoin="round"/>')

    def text(self, x: float, y: float, text: str, size: float = 28, weight: str = "600", fill: str = "#111827", anchor: str = "middle") -> None:
        self.add(
            f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" font-family="Arial, Helvetica, sans-serif" '
            f'font-size="{size:.2f}" font-weight="{weight}" fill="{fill}">{escape(text)}</text>'
        )

    def arrow(self, x1: float, y1: float, x2: float, y2: float, stroke: str = "#64748B", sw: float = 8, head: float = 28) -> None:
        self.line(x1, y1, x2, y2, stroke=stroke, sw=sw)
        angle = math.atan2(y2 - y1, x2 - x1)
        left = (x2 - head * math.cos(angle - math.pi / 6), y2 - head * math.sin(angle - math.pi / 6))
        right = (x2 - head * math.cos(angle + math.pi / 6), y2 - head * math.sin(angle + math.pi / 6))
        self.polygon([(x2, y2), left, right], fill=stroke, stroke=stroke, sw=1)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        body = "\n  ".join(self.items)
        path.write_text(
            f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" viewBox="0 0 {self.width} {self.height}">
  <defs>
    <style>
      .small {{ font-family: Arial, Helvetica, sans-serif; font-size: 18px; font-weight: 600; fill: #111827; }}
      .tiny {{ font-family: Arial, Helvetica, sans-serif; font-size: 14px; font-weight: 500; fill: #374151; }}
    </style>
  </defs>
  {body}
</svg>
''',
            encoding="utf-8",
        )


def panel(svg: SVG, x: float, y: float, w: float, h: float, fill: str, stroke: str) -> None:
    svg.rect(x + 8, y + 10, w, h, fill="#E6ECF3", stroke="none", sw=0, rx=28, opacity=0.8)
    svg.rect(x, y, w, h, fill=fill, stroke=stroke, sw=4, rx=28)


def receptor(svg: SVG, cx: float, cy: float, scale: float = 1.0) -> None:
    pts = [
        (-105, -20), (-75, -92), (15, -110), (78, -72), (110, -10),
        (78, 72), (5, 105), (-82, 70), (-118, 18),
    ]
    pts = [(cx + x * scale, cy + y * scale) for x, y in pts]
    svg.polygon(pts, fill="#BFE0EF", stroke="#4A90B7", sw=4)
    for i in range(4):
        y = cy - 55 * scale + i * 34 * scale
        svg.path(
            f"M {cx-75*scale:.1f} {y:.1f} C {cx-20*scale:.1f} {y-38*scale:.1f}, {cx+25*scale:.1f} {y+38*scale:.1f}, {cx+82*scale:.1f} {y:.1f}",
            stroke="#FFFFFF",
            sw=5 * scale,
            opacity=0.85,
        )


def peptide_surface(svg: SVG, cx: float, cy: float, scale: float = 1.0) -> None:
    pts = [
        (-115, 0), (-84, -42), (-38, -32), (-5, -62), (42, -44),
        (92, -20), (118, 25), (72, 56), (25, 42), (-20, 68), (-78, 45),
    ]
    pts = [(cx + x * scale, cy + y * scale) for x, y in pts]
    svg.polygon(pts, fill="#F3C99B", stroke="#D58B43", sw=3)
    peptide_chain(svg, cx, cy, scale=scale * 0.9, stroke="#C76A1A", nodes=False)


def peptide_chain(svg: SVG, cx: float, cy: float, scale: float = 1.0, stroke: str = "#E87945", nodes: bool = True, rot: float = 0.0) -> None:
    raw = [(-82, 15), (-55, -20), (-20, -5), (12, -34), (45, -12), (78, 20), (102, -5)]
    cr = math.cos(rot)
    sr = math.sin(rot)
    pts = []
    for x, y in raw:
        sx, sy = x * scale, y * scale
        pts.append((cx + sx * cr - sy * sr, cy + sx * sr + sy * cr))
    svg.polyline(pts, stroke=stroke, sw=7 * scale)
    if nodes:
        for i, (x, y) in enumerate(pts):
            if i % 2 == 0:
                svg.circle(x, y, 9 * scale, fill="#F29970", stroke="#A94724", sw=2)


def graph_icon(svg: SVG, x: float, y: float, kind: str, scale: float = 1.0) -> None:
    if kind == "receptor":
        pts = [(x, y), (x + 35 * scale, y - 28 * scale), (x + 68 * scale, y + 18 * scale), (x + 92 * scale, y - 48 * scale), (x + 126 * scale, y - 5 * scale)]
        svg.polyline(pts, stroke="#555", sw=3 * scale)
        for px, py in pts:
            svg.circle(px, py, 10 * scale, fill="#2D86C4", stroke="#174A6A", sw=2)
    elif kind == "cross":
        left = [(x, y), (x + 25 * scale, y + 45 * scale), (x + 45 * scale, y + 92 * scale), (x + 70 * scale, y + 135 * scale)]
        right = [(x + 105 * scale, y - 5 * scale), (x + 130 * scale, y + 42 * scale), (x + 115 * scale, y + 92 * scale), (x + 150 * scale, y + 135 * scale)]
        svg.polyline(left, stroke="#555", sw=3 * scale)
        svg.polyline(right, stroke="#555", sw=3 * scale)
        for a in left:
            for b in right:
                if abs(a[1] - b[1]) < 70 * scale:
                    svg.line(a[0], a[1], b[0], b[1], stroke="#39A96B", sw=2 * scale)
        for px, py in left:
            svg.circle(px, py, 9 * scale, fill="#2D86C4", stroke="#174A6A", sw=2)
        for px, py in right:
            svg.circle(px, py, 9 * scale, fill="#E86B23", stroke="#8F3B13", sw=2)
    elif kind == "residue":
        pts = [(x, y), (x + 22 * scale, y + 35 * scale), (x + 36 * scale, y + 70 * scale), (x + 58 * scale, y + 105 * scale)]
        svg.polyline(pts, stroke="#555", sw=3 * scale)
        for px, py in pts:
            svg.circle(px, py, 9 * scale, fill="#E86B23", stroke="#8F3B13", sw=2)
    else:
        pts = [(x, y), (x + 35 * scale, y + 42 * scale), (x + 88 * scale, y + 22 * scale), (x + 125 * scale, y + 65 * scale), (x + 72 * scale, y + 100 * scale), (x + 18 * scale, y + 85 * scale)]
        for i in range(len(pts)):
            svg.line(pts[i][0], pts[i][1], pts[(i + 1) % len(pts)][0], pts[(i + 1) % len(pts)][1], stroke="#555", sw=3 * scale)
        colors = ["#E86B23", "#38A6D9", "#E86B23", "#38A6D9", "#D11F2F", "#E86B23"]
        for (px, py), c in zip(pts, colors):
            svg.circle(px, py, 8 * scale, fill=c, stroke="#333", sw=1.5)


def cgtpel_stack(svg: SVG, x: float, y: float, w: float, h: float) -> None:
    svg.rect(x, y, w, h, fill="#E6EFF8", stroke="#222", sw=4, rx=26)
    rows = [
        (y + 35, "#CFE1F2", "#5F96BF", "Receptor Stream"),
        (y + 150, "#CFE5D0", "#77B88A", "Bidirectional Cross-Graph Interaction"),
        (y + 265, "#F9E7DD", "#E5985D", "Peptide Stream"),
    ]
    for yy, fill, accent, label in rows:
        svg.rect(x + 45, yy, w - 90, 78, fill=fill, stroke="#222", sw=4, rx=18)
        svg.rect(x + 175, yy + 18, 140, 42, fill=accent, stroke="#222", sw=3, rx=18)
        svg.text(x + w / 2, yy + 51, label, size=22 if "Bidirectional" not in label else 20, weight="700")
    svg.arrow(x + w / 2, y + 115, x + w / 2, y + 145, stroke="#222", sw=4, head=14)
    svg.arrow(x + w / 2, y + 230, x + w / 2, y + 260, stroke="#222", sw=4, head=14)


def flow_heads(svg: SVG, x: float, y: float, w: float, h: float) -> None:
    svg.rect(x, y, w, h, fill="#EEF7F3", stroke="#222", sw=4, rx=24)
    svg.text(x + 185, y + 36, "Atom-Level Torsion Branch", size=22, weight="700")
    svg.text(x + 465, y + 36, "Rigid-Body Flow Head", size=22, weight="700")
    for i, c in enumerate(["#F28E2B", "#F3C46B", "#F7D79D", "#56B4E9"]):
        svg.rect(x + 38, y + 70 + i * 26, 32, 25, fill=c, stroke="#222", sw=2)
    graph_icon(svg, x + 155, y + 95, "atom", scale=0.55)
    svg.arrow(x + 78, y + 122, x + 145, y + 122, stroke="#6B7280", sw=4, head=15)
    svg.arrow(x + 300, y + 122, x + 365, y + 122, stroke="#6B7280", sw=4, head=15)
    svg.polygon([(x + 395, y + 90), (x + 465, y + 122), (x + 395, y + 154)], fill="#F7D79D", stroke="#222", sw=3)
    svg.rect(x + 95, y + 220, 165, 30, fill="#F3C46B", stroke="#222", sw=3)
    cx, cy = x + 365, y + 235
    svg.rect(cx - 45, cy - 62, 90, 124, fill="#FFFFFF", stroke="#222", sw=3, rx=18)
    for px, py, c in [(cx, cy - 34, "#56B4E9"), (cx - 28, cy + 24, "#56B4E9"), (cx + 28, cy + 24, "#F28E2B")]:
        svg.circle(px, py, 10, fill=c, stroke="#222", sw=2)
    svg.polyline([(cx, cy - 34), (cx - 28, cy + 24), (cx + 28, cy + 24), (cx, cy - 34)], stroke="#6B7280", sw=3)
    svg.polygon([(x + 490, y + 205), (x + 565, y + 235), (x + 490, y + 265)], fill="#F7D79D", stroke="#222", sw=3)


def candidate_bank(svg: SVG, x: float, y: float, w: float, h: float) -> None:
    panel(svg, x, y, w, h, "#F6FAFE", "#B8C6D6")
    card_w, card_h = 190, 130
    positions = [(x + 55, y + 65), (x + 285, y + 65), (x + 55, y + 245), (x + 285, y + 245), (x + 55, y + 425), (x + 285, y + 425)]
    for i, (px, py) in enumerate(positions):
        svg.rect(px, py, card_w, card_h, fill="#FFFFFF", stroke="#2F9E6D" if i == 5 else "#B8C6D6", sw=5 if i == 5 else 3, rx=16)
        peptide_chain(svg, px + card_w / 2, py + card_h / 2, scale=0.42, rot=[0, 0.25, -0.2, 0.35, -0.35, 0.1][i])
    for dx in [0, 34, 68]:
        svg.circle(x + 250 + dx, y + 590, 7, fill="#94A3B8", stroke="#94A3B8")


def ipg_stage(svg: SVG, x: float, y: float, w: float, h: float) -> None:
    panel(svg, x, y, w, h, "#F8F3FB", "#C8B7D8")
    svg.rect(x + 70, y + 65, 310, 335, fill="#F4F8FB", stroke="#B8C6D6", sw=3, rx=24)
    graph_icon(svg, x + 150, y + 150, "cross", scale=1.0)
    svg.text(x + 225, y + 110, "Interface Pair Graph", size=24, weight="700")
    svg.arrow(x + 395, y + 250, x + 460, y + 250, stroke="#8A9AAC", sw=6, head=20)
    svg.rect(x + 500, y + 60, 430, 355, fill="#F4EAF6", stroke="#222", sw=4, rx=28)
    svg.text(x + 715, y + 105, "GINE Layer", size=22, weight="700")
    svg.rect(x + 540, y + 125, 105, 235, fill="#F7CF99", stroke="#222", sw=4, rx=28)
    svg.text(x + 592, y + 250, "Edge\nMLP", size=20, weight="700")
    svg.rect(x + 710, y + 125, 135, 180, fill="#A7D6CF", stroke="#222", sw=4, rx=28)
    svg.text(x + 777, y + 225, "Aggregation", size=20, weight="700")
    svg.rect(x + 855, y + 240, 120, 60, fill="#F7CF99", stroke="#222", sw=3, rx=20)
    svg.text(x + 915, y + 277, "Update", size=18, weight="700")
    svg.arrow(x + 650, y + 245, x + 705, y + 245, stroke="#222", sw=4, head=14)
    svg.arrow(x + 845, y + 245, x + 855, y + 265, stroke="#222", sw=4, head=14)
    svg.arrow(x + w / 2, y + 445, x + w / 2, y + 520, stroke="#8A9AAC", sw=6, head=20)
    svg.rect(x + 70, y + 560, w - 140, 410, fill="#EAF6F3", stroke="#222", sw=4, rx=24)
    for i, label in enumerate(["Attention Pooling", "Mean Pooling", "Max Pooling"]):
        px = x + 120 + i * 260
        svg.rect(px, y + 625, 220, 130, fill="#FFFFFF", stroke="#222", sw=3, rx=18)
        for j, c in enumerate(["#E8EEF6", "#C5D8EA", "#8FB7D8", "#3D77A4"]):
            svg.rect(px + 55 + j * 25, y + 650, 22, 22, fill=c, stroke="#222", sw=1)
        svg.rect(px + 35, y + 690, 150, 45, fill="#B8D8F0", stroke="#222", sw=2, rx=16)
        svg.text(px + 110, y + 720, label, size=18, weight="700")
        svg.arrow(px + 110, y + 755, px + 110, y + 830, stroke="#222", sw=4, head=14)
    svg.rect(x + 120, y + 840, w - 240, 50, fill="#E5E5E5", stroke="#222", sw=3, rx=14)
    svg.text(x + w / 2, y + 874, "Concat", size=18, weight="500")
    for j, c in enumerate(["#E8EEF6", "#C5D8EA", "#8FB7D8", "#3D77A4"] * 5):
        svg.rect(x + 330 + j * 22, y + 930, 20, 20, fill=c, stroke="#222", sw=1)


def score_output(svg: SVG, x: float, y: float, w: float, h: float) -> None:
    panel(svg, x, y, w, h, "#FFF8EC", "#D7BE7A")
    svg.rect(x + 75, y + 65, w - 150, 170, fill="#E5E5E5", stroke="#222", sw=4, rx=22)
    svg.text(x + w / 2, y + 105, "Shared Readout", size=22, weight="700")
    svg.rect(x + 110, y + 130, 110, 60, fill="#F4F4F4", stroke="#222", sw=3, rx=16)
    svg.rect(x + 250, y + 130, 110, 60, fill="#F4F4F4", stroke="#222", sw=3, rx=16)
    svg.rect(x + 128, y + 158, 74, 22, fill="#F7CF99", stroke="#222", sw=2, rx=7)
    svg.rect(x + 268, y + 158, 74, 22, fill="#F7CF99", stroke="#222", sw=2, rx=7)
    svg.arrow(x + w / 2, y + 255, x + w / 2, y + 340, stroke="#A88740", sw=7, head=24)
    for i, angle in enumerate([0, -0.2, 0.2, 0.35]):
        cy = y + 360 + i * 145
        svg.rect(x + 80, cy, w - 160, 105, fill="#FFFFFF", stroke="#2F9E6D" if i == 0 else "#B8C6D6", sw=5 if i == 0 else 3, rx=16)
        peptide_chain(svg, x + w / 2, cy + 55, scale=0.42, rot=angle)
        if i < 3:
            svg.arrow(x + w - 45, cy + 55, x + w - 45, cy + 128, stroke="#B8A16B", sw=5, head=18)


def render() -> SVG:
    svg = SVG(W, H)
    svg.rect(30, 35, W - 60, H - 70, fill="#FFFFFF", stroke="#D8E0EA", sw=4, rx=42)

    input_box = (80, 120, 390, 1010)
    flow_box = (555, 120, 910, 1010)
    bank_box = (1550, 120, 500, 1010)
    ipg_box = (2135, 120, 925, 1010)
    output_box = (3145, 120, 375, 1010)

    panel(svg, *input_box, fill="#F4F8FB", stroke="#B8C6D6")
    receptor(svg, 275, 270, scale=0.8)
    svg.arrow(275, 420, 275, 500, stroke="#8A9AAC", sw=6, head=22)
    peptide_surface(svg, 275, 610, scale=0.78)
    svg.arrow(275, 745, 275, 835, stroke="#8A9AAC", sw=6, head=22)
    graph_icon(svg, 140, 910, "receptor", scale=0.7)
    graph_icon(svg, 215, 880, "cross", scale=0.6)
    graph_icon(svg, 320, 905, "residue", scale=0.55)
    graph_icon(svg, 370, 900, "atom", scale=0.45)

    svg.arrow(495, 625, 535, 625, stroke="#5E6D7E", sw=9, head=28)

    panel(svg, *flow_box, fill="#EEF7F2", stroke="#AFC5B5")
    graph_icon(svg, 620, 245, "receptor", scale=0.7)
    graph_icon(svg, 610, 420, "cross", scale=0.7)
    graph_icon(svg, 650, 610, "residue", scale=0.7)
    graph_icon(svg, 610, 780, "atom", scale=0.6)
    svg.arrow(780, 360, 830, 360, stroke="#8A9AAC", sw=6, head=22)
    cgtpel_stack(svg, 835, 170, 570, 395)
    svg.arrow(1120, 585, 1120, 670, stroke="#8A9AAC", sw=6, head=22)
    flow_heads(svg, 790, 705, 620, 305)
    svg.arrow(1410, 858, 1495, 858, stroke="#5E6D7E", sw=9, head=28)

    candidate_bank(svg, *bank_box)
    svg.arrow(2075, 625, 2115, 625, stroke="#5E6D7E", sw=9, head=28)
    ipg_stage(svg, *ipg_box)
    svg.arrow(3045, 830, 3125, 830, stroke="#5E6D7E", sw=9, head=28)
    score_output(svg, *output_box)
    return svg


def main() -> None:
    args = parse_args()
    svg = render()
    svg.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
