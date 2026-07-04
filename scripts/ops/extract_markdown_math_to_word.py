#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


BLOCK_PATTERNS = (
    ("块公式", re.compile(r"\$\$(.*?)\$\$", flags=re.S)),
    ("块公式", re.compile(r"\\\[(.*?)\\\]", flags=re.S)),
    (
        "公式环境",
        re.compile(
            r"\\begin\{(equation\*?|align\*?|gather\*?|multline\*?|eqnarray\*?)\}(.*?)\\end\{\1\}",
            flags=re.S,
        ),
    ),
)
INLINE_PATTERNS = (
    ("行内公式", re.compile(r"\\\((.*?)\\\)", flags=re.S)),
    ("行内公式", re.compile(r"(?<!\\)\$(?!\$)([^$\n]+?)(?<!\\)\$(?!\$)")),
)

FORMAT_COMMANDS = (
    "mathbf",
    "boldsymbol",
    "mathrm",
    "mathbb",
    "mathcal",
    "mathsf",
    "mathtt",
    "operatorname",
    "text",
    "hat",
    "bar",
    "tilde",
    "vec",
)
IGNORED_VAR_BASES = {
    "left",
    "right",
    "middle",
    "qquad",
    "quad",
    "text",
    "mathrm",
    "mathbf",
    "boldsymbol",
    "mathbb",
    "mathcal",
    "mathsf",
    "mathtt",
    "operatorname",
    "min",
    "max",
    "arg",
    "softmax",
    "frac",
    "sqrt",
}
STRUCTURE_CHARS = set("+-=*/^_()[]{}|,:;<>")
SNAKE_CASE_RE = re.compile(r"\b[a-z]+(?:_[A-Za-z0-9]+)+\b")
HEADING_RE = re.compile(r"^(?P<code>\d+(?:\.\d+){1,3})\s+(?P<title>.+)$")
VAR_TOKEN_RE = re.compile(
    r"(?:\\[A-Za-z]+|[A-Za-z])"
    r"(?:_\{[^{}]+\}|_[A-Za-z0-9()]+)?"
    r"(?:\^\{[^{}]+\}|\^[A-Za-z0-9()]+)?"
    r"(?:_\{[^{}]+\}|_[A-Za-z0-9()]+)?"
)


@dataclass(frozen=True)
class MathItem:
    index: int
    kind: str
    line: int
    raw: str
    content: str
    start: int
    end: int


@dataclass(frozen=True)
class BlockContext:
    index: int
    kind: str
    start_line: int
    end_line: int
    section_code: str
    section_title: str
    paragraph_no: int
    text: str

    @property
    def location(self) -> str:
        if self.section_code:
            return f"{self.section_code} 第{self.paragraph_no}段"
        return f"未编号区域第{self.paragraph_no}段"

    @property
    def section_label(self) -> str:
        if self.section_code and self.section_title:
            return f"{self.section_code} {self.section_title}"
        if self.section_code:
            return self.section_code
        return "未编号区域"


def line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def overlaps(start: int, end: int, spans: Iterable[tuple[int, int]]) -> bool:
    for span_start, span_end in spans:
        if start < span_end and end > span_start:
            return True
    return False


def extract_math_items(text: str) -> list[MathItem]:
    items: list[MathItem] = []
    spans: list[tuple[int, int]] = []

    for kind, pattern in BLOCK_PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.span()
            if overlaps(start, end, spans):
                continue
            content = match.group(2) if pattern.groups == 2 else match.group(1)
            items.append(
                MathItem(
                    index=0,
                    kind=kind,
                    line=line_number(text, start),
                    raw=text[start:end],
                    content=content.strip(),
                    start=start,
                    end=end,
                )
            )
            spans.append((start, end))

    for kind, pattern in INLINE_PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.span()
            if overlaps(start, end, spans):
                continue
            items.append(
                MathItem(
                    index=0,
                    kind=kind,
                    line=line_number(text, start),
                    raw=text[start:end],
                    content=match.group(1).strip(),
                    start=start,
                    end=end,
                )
            )

    items.sort(key=lambda item: (item.start, item.end))
    return [
        MathItem(
            index=index,
            kind=item.kind,
            line=item.line,
            raw=item.raw,
            content=item.content,
            start=item.start,
            end=item.end,
        )
        for index, item in enumerate(items, start=1)
    ]


def classify_block(lines: list[str]) -> str:
    stripped = [line.strip() for line in lines if line.strip()]
    if not stripped:
        return "空块"
    if HEADING_RE.match(stripped[0]):
        return "标题"
    if stripped[0].startswith("$$") or stripped[0].startswith("\\[") or stripped[0].startswith("\\begin{"):
        return "公式块"
    if stripped[0].startswith("!["):
        return "图片"
    if all(line.startswith("|") for line in stripped):
        return "表格"
    if all(
        line.startswith(("- ", "* "))
        or re.match(r"^\d+[.)]\s+", line)
        for line in stripped
    ):
        return "列表"
    return "正文"


def extract_blocks(text: str) -> list[BlockContext]:
    lines = text.splitlines()
    blocks: list[BlockContext] = []
    buffer: list[tuple[int, str]] = []
    current_section_code = ""
    current_section_title = ""
    paragraph_counters: dict[str, int] = {}
    block_index = 0

    def flush_buffer() -> None:
        nonlocal buffer, current_section_code, current_section_title, block_index
        if not buffer:
            return
        block_lines = [line for _, line in buffer]
        stripped_first = block_lines[0].strip()
        heading_match = HEADING_RE.match(stripped_first)
        if heading_match:
            current_section_code = heading_match.group("code")
            current_section_title = heading_match.group("title").strip()
            paragraph_counters.setdefault(current_section_code, 0)
            buffer = []
            return

        section_code = current_section_code
        section_title = current_section_title
        paragraph_counters.setdefault(section_code, 0)
        paragraph_counters[section_code] += 1
        block_index += 1
        blocks.append(
            BlockContext(
                index=block_index,
                kind=classify_block(block_lines),
                start_line=buffer[0][0],
                end_line=buffer[-1][0],
                section_code=section_code,
                section_title=section_title,
                paragraph_no=paragraph_counters[section_code],
                text="\n".join(block_lines).strip(),
            )
        )
        buffer = []

    for line_no, line in enumerate(lines, start=1):
        if line.strip():
            buffer.append((line_no, line))
        else:
            flush_buffer()
    flush_buffer()
    return blocks


def build_line_to_block_map(blocks: Iterable[BlockContext]) -> dict[int, BlockContext]:
    mapping: dict[int, BlockContext] = {}
    for block in blocks:
        for line_no in range(block.start_line, block.end_line + 1):
            mapping[line_no] = block
    return mapping


def shorten_text(text: str, max_length: int = 140) -> str:
    single_line = re.sub(r"`([^`]+)`", r"“\1”", text)
    single_line = re.sub(r"\s+", " ", single_line).strip()
    if len(single_line) <= max_length:
        return single_line
    return single_line[: max_length - 1] + "…"


def quote_text(text: str) -> str:
    return f"“{text}”"


def unwrap_format_commands(expr: str) -> str:
    result = expr
    for command in FORMAT_COMMANDS:
        pattern = re.compile(rf"\\{command}\{{([^{{}}]+)\}}")
        while True:
            updated = pattern.sub(r"\1", result)
            if updated == result:
                break
            result = updated
    return result


def extract_command_tokens(contents: Iterable[str]) -> list[str]:
    counter: Counter[str] = Counter()
    for content in contents:
        for token in re.findall(r"\\[A-Za-z]+", content):
            counter[token] += 1
    return [token for token, _ in sorted(counter.items(), key=lambda item: (-item[1], item[0]))]


def extract_structure_tokens(contents: Iterable[str]) -> list[str]:
    counter: Counter[str] = Counter()
    for content in contents:
        for char in content:
            if char in STRUCTURE_CHARS:
                counter[char] += 1
    return [token for token, _ in sorted(counter.items(), key=lambda item: (-item[1], item[0]))]


def extract_formula_variables(items: Iterable[MathItem]) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for item in items:
        content = unwrap_format_commands(item.content)
        for token in VAR_TOKEN_RE.findall(content):
            base = token.lstrip("\\")
            if base in IGNORED_VAR_BASES:
                continue
            if len(base) == 1 and not any(mark in token for mark in ("_", "^", "\\")):
                continue
            record = records.setdefault(
                token,
                {"count": 0, "lines": set()},
            )
            record["count"] += 1
            record["lines"].add(item.line)
    return dict(sorted(records.items(), key=lambda item: (-item[1]["count"], item[0])))


def mask_spans(text: str, spans: Iterable[tuple[int, int]]) -> str:
    chars = list(text)
    for start, end in spans:
        for index in range(start, end):
            if chars[index] != "\n":
                chars[index] = " "
    return "".join(chars)


def extract_snake_case_identifiers(
    text: str,
    spans: Iterable[tuple[int, int]],
    line_to_block: dict[int, BlockContext],
) -> dict[str, dict[str, object]]:
    masked_text = mask_spans(text, spans)
    records: dict[str, dict[str, object]] = {}
    for line_no, line in enumerate(masked_text.splitlines(), start=1):
        for match in SNAKE_CASE_RE.finditer(line):
            token = match.group()
            block = line_to_block.get(line_no)
            record = records.setdefault(
                token,
                {"count": 0, "lines": set(), "occurrences": [], "seen_keys": set()},
            )
            record["count"] += 1
            record["lines"].add(line_no)
            occurrence_key = (line_no, block.location if block else "")
            if occurrence_key not in record["seen_keys"]:
                record["occurrences"].append(
                    {
                        "line": line_no,
                        "location": block.location if block else f"行 {line_no}",
                        "section_label": block.section_label if block else "未编号区域",
                        "block_kind": block.kind if block else "未知",
                        "excerpt": shorten_text(text.splitlines()[line_no - 1].strip()),
                    }
                )
                record["seen_keys"].add(occurrence_key)
    for meta in records.values():
        meta.pop("seen_keys", None)
    return dict(sorted(records.items(), key=lambda item: (-item[1]["count"], item[0])))


def render_location_rules() -> str:
    parts = [
        "## 替换定位规则",
        "",
        "- 定位格式统一写为“x.x / x.x.x 小节的第 n 段”。",
        "- 段号按对应小节下自上而下计数；正文段、列表、表格、图片块、独立公式块都各算 1 段。",
        "- 标题本身不计段号；行号保留为辅助定位，避免你在 Word 里来回翻得脑壳疼。",
        "",
    ]
    return "\n".join(parts)


def render_formula_section(items: list[MathItem], line_to_block: dict[int, BlockContext]) -> str:
    parts = ["## 公式清单", ""]
    for item in items:
        block = line_to_block.get(item.line)
        parts.append(f"### 公式 {item.index}")
        parts.append(f"- 类型：{item.kind}")
        parts.append(f"- 起始行：{item.line}")
        if block:
            parts.append(f"- 所在位置：{quote_text(block.location)}")
            parts.append(f"- 所在小节：{quote_text(block.section_label)}")
            parts.append(f"- 所在块类型：{quote_text(block.kind)}")
        parts.append("")
        if item.raw.startswith("$$") or item.raw.startswith("\\[") or item.raw.startswith("\\begin"):
            parts.append(item.raw.strip())
        else:
            parts.append(f"- 原文：{item.raw.strip()}")
        parts.append("")
    return "\n".join(parts)


def render_symbol_section(command_tokens: list[str], structure_tokens: list[str]) -> str:
    parts = [
        "## 数学符号与命令",
        "",
        "### LaTeX 数学命令",
        "",
        *[f"- {quote_text(token)}" for token in command_tokens],
        "",
        "### 结构与运算符字符",
        "",
        *[f"- {quote_text(token)}" for token in structure_tokens],
        "",
    ]
    return "\n".join(parts)


def render_formula_variable_section(records: dict[str, dict[str, object]]) -> str:
    parts = ["## 公式中的变量表达式", ""]
    for token, meta in records.items():
        line_text = ", ".join(str(line) for line in sorted(meta["lines"]))
        parts.append(f"- {quote_text(token)}：出现 {meta['count']} 次；行号：{line_text}")
    parts.append("")
    return "\n".join(parts)


def render_snake_case_section(records: dict[str, dict[str, object]]) -> str:
    parts = ["## 正文中的下划线变量/标识符候选", ""]
    for token, meta in records.items():
        line_text = ", ".join(str(line) for line in sorted(meta["lines"]))
        parts.append(f"- {quote_text(token)}：出现 {meta['count']} 次；行号：{line_text}")
        for occurrence in meta["occurrences"]:
            parts.append(
                f"  - 位置：{quote_text(occurrence['location'])}；所在小节：{quote_text(occurrence['section_label'])}；"
                f"行号：{occurrence['line']}；块类型：{quote_text(occurrence['block_kind'])}"
            )
            parts.append(f"  - 替换对象：{quote_text(token)}；上下文：{occurrence['excerpt']}")
    parts.append("")
    return "\n".join(parts)


def build_markdown(
    source_path: Path,
    items: list[MathItem],
    blocks: list[BlockContext],
    line_to_block: dict[int, BlockContext],
    command_tokens: list[str],
    structure_tokens: list[str],
    formula_variables: dict[str, dict[str, object]],
    snake_case_records: dict[str, dict[str, object]],
) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    inline_count = sum(1 for item in items if item.kind == "行内公式")
    block_count = len(items) - inline_count
    sections = [
        f"# {source_path.name} 公式与数学符号提取汇总",
        "",
        "## 提取说明",
        "",
        f"- 源文件：{source_path.as_posix()}",
        f"- 生成时间：{generated_at}",
        f"- 公式总数：{len(items)}（行内公式 {inline_count}，块公式/公式环境 {block_count}）",
        f"- 数学命令数：{len(command_tokens)}",
        f"- 结构字符数：{len(structure_tokens)}",
        f"- 公式变量表达式数：{len(formula_variables)}",
        f"- 下划线变量/标识符候选数：{len(snake_case_records)}",
        f"- 可用于段落定位的非空内容块数：{len(blocks)}",
        "",
        render_location_rules(),
        render_formula_section(items, line_to_block),
        render_symbol_section(command_tokens, structure_tokens),
        render_formula_variable_section(formula_variables),
        render_snake_case_section(snake_case_records),
    ]
    return "\n".join(sections).strip() + "\n"


def run_pandoc(input_md: Path, output_docx: Path) -> None:
    subprocess.run(
        [
            "pandoc",
            str(input_md),
            "-f",
            "markdown+tex_math_dollars+tex_math_single_backslash",
            "-t",
            "docx",
            "-o",
            str(output_docx),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="提取 Markdown 中的公式、数学符号和变量名并导出为 Word。")
    parser.add_argument("input_md", type=Path, help="输入 Markdown 文件路径")
    parser.add_argument("--output-md", type=Path, required=True, help="汇总 Markdown 输出路径")
    parser.add_argument("--output-docx", type=Path, required=True, help="Word 输出路径")
    args = parser.parse_args()

    source_path = args.input_md
    text = source_path.read_text(encoding="utf-8")

    items = extract_math_items(text)
    blocks = extract_blocks(text)
    line_to_block = build_line_to_block_map(blocks)
    command_tokens = extract_command_tokens(item.content for item in items)
    structure_tokens = extract_structure_tokens(item.content for item in items)
    formula_variables = extract_formula_variables(items)
    snake_case_records = extract_snake_case_identifiers(
        text,
        spans=[(item.start, item.end) for item in items],
        line_to_block=line_to_block,
    )

    markdown_output = build_markdown(
        source_path=source_path,
        items=items,
        blocks=blocks,
        line_to_block=line_to_block,
        command_tokens=command_tokens,
        structure_tokens=structure_tokens,
        formula_variables=formula_variables,
        snake_case_records=snake_case_records,
    )

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_docx.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown_output, encoding="utf-8")
    run_pandoc(args.output_md, args.output_docx)


if __name__ == "__main__":
    main()
