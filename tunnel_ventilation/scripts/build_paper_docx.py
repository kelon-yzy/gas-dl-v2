"""把 docs/paper 下的 markdown 章节合并成一份 .docx。

正文 = §1–§9（中文手稿），附录 A = 文献核实与领地判定，附录 B = 文档说明与产物索引。
附录两份是中文工作记录，标题层级整体下移一级挂到附录 H1 下。

依赖 pandoc（转换）与标准库（参考文档字体/版式补丁）。

    python scripts/build_paper_docx.py
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
import zipfile
from pathlib import Path

PAPER_DIR = Path("docs/paper")
OUTPUT = PAPER_DIR / "tv3_methods_compendium.docx"

TITLE = "tv3 方法梳理文档"
SUBTITLE = (
    "三组分声学气体组分方案的前向模型、测量级信号处理、审计链与实验结果"
)

# 正文按手稿顺序；附录单独处理
BODY_FILES = [
    "01_introduction.md",
    "02_problem_and_forward.md",
    "03_front_end.md",
    "04_audit_chain.md",
    "05_protocol.md",
    "06_results.md",
    "07_discussion.md",
    "08_conclusion.md",
    "09_limitations.md",
]
APPENDICES = [
    ("00_related_work.md", "附录 A　文献核实与领地判定"),
    ("README.md", "附录 B　文档说明、数值溯源与产物索引"),
]

PAGE_BREAK = '\n```{=openxml}\n<w:p><w:r><w:br w:type="page"/></w:r></w:p>\n```\n'

FRONT_MATTER = """*上面的目录是 Word 域，首次打开时是空的：全选后按 F9，或右键“更新域”，即可填出条目与页码。*

# 文档说明

| 项 | 内容 |
| --- | --- |
| 场景 | 掘进通风 CO₂ / O₂ / N₂ 三组分反演（`tunnel_ventilation`，tv3） |
| 文档性质 | **方法梳理稿，不是投稿论文。** 期刊模板、投稿英文稿润色、绘图、字数裁剪、投稿附信均未做，详见附录 B |
| 正文语言 | 中文（§1–§9）；附录 A / B 为中文工作记录 |
| 数值溯源 | 正文引用的数值由 `scripts/build_paper_artifacts.py` 从冻结产物生成，17 个产物 / 46 个源文件，逐个记录路径与 SHA-256 |
| 本合并稿生成 | `python scripts/build_paper_docx.py`（源文件不变则输出可复现） |

正文 §1 至 §9 按手稿顺序排列，附录 A、B 的标题层级整体下移一级。
"""


def shift_headings(text: str, by: int = 1) -> str:
    """把 ATX 标题下移 by 级，跳过围栏代码块内的 # 行。"""
    lines: list[str] = []
    fence: str | None = None
    for line in text.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith(("```", "~~~")):
            marker = stripped[:3]
            if fence is None:
                fence = marker
            elif stripped.startswith(fence):
                fence = None
            lines.append(line)
            continue
        if fence is None and re.match(r"^#{1,6}\s", line):
            line = "#" * by + line
        lines.append(line)
    return "\n".join(lines)


def assemble(paper_dir: Path) -> str:
    parts = [FRONT_MATTER]
    for name in BODY_FILES:
        path = paper_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        parts.append(PAGE_BREAK)
        parts.append(path.read_text(encoding="utf-8").strip())
    for name, heading in APPENDICES:
        path = paper_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        parts.append(PAGE_BREAK)
        parts.append(f"# {heading}\n")
        parts.append(shift_headings(path.read_text(encoding="utf-8").strip()))
    return "\n\n".join(parts) + "\n"


def _patch_theme(xml: str) -> str:
    xml = xml.replace('typeface="Aptos Display"', 'typeface="Cambria"')
    xml = xml.replace('typeface="Aptos"', 'typeface="Calibri"')
    xml = xml.replace('<a:ea typeface=""/>', '<a:ea typeface="Microsoft YaHei"/>')
    xml = xml.replace('script="Hans" typeface="等线 Light"', 'script="Hans" typeface="Microsoft YaHei"')
    xml = xml.replace('script="Hans" typeface="等线"', 'script="Hans" typeface="Microsoft YaHei"')
    return xml


def _patch_styles(xml: str) -> str:
    # 正文 11pt：篇幅较长，12pt 过大
    xml = xml.replace('<w:sz w:val="24" />\n        <w:szCs w:val="24" />',
                      '<w:sz w:val="22" />\n        <w:szCs w:val="22" />')
    # 表格 8pt：§6.4 / §6.6 / §6.7 是 6–7 列，9pt 下数字列仍会断行
    xml = xml.replace(
        '<w:name w:val="Table" />\n    <w:basedOn w:val="TableNormal" />',
        '<w:name w:val="Table" />\n    <w:basedOn w:val="TableNormal" />'
        '\n    <w:rPr><w:sz w:val="16" /><w:szCs w:val="16" /></w:rPr>',
    )
    return xml


def _patch_document(xml: str) -> str:
    """A4 + 2cm 左右边距，给 §6.4 / §6.6 那类宽表让出版心。

    pandoc 默认参考文档的 sectPr 只带 footnotePr，没有 pgSz/pgMar，
    落到阅读器的默认纸型（实测 LibreOffice 给的是 Letter），所以这里是插入而非替换。
    """
    if "<w:pgSz" in xml:
        raise RuntimeError("reference sectPr already carries a page size; revisit this patch")
    page = (
        '<w:pgSz w:w="11906" w:h="16838" />'
        '<w:pgMar w:top="1418" w:right="1134" w:bottom="1418" w:left="1134" '
        'w:header="709" w:footer="709" w:gutter="0" />'
    )
    patched = xml.replace("</w:sectPr>", page + "</w:sectPr>", 1)
    if patched == xml:
        raise RuntimeError("no sectPr found in the reference document")
    return patched


def build_reference(dest: Path) -> None:
    base = dest.with_suffix(".base.docx")
    with base.open("wb") as fh:
        subprocess.run(
            ["pandoc", "--print-default-data-file", "reference.docx"],
            stdout=fh, check=True,
        )
    patches = {
        "word/theme/theme1.xml": _patch_theme,
        "word/styles.xml": _patch_styles,
        "word/document.xml": _patch_document,
    }
    with zipfile.ZipFile(base) as src, zipfile.ZipFile(
        dest, "w", zipfile.ZIP_DEFLATED
    ) as out:
        for item in src.infolist():
            data = src.read(item.filename)
            patch = patches.get(item.filename)
            if patch is not None:
                data = patch(data.decode("utf-8")).encode("utf-8")
            out.writestr(item, data)
    base.unlink()


def autofit_tables(path: Path) -> int:
    """把 pandoc 的固定表格布局改成自适应。

    pandoc 按 markdown 源里各列的字符宽度分配 gridCol，源里列宽本来就不均匀，
    固定布局下 §6.4 这类表会出现 "LHS boundar / y" 这种断词。改成 autofit 后
    由阅读器按内容分列宽。
    """
    with zipfile.ZipFile(path) as src:
        items = [(i, src.read(i.filename)) for i in src.infolist()]
    replaced = 0
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as out:
        for item, data in items:
            if item.filename == "word/document.xml":
                xml = data.decode("utf-8")
                xml, replaced = re.subn(
                    r'<w:tblLayout w:type="fixed"\s*/>',
                    '<w:tblLayout w:type="autofit" />',
                    xml,
                )
                data = xml.encode("utf-8")
            out.writestr(item, data)
    return replaced


def set_chapter_page_breaks(path: Path) -> int:
    """为正文和附录的一级标题设置段前分页。

    Pandoc 对 raw OpenXML 分页块的处理随版本和输入扩展而变化，不能把它
    作为分页真相源。生成文档后直接处理 Heading1：保留第一个“文档说明”
    标题，其余正文与附录标题统一设置 pageBreakBefore。
    """
    with zipfile.ZipFile(path) as src:
        items = [(item, src.read(item.filename)) for item in src.infolist()]

    page_break_paragraph = re.compile(
        r'<w:p>\s*<w:r>\s*<w:br\s+w:type="page"\s*/>\s*</w:r>\s*</w:p>\s*',
        re.DOTALL,
    )
    heading_pattern = re.compile(
        r'<w:p>\s*<w:pPr>\s*<w:pStyle\s+w:val="Heading1"\s*/>'
        r'.*?</w:pPr>.*?</w:p>',
        re.DOTALL,
    )

    heading_index = 0
    chapter_breaks = 0

    def add_break(match: re.Match[str]) -> str:
        nonlocal heading_index, chapter_breaks
        heading = match.group(0)
        heading_index += 1
        if heading_index == 1:
            return heading
        chapter_breaks += 1
        if "<w:pageBreakBefore" in heading:
            return heading
        return heading.replace(
            "<w:pPr>", "<w:pPr><w:pageBreakBefore />", 1
        )

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as out:
        for item, data in items:
            if item.filename == "word/document.xml":
                xml = data.decode("utf-8")
                xml = page_break_paragraph.sub("", xml)
                xml = heading_pattern.sub(add_break, xml)
                data = xml.encode("utf-8")
            out.writestr(item, data)
    return chapter_breaks


def _display_width(text: str) -> int:
    """去掉行内 markdown 标记后的显示宽度，CJK 与全角标点按 2 计。"""
    plain = re.sub(r"[*`]", "", text).strip()
    width = 0
    for ch in plain:
        width += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return width


def _split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def rebalance_pipe_tables(text: str, min_w: int = 8, max_w: int = 36) -> str:
    """按各列内容宽度重写分隔行。

    pandoc 的 docx writer 用分隔行各段的长度决定 gridCol，源文件里写的是等宽
    `| --- | --- |`，于是七列表格拿到七等分列宽，"SPXY, target-margin selector"
    这类长标签就会断在词中间。这里只改分隔行，单元格内容一个字符不动。
    """
    lines = text.split("\n")
    out: list[str] = []
    fence: str | None = None
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        if stripped.startswith(("```", "~~~")):
            marker = stripped[:3]
            fence = marker if fence is None else (None if stripped.startswith(fence) else fence)
            out.append(line)
            i += 1
            continue
        is_sep = (
            fence is None
            and i > 0
            and lines[i - 1].lstrip().startswith("|")
            and re.fullmatch(r"\|[\s:\-|]+\|", stripped or "")
            and "-" in stripped
        )
        if not is_sep:
            out.append(line)
            i += 1
            continue

        seps = _split_row(line)
        # 表格体：从表头往上、分隔行往下，收集所有单元格
        start = i - 1
        while start > 0 and lines[start - 1].lstrip().startswith("|"):
            start -= 1
        end = i + 1
        while end < len(lines) and lines[end].lstrip().startswith("|"):
            end += 1

        # 列宽取该列最宽的单元格（含表头）加 1 的余量。
        # 试过两种"让数据列更宽"的变体——给表头打 0.6 折、限制表头只能比数据宽
        # 6 个字符——两次都是把末列的断行换成了表头自己断成 "evid/ence"。
        # §6.4 那张七列表的内容本来就接近 A4 版心的容量，这里不再调。
        widths = [min_w] * len(seps)
        for row in lines[start:i] + lines[i + 1 : end]:
            for col, cell in enumerate(_split_row(row)[: len(seps)]):
                widths[col] = max(widths[col], _display_width(cell) + 1)
        widths = [max(min_w, min(max_w, w)) for w in widths]

        rebuilt = []
        for sep, width in zip(seps, widths):
            left, right = sep.startswith(":"), sep.endswith(":")
            dashes = max(3, width - int(left) - int(right))
            rebuilt.append((":" if left else "") + "-" * dashes + (":" if right else ""))
        out.append("| " + " | ".join(rebuilt) + " |")
        i += 1
    return "\n".join(out)


def main() -> int:
    if shutil.which("pandoc") is None:
        print("pandoc not found on PATH", file=sys.stderr)
        return 1
    paper_dir = Path(PAPER_DIR)
    if not paper_dir.is_dir():
        print(f"missing {paper_dir}; run from the tunnel_ventilation root", file=sys.stderr)
        return 1

    merged = rebalance_pipe_tables(assemble(paper_dir))
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        md = tmp_path / "merged.md"
        md.write_text(merged, encoding="utf-8")
        reference = tmp_path / "reference.docx"
        build_reference(reference)
        subprocess.run(
            [
                "pandoc",
                str(md),
                "--from=markdown",
                "--to=docx",
                f"--reference-doc={reference}",
                "--toc",
                "--toc-depth=3",
                "--metadata", f"title={TITLE}",
                "--metadata", f"subtitle={SUBTITLE}",
                "--metadata", "toc-title=目录",
                "-o", str(OUTPUT),
            ],
            check=True,
        )

    chapter_breaks = set_chapter_page_breaks(OUTPUT)
    expected_breaks = len(BODY_FILES) + len(APPENDICES)
    if chapter_breaks != expected_breaks:
        raise RuntimeError(
            f"expected {expected_breaks} chapter page breaks, found {chapter_breaks}"
        )
    tables = autofit_tables(OUTPUT)
    words = len(merged.split())
    print(
        f"{OUTPUT}  ({OUTPUT.stat().st_size / 1024:.0f} KB, "
        f"merged source {words} words, {chapter_breaks} chapter starts, "
        f"{tables} tables set to autofit)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
