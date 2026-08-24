"""P1 引用一致性检查。

把 P0 的纪律"引用数值必须由脚本从冻结产物生成，不手工誊写"迁到 P1 的引用元数据上：
散文里出现的每个 DOI 必须在 references.json 登记，且登记的元数据必须经 CrossRef 核实。

用法：
    python docs/p1/tools/check_citations.py                # 离线检查
    python docs/p1/tools/check_citations.py --online       # 额外向 CrossRef 复核每条元数据

退出码 0 表示全部通过，1 表示存在不一致。
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger("p1.check_citations")

P1_ROOT = Path(__file__).resolve().parents[1]
REFERENCES_PATH = P1_ROOT / "references" / "references.json"

REQUIRED_ENTRY_FIELDS: Tuple[str, ...] = (
    "key",
    "doi",
    "title",
    "authors",
    "year",
    "container",
    "publisher",
    "type",
    "verified_via",
    "verified_at",
    "perspective",
    "evidence_grade",
)
ALLOWED_VERIFICATION_SOURCES: Tuple[str, ...] = ("crossref",)

# DOI 主体允许的字符集按 CrossRef 实践取，匹配后再剥离尾随的 Markdown 标点。
DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")
DOI_TRAILING_JUNK = "`.,;:)]}>\"'*"

CROSSREF_API = "https://api.crossref.org/works/{doi}"
CROSSREF_TIMEOUT_S = 20
USER_AGENT = "p1-citation-check (mailto:unset; academic use)"


@dataclass(frozen=True)
class Finding:
    """一条不一致。"""

    severity: str
    where: str
    message: str


@dataclass
class CheckReport:
    """检查结果汇总。"""

    n_entries: int = 0
    n_prose_dois: int = 0
    n_online_checked: int = 0
    findings: List[Finding] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(f.severity == "error" for f in self.findings)

    def add(self, severity: str, where: str, message: str) -> None:
        self.findings.append(Finding(severity=severity, where=where, message=message))


def normalize_doi(raw: str) -> str:
    """DOI 按规范大小写不敏感，统一转小写并剥离尾随标点。"""
    return raw.rstrip(DOI_TRAILING_JUNK).lower()


def load_references(path: Path) -> Dict[str, Any]:
    """读取结构化引用存储。"""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.error("引用存储不存在：%s", path)
        raise
    except json.JSONDecodeError:
        logger.error("引用存储不是合法 JSON：%s", path)
        raise


def check_entries(store: Dict[str, Any], report: CheckReport) -> Dict[str, Dict[str, Any]]:
    """校验每条登记项的字段完整性与核实来源，返回 doi -> entry 映射。"""
    entries: Sequence[Dict[str, Any]] = store.get("entries", [])
    report.n_entries = len(entries)

    by_doi: Dict[str, Dict[str, Any]] = {}
    seen_keys: Dict[str, int] = {}

    for index, entry in enumerate(entries):
        where = f"entries[{index}]"
        key = entry.get("key", "")

        for name in REQUIRED_ENTRY_FIELDS:
            value = entry.get(name)
            if value is None or value == "" or value == []:
                report.add("error", where, f"{key or '(无 key)'} 缺少必填字段 {name}")

        source = entry.get("verified_via", "")
        if source not in ALLOWED_VERIFICATION_SOURCES:
            report.add(
                "error",
                where,
                f"{key} 的 verified_via={source!r} 不在允许的核实来源 {ALLOWED_VERIFICATION_SOURCES} 内",
            )

        if key:
            if key in seen_keys:
                report.add("error", where, f"key 重复：{key}（首次出现在 entries[{seen_keys[key]}]）")
            else:
                seen_keys[key] = index

        doi = normalize_doi(str(entry.get("doi", "")))
        if not doi:
            continue
        if doi in by_doi:
            report.add("error", where, f"DOI 重复登记：{doi}")
        else:
            by_doi[doi] = entry

    for index, pending in enumerate(store.get("pending_verification", [])):
        doi = normalize_doi(str(pending.get("doi", "")))
        if doi and doi in by_doi:
            report.add(
                "error",
                f"pending_verification[{index}]",
                f"{doi} 同时出现在 entries 与 pending_verification",
            )

    return by_doi


def iter_prose_files(root: Path) -> Iterable[Path]:
    """P1 目录下的散文文档。"""
    for path in sorted(root.rglob("*.md")):
        yield path


def collect_prose_dois(root: Path) -> Dict[str, List[str]]:
    """扫描散文里出现的 DOI，返回 doi -> [出现位置]。"""
    found: Dict[str, List[str]] = {}
    for path in iter_prose_files(root):
        rel = path.relative_to(root.parent).as_posix()
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for match in DOI_PATTERN.finditer(line):
                doi = normalize_doi(match.group(0))
                if doi:
                    found.setdefault(doi, []).append(f"{rel}:{lineno}")
    return found


def check_prose_coverage(
    prose_dois: Dict[str, List[str]],
    registered: Dict[str, Dict[str, Any]],
    report: CheckReport,
) -> None:
    """散文里的每个 DOI 必须已登记。"""
    report.n_prose_dois = len(prose_dois)
    for doi, locations in sorted(prose_dois.items()):
        if doi not in registered:
            report.add("error", locations[0], f"散文引用了未登记的 DOI：{doi}")


def fetch_crossref(doi: str) -> Optional[Dict[str, Any]]:
    """向 CrossRef 取一条记录，网络或解析失败返回 None。"""
    request = urllib.request.Request(
        CROSSREF_API.format(doi=urllib.parse.quote(doi, safe="")),
        headers={"Accept": "application/json", "User-Agent": USER_AGENT},
    )
    try:
        with urllib.request.urlopen(request, timeout=CROSSREF_TIMEOUT_S) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        logger.warning("CrossRef 返回 HTTP %s：%s", exc.code, doi)
        return None
    except (urllib.error.URLError, TimeoutError) as exc:
        logger.warning("CrossRef 网络失败：%s（%s）", doi, exc)
        return None
    except json.JSONDecodeError:
        logger.warning("CrossRef 响应不是合法 JSON：%s", doi)
        return None
    return payload.get("message")


def crossref_years(message: Dict[str, Any]) -> Dict[str, int]:
    """取 CrossRef 的各口径出版年。

    online-first 与 print 分属不同年份是期刊常态（IOP / Elsevier 尤为常见），
    因此不预设唯一正确年份，返回全部口径供比对，由调用方判断本地年份是否落在其中。
    """
    years: Dict[str, int] = {}
    for field_name in ("issued", "published", "published-online", "published-print"):
        parts = message.get(field_name, {}).get("date-parts") or []
        if parts and parts[0] and parts[0][0]:
            years[field_name] = int(parts[0][0])
    return years


def normalize_title(title: str) -> str:
    """题名比对前归一：解 HTML 实体、小写、非字母数字归为空格、压缩空白。"""
    lowered = re.sub(r"[^a-z0-9]+", " ", html.unescape(title).lower())
    return " ".join(lowered.split())


def check_entry_online(entry: Dict[str, Any], report: CheckReport) -> bool:
    """向 CrossRef 复核一条登记项的题名、首作者姓、年份与刊名。"""
    doi = normalize_doi(str(entry.get("doi", "")))
    key = entry.get("key", doi)
    message = fetch_crossref(doi)
    if message is None:
        report.add("warning", key, f"未能从 CrossRef 取回 {doi}，本轮无法复核")
        return False

    remote_titles = message.get("title") or []
    if remote_titles:
        local = normalize_title(str(entry.get("title", "")))
        remote = normalize_title(str(remote_titles[0]))
        if local != remote:
            report.add("error", key, f"题名与 CrossRef 不一致\n  本地：{entry.get('title')}\n  远端：{remote_titles[0]}")

    remote_authors = message.get("author") or []
    local_authors = entry.get("authors") or []
    if remote_authors and local_authors:
        remote_family = str(remote_authors[0].get("family", "")).lower()
        local_first = str(local_authors[0]).lower()
        if remote_family and remote_family not in local_first:
            report.add(
                "error",
                key,
                f"首作者与 CrossRef 不一致（本地 {local_authors[0]!r}，远端姓 {remote_authors[0].get('family')!r}）",
            )
    if remote_authors and len(remote_authors) != len(local_authors):
        report.add(
            "warning",
            key,
            f"作者数不一致（本地 {len(local_authors)}，远端 {len(remote_authors)}）",
        )

    remote_years = crossref_years(message)
    local_year = int(entry.get("year", 0))
    if remote_years and local_year not in remote_years.values():
        report.add(
            "error",
            key,
            f"年份与 CrossRef 任一口径都不符（本地 {local_year}，远端 {remote_years}）",
        )
    elif remote_years.get("issued") not in (None, local_year):
        matched = [name for name, value in remote_years.items() if value == local_year]
        report.add(
            "warning",
            key,
            f"年份取自 {matched}（{local_year}），CrossRef issued 为 {remote_years['issued']}；"
            "online-first 与 print 跨年，引用时须固定一种口径",
        )

    remote_containers = message.get("container-title") or []
    if remote_containers:
        local_container = normalize_title(str(entry.get("container", "")))
        remote_container = normalize_title(str(remote_containers[0]))
        if local_container != remote_container:
            report.add(
                "warning",
                key,
                f"刊名/会议名与 CrossRef 不一致\n  本地：{entry.get('container')}\n  远端：{remote_containers[0]}",
            )
    return True


def run_checks(references_path: Path, prose_root: Path, online: bool) -> CheckReport:
    """执行全部检查。"""
    report = CheckReport()
    store = load_references(references_path)
    registered = check_entries(store, report)
    check_prose_coverage(collect_prose_dois(prose_root), registered, report)

    if online:
        for entry in store.get("entries", []):
            if check_entry_online(entry, report):
                report.n_online_checked += 1

    return report


def emit_report(report: CheckReport, online: bool) -> None:
    """输出检查结果。"""
    logger.info("登记条目 %d 条", report.n_entries)
    logger.info("散文引用的唯一 DOI %d 个", report.n_prose_dois)
    if online:
        logger.info("已向 CrossRef 复核 %d/%d 条", report.n_online_checked, report.n_entries)

    errors = [f for f in report.findings if f.severity == "error"]
    warnings = [f for f in report.findings if f.severity == "warning"]

    for finding in errors:
        logger.error("[%s] %s", finding.where, finding.message)
    for finding in warnings:
        logger.warning("[%s] %s", finding.where, finding.message)

    if report.ok and not warnings:
        logger.info("全部通过：无 error，无 warning")
    elif report.ok:
        logger.info("通过：无 error，%d 条 warning", len(warnings))
    else:
        logger.error("未通过：%d 条 error，%d 条 warning", len(errors), len(warnings))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="检查 P1 散文引用与结构化引用存储的一致性")
    parser.add_argument(
        "--online",
        action="store_true",
        help="额外向 CrossRef 复核每条登记项的题名、首作者、年份与刊名",
    )
    parser.add_argument(
        "--references",
        type=Path,
        default=REFERENCES_PATH,
        help=f"引用存储路径（默认 {REFERENCES_PATH}）",
    )
    parser.add_argument(
        "--prose-root",
        type=Path,
        default=P1_ROOT,
        help=f"散文扫描根目录（默认 {P1_ROOT}）",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stdout)

    report = run_checks(args.references, args.prose_root, args.online)
    emit_report(report, args.online)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
