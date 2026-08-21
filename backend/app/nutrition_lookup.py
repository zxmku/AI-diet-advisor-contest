"""确定性营养数值查询工具（BUG-5 修复 / 团长「数值必须工具计算」指令）。

设计目标：
- 用户问「<食材> 多少/几 <营养指标>」这类数值问题时，必须返回知识库表格中的
  **精确权威值**，而不是依赖 BM25 模糊检索拼出的「方案块」（例如把鸡胸肉的 165
  千卡/100g 误答成 7 日食谱里的「午餐约 520 千卡」）。
- 完全脱离模型推理：读 core_nutrition_A.json 的 3.1~3.4 章节权威表，按归一化
  食材名做确定性匹配，返回与表格逐字一致的数值。

对外 API：
- is_numeric_lookup_query(query) -> (bool, food|None)
- lookup(food) -> dict|None
- format_reply(row, metric=None) -> str
- get_table_markdown(section) -> str|None   （供 UI 来源溯源）
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger("healthpick.nutrition_lookup")

# ── 路径：优先使用 app.config.KNOWLEDGE_DIR， standalone 时回退 ──
try:  # 作为 app 包的一部分被导入（PYTHONPATH=backend）
    from app import config

    _KNOWLEDGE_DIR = config.KNOWLEDGE_DIR
except Exception:  # noqa: BLE001  # 独立运行（cwd=backend/app）时直接 import config
    try:
        import config as _config  # type: ignore

        _KNOWLEDGE_DIR = _config.KNOWLEDGE_DIR
    except Exception:  # noqa: BLE001  # 终极兜底：相对本文件推算
        _KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent.parent / "knowledge"

_JSON_PATH = Path(_KNOWLEDGE_DIR) / "core_nutrition_A.json"

# 数值字段 → (展示标签, 单位)
_FIELD_META: dict[str, tuple[str, str]] = {
    "kcal": ("热量", "千卡"),
    "protein": ("蛋白质", "克"),
    "fat": ("脂肪", "克"),
    "carb": ("碳水化合物", "克"),
    "fiber": ("膳食纤维", "克"),
    "cholesterol": ("胆固醇", "毫克"),
    "calcium": ("钙", "毫克"),
    "gi": ("GI值", ""),
    "vitc": ("维生素C", "毫克"),
}

# 表头单元格（去空白后按子串） → 字段名
_HEADER_MAP: dict[str, str] = {
    "热量": "kcal",
    "蛋白质": "protein",
    "脂肪": "fat",
    "胆固醇": "cholesterol",
    "钙": "calcium",
    "碳水化合物": "carb",
    "膳食纤维": "fiber",
    "GI": "gi",
    "维生素C": "vitc",
    "推荐人群": "audience",
    "推荐做法": "method",
}

# 仅取这 4 个权威速查章节
_TARGET_SECTIONS = ("3.1", "3.2", "3.3", "3.4")

# 数值问答可识别的指标词（长词优先，避免「卡」误命中「千卡/卡路里」）
_METRICS: list[str] = [
    "卡路里", "千卡", "kcal", "热量", "蛋白质", "脂肪", "碳水", "克", "g", "卡",
]

# 疑问/停用词（提取食材前的噪声）：长词优先
_QUERY_STOP: set[str] = {
    "多少", "几", "吃", "吗", "的", "能", "有", "是", "怎么", "如何", "什么",
    "可以", "想", "要", "该", "这", "那", "我", "你", "它", "一个", "了", "啊",
    "呢", "呀", "请问", "问", "下", "一下", "知道", "了解", "查询", "查", "告诉",
    "看", "啥", "些", "里", "中", "上", "对", "为", "与", "和", "及", "或",
}

NUTRITION_TABLE: dict[str, dict] = {}
_SECTION_CONTENT: dict[str, str] = {}

# ── 健康自检状态（BUG-5 加固：让「静默降级」变为可观测）──
# _NUTRITION_READY: 营养速查表是否成功解析出权威行；False 表示 KB 缺失/损坏/格式不符，
#         数值问答将降级到模糊 BM25（precision 受损但不崩溃）。
# _NUTRITION_ROWS:  当前已加载的权威行数（3.1~3.4）。
# _STATUS_DETAIL: 人类可读的状态详情，供 /health 与启动日志观测。
_NUTRITION_READY: bool = False
_NUTRITION_ROWS: int = 0
_STATUS_DETAIL: str = ""


def _normalize_food(name: str) -> str:
    """归一化食材名：去掉全角/半角括号里的限定词（如「（去皮）」），并去首尾空白。

    「鸡胸肉（去皮）」 → 「鸡胸肉」；展示名保留原全称。
    """
    s = re.sub(r"[（(][^）)]*[）)]", "", name or "")
    return s.strip()


def _to_number(value: str) -> float | int | None:
    """把单元格文本转成数值；空/非数字返回 None。整型源存 int，带小数点存 float。"""
    s = (value or "").strip()
    if s == "" or s == "-":
        return None
    try:
        return int(s) if "." not in s else float(s)
    except ValueError:
        return None


def _map_header(cell: str) -> str | None:
    """表头单元格 → 字段名（去空白后子串匹配）。"""
    norm = cell.replace(" ", "")
    for key, field in _HEADER_MAP.items():
        if key in norm:
            return field
    return None


def _build_row(cells: list[str], header_fields: list[str | None], chapter: str, section: str) -> dict | None:
    """把一行数据单元格按 header 映射成结构化行；首列为食材名。"""
    if not cells or cells[0].replace(" ", "") == "食材":
        return None
    display_name = cells[0].strip()
    key = _normalize_food(display_name)
    if not key:
        return None
    row: dict[str, Any] = {
        "key": key,
        "display_name": display_name,
        "source_chapter": chapter,
        "source_section": section,
    }
    for field, cell in zip(header_fields, cells):
        if field is None:
            continue
        if field in ("audience", "method"):
            row[field] = cell.strip()
        else:
            row[field] = _to_number(cell)
    return row


def _parse_table_chunk(content: str, chapter: str, section: str) -> tuple[list[dict], str]:
    """解析一个 markdown 表格块（可能含重复表头，如 3.1 肉类+蛋两类合并）。

    返回 (rows, raw_markdown)。分隔行「|---|」与重复表头行均被跳过/重新映射。
    """
    lines = [ln.strip() for ln in (content or "").split("\n") if ln.strip().startswith("|")]
    header_fields: list[str | None] | None = None
    rows: list[dict] = []
    for line in lines:
        cells = [c.strip() for c in line.strip("|").split("|")]
        # 分隔行：全部由 - : 空格 组成
        if cells and all(re.fullmatch(r"[:\- ]*", c) for c in cells):
            continue
        # 表头行：首列为「食材」
        if cells and cells[0].replace(" ", "") == "食材":
            header_fields = [_map_header(c) for c in cells]
            continue
        # 数据行
        if header_fields is None:
            continue
        row = _build_row(cells, header_fields, chapter, section)
        if row:
            rows.append(row)
    return rows, content or ""


def _load() -> None:
    """导入时加载 3.1~3.4 权威表到内存表。

    BUG-5 加固：任何异常都「优雅降级」（不阻断服务启动），但**不再静默吞掉**——
    加载失败/空表会写入 _NUTRITION_READY/_STATUS_DETAIL 并经 logger.warning 暴露，/health
    端点与启动自检可一眼发现，避免数值问答 precision 被悄悄牺牲。
    """
    global _NUTRITION_READY, _NUTRITION_ROWS, _STATUS_DETAIL
    # 重置已加载状态，使 _load() 可重复调用（测试/重载场景）且幂等。
    NUTRITION_TABLE.clear()
    _SECTION_CONTENT.clear()
    _NUTRITION_READY = False
    _NUTRITION_ROWS = 0
    _STATUS_DETAIL = ""
    try:
        with open(_JSON_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("营养速查表加载失败: %s", exc)
        _NUTRITION_READY = False
        _NUTRITION_ROWS = 0
        _STATUS_DETAIL = f"load_error: {exc}"
        return
    for chunk in data.get("chunks", []):
        section = chunk.get("section") or ""
        if not section.startswith(_TARGET_SECTIONS):
            continue
        content = chunk.get("content", "")
        _SECTION_CONTENT[section] = content
        rows, _ = _parse_table_chunk(content, chunk.get("chapter", ""), section)
        for r in rows:
            NUTRITION_TABLE.setdefault(r["key"], r)
    _NUTRITION_ROWS = len(NUTRITION_TABLE)
    if _NUTRITION_ROWS == 0:
        _NUTRITION_READY = False
        _STATUS_DETAIL = "empty: no rows parsed from 3.1-3.4"
        logger.warning("营养速查表加载失败: %s", _STATUS_DETAIL)
    else:
        _NUTRITION_READY = True
        _STATUS_DETAIL = "ok"


_load()


def is_numeric_lookup_query(query: str) -> tuple[bool, str | None]:
    """判断是否为「<食材> 多少/几 <营养指标>」数值查询。

    指标词 ∈ {千卡,卡路里,卡,kcal,热量,蛋白质,脂肪,碳水,克,g}。
    命中则返回 (True, 食材token)；否则 (False, None)。
    """
    if not query:
        return (False, None)
    q = query
    metric = None
    for m in sorted(_METRICS, key=len, reverse=True):
        if m.lower() in q.lower():
            metric = m
            break
    if metric is None:
        return (False, None)
    idx = q.lower().find(metric.lower())
    prefix = q[:idx]
    food = _extract_food(prefix)
    if not food:
        return (False, None)
    return (True, food)


def _extract_food(prefix: str) -> str:
    """从指标词前的前缀里剔除疑问/停用词与指标词，得到食材 token。"""
    p = prefix
    for st in sorted(_QUERY_STOP, key=len, reverse=True):
        p = p.replace(st, "")
    for m in sorted(_METRICS, key=len, reverse=True):
        p = p.replace(m, "")
    p = re.sub(r"[\s，。、？！,.;:：（）()\[\]【】\"'“”]", "", p)
    return p.strip()


def lookup(food: str) -> dict | None:
    """按归一化食材名做包含匹配（query⊆key 或 key⊆query），返回首个命中行。

    返回行含：key/display_name/source_chapter/source_section 及全部解析字段。
    """
    q = _normalize_food(food)
    if not q:
        return None
    for key, row in NUTRITION_TABLE.items():
        if q in key or key in q:
            return row
    return None


def _fmt_num(value: float | int | None) -> float | int | None:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def format_reply(row: dict, metric: str | None = None) -> str:
    """把命中行格式化为带溯源标题的确定性文本（展示全称 + 完整一行数值）。"""
    chapter = row.get("source_chapter", "")
    section = row.get("source_section", "")
    name = row.get("display_name", "")
    header = f"【{chapter} · {section}】"

    parts: list[str] = []
    for field, (label, unit) in _FIELD_META.items():
        if field in row and row[field] is not None:
            val = _fmt_num(row[field])
            parts.append(f"{label} {val} {unit}".strip() if unit else f"{label} {val}")

    extra: list[str] = []
    if row.get("audience"):
        extra.append(f"（推荐人群：{row['audience']}）")
    if row.get("method"):
        extra.append(f"（推荐做法：{row['method']}）")

    body = f"{name}：每100克可食部约 " + "，".join(parts) + "。" + "".join(extra)
    return f"{header}\n{body}"


def get_table_markdown(section: str) -> str | None:
    """返回某章节表格块的原文 markdown（供 UI 来源溯源）。"""
    return _SECTION_CONTENT.get(section)


def is_nutrition_table_ready() -> bool:
    """营养速查表是否就绪（可被 /health 与启动自检观测）。

    返回 False 表示 KB 缺失/损坏/格式不符导致未解析出任何权威行，数值问答将
    降级到模糊 BM25 检索（precision 受损），但服务不会崩溃。
    """
    return _NUTRITION_READY


def nutrition_table_status() -> dict:
    """返回营养速查表加载状态快照（供 /health 端点与日志观测）。

    返回: {"ready": bool, "rows": int, "detail": str}
    """
    return {"ready": _NUTRITION_READY, "rows": _NUTRITION_ROWS, "detail": _STATUS_DETAIL}
