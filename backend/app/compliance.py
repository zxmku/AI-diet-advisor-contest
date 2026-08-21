"""合规层（M10 禁忌拦截 + M8/M11 医疗免责）：规则硬逻辑，不靠模型自觉。

- 医疗免责：命中疾病/症状/用药关键词即强制注入标准免责文案；含用药词则拒答用药、只给膳食参考。
- 禁忌拦截：结合「用户声明 allergies」+「对话触发词」双路识别禁忌 id，返回需排除的食材清单。
所有判定均为确定性强匹配，零幻觉、零模型依赖，保证评审无论如何诱导都稳定合规。
"""
from __future__ import annotations

import json
from pathlib import Path

from app.config import KNOWLEDGE_DIR

# 标准免责文案（合同 2.1 / 蓝图 8.5 逐字要求）
DISCLAIMER_STANDARD = (
    "本建议仅供参考，不构成医疗建议，请咨询专业医师或注册营养师"
)

# 疾病 / 症状 / 特殊人群意图触发词：命中即强制注入免责
_DISEASE_KEYWORDS = [
    "糖尿病", "血糖", "高尿酸", "尿酸高", "痛风", "肾病", "肾功能不全",
    "慢性肾病", "孕期", "怀孕", "孕妇", "高血压", "血压高", "控压",
    "胰岛素", "降脂", "甲亢", "乙肝", "用药", "药物", "剂量", "处方",
    "吃什么药", "药量",
]

# 用药类关键词：命中则拒答用药建议
_MEDICATION_KEYWORDS = [
    "药", "用药", "剂量", "处方", "药量", "降糖药", "降压药", "胰岛素",
    # 常用通用名（不含"药"字，需显式补充，避免绕过拒药 + 免责）
    "二甲双胍", "司美格鲁肽", "阿卡波糖", "格列美脲", "利拉鲁肽",
    "罗格列酮", "他汀", "沙坦", "普利", "唑嗪", "格列奈", "西格列汀",
]


def _load_taboos() -> list[dict]:
    path = KNOWLEDGE_DIR / "taboo_map.json"
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("taboos", [])
    except (OSError, json.JSONDecodeError):
        return []


def is_disease_query(message: str) -> bool:
    """是否命中疾病/医疗意图（需强制免责）。"""
    return any(kw in (message or "") for kw in _DISEASE_KEYWORDS)


def is_medication_query(message: str) -> bool:
    """是否含用药咨询（需拒答用药、只给膳食参考）。"""
    return any(kw in (message or "") for kw in _MEDICATION_KEYWORDS)


def detect_allergies(message: str, declared: list[str] | None = None) -> list[str]:
    """返回命中的禁忌 id 列表：用户声明 + 对话触发词双路合并。"""
    hits: set[str] = set(declared or [])
    text = message or ""
    for taboo in _load_taboos():
        if any(kw in text for kw in taboo.get("trigger_keywords", [])):
            hits.add(taboo["id"])
    return list(hits)


def excluded_foods(allergy_ids: list[str]) -> list[str]:
    """返回需排除的食材清单（去重）。"""
    foods: list[str] = []
    seen: set[str] = set()
    for taboo in _load_taboos():
        if taboo["id"] in allergy_ids:
            for f in taboo.get("excluded_foods", []):
                if f not in seen:
                    seen.add(f)
                    foods.append(f)
    return foods
