"""AI 自动维护用户档案（UserProfile）：users.profile_json 的读写 + 对话规则检测。

背景（产品决策）：用户档案 UI 端不再允许人工编辑，改由 AI（系统）从对话中自动识别
并写入。写入前判断「档案里已有该字段就不写（保留用户先说的），没有才写入」——
字段级去重，绝不覆盖用户更早的陈述。

红线约束（本模块只做记录，绝不越权）：
- 档案**不参与任何合规判定**：不新增免免责/免拒药通道；
- 不覆盖 sessions.allergies / goal_tag（本模块只读写 users.profile_json）；
- 档案**不注入 LLM 上下文**（守卫 3 保持，LLM 只看到检索块 + 合规层产物）。

检测策略：关键词/正则规则（第一人称陈述才提取，问句/第三人称不提取），
零 LLM 成本、稳定可测。规则刻意从简、可解释，不追求 NLP 全覆盖。
"""
from __future__ import annotations

import json
import logging
import re

from app import models
from app.database import SessionLocal

logger = logging.getLogger("healthpick.user_profile")

# ── 档案字段白名单：仅允许写入这些键，杜绝任意键污染 ──
PROFILE_FIELDS: tuple[str, ...] = ("nickname", "pregnancy", "taste", "meal_style", "exercise")

# ── 问句提示词：命中即视为问句，不做档案提取（只匹配陈述） ──
_QUESTION_HINTS: tuple[str, ...] = ("吗", "呢", "？", "?", "怎么样", "怎样", "为什么", "为啥")

# ── 第三人称/他人前缀：涉及他人（朋友/家人等）的陈述不写进本人档案 ──
_THIRD_PARTY_PATTERNS: tuple[str, ...] = (
    "我朋友", "我同事", "我同学", "我邻居", "我家人",
    "我妈妈", "我妈", "我爸爸", "我爸",
    "我姐姐", "我妹妹", "我姐", "我妹", "我哥哥", "我弟弟", "我哥", "我弟",
    "我老婆", "我老公", "我妻子", "我丈夫",
    "我孩子", "我儿子", "我女儿", "我宝宝",
)

# ── 昵称黑名单：「我是孕妇/我是孕早期/我是学生」是自我描述，不是名字，禁止写入 ──
_NICKNAME_BLACKLIST: frozenset[str] = frozenset({
    "孕妇", "孕早期", "孕中期", "孕晚期", "孕期", "学生", "上班族", "宝妈", "奶爸", "哺乳期",
    "减脂", "增肌", "调理", "过敏", "糖尿病", "高血压", "高血糖",
    "肠胃", "便秘", "失眠", "感冒",
})

# ── 规则（按团队规格原文）：nickname / pregnancy / taste / meal_style / exercise ──
_NICKNAME_RE = re.compile(
    r"(?:我是|我叫|叫我|可以叫我|大家都叫我)[（(]?([\u4e00-\u9fa5A-Za-z0-9]{1,8})[）)]?"
)
_PREGNANCY_KW: tuple[str, ...] = (
    "怀孕", "孕期", "孕早期", "孕中期", "孕晚期", "孕妇", "有宝宝了", "要当妈妈",
)
# 喜欢/爱吃 → 「爱吃X」；不吃 → 「不吃X」
_TASTE_RE = re.compile(r"(?:喜欢(?:吃)?|爱吃)(辣|甜|咸|清淡|重口味|酸)|不吃(辣|甜|香菜|海鲜)")
_MEAL_STYLE_RE = re.compile(
    r"(?:主食|平时)喜欢(?:吃)?(米饭|面食|面条)|素食|吃素|不吃肉|爱吃肉|肉食"
)
_EXERCISE_RE = re.compile(r"(?:每天|经常|平时)(?:跑步|健身|运动|锻炼|走路)|不运动|久坐|运动量(?:大|小)")

_MEAL_STYLE_VALUE_MAP: dict[str, str] = {
    "素食": "素食",
    "吃素": "素食",
    "不吃肉": "不吃肉",
    "爱吃肉": "爱吃肉",
    "肉食": "爱吃肉",
}


# ── 档案读写：get / set（字段级去重）/ 批量 ────────────────────────────
def _parse_profile(raw: str | None) -> dict:
    """解析 profile_json 字符串为 dict；空串/损坏 JSON 一律返回 {}。"""
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except (TypeError, ValueError):
        logger.warning("profile_json 解析失败，按空档案处理: %r", raw)
        return {}


def get_profile(user_id: str) -> dict:
    """读 users.profile_json 解析成 dict（空/损坏返回 {}）。只读，零副作用。"""
    db = SessionLocal()
    try:
        user = db.get(models.User, user_id)
        if user is None:
            return {}
        return _parse_profile(user.profile_json)
    except Exception:  # noqa: BLE001
        logger.exception("读取用户档案失败（降级为空档案）: user=%s", user_id)
        return {}
    finally:
        db.close()


def set_profile_field(user_id: str, field: str, value: str) -> bool:
    """写入单个档案字段（字段级去重）：已有非空值则不覆盖，返回 False；否则写入返回 True。

    - 读现有 dict → 若 field 已有非空值 → 返回 False（保留用户先说的，不覆盖）；
    - 否则写入该 field → 保存 JSON 回库 → 返回 True。
    - 用户不存在时按需创建（与 _persist_turn 同一套兜底）。
    - 任何异常返回 False，绝不打断主流程。
    """
    if not value or field not in PROFILE_FIELDS:
        return False
    db = SessionLocal()
    try:
        user = db.get(models.User, user_id)
        if user is None:
            user = models.User(id=user_id, nickname=None)
            db.add(user)
            db.flush()
        profile = _parse_profile(user.profile_json)
        existing = profile.get(field)
        if existing is not None and str(existing).strip():
            return False  # 已有非空值：保留用户先说的
        profile[field] = str(value).strip()
        user.profile_json = json.dumps(profile, ensure_ascii=False)
        db.commit()
        return True
    except Exception:  # noqa: BLE001
        db.rollback()
        logger.exception("用户档案字段写入失败（非致命）: user=%s field=%s", user_id, field)
        return False
    finally:
        db.close()


def add_profile_fields(user_id: str, updates: dict) -> dict:
    """批量调 set_profile_field（每字段独立去重），返回实际写入的字段集。

    例：updates={"pregnancy": "孕期", "taste": "爱吃辣"}，若 pregnancy 已有值
    而 taste 没有 → 返回 {"taste": "爱吃辣"}（pregnancy 保留旧值不覆盖）。
    """
    written: dict[str, str] = {}
    for field, value in updates.items():
        if set_profile_field(user_id, field, str(value)):
            written[field] = str(value)
    return written


# ── 对话规则检测：只提取第一人称陈述 ───────────────────────────────────
def _is_first_person_statement(text: str) -> bool:
    """是否第一人称陈述（非问句、非第三人称）。"""
    if not text:
        return False
    if any(q in text for q in _QUESTION_HINTS):
        return False  # 问句不提取
    if any(p in text for p in _THIRD_PARTY_PATTERNS):
        return False  # 涉及他人，不写本人档案
    return any(p in text for p in ("我", "本人"))


def _taste_value(match: re.Match) -> str:
    """taste 取值：第一分支（喜欢/爱吃）→ 爱吃X；第二分支（不吃）→ 不吃X。"""
    if match.group(1):
        return "爱吃" + match.group(1)
    return "不吃" + match.group(2)


def _meal_style_value(text: str, match: re.Match) -> str:
    """meal_style 取值：主食/平时喜欢+主食 → 爱吃X；其余裸词映射为规范说法。"""
    if match.group(1):
        return "爱吃" + match.group(1)
    return _MEAL_STYLE_VALUE_MAP.get(match.group(0), match.group(0))


def detect_user_profile(message: str) -> dict:
    """从消息中用关键词/正则规则提取用户属性（仅第一人称陈述）。

    返回 dict（可能为空），键 ∈ {nickname, pregnancy, taste, meal_style, exercise}。
    规则稳定零成本，只做档案记录；不参与合规判定、不注入 LLM 上下文。
    """
    text = (message or "").strip()
    if not _is_first_person_statement(text):
        return {}

    profile: dict = {}

    # nickname：「我是小优」「叫我小优」「可以叫我小优」
    m = _NICKNAME_RE.search(text)
    if m and m.group(1) not in _NICKNAME_BLACKLIST:
        profile["nickname"] = m.group(1)

    # pregnancy：怀孕/孕期/孕妇… → 孕期
    if any(kw in text for kw in _PREGNANCY_KW):
        profile["pregnancy"] = "孕期"

    # taste：喜欢/爱吃/不吃 + 口味词 → 爱吃辣/不吃香菜 等
    m = _TASTE_RE.search(text)
    if m:
        profile["taste"] = _taste_value(m)

    # meal_style：主食偏好/素食/肉食 → 爱吃面食/素食 等
    m = _MEAL_STYLE_RE.search(text)
    if m:
        profile["meal_style"] = _meal_style_value(text, m)

    # exercise：每天/经常/平时+运动词，或 不运动/久坐/运动量大(小) → 原文
    m = _EXERCISE_RE.search(text)
    if m:
        profile["exercise"] = m.group(0)

    return profile
