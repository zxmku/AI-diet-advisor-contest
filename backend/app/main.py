"""FastAPI 入口（M3 API 网关）：注册全部 6 个端点 + 全局中间件。

当前为「检索 grounding + 规则合规」实现（local-rules 降级模式）：
- 回答 = BM25 检索素材 A/B 原文块（零编造、带来源标注）；
- 合规层（compliance.py）硬注入医疗免责 + 拒答用药 + 禁忌排除；
- 多轮对话落库（MOD-04），history 真实可读；
- 推荐/快捷基于素材 B 真实方案生成理由（真实可读，非示意文案）；
- P3 一餐生成：命中一餐意图（晚餐/今晚/给我做…）时由 _build_meal 确定性出餐——
  数值（热量/蛋白/碳水/克数）仅来自营养速查表真实值 × 份量，绝不编造，天然剔除禁忌；
- P5 饮食管理 + 情感陪伴：命中记录/查询/陪伴意图时互斥优先返回——记录（我吃了…）落
  DietLog 台账并按速查表估算热量，查询（最近吃了什么）汇总最近 5 条，陪伴（一个人吃饭/
  好累/没胃口）先共情再轻建议，LLM 失败降级本地模板；
- 未配置 DeepSeek Key 时以上述规则兜底，配置后可由 app/llm.py 升级为 LLM 合成（M13）。
"""
from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Sequence
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from jieba import lcut
from pathlib import Path
from pydantic import BaseModel, Field

from app import config
from app.api.schemas import (
    ChatRequest,
    QuickRequest,
    RecommendRequest,
    SessionRequest,
    UnifiedResponse,
    UserCreateRequest,
    make_response,
)
from app.compliance import (
    ALLERGY_FOLLOWUP,
    DISCLAIMER_STANDARD,
    _load_taboos,
    detect_allergies,
    excluded_foods,
    is_dietary_domain,
    is_disease_query,
    is_medication_query,
    strip_disease_negations,
)
from app.cost_gate import cost_gate
from app.database import SessionLocal, init_db
from app import models
from app.retrieval import PLATFORM_HINTS, SourceChunk, get_retriever
from app import nutrition_lookup
from app import health_state
from app import decision_tool
from app import user_profile
from app import llm
from app.middleware.guard import (
    RateLimitMiddleware,
    SecurityHeadersMiddleware,
    register_exception_handlers,
    validate_message,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("healthpick.main")

_is_prod = config.ENV == "prod"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """生命周期：建表、预热检索索引（替代已弃用的 on_event 钩子）。"""
    init_db()
    logger.info("HealthPick API 启动完成（env=%s）", config.ENV)
    try:
        get_retriever()  # 预热 BM25 索引，提前暴露知识库加载异常
        logger.info("知识库检索索引预热完成")
    except Exception as exc:  # noqa: BLE001
        logger.error("知识库索引预热失败（检索降级）: %s", exc)
    # BUG-5 加固：营养速查表未就绪时启动告警（但不阻断启动，优雅降级）。
    if not nutrition_lookup.is_nutrition_table_ready():
        logger.warning("营养速查表未就绪: %s", nutrition_lookup.nutrition_table_status()["detail"])
    yield


app = FastAPI(
    title="HealthPick AI 智能膳食顾问 API",
    version="0.1.0",
    description="健康优选 7×24 小时 AI 膳食顾问后端服务",
    lifespan=lifespan,
    docs_url=None if _is_prod else "/docs",
    redoc_url=None if _is_prod else "/redoc",
    openapi_url=None if _is_prod else "/openapi.json",
)

# ── 中间件（安全响应头最外层 → CORS → 限流）──
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Session-Id", "X-User-Id"],
)
app.add_middleware(RateLimitMiddleware)
register_exception_handlers(app)


@app.get("/health")
def health() -> dict:
    """健康检查：存活 + 关键配置自检（不回显任何密钥）。"""
    try:
        from app import tts_service
        _tts_ok = tts_service.tts_available()
    except Exception:  # noqa: BLE001
        _tts_ok = False
    return {
        "status": "ok",
        "llm_key_set": bool(config.DEEPSEEK_API_KEY),
        "llm_model": config.DEEPSEEK_MODEL if config.DEEPSEEK_API_KEY else None,
        "database": "sqlite" if config.DATABASE_URL.startswith("sqlite") else "external",
        # BUG-5 加固：暴露营养速查表就绪状态，使静默降级可被一眼观测。
        "nutrition_table_ready": nutrition_lookup.is_nutrition_table_ready(),
        "nutrition_table_rows": nutrition_lookup.nutrition_table_status()["rows"],
        # M17 成本闸门：仅当 LLM 启用时暴露成本状态，未启用为 None。
        "llm_cost": cost_gate.status() if llm.is_enabled() else None,
        # 语音输出健康状态：edge-tts 可用性（前端指示灯：绿=健康/黑=不健康）。
        "tts_available": _tts_ok,
    }


# ── 快捷操作 → 人群标签映射（素材 B 三套方案）──
_ACTION_GOAL_MAP: dict[str, str | None] = {
    "lose_fat": "减脂",
    "gain_muscle": "增肌",
    "control_sugar": "调理",
    "today_meal": None,
}

# 三套方案的结构化字段（名称/热量区间/代表食材来自素材 B 核对，绝不混 C）
_GOAL_PLAN_MAP: dict[str, dict] = {
    "减脂": {
        "name": "轻盈减脂方案",
        "kcal_range": "1200-1500 千卡（女性）/ 1500-1800 千卡（男性）",
        "foods": ["鸡胸肉（去皮）", "糙米", "西兰花", "鳕鱼", "燕麦"],
    },
    "增肌": {
        "name": "力量增肌方案",
        "kcal_range": "2500-3000 千卡（训练日）/ 2200-2600 千卡（休息日）",
        "foods": ["鸡胸肉（去皮）", "牛里脊", "鸡蛋（全蛋）", "白米饭", "希腊酸奶（无糖）"],
    },
    "调理": {
        "name": "稳糖调理方案",
        "kcal_range": "按控糖 321 餐盘执行，低 GI 碳水约拳头大小",
        "foods": ["糙米", "燕麦", "红薯", "藜麦", "绿叶菜"],
    },
}

# P2 人群目标关键词 → goal_tag（与 _GOAL_PLAN_MAP 三套方案一致；控糖/血糖/稳糖→调理）
_GOAL_KEYWORDS: dict[str, tuple[str, ...]] = {
    "减脂": ("减脂", "减肥", "瘦身", "减重", "燃脂", "掉秤", "瘦下来"),
    "增肌": ("增肌", "增重", "长肌肉", "练肌肉", "力量训练"),
    "调理": ("控糖", "血糖", "稳糖", "糖尿病", "高血糖", "血糖高", "糖友", "慢病", "三高"),
}

# DEFECT-B 修复：纯请求语义关键词——命中说明是「求方案/要推荐」而非个人目标声明；
# 与「我的/本人」（裸"我"不算，避免「给我/帮我」这类请求被误判为个人声明）或健康状态
# 表述同现时才视为个人声明（避免把「推荐减脂方案」这类请求语义误判为个人目标，从而
# 覆盖已有 goal_tag）。
_REQUEST_KEYWORDS: tuple[str, ...] = ("推荐", "方案", "食谱", "计划", "怎么吃", "给我")
_PERSONAL_PRONOUNS: tuple[str, ...] = ("我的", "本人")
_HEALTH_STATE_PHRASES: tuple[str, ...] = (
    "血糖高", "血糖偏高", "高血糖", "糖尿病", "三高", "血压高", "尿酸高",
    "减肥中", "瘦身中", "在减肥", "想减肥", "想减脂", "在减脂", "要减脂",
    "想增肌", "在增肌", "增肌中", "想控糖",
)


def _detect_goal(message: str) -> str | None:
    """从消息中识别人群目标（减脂/增肌/调理）；未命中返回 None，不覆盖既有画像。

    DEFECT-B 修复：纯子串匹配会把「推荐减脂方案」这类**请求语义**误判为个人目标，
    从而覆盖已有 goal_tag。此处仅当消息含个人意图/健康状态声明时才判定：
    - 命中「我的/本人」（裸"我"不算，避免「给我/帮我」被误判）且非纯请求词
      （推荐/方案/食谱/计划/怎么吃/给我）→ 个人声明；
    - 命中健康状态表述（血糖高/血糖偏高/想增肌/想控糖等）→ 个人声明；
    - 纯请求词命中且无个人/健康声明（如「推荐减脂方案」「给我推荐减脂方案」）
      → 一律返回 None，不写库。
    """
    text = message or ""
    request_hit = any(rk in text for rk in _REQUEST_KEYWORDS)
    personal_claim = any(p in text for p in _PERSONAL_PRONOUNS)
    health_claim = any(h in text for h in _HEALTH_STATE_PHRASES)
    # 纯请求语义：用户在「要方案/求推荐」而非陈述自身状态 → 不算目标声明
    if request_hit and not personal_claim and not health_claim:
        return None
    for goal, keywords in _GOAL_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return goal
    return None


# ── P3 一餐意图：触发词 / 目标追问话术 / 餐次识别 ──
# V08 修复：补 早上吃/中午吃/下午吃/晚上吃（「晚上吃什么」「中午吃什么」此前漏判成闲聊）。
# 注意用「时段+吃」而非裸时段词：裸「晚上/中午/早上/下午」会把「晚上好」等问候
# 劫持进一餐追问目标分支（meal_goal_ask），属回归；「时段+吃」精准命中一餐请求。
# 二审 P1-7：裸「今晚」也从触发词中移除（「今晚天气怎么样」不得劫持成追问目标），
# 由 _is_meal_intent 的正则「今晚+餐次词同现」接管（今晚吃什么/今晚减脂餐仍命中）。
_MEAL_INTENT_TRIGGERS: tuple[str, ...] = (
    "早餐", "早饭", "午餐", "午饭", "晚餐", "晚饭", "夜宵",
    "今天吃什么", "一餐", "一顿", "搭一餐", "做一餐", "做一顿", "来一餐",
    "给我做", "帮我做", "想吃点啥", "吃啥", "吃点啥", "吃什么好",
    "早上吃", "中午吃", "下午吃", "晚上吃",
)

_MEAL_GOAL_ASK = "想减脂、增肌还是控糖？告诉我你的目标，我先帮你搭一餐。"

_MEAL_TIME_WORDS: dict[str, tuple[str, ...]] = {
    "早餐": ("早餐", "早饭", "早上"),
    "午餐": ("午餐", "午饭", "中午"),
    "晚餐": ("晚餐", "晚饭", "今晚", "晚上"),
}

# ── P3 一餐生成器：各人群槽位食材（均来自素材 B 方案/食谱/速查表，绝不混 C）──
# 主蛋白/碳水/蔬菜候选：优先取自 _GOAL_PLAN_MAP[goal] 方案食材，其余以素材 B 的
# 7 日循环食谱 / 控糖餐盘法则 / 食材替换指南为据（如减脂「虾仁/北豆腐」见 B 2.3）。
_MEAL_PROTEIN_FOODS: dict[str, list[str]] = {
    "减脂": ["鸡胸肉（去皮）", "鳕鱼", "虾仁"],
    "增肌": ["鸡胸肉（去皮）", "牛里脊", "鸡蛋（全蛋）"],
    "调理": ["鸡胸肉（去皮）", "鳕鱼", "北豆腐"],
}
_MEAL_CARB_FOODS: dict[str, list[str]] = {
    "减脂": ["糙米", "燕麦", "红薯"],
    "增肌": ["白米饭", "糙米", "燕麦"],
    "调理": ["糙米", "藜麦", "红薯"],
}
_MEAL_VEGGIE_FOODS: dict[str, list[str]] = {
    "减脂": ["西兰花", "菠菜", "番茄"],
    "增肌": ["西兰花", "菠菜", "番茄"],
    "调理": ["菠菜", "西兰花", "黄瓜"],
}
# 同族替换项（素材 B 2.4 食材替换指南 / 3.3 / 4.2 为据；全部在速查表有真实值）
_MEAL_PROTEIN_SWAPS: dict[str, list[str]] = {
    "减脂": ["鳕鱼", "虾仁", "北豆腐"],
    "增肌": ["牛里脊", "鸡蛋（全蛋）", "希腊酸奶（无糖）"],
    "调理": ["北豆腐", "鳕鱼", "鸡蛋（全蛋）"],
}
_MEAL_CARB_SWAPS: dict[str, list[str]] = {
    "减脂": ["燕麦", "红薯", "藜麦"],
    "增肌": ["糙米", "燕麦", "藜麦"],
    "调理": ["藜麦", "红薯", "燕麦"],
}
# 做法步骤模板：仅烹饪指导（无营养素数值），措辞源自素材 B 方案原文（香煎/清蒸/
# 焯水/控糖 321 餐盘等），来源标注指向检索到的 B 章节；蔬菜做法优先用速查表「推荐做法」。
_MEAL_METHOD_TEMPLATES: dict[str, list[str]] = {
    "减脂": [
        "{protein}切块腌制，平底锅加少量橄榄油，中小火煎至两面金黄（参考素材 B 香煎做法）。",
        "{carb}淘洗后加水蒸煮成杂粮饭（生重约 100 克），替代精白米面。",
        "{veggie}焯水后蒜蓉快炒或凉拌（参考速查表推荐做法）。",
    ],
    "增肌": [
        "{protein}切块腌制，平底锅加少量橄榄油煎熟，作为优质蛋白来源。",
        "{carb}加水蒸煮至熟，训练日可作为主力碳水补充能量。",
        "{veggie}焯水后快炒，搭配高蛋白餐补足膳食纤维。",
    ],
    "调理": [
        "{protein}清蒸或水煮，避免红烧、油炸等重油做法。",
        "{carb}选低 GI 主食，按「先吃蔬菜→再吃蛋白→最后吃碳水」的顺序进食。",
        "{veggie}凉拌或清炒，少盐烹调（每日食盐建议低于 5 克）。",
    ],
}


def _is_meal_intent(message: str) -> bool:
    """是否命中「一餐」意图（晚餐/午餐/早餐/今晚/给我做/想吃点啥 等）。

    二审 P1-7 修复：裸「今晚」不再直接判一餐——「今晚天气怎么样」「今晚有空吗」
    等闲聊会被劫持成 meal_goal_ask。改为「今晚」须与餐次词（吃/餐/饭/做/搭/点/
    宵夜）同现才命中；「今晚减脂餐/今晚吃什么/今晚吃啥」不受影响。
    """
    text = message or ""
    if any(t in text for t in _MEAL_INTENT_TRIGGERS):
        return True
    if "今晚" in text and re.search(r"今晚.{0,6}(?:吃|餐|饭|做|搭|点|宵夜)", text):
        return True
    return False


def _meal_goal(message: str, session_goal: str | None) -> str | None:
    """一餐目标解析：消息内目标词（纯子串，放宽到请求语义）优先，其次会话 goal。

    注意：这里不走 _detect_goal 的「个人声明」过滤——「给我做减脂晚餐」是请求语义，
    不应覆盖 session.goal_tag（DEFECT-B 逻辑照旧），但本餐仍按消息里的「减脂」来搭。
    """
    text = message or ""
    for goal, keywords in _GOAL_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return goal
    return session_goal


def _meal_time_from_message(message: str) -> str:
    """从消息识别餐次（早餐/午餐/晚餐）；未命中返回「正餐」。"""
    text = message or ""
    for meal_time, words in _MEAL_TIME_WORDS.items():
        if any(w in text for w in words):
            return meal_time
    return "正餐"


def _is_food_excluded(food: str, excluded_set: set[str]) -> bool:
    """食材是否命中禁忌清单（双向包含匹配，如「希腊酸奶（无糖）」命中「酸奶」）。"""
    norm = nutrition_lookup._normalize_food(food)
    for ex in excluded_set:
        ex_norm = nutrition_lookup._normalize_food(ex)
        if ex_norm and (ex_norm in norm or norm in ex_norm):
            return True
    return False


_REDACT_MARK = "（已按禁忌剔除）"
_REDACT_RUN = re.compile(r"（已按禁忌剔除）(?:[、/，,]*（已按禁忌剔除）)+")
# V04：「...」引号内容（营养速查表「暂未收录『食材』」等结构化引用），R3 剔除时须保护。
_QUOTED_RE = re.compile(r"「[^」]*」")


def _collapse_redaction_marks(text: str) -> str:
    """折叠连续的剔除标记（可被 、/，, 分隔），避免「（已按禁忌剔除）/（已按禁忌剔除）」观感。"""
    return _REDACT_RUN.sub(_REDACT_MARK, text)


def _redact_excluded(text: str, excluded: Sequence[str]) -> str:
    """R3 禁忌剔除：剔除回复正文中的禁忌食材名（长名优先，避免短名替换破坏长名）。

    V04 修复：剔除前先把「...」引号内容临时保护（\\x1fQ<随机>N\\x1f 标记），剔除完成后再还原——
    否则「表中暂未收录『坚果』」这类引号内的食材名会被替换成「（已按禁忌剔除）」，
    造成「『（已按禁忌剔除）』的精确营养数据」文本损坏。
    遗留①加固：引号保护标记用 C0 控制符 \\x1f + 随机串（validate_message 已剥离用户消息中的
    C0 控制符；即便绕过入口，随机串也使伪造标记无法命中），杜绝引号内容错位。
    """
    protected: dict[str, str] = {}

    def _hold_quoted(m: re.Match) -> str:
        token = f"\x1fQ{uuid.uuid4().hex[:8]}:{len(protected)}\x1f"
        protected[token] = m.group(0)
        return token

    out = _QUOTED_RE.sub(_hold_quoted, text)
    for f in sorted(excluded, key=len, reverse=True):
        # P0-3 修复：单字排除词（如 egg_allergy 补的「蛋」）不参与正文替换——
        # 否则「蛋白质」「蛋糕」等含「蛋」字词会被替换成「（已按禁忌剔除）白质」。
        # 单字词仅用于 _is_food_excluded 的食材匹配（候选名不含「蛋白质」等字样）。
        if len(f) < 2:
            continue
        out = out.replace(f, _REDACT_MARK)
    out = _collapse_redaction_marks(out)
    for token, original in protected.items():
        out = out.replace(token, original)
    return out


def _meal_method(
    goal: str,
    protein_food: str | None,
    carb_food: str | None,
    veggie_food: str | None,
) -> tuple[list[str], SourceChunk | None]:
    """做法步骤：取自素材 B 检索块（方案章节/食谱/餐盘）+ 速查表「推荐做法」。

    返回 (steps, chunk)。chunk 为方法来源的 B 检索块（用于来源标注）；未命中时
    chunk=None（步骤仍以方案模板给出，但不在 sources 里挂假章节）。
    """
    chunks = get_retriever().retrieve(f"{goal} 食谱 做法 烹饪", top_k=6)
    kw = _GOAL_CHAPTER_KW.get(goal, goal)
    best: SourceChunk | None = None
    # 优先含「食谱/餐盘/做法」的方案章节（如减脂 2.3 7 日循环食谱、调理 4.3 控糖餐盘）
    for c in chunks:
        if c.source == "B" and kw in (c.chapter or ""):
            if any(sec in (c.section or "") for sec in ("食谱", "餐盘", "做法")):
                best = c
                break
    if best is None:
        for c in chunks:
            if c.source == "B" and kw in (c.chapter or ""):
                best = c
                break

    template = _MEAL_METHOD_TEMPLATES.get(goal, _MEAL_METHOD_TEMPLATES["减脂"])
    steps = [
        t.format(
            protein=protein_food or "优质蛋白",
            carb=carb_food or "低 GI 主食",
            veggie=veggie_food or "时令蔬菜",
        )
        for t in template
    ]
    # 蔬菜做法优先采用速查表「推荐做法」列（素材 A 权威值，零编造）
    if veggie_food:
        row = nutrition_lookup.lookup(veggie_food)
        if row and row.get("method"):
            steps[-1] = f"{veggie_food}：{row['method']}（速查表推荐做法）。"
    return steps, best


def _build_meal(goal: str, excluded: list[str], meal_time: str = "正餐") -> dict:
    """确定性一餐生成器（P3）：主蛋白 + 主食碳水 + 蔬菜 + 好脂肪 + 总热量 + 做法 + 同族替换。

    红线「数值只来自素材」：
    - 每个食材的热量/蛋白/碳水/脂肪一律经 nutrition_lookup.lookup 取速查表真实值
      × 份量（克/100）计算，绝不由模型/模板推算；
    - 表外食材（如橄榄油）热量标注 None（按约计、不编造精确数值，不计入总热量）；
    - 禁忌排除：主蛋白/碳水/蔬菜/替换项全部剔除 excluded 中的食材；
    - 做法步骤与来源均来自知识库（B 方案章节 + A 速查表），绝不混 C 库。

    返回结构化 dict（响应契约 data.meal）：
    {name, items:[{category, food, grams, kcal}], total_kcal,
     macros:{protein,carbs,fat}, method:[...], swaps:{protein,carbs}, sources:[...]}
    """
    base = _GOAL_PLAN_MAP.get(goal) or _GOAL_PLAN_MAP["减脂"]
    excluded_set = set(excluded)

    def pick(candidates: Sequence[str]) -> str | None:
        for f in candidates:
            if not _is_food_excluded(f, excluded_set):
                return f
        return None

    def clean_swaps(candidates: Sequence[str]) -> list[str]:
        return [f for f in candidates if not _is_food_excluded(f, excluded_set)]

    protein = pick(_MEAL_PROTEIN_FOODS[goal])
    carb = pick(_MEAL_CARB_FOODS[goal])
    veggie = pick(_MEAL_VEGGIE_FOODS[goal])

    items: list[dict] = []
    total_kcal = 0.0
    macros: dict[str, float] = {"protein": 0.0, "carbs": 0.0, "fat": 0.0}
    meal_sources: list[dict] = []
    seen_sources: set[tuple[str, str]] = set()

    def add_item(category: str, food: str, grams: int) -> None:
        """取速查表真实值 × 份量计算；表外食材 kcal=None（按约计，不编造）。"""
        nonlocal total_kcal
        row = nutrition_lookup.lookup(food)
        kcal: float | None = None
        if row is not None:
            if row.get("kcal") is not None:
                kcal = round(float(row["kcal"]) * grams / 100, 1)
                total_kcal += kcal
            if row.get("protein") is not None:
                macros["protein"] += float(row["protein"]) * grams / 100
            if row.get("carb") is not None:
                macros["carbs"] += float(row["carb"]) * grams / 100
            if row.get("fat") is not None:
                macros["fat"] += float(row["fat"]) * grams / 100
            src_key = (row["source_chapter"], row["source_section"])
            if src_key not in seen_sources:
                seen_sources.add(src_key)
                meal_sources.append(
                    {"source": "A", "chapter": row["source_chapter"], "section": row["source_section"]}
                )
        items.append({"category": category, "food": food, "grams": grams, "kcal": kcal})

    # 结构：主蛋白（150g 级）+ 主食碳水（100g 级）+ 蔬菜（200g 级）+ 好脂肪（10g 级）
    if protein is not None:
        add_item("主蛋白", protein, 150)
    if carb is not None:
        add_item("主食碳水", carb, 100)
    if veggie is not None:
        add_item("蔬菜", veggie, 200)
    # 好脂肪：橄榄油不在速查表 → 热量按约计（kcal=None），绝不编造精确值
    add_item("好脂肪", "橄榄油", 10)

    method, method_chunk = _meal_method(goal, protein, carb, veggie)
    if method_chunk is not None:
        src_key = (method_chunk.chapter, method_chunk.section)
        if src_key not in seen_sources:
            meal_sources.append(
                {"source": method_chunk.source, "chapter": method_chunk.chapter, "section": method_chunk.section}
            )

    meal: dict = {
        "name": f"{base['name']}·{meal_time}",
        "goal": goal,
        "meal_time": meal_time,
        "items": items,
        "shopping_list": [
            {"food": it["food"], "grams": it.get("grams")}
            for it in items
            if it.get("food")
        ],  # 采购清单（P-迭代）：由一餐 items 派生，供前端一键展示"要买什么"
        "total_kcal": round(total_kcal, 1),
        "macros": {k: round(v, 1) for k, v in macros.items()},
        "method": method,
        "swaps": {
            "protein": clean_swaps(_MEAL_PROTEIN_SWAPS[goal]),
            "carbs": clean_swaps(_MEAL_CARB_SWAPS[goal]),
        },
        "sources": meal_sources,
        "note": "好脂肪（橄榄油）热量按约计：营养速查表未收录其精确数值，未计入总热量。",
    }
    if excluded_set:
        meal["excluded_for_allergy"] = sorted(excluded_set)
    return meal


def _meal_response_sources(meal: dict) -> list[dict]:
    """把 meal.sources（source/chapter/section）补全为响应级 sources（含原文 content）。"""
    out: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for s in meal["sources"]:
        key = (s["source"], s["chapter"], s["section"])
        if key in seen:
            continue
        seen.add(key)
        content = ""
        if s["source"] == "A":
            content = nutrition_lookup.get_table_markdown(s["section"]) or ""
        else:
            for c in get_retriever().retrieve(f"{meal.get('goal', '')} 饮食计划 食谱", top_k=8):
                if c.source == "B" and c.section == s["section"]:
                    content = c.content
                    break
        out.append(
            {"source": s["source"], "chapter": s["chapter"], "section": s["section"], "content": content, "score": None}
        )
    return out


def _token_overlap(content: str, query_tokens: set[str]) -> int:
    """回复块优选：统计块中命中的查询词数（同义重叠优先）。"""
    if not query_tokens:
        return 0
    return sum(1 for t in query_tokens if t in content)


# ── P5a 饮食管理（记住饮食）：记录/查询意图触发词 ──
# 记录触发词只取「已吃/记录动作」语义（吃了/记一下/帮我记…），不把裸餐次词（早餐/晚餐）
# 当记录——否则「晚餐给我做减脂的」这类 P3 一餐请求会被误记成台账。餐次仅用于 meal_tag 判定。
_DIET_RECORD_TRIGGERS: tuple[str, ...] = (
    "我吃了", "今天吃了", "吃了", "记录一下", "记一下", "帮我记", "帮我记录", "记一笔", "加餐",
)
# 查询触发词：问「吃了什么/饮食记录」等。必须优先于记录判定（「我最近吃了什么」含「吃了」）。
_DIET_QUERY_TRIGGERS: tuple[str, ...] = (
    "吃了什么", "吃了啥", "饮食记录", "最近吃什么", "我吃了什么", "我吃了啥",
    "记了什么", "台账",
)
# 问句/求方案语义：命中则不当作记录（如「帮我记一下今天吃什么」是求建议，不是记台账）。
_MEAL_REQUEST_HINTS: tuple[str, ...] = (
    "什么", "啥", "吗", "呢", "推荐", "给我做", "帮我做", "想吃", "该吃", "怎么吃", "安排",
)

# ── P5b 情感陪伴（一人食陪聊）：触发词 ──
_COMPANION_TRIGGERS: tuple[str, ...] = (
    "一个人吃饭", "一个人吃", "没人陪", "好累", "没胃口", "心情不好", "不开心",
    "加班", "孤独", "孤单", "不想动", "一个人住",
)


def _is_diet_query(message: str) -> bool:
    """是否命中「查询饮食台账」意图。"""
    return any(t in (message or "") for t in _DIET_QUERY_TRIGGERS)


def _is_diet_record(message: str) -> bool:
    """是否命中「记录饮食台账」意图（互斥：查询/问句语义一律不算记录）。"""
    text = message or ""
    if _is_diet_query(text):
        return False
    if any(h in text for h in _MEAL_REQUEST_HINTS):
        return False
    return any(t in text for t in _DIET_RECORD_TRIGGERS)


def _is_companion(message: str) -> bool:
    """是否命中「一人食陪聊/情感陪伴」意图。"""
    return any(t in (message or "") for t in _COMPANION_TRIGGERS)


def _diet_meal_tag_from_message(message: str) -> str:
    """从消息提取餐次标签：含早/午/晚/加餐 → 早餐/午餐/晚餐/加餐；未命中默认正餐。"""
    text = message or ""
    if "加餐" in text:
        return "加餐"
    if "早" in text:
        return "早餐"
    if "午" in text:
        return "午餐"
    if "晚" in text:
        return "晚餐"
    return "正餐"


def _grams_near(text: str, start: int, length: int) -> int:
    """在食材名前后 8 个字符窗口内找「数字+克/g」份量；未找到返回 100（常见份量）。

    红线「数值只来自速查表」：份量只取消息里显式写的克数，其余按 100g 常见份量估算，
    不猜单位换算（「两个鸡蛋」仍按 100g 计，不虚构鸡蛋单重）。
    """
    window = text[max(0, start - 8): start + length + 8]
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:克|g|G)", window)
    if m:
        return int(float(m.group(1)))
    return 100


def _estimate_diet_kcal(message: str) -> tuple[float | None, list[dict]]:
    """尽力估算消息中食材的热量（仅速查表真实值 × 份量；表外不编、不显示数字）。

    对消息里的每个表内食材（长名优先，避免「鸡蛋（全蛋）」被短名抢占）逐个 lookup，
    累加 kcal；返回 (total_kcal|None, items)。无任何表内食材时 total_kcal=None，
    调用方不显示任何数字。
    """
    text = message or ""
    matches: list[tuple[str, int, int]] = []
    for key in sorted(nutrition_lookup.NUTRITION_TABLE, key=len, reverse=True):
        idx = text.find(key)
        while idx != -1:
            matches.append((key, idx, _grams_near(text, idx, len(key))))
            idx = text.find(key, idx + len(key))
    if not matches:
        return None, []
    matches.sort(key=lambda m: m[1])
    items: list[dict] = []
    total = 0.0
    for key, _idx, grams in matches:
        row = nutrition_lookup.NUTRITION_TABLE[key]
        kcal: float | None = None
        if row.get("kcal") is not None:
            kcal = round(float(row["kcal"]) * grams / 100, 1)
            total += kcal
        items.append({"food": row["display_name"], "grams": grams, "kcal": kcal})
    return round(total, 1), items


def _handle_diet_record(req: ChatRequest, message: str, excluded: list[str]) -> str:
    """记录饮食台账：DietLog 入库（非致命）+ 尽力估算热量（仅速查表真实值）。

    回复语气像真人营养师记台账：先友好确认，再给「约 X 千卡」估算；
    表外食材不编造数字、不显示，仅当有表内食材才给出合计。
    """
    meal_tag = _diet_meal_tag_from_message(message)
    db = SessionLocal()
    try:
        user = db.get(models.User, req.user_id)
        if user is None:
            user = models.User(id=req.user_id, nickname=None)
            db.add(user)
            db.flush()
        sess = db.get(models.Session, req.session_id) if req.session_id else None
        # 二审 P1-2：会话归属校验——session 已存在但属于其他用户时禁止写入台账
        # （userB 借 userA 的 session_id 不得污染 A 的会话/台账）。
        if sess is not None and sess.user_id != req.user_id:
            return "抱歉，会话不存在或无权访问。"
        if sess is None:
            sess = models.Session(
                id=req.session_id or uuid.uuid4().hex,
                user_id=req.user_id,
                goal_tag=None,
                allergies=[],
            )
            db.add(sess)
            db.flush()
        db.add(
            models.DietLog(
                session_id=sess.id,
                user_id=req.user_id,
                meal_tag=meal_tag,
                content=message,
            )
        )
        db.commit()
    except Exception:  # noqa: BLE001
        db.rollback()
        logger.exception("饮食台账持久化失败（非致命，已跳过）")
    finally:
        db.close()

    total_kcal, items = _estimate_diet_kcal(message)
    head = f"好的，已帮你记下{meal_tag}台账📒：{message}。"
    if total_kcal is None:
        return (
            head
            + "\n这些食材暂时不在营养速查表里，先不估热量；以后你吃了什么都可以告诉我，我帮你记。"
        )
    parts = "、".join(
        f"{it['food']}约 {int(round(it['kcal']))} 千卡"
        if it["kcal"] is not None
        else f"{it['food']}（速查表未收录，不计）"
        for it in items
    )
    # 份量说明：消息带克数按用户份量估算；否则按常见份量 100g 估算。
    portion_note = (
        "按常见份量 100g 估算"
        if all(it["grams"] == 100 for it in items)
        else "按你提到的份量估算"
    )
    return (
        head
        + f"\n按速查表估算，这一餐约 {int(round(total_kcal))} 千卡"
        f"（{parts}，{portion_note}）。"
    )


def _handle_diet_query(user_id: str, session_id: str | None) -> str:
    """查询该用户最近 5 条饮食台账（created_at desc），自然汇总；无记录则引导记录。"""
    db = SessionLocal()
    try:
        logs = (
            db.query(models.DietLog)
            .filter(models.DietLog.user_id == user_id)
            .order_by(models.DietLog.created_at.desc(), models.DietLog.id.desc())
            .limit(5)
            .all()
        )
    except Exception:  # noqa: BLE001
        logs = []
        logger.exception("饮食台账查询失败（非致命，已降级为空）")
    finally:
        db.close()
    if not logs:
        return "你还没有饮食记录哦。告诉我你吃了什么（比如「早餐吃了两个鸡蛋」），我帮你记下来。"
    lines = [f"· {log.meal_tag}：{log.content}" for log in logs]
    return "你最近记了：\n" + "\n".join(lines)


def _companion_local_reply(req: ChatRequest, excluded: list[str]) -> str:
    """本地陪伴模板：先共情再轻建议；会话有目标时复用 P3 一餐生成器给轻量建议。

    建议里的食材全部来自速查表（P3 生成器保证），不编造数值。
    """
    goal = _session_goal_tag(req.session_id, req.user_id)
    meal = _build_meal(goal, excluded, "正餐") if goal else None
    if meal is not None:
        foods = "、".join(
            it["food"]
            for it in meal["items"]
            if it["category"] in ("主蛋白", "主食碳水", "蔬菜")
        )
        return (
            f"一个人吃饭更要好好吃🍚 我陪你。来点简单快手的？{foods} 一份就够，"
            "慢点吃，别让胃空着；需要的话我把做法也给你。"
        )
    return (
        "一个人吃饭更要好好吃🍚 我陪你。来点简单快手的？鸡胸肉+西兰花 15 分钟就能搞定，"
        "慢点吃，别让胃空着。"
    )


# 问候/情感触发词：本地降级（无 LLM）时区分闲聊话术，避免「你好」也回「抱歉没找到」的生硬感。
_GREETING_TRIGGERS = (
    "你好", "您好", "嗨", "哈喽", "在吗", "早上好", "下午好", "晚上好", "早安", "晚安",
    "hello", "hi",
)
_EMOTION_TRIGGERS = ("想你", "爱你", "喜欢你", "抱抱", "亲亲", "想哭", "委屈", "难过")


def _chitchat_local_reply(message: str) -> str:
    """本地闲聊兜底（无 LLM 时）：区分问候/情感/其他，告别一律「抱歉没找到」。"""
    text = message or ""
    if any(t in text for t in _GREETING_TRIGGERS):
        return (
            "你好呀！我是你的 AI 膳食顾问小优 👋 减脂、增肌还是慢病调理，"
            "或者直接告诉我「今晚想吃什么」，我帮你搭一餐～"
        )
    if any(t in text for t in _EMOTION_TRIGGERS):
        return "我也在呢～有我在，一个人也要好好吃饭。想聊聊今天吃什么，还是单纯想找人说说话？"
    return (
        "这个我可能帮不上忙，我主要擅长膳食营养（减脂 / 增肌 / 慢病调理）。"
        "要不要换个角度，聊聊你今天想吃什么、怎么吃更健康？"
    )


def _handle_companion(
    req: ChatRequest, message: str, excluded: list[str], recent: Sequence[str]
) -> tuple[str, str, bool]:
    """情感陪伴：先共情再轻建议。返回 (reply, model_tag, degraded)。

    模型来源标注诚实（进阶2）：LLM 启用且合成成功 → (reply, DEEPSEEK_MODEL, False)，
    与 chat() 其他 LLM 路径一致；LLM 未启用/失败 → 本地模板 (reply, "local-rules", True)。
    """
    if llm.is_enabled():
        llm_reply = llm.synthesize(
            message,
            [],
            history=_recent_chat_turns(req.session_id, 4),
            session_id=req.session_id,
            user_id=req.user_id,
            excluded_foods=excluded or None,
            state_context=health_state.build_state_context(req.user_id),
        )
        if llm_reply:
            return llm_reply, config.DEEPSEEK_MODEL, False
    return _companion_local_reply(req, excluded), "local-rules", True


def _try_p5_intent(
    req: ChatRequest, message: str, excluded: list[str], recent: Sequence[str]
) -> tuple[str, str, str, bool] | None:
    """P5 互斥意图派发：饮食查询 → 饮食记录 → 情感陪伴。

    命中返回 (reply, intent, model_tag, degraded)，chat() 直接采用（不落入 P3/检索）；
    未命中返回 None，保持既有流程。
    - 饮食记录/查询为规则路径 → model_tag="local-rules"、degraded=True；
    - 陪伴分支 LLM 成功 → model_tag=DEEPSEEK_MODEL、degraded=False；失败/未启用 → local-rules。
    """
    if _is_diet_query(message):
        return _handle_diet_query(req.user_id, req.session_id), "diet_query", "local-rules", True
    if _is_diet_record(message):
        return _handle_diet_record(req, message, excluded), "diet_record", "local-rules", True
    if _is_companion(message):
        reply, model_tag, degraded = _handle_companion(req, message, excluded, recent)
        return reply, "companion", model_tag, degraded
    return None


# BUG-5：数值问答意图词 & 数值单位（命中则进入二次排序）
_NUMERIC_HINTS = ("千卡", "卡路里", "卡", "kcal", "热量", "蛋白质", "脂肪", "碳水", "克", "g", "多少", "几")
_VALUE_UNITS = ("千卡", "卡路里", "卡", "kcal", "g", "克", "蛋白质", "脂肪", "碳水", "热量")
# 数值问答里需剔除的疑问/停用词，剩下的实词当作候选食材名
_NUM_STOP = {"多少", "几", "吃", "吗", "的", "能", "有", "是", "怎么", "如何", "什么", "可以", "想", "要", "该", "这", "那", "我", "你", "它"}
# 平台/定价类问题优先选自 C 库（平台私有内容），避免被营养块误排（缺陷 A 修复）。
# 二审 P1-1：统一引用 retrieval.PLATFORM_HINTS（含 收费/开通/升级/付费/定价/企业/SaaS
# /API/案例/白皮书），消除与检索闸门词表的漂移。
_PLATFORM_HINTS = PLATFORM_HINTS
# 定价类问法：在 C 库内部再优先含实际价格信息的块（缺陷 A 修复）
_PRICE_HINTS = ("多少钱", "价格", "收费", "费用", "报价", "钱", "付费", "套餐")


def _pick_reply_chunk(chunks: Sequence[SourceChunk], query: str) -> SourceChunk | None:
    """在 BM25 top-k 中，优先选与查询词重叠最多的块（解决 A4.1 过拟合问题）。

    BUG-5 增强：当 query 含营养数值意图（如"鸡胸肉多少千卡"）时，对 chunks 做
    二次排序——优先挑选 content 同时含候选食材名与"数字+营养单位"权威数值的块；
    无命中再退回原 token 重叠 heuristic，保证普通问答不回归。
    """
    if not chunks:
        return None
    q_tokens = {w for w in lcut(query or "") if w.strip()}
    # 缺陷 A 修复：平台/定价类问题优先自 C 库（私有平台内容）挑主答案。
    # 必须放在营养数值二次排序**之前**：否则「平台多少钱」因含「多少」被下方
    # value_chunks 分支提前 return 劫持，仍会误答 A 库「2.3 三餐热量分配建议」。
    if any(h in (query or "") for h in _PLATFORM_HINTS):
        c_chunks = [c for c in chunks if c.source == "C"]
        if c_chunks:
            # 定价类问法再收窄到「含实际价格信息」的 C 块：否则 _token_overlap 会因
            # 「会员」字面命中把「会员多少钱」答到「5.2 渠道合作」而漏掉真正的价目表。
            if any(p in (query or "") for p in _PRICE_HINTS):
                priced = [c for c in c_chunks if "价格" in c.content or "元" in c.content]
                if priced:
                    c_chunks = priced
            return max(c_chunks, key=lambda c: (_token_overlap(c.content, q_tokens), c.score or 0))
    if any(hint in (query or "") for hint in _NUMERIC_HINTS):
        food_tokens = {
            t for t in q_tokens
            if t not in _NUM_STOP and t not in _NUMERIC_HINTS and len(t) >= 2
        }
        value_chunks = [
            c for c in chunks
            if any(u in c.content for u in _VALUE_UNITS)
            and any(ch.isdigit() for ch in c.content)
            and any(food in c.content for food in food_tokens)
        ]
        if value_chunks:
            return max(
                value_chunks,
                key=lambda c: (_token_overlap(c.content, q_tokens), c.score or 0),
            )
    return max(chunks, key=lambda c: (_token_overlap(c.content, q_tokens), c.score or 0))


def _session_allergies(user_id: str, session_id: str | None) -> list[str]:
    """从 DB 读取会话已声明的禁忌（与对话触发、请求声明合并）。

    二审 P1-2：会话归属校验——session 已存在但属于其他用户时返回 []，
    杜绝 userB 借用 userA 的 session_id 继承 A 的禁忌画像。
    """
    if not session_id:
        return []
    db = SessionLocal()
    try:
        sess = db.get(models.Session, session_id)
        if sess is None or sess.user_id != user_id:
            return []
        return list(sess.allergies)
    except Exception:  # noqa: BLE001
        return []
    finally:
        db.close()


def _session_goal_tag(session_id: str | None, user_id: str | None = None) -> str | None:
    """从 DB 读取会话当前人群标签；未设置/异常返回 None。

    二审 P1-2：user_id 非空时做会话归属校验——session 属于其他用户返回 None，
    杜绝 userB 借用 userA 的 session_id 继承 A 的目标画像。
    """
    if not session_id:
        return None
    db = SessionLocal()
    try:
        sess = db.get(models.Session, session_id)
        if sess is None:
            return None
        if user_id is not None and sess.user_id != user_id:
            return None
        return sess.goal_tag
    except Exception:  # noqa: BLE001
        return None
    finally:
        db.close()


def _recent_user_messages(session_id: str | None, n: int = 2) -> list[str]:
    """取最近 n 条用户消息（旧→新），用于多轮上下文检索增强。"""
    if not session_id:
        return []
    db = SessionLocal()
    try:
        msgs = (
            db.query(models.Message)
            .filter(models.Message.session_id == session_id, models.Message.role == "user")
            .order_by(models.Message.id.desc())
            .limit(n)
            .all()
        )
        return [m.content for m in reversed(msgs)]
    except Exception:  # noqa: BLE001
        return []
    finally:
        db.close()


def _recent_chat_turns(session_id: str | None, n: int = 4) -> list[dict]:
    """取最近 n 条对话消息（含 user 与 assistant，旧→新），供 LLM 真实感知多轮上下文。

    外部审查发现：此前只传 user 消息给 LLM，用户「谢谢」时模型看到两个连续 user
    消息（以为还没回答过），导致复读上一轮方案。带上 assistant 回复后模型知道
    自己已答过，礼貌收尾即可。仅用于 LLM 对话历史；检索增强仍用 _recent_user_messages。
    """
    if not session_id:
        return []
    db = SessionLocal()
    try:
        msgs = (
            db.query(models.Message)
            .filter(models.Message.session_id == session_id)
            .order_by(models.Message.id.desc())
            .limit(n)
            .all()
        )
        return [{"role": m.role, "content": m.content} for m in reversed(msgs)]
    except Exception:  # noqa: BLE001
        return []
    finally:
        db.close()


def _session_has_disease(session_id: str | None) -> bool:
    """会话历史中是否曾出现疾病/医疗意图（跨轮延续免责）。

    二审 P0-5 修复：只扫描 **user** 消息——assistant 回复的素材块原文常含
    「血糖/糖尿病/三高」等词，全量扫描会把普通增肌用户误判为「控糖用户」，
    导致次轮无关问题也被贴控糖引导+过度免责（产品记忆错乱）。模型/检索输出
    永不当用户事实。
    二审 P1-4：对历史 user 消息同样做否定剥离——「我没有糖尿病」不得记为患病。
    """
    if not session_id:
        return False
    db = SessionLocal()
    try:
        msgs = (
            db.query(models.Message)
            .filter(
                models.Message.session_id == session_id,
                models.Message.role == "user",
            )
            .all()
        )
        return any(is_disease_query(strip_disease_negations(m.content)) for m in msgs)
    except Exception:  # noqa: BLE001
        return False
    finally:
        db.close()


def _session_profile_has_disease(session_id: str | None) -> bool:
    """会话画像中是否含疾病类禁忌（taboo_map disclaimer_type == disease）。

    守卫 2（P0）：用户把「高血压/糖尿病/痛风」写在档案（/api/session allergies）
    而非对话里时，仍需跨轮免责延续 + 引导语差异化；但**不改变消息级拒药/免责
    硬判定**——is_disease_query/is_medication_query 仍以每条消息关键词为准。
    """
    if not session_id:
        return False
    db = SessionLocal()
    try:
        sess = db.get(models.Session, session_id)
        if not sess or not sess.allergies:
            return False
        by_id = {t["id"]: t for t in _load_taboos()}
        return any(
            aid in by_id and by_id[aid].get("disclaimer_type") == "disease"
            for aid in sess.allergies
        )
    except Exception:  # noqa: BLE001
        return False
    finally:
        db.close()


# M11 疾病标签映射：疾病/症状关键词 → 引导语标签（「结合您的控糖需求，」）
_DISEASE_LABELS: tuple[tuple[str, str], ...] = (
    ("糖尿病", "控糖"), ("血糖", "控糖"), ("糖友", "控糖"),
    ("痛风", "低嘌呤"), ("高尿酸", "低嘌呤"), ("尿酸高", "低嘌呤"),
    ("高血压", "控盐"), ("血压高", "控盐"),
    ("肾病", "护肾"), ("肾功能不全", "护肾"),
    ("孕期", "孕期营养"), ("孕妇", "孕期营养"), ("怀孕", "孕期营养"),
    ("甲亢", "甲状腺管理"), ("甲减", "甲状腺管理"),
)


def _profile_guide(message: str, disease: bool, session_id: str | None = None) -> str:
    """M11 画像感知引导语：疾病用户回复差异化（「结合您的控糖需求，」）。

    顾问感的核心：不是所有用户都得到同一句模板，而是「它记得我的情况，为我定制」。
    疾病标签先从当前消息识别；当前消息无病名时扫会话历史（跨轮延续，「我有糖尿病」
    之后随便问什么都能带「结合您的控糖需求」）；再查会话画像的疾病类禁忌
    （守卫 2：用户把「高血压」写在档案而非对话里时同样带「结合您的控盐需求」）；
    都识别不到才用通用「健康管理」兜底。
    """
    if not disease:
        return ""
    texts = [strip_disease_negations(message or "")]
    sess_disease_name: str | None = None
    if session_id:
        db = SessionLocal()
        try:
            # 二审 P0-5 修复：只扫 user 消息——assistant 回复的素材块原文常含
            # 「血糖/糖尿病」等词，全量扫描会污染画像（同 _session_has_disease）。
            # 二审 P1-4：历史 user 消息同样做否定剥离（「我没有糖尿病」不引导）。
            recent = (
                db.query(models.Message)
                .filter(
                    models.Message.session_id == session_id,
                    models.Message.role == "user",
                )
                .order_by(models.Message.created_at.desc())
                .limit(6)
                .all()
            )
            texts += [strip_disease_negations(m.content) for m in recent]
            # 守卫 2：会话画像疾病类禁忌 → 对应引导语标签（档案写入而非对话声明时）
            sess = db.get(models.Session, session_id)
            if sess and sess.allergies:
                by_id = {t["id"]: t for t in _load_taboos()}
                for aid in sess.allergies:
                    t = by_id.get(aid)
                    if t and t.get("disclaimer_type") == "disease":
                        sess_disease_name = t["name"]
                        break
        except Exception:  # noqa: BLE001
            pass
        finally:
            db.close()
    for t in texts:
        for kw, label in _DISEASE_LABELS:
            if kw in (t or ""):
                return f"结合您的{label}需求，"
    if sess_disease_name:
        for kw, label in _DISEASE_LABELS:
            if kw in sess_disease_name:
                return f"结合您的{label}需求，"
    return "结合您的健康管理需求，"


def _update_session_profile(
    user_id: str,
    session_id: str | None,
    *,
    allergy_ids: list[str] | None = None,
    goal_tag: str | None = None,
) -> None:
    """P2：把对话中新识别的过敏/目标写回会话画像（非致命，异常不影响主流程）。

    会话不存在时按需创建（与 _persist_turn 同一套兜底），保证「中途声明」的
    过敏在下一轮仍生效——即「记得我」。写库放在 _persist_turn 之前。

    守卫 1（P0）：写库前对 allergy_ids 做 taboo_map 白名单过滤——未知 id 一律丢弃
    （安全失败优先），杜绝伪造 id 注入档案；与 POST /api/session 入口过滤双保险。
    """
    if not session_id:
        return
    db = SessionLocal()
    try:
        user = db.get(models.User, user_id)
        if user is None:
            user = models.User(id=user_id, nickname=None)
            db.add(user)
            db.flush()
        sess = db.get(models.Session, session_id)
        # 二审 P1-2：会话归属校验——session 已存在但属于其他用户时禁止改写画像。
        if sess is not None and sess.user_id != user_id:
            return
        if sess is None:
            sess = models.Session(id=session_id, user_id=user_id, goal_tag=None, allergies=[])
            db.add(sess)
            db.flush()
        if allergy_ids is not None:
            valid_ids = {t["id"] for t in _load_taboos()}
            sess.allergies = [a for a in allergy_ids if a in valid_ids]
        if goal_tag is not None:
            sess.goal_tag = goal_tag
        db.commit()
    except Exception:  # noqa: BLE001
        db.rollback()
        logger.exception("会话画像持久化失败（非致命，已跳过）")
    finally:
        db.close()


def _persist_turn(
    user_id: str,
    session_id: str | None,
    user_msg: str,
    bot_reply: str,
    sources: list[dict],
) -> None:
    """对话落库（MOD-04）：非致命，DB 异常不影响返回。"""
    if not session_id:
        return
    db = SessionLocal()
    try:
        user = db.get(models.User, user_id)
        if user is None:
            user = models.User(id=user_id, nickname=None)
            db.add(user)
            db.flush()
        sess = db.get(models.Session, session_id)
        # 二审 P1-2：会话归属校验——session 已存在但属于其他用户时禁止落库。
        if sess is not None and sess.user_id != user_id:
            return
        if sess is None:
            sess = models.Session(id=session_id, user_id=user_id, goal_tag=None, allergies=[])
            db.add(sess)
            db.flush()
        db.add(models.Message(session_id=session_id, role="user", content=user_msg, sources=[]))
        db.add(models.Message(session_id=session_id, role="assistant", content=bot_reply, sources=sources))
        db.commit()
    except Exception:  # noqa: BLE001
        db.rollback()
        logger.exception("对话持久化失败（非致命，已跳过）")
    finally:
        db.close()


@app.post("/api/chat", response_model=UnifiedResponse)
def chat(req: ChatRequest) -> UnifiedResponse:
    """多轮对话：检索 grounding 回答 + 合规层（免责/拒药/禁忌）+ 过敏/目标画像持久化 + 对话落库。"""
    message = validate_message(req.message)
    # 二审 P1-4：疾病/目标判定用否定剥离后的消息（「我没有糖尿病」「我血糖正常」
    # 不得误判患病）；落库/检索仍保留原始 message（ctx_query 用原文保证召回）。
    message_scan = strip_disease_negations(message)
    # 多轮上下文：最近用户消息参与检索；疾病意图跨轮延续免责
    recent = _recent_user_messages(req.session_id, 2)
    ctx_query = " ".join([message] + recent)
    disease = (
        is_disease_query(message_scan)
        or _session_has_disease(req.session_id)
        or _session_profile_has_disease(req.session_id)
    )

    chunks = get_retriever().retrieve(ctx_query, top_k=4, current_query=message)

    reply = ""
    intent = "nutrition_qa"
    model_tag = "local-rules"
    degraded = True
    sources: list = list(chunks)

    # 禁忌识别：请求声明 + 对话触发 + 会话已声明，三路合并
    # （提前到回复逻辑之前，供 P3 一餐生成器做禁忌排除）
    session_allergy_ids = set(_session_allergies(req.user_id, req.session_id))
    detected_allergies = set(detect_allergies(message))
    # P2：本轮「新检测到」的过敏（尚未写回会话）——只在首次声明时非空 → 只追问一次
    new_allergies = detected_allergies - session_allergy_ids
    allergy_ids = set(req.allergies) | detected_allergies | session_allergy_ids
    excluded = excluded_foods(list(allergy_ids))

    disclaimer = DISCLAIMER_STANDARD if (disease or is_medication_query(message)) else None

    # ── R1/R2 修复：一餐前置门槛 ──
    # 判定顺序：用药硬拦截 → 数值精确问答 → 一餐 → 检索/LLM/其他。
    # （R1）「晚餐鸡胸肉多少千卡」不得被一餐劫持成追问目标，必须走数值路径返回速查表精确值；
    # （R2）「吃布洛芬后晚餐吃什么好」不得进一餐，必须走现有拒药路径（拒答+免责）。
    is_med_query = is_medication_query(message)
    is_num_query, num_food = nutrition_lookup.is_numeric_lookup_query(message)

    # ── 点单决策（Decision Tool · THE LAST 30 SECONDS）：分支前解析并按禁忌过滤，
    # 全部被过滤则候选为 None → 回退常规流程，避免空答复。 ──
    _decision_cand = decision_tool.resolve(
        message, _meal_goal(message, _session_goal_tag(req.session_id, req.user_id)) or "减脂"
    )
    if _decision_cand and excluded:
        # P0-3 修复：决策候选按禁忌过滤改用 _is_food_excluded（与 P3 一餐生成器统一，
        # 双向包含+归一化）。修复前用 `f in it` 弱子串匹配，「白煮蛋×2」不含「鸡蛋」
        # 子串导致蛋过敏用户仍被推荐白煮蛋（红线②击穿）。
        _decision_cand["items"] = [
            it for it in _decision_cand["items"]
            if not _is_food_excluded(it, set(excluded))
        ]
    if _decision_cand and not _decision_cand["items"]:
        _decision_cand = None

    # ── P5 饮食管理（记住饮食）+ 情感陪伴（一人食陪聊）：互斥优先于 P3/检索 ──
    # 命中记录/查询/陪伴意图即返回，不落入一餐生成/BM25 检索；问句/求方案语义
    # （什么/推荐/给我做）不当作记录，避免「早餐吃什么好」被误记成台账。
    # V01 修复：用药硬拦截优先于一切——is_med_query 为 True 时跳过 P5，
    # 「我吃了布洛芬怎么办」不得被记成饮食台账、「一个人吃饭可以吃布洛芬吗」
    # 不得走情感陪伴，一律进入现有拒药路径。
    meal: dict | None = None
    decision: dict | None = None
    p5 = None if is_med_query else _try_p5_intent(req, message, excluded, recent)
    if p5 is not None:
        reply, intent, model_tag, degraded = p5
        sources = []
    elif _is_meal_intent(message) and not is_med_query and not (is_num_query and num_food):
        goal = _meal_goal(message, _session_goal_tag(req.session_id, req.user_id))
        if goal is None:
            # 消息无目标词且会话也无 goal → 先追问目标（复用追问风格，不直接出餐）
            reply = _MEAL_GOAL_ASK
            intent = "meal_goal_ask"
            sources = []
        else:
            meal = _build_meal(goal, excluded, _meal_time_from_message(message))
            reply = (
                f"为你搭配的{meal['name']}👇\n"
                f"一餐约 {int(round(meal['total_kcal']))} 千卡"
                "（热量与营养素均按营养速查表数值计算）。"
            )
            intent = "meal"
            model_tag = "local-rules"
            sources = _meal_response_sources(meal)
    elif _decision_cand and not is_med_query and not (is_num_query and num_food):
        # ── 点单决策（Decision Tool · THE LAST 30 SECONDS）──
        # 候选已在分支前解析并按禁忌过滤；纯行为决策、不引用素材外精确数值（红线⑤）。
        reply = (
            f"【点单决策】在{_decision_cand['scenario']}，按{_decision_cand['goal']}目标这样点👇\n"
            + "\n".join(f"· {it}" for it in _decision_cand["items"])
            + (f"\n💡 {_decision_cand['note']}" if _decision_cand.get("note") else "")
            + "\n（具体以餐厅实际出品为准）"
        )
        intent = "decision"
        model_tag = "local-rules"
        sources = []
        decision = _decision_cand
    else:
        # ── BUG-5 数值问答确定性工具命中（红线要求：数值必须工具计算，禁止模型推理）──
        # 在模糊检索之上插入确定性强校验：命中权威表则直接返回精确值，跳过 _pick_reply_chunk。
        numeric_hit = False
        numeric_miss = False  # 数值意图命中但表中无该食材：诚实告知，不臆测、不回退 BM25
        is_num, food = is_num_query, num_food
        if is_num and food and not is_med_query:  # V01 扩展：用药问题（如「头孢克肟…几天」）
            # 不得被数值分支截获——is_med_query 时跳过数值，交给拒药分支
            row = nutrition_lookup.lookup(food)
            if row:
                reply = nutrition_lookup.format_reply(row)
                intent = "nutrition_lookup"
                sources = [{
                    "source": "A",
                    "chapter": row["source_chapter"],
                    "section": row["source_section"],
                    "content": nutrition_lookup.get_table_markdown(row["source_section"]) or "",
                    "score": None,
                }]
                numeric_hit = True
            else:
                # 仅当候选名「像食材」（≥2 字且含中文）时才走诚实告知分支。
                # is_numeric_lookup_query 把裸「g」「卡」也当指标词，会从
                # 「低GI食物有哪些」「yogurt 推荐吗」误提取出 food='低' / 'yo'；
                # 这类碎片保持原 BM25 回退，避免误报「表中暂未收录」造成回归。
                numeric_miss = len(food) >= 2 and any("\u4e00" <= ch <= "\u9fff" for ch in food)

        if numeric_miss:
            # 暴风雪测试回归：非膳食问法（如「世界首富马斯克今天吃什么」）误入
            # 数值 miss 时不得回 miss——交给下方领域门控走闲聊，避免答
            # 「暂未收录「世界首富马斯」」；膳食问法（如「牛油果」）保持诚实 miss。
            if is_dietary_domain(message):
                reply = (
                    f"【营养速查表】表中暂未收录「{food}」的精确营养数据，无法给出确定数值，恕不臆测。\n"
                    "您可以换问表中已有食材（如鸡胸肉、糙米、西兰花、三文鱼、鸡蛋等），"
                    "或告诉我您想了解的营养维度（热量 / 蛋白质 / 脂肪 / 碳水）。"
                )
                intent = "nutrition_lookup_miss"
                sources = []
            else:
                numeric_miss = False

        if not numeric_hit and not numeric_miss:
            # 领域路由门控：区分「膳食顾问需求」与「闲聊/非膳食」。
            # 无检索命中，或命中了检索块但用户问题不属于膳食领域（如「大龙虾吃什么」是
            # 动物习性、「世界首富是谁」是常识）→ 走闲聊分支，绝不硬套无关知识块。
            top = _pick_reply_chunk(chunks, message) if chunks else None
            is_platform = bool(top and top.source == "C")
            in_domain = is_platform or is_dietary_domain(message)
            if not chunks or not in_domain:
                # 无检索命中 / 非膳食领域：闲聊/客套绝不外挂来源分块（外部审查发现：
                # 此前 sources 带着上一轮 chunks 的 4 个分块，闲聊底下误挂来源标签）。
                sources = []
                # 用药类硬拦截（不经 LLM），其余尝试 LLM 自然回应
                # （打招呼/闲聊/情感陪伴），失败再降级原话术。
                if is_medication_query(message):
                    reply = (
                        "本工具不提供用药建议，请遵医嘱。如有膳食搭配、营养方面的疑问，"
                        "我很乐意为您解答。"
                    )
                    intent = "medication_refuse"
                else:
                    llm_reply = None
                    if llm.is_enabled():
                        llm_reply = llm.synthesize(
                            message,
                            [],  # 空 grounding：靠系统提示第 5 条约束行为，绝不裸奔
                            history=_recent_chat_turns(req.session_id, 4),
                            session_id=req.session_id,
                            user_id=req.user_id,
                            excluded_foods=excluded or None,
                            state_context=health_state.build_state_context(req.user_id),
                        )
                    if llm_reply:
                        reply = llm_reply
                        model_tag = config.DEEPSEEK_MODEL
                        degraded = False
                        intent = "chitchat"
                    else:
                        reply = _chitchat_local_reply(message)
                        intent = "chitchat"
            else:
                # 用药类问题：即使检索到膳食参考块，意图标签也必须是拒药（红线语义一致，
                # 否则「司美格鲁肽减肥」扩展命中减脂块后 intent 会误标成 nutrition_qa）。
                if is_med_query:
                    intent = "medication_refuse"
                else:
                    intent = "platform" if is_platform else "nutrition_qa"
                # 命中检索块 -> 优先尝试 LLM 合成（Key 就绪且非用药类问题时）。
                # 用药类问题由合规层硬拦截，不经 LLM，避免任何用药建议风险。
                used_llm = False
                if llm.is_enabled() and not is_medication_query(message):
                    llm_reply = llm.synthesize(
                        message,
                        chunks,
                        history=_recent_chat_turns(req.session_id, 4),
                        session_id=req.session_id,
                        user_id=req.user_id,
                        excluded_foods=excluded or None,
                        state_context=health_state.build_state_context(req.user_id),
                    )
                    if llm_reply:
                        reply = llm_reply
                        model_tag = config.DEEPSEEK_MODEL
                        degraded = False
                        used_llm = True
                # 未启用 LLM 或合成失败 -> 本地规则兜底（绝不裸奔）
                if not used_llm:
                    snippet = top.content.strip()
                    if len(snippet) > 600:
                        snippet = snippet[:600] + "…（更多见下方来源）"
                    if is_medication_query(message):
                        # 拒答用药，仅给膳食参考
                        reply = (
                            "本工具不提供用药建议，请遵医嘱。以下为相关膳食参考：\n"
                            f"【{top.chapter} · {top.section}】\n{snippet}"
                        )
                    else:
                        reply = f"【{top.chapter} · {top.section}】\n{snippet}"

    # P2 画像持久化：过敏全集写回 session + 隐含人群目标动态更新
    # （写库放在 _persist_turn 之前；异常不影响主流程）
    if new_allergies:
        _update_session_profile(req.user_id, req.session_id, allergy_ids=list(allergy_ids))
    # 二审 P1-4：目标识别用否定剥离后的消息（「我血糖正常，推荐减脂」不得写成调理）
    detected_goal = _detect_goal(message_scan)
    if detected_goal and detected_goal != _session_goal_tag(req.session_id, req.user_id):
        _update_session_profile(req.user_id, req.session_id, goal_tag=detected_goal)

    # M11 画像感知引导语：疾病用户回复差异化（「结合您的控糖需求，」）——加在正文前，
    # 位于 R3 剔除之前（引导语无禁忌词，不受剔除影响）；首次过敏追问（new_allergies）
    # 时跳过，避免「追问 + 引导语」双重前缀叠罗汉。
    if not new_allergies and reply:
        guide = _profile_guide(message, disease, req.session_id)
        if guide:
            reply = f"{guide}\n\n{reply}"

    # P2 过敏追问：仅首次声明时，以追问开头确认「已记录」；未配置话术的 id 跳过
    # DEFECT-A 修复：R3 剔除循环只作用于**原回复正文**——先对原 reply 剔除再拼接追问
    # （ALLERGY_FOLLOWUP 含花生/牛奶等精确食材名，绝不经剔除循环，避免被替换成
    # 「（已按禁忌剔除）」乱码）；排除提示仍追加在最终 reply 尾部（追问之后）。
    if excluded:
        # 红线②「禁忌必排除」：先剔除原回复正文中出现的禁忌食材
        # （长名优先 + 「...」引号内容保护，V04）。
        reply = _redact_excluded(reply, excluded)

    if new_allergies:
        follow_up_parts = [
            ALLERGY_FOLLOWUP[aid]
            for aid in sorted(new_allergies)
            if aid in ALLERGY_FOLLOWUP
        ]
        if follow_up_parts:
            follow_up = "\n\n".join(follow_up_parts)
            reply = f"{follow_up}\n\n{reply}" if reply else follow_up

    if excluded:
        # P0-3 修复③：排除提示仅在「确有剔除/回避处理」时展示——
        # ①正文实际剔除了禁忌食材（出现剔除标记）；②一餐生成/点单决策按禁忌
        # 排除了候选；③本轮首次声明过敏（追问确认）。修复前只要 excluded 非空
        # 就无条件展示，导致「一边推荐白煮蛋、一边提示已排除鸡蛋」的自相矛盾。
        was_redacted = _REDACT_MARK in reply
        if was_redacted or meal is not None or decision is not None or new_allergies:
            # 确认排除提示保留在 reply 尾部（追问之后），列全清单供用户知晓。
            reply += f"\n\n⚠️ 已按您的禁忌排除以下食材：{', '.join(excluded)}"

    # AI 自动维护用户档案（UI 端不可人工编辑，由 AI 从对话中自动识别写入）：
    # 规则检测（仅第一人称陈述，问句/第三人称不提取）→ 字段级去重写入
    # （档案里已有该字段则不覆盖，保留用户先说的；没有才写入）。
    # 红线：档案只做记录用途——不参与合规判定、不覆盖 allergies/goal_tag、
    # 不注入 LLM 上下文（守卫 3 保持）；异常不影响主流程。
    try:
        _profile_updates = user_profile.detect_user_profile(message)
        if _profile_updates:
            user_profile.add_profile_fields(req.user_id, _profile_updates)
    except Exception:  # noqa: BLE001
        logger.exception("用户档案自动写入失败（非致命，已跳过）")

    _persist_turn(req.user_id, req.session_id, message, reply, [c.model_dump() for c in chunks])
    data: dict = {"reply": reply, "intent": intent, "goal_tag": None}
    if meal is not None:
        # P3 响应契约：命中一餐意图时附加结构化 meal（可选字段）；未命中保持原契约，
        # 前端检测到 meal 才渲染一餐卡，不破坏现有渲染逻辑。
        data["meal"] = meal
    if decision is not None:
        # 点单决策契约：命中点单场景时附加结构化 decision（可选字段）。
        data["decision"] = decision
    return make_response(
        data,
        sources=sources,
        model=model_tag,
        degraded=degraded,
        disclaimer=disclaimer,
    )


_GOAL_CHAPTER_KW: dict[str, str] = {
    "减脂": "减脂方案",
    "增肌": "增肌方案",
    "调理": "稳糖调理方案",
}


def _first_sentence(text: str) -> str:
    """取知识块首句完整文本（换行/句末标点切分），不截断到标签。

    BUG-6 修复：素材 B 块首行常为「目标人群：」标签（其后换行才是正文），
    原逻辑按换行取首行只剩标签。此处先合并换行，再按句末标点切出首句；
    若无句末标点则回退完整文本，保证 reason 显示完整首句而非被截断的标签。
    """
    text = (text or "").strip().replace("\n", " ")
    if not text:
        return ""
    for sep in ("。", ".", "！", "?", "；", ";"):
        idx = text.find(sep)
        if idx > 0:
            return text[:idx].strip()
    return text


def _plan_from_goal(goal: str, allergy_ids: list[str]) -> dict:
    """基于素材 B 检索生成真实推荐（非示意文案），并按禁忌排除食材。"""
    base = _GOAL_PLAN_MAP.get(goal) or _GOAL_PLAN_MAP["减脂"]
    chunks = get_retriever().retrieve(f"{goal} 饮食计划 营养素目标 推荐食材", top_k=6)
    kw = _GOAL_CHAPTER_KW.get(goal, goal)
    reason = ""
    for c in chunks:
        if c.source == "B" and kw in (c.chapter or ""):
            reason = _first_sentence(c.content)
            break
    if not reason and chunks:
        reason = _first_sentence(chunks[0].content)
    foods = [f for f in base["foods"] if f not in excluded_foods(allergy_ids)]
    excl = [f for f in base["foods"] if f in excluded_foods(allergy_ids)]
    # 红线②「禁忌必排除」：推荐理由（来自素材 B 检索块）若含禁忌食材同样剔除
    # （长名优先 + 「...」引号内容保护，V04）。
    if excl:
        reason = _redact_excluded(reason, excl)
    plan = {
        "name": base["name"],
        "kcal_range": base["kcal_range"],
        "foods": foods,
        "reason": reason or f"{base['name']}：基于《中国居民膳食指南》与素材 B 方案生成。",
        "source_chapter": chunks[0].chapter if chunks else None,
    }
    # M5 Explain My Plan：结构化"为什么推荐"——基于方案字段（素材 B 数据）给目标对应解释，
    # 不引入素材外数值；frontend renderRec 在 reason 后展示。
    plan["why"] = (
        f"为什么适合你：{goal}阶段的热量与蛋白质目标落在 {base['kcal_range']}，"
        f"{base['name']}的食材组合（{'、'.join(foods[:3])}{'…' if len(foods) > 3 else ''}）"
        f"匹配{goal}阶段的蛋白/控卡/稳糖需求。"
    )
    if excl:
        plan["excluded_for_allergy"] = excl
    return plan


@app.post("/api/recommend", response_model=UnifiedResponse)
def recommend(req: RecommendRequest) -> UnifiedResponse:
    """按人群标签推荐方案 + 食材 + 理由（基于素材 B 检索 grounding，含禁忌排除）。"""
    goal = req.goal_tag or "减脂"
    allergy_ids = set(req.allergies) | set(_session_allergies(req.user_id, req.session_id))
    plan = _plan_from_goal(goal, list(allergy_ids))
    disclaimer = DISCLAIMER_STANDARD if goal == "调理" else None
    return make_response(
        {"plans": [plan], "intent": "recommend", "goal_tag": goal},
        sources=get_retriever().retrieve(f"{goal} 饮食计划 营养素目标 推荐食谱", top_k=3),
        model="local-rules",
        degraded=True,
        disclaimer=disclaimer,
    )


@app.post("/api/quick", response_model=UnifiedResponse)
def quick(req: QuickRequest) -> UnifiedResponse:
    """快捷操作：减脂 / 增肌 / 控糖 / 今日食谱（检索 grounding 来源标注）。"""
    goal = _ACTION_GOAL_MAP[req.action]
    allergy_ids = set(req.allergies) | set(_session_allergies(req.user_id, req.session_id))
    if goal is None:
        # 今日食谱：按减脂方案给出（素材 B 7 日循环），真实生成
        plan = _plan_from_goal("减脂", list(allergy_ids))
        plan = dict(plan)
        plan["note"] = "今日食谱依据素材 B 7 日循环食谱生成，可循环使用。"
    else:
        plan = _plan_from_goal(goal, list(allergy_ids))
    # 二审 P0-4（红线②）：控糖（调理）快捷入口此前漏免责（recommend 有、quick 无，
    # 同一产品两入口合规不一致）；补上与 recommend 一致的免责。
    disclaimer = DISCLAIMER_STANDARD if goal == "调理" else None
    return make_response(
        {"plans": [plan], "intent": "recommend", "goal_tag": goal},
        sources=get_retriever().retrieve(f"{goal or '减脂'} 饮食计划 营养素目标 推荐食材", top_k=2),
        model="local-rules",
        degraded=True,
        disclaimer=disclaimer,
    )


@app.get("/api/history", response_model=UnifiedResponse)
def history(
    user_id: str = Query(min_length=1, max_length=64),
    session_id: str | None = Query(default=None, max_length=64),
) -> UnifiedResponse:
    """拉取用户对话历史（前端历史面板调用，真实读库）。"""
    db = SessionLocal()
    try:
        q = db.query(models.Message).join(
            models.Session, models.Session.id == models.Message.session_id
        ).filter(models.Session.user_id == user_id)
        if session_id:
            q = q.filter(models.Message.session_id == session_id)
        msgs = q.order_by(models.Message.id).all()
        messages = [
            {"role": m.role, "content": m.content, "sources": m.sources} for m in msgs
        ]
        if session_id:
            sessions = (
                [{"session_id": session_id, "message_count": len(messages)}]
                if messages
                else []
            )
        else:
            # 未指定 session_id：按会话聚合，供前端历史面板列出全部会话。
            # messages 按 id 升序；按每个会话最后一条消息 id 排序，最新活跃的会话在前。
            agg: dict[str, dict] = {}
            for m in msgs:
                sid = m.session_id
                if sid not in agg:
                    agg[sid] = {
                        "session_id": sid,
                        "message_count": 0,
                        "first_message": "",
                        "_last_id": m.id,
                    }
                agg[sid]["message_count"] += 1
                agg[sid]["_last_id"] = m.id
                if not agg[sid]["first_message"] and m.role == "user":
                    agg[sid]["first_message"] = m.content
            sessions = [
                agg[sid]
                for sid in sorted(agg, key=lambda s: agg[s]["_last_id"], reverse=True)
            ]
            for s in sessions:
                s.pop("_last_id", None)
    finally:
        db.close()
    return make_response(
        {"sessions": sessions, "messages": messages},
        sources=[],
        model="local-rules",
        degraded=True,
    )


@app.post("/api/user", response_model=UnifiedResponse)
def create_user(req: UserCreateRequest) -> UnifiedResponse:
    """创建用户身份（真实写入 users 表）。"""
    db = SessionLocal()
    try:
        user = models.User(id=uuid.uuid4().hex, nickname=req.nickname)
        db.add(user)
        db.commit()
        uid = user.id
    finally:
        db.close()
    return make_response(
        {"user_id": uid, "nickname": req.nickname, "created_at": datetime.now(timezone.utc).isoformat()},
        sources=[],
        model="local-rules",
        degraded=True,
    )


@app.get("/api/user", response_model=UnifiedResponse)
def get_user(user_id: str = Query(min_length=1, max_length=64)) -> UnifiedResponse:
    """识别用户身份（真实读库）。"""
    db = SessionLocal()
    try:
        user = db.get(models.User, user_id)
        nick = user.nickname if user else None
    finally:
        db.close()
    return make_response(
        {"user_id": user_id, "nickname": nick, "created_at": None},
        sources=[],
        model="local-rules",
        degraded=True,
    )


@app.post("/api/session", response_model=UnifiedResponse)
def session_op(req: SessionRequest) -> UnifiedResponse:
    """新建/切换会话，携带人群标签与禁忌列表（真实写入 sessions 表）。

    守卫 1（P0）：allergies 白名单校验——仅接受 taboo_map.json 已知 id，
    未知 id 过滤掉（安全失败优先）；「仅含未知 id」对既有会话视为无效更新
    （保持原样，不误清真实禁忌），杜绝伪造 id 注入档案。
    """
    valid_ids = {t["id"] for t in _load_taboos()}
    req_allergies = [a for a in list(req.allergies) if a in valid_ids]
    db = SessionLocal()
    try:
        # 确保 users 表存在对应行：FK 约束开启时，缺失 user 行会触发 IntegrityError → 500
        user = db.get(models.User, req.user_id)
        if user is None:
            user = models.User(id=req.user_id, nickname=None)
            db.add(user)
            db.flush()

        sess = db.get(models.Session, req.session_id) if req.session_id else None
        # 二审 P1-2：跨用户会话串号防护——userB 不得通过 action=switch 改写
        # userA 的会话画像（否则可继承/篡改他人禁忌与目标）。
        if sess is not None and sess.user_id != req.user_id:
            db.rollback()
            return make_response(
                {"error": "会话不存在或无权访问"},
                sources=[],
                model="local-rules",
                degraded=True,
            )
        if sess is None:
            sess = models.Session(
                id=req.session_id or uuid.uuid4().hex,
                user_id=req.user_id,
                goal_tag=req.goal_tag,
                allergies=list(req_allergies),
            )
            db.add(sess)
        else:
            if req.goal_tag is not None:
                sess.goal_tag = req.goal_tag
            if req.allergies:
                # 守卫 1：白名单过滤后写入；过滤结果为空（原请求仅含未知 id）→ 保持原样
                sess.allergies = (
                    list(req_allergies) if req_allergies else list(sess.allergies or [])
                )
        db.commit()
        sid = sess.id
        goal = sess.goal_tag
        allergies = list(sess.allergies)
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        logger.exception("创建/切换会话失败: %s", exc)
        return make_response(
            {"error": "创建会话失败，请稍后重试"},
            sources=[],
            model="local-rules",
            degraded=True,
        )
    finally:
        db.close()
    return make_response(
        {"session_id": sid, "goal_tag": goal, "allergies": allergies},
        sources=[],
        model="local-rules",
        degraded=True,
    )


# ── 语音输出（TTS 适配器，热插拔；edge-tts 缺失/失败 → 503 优雅降级）──
class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


@app.post("/api/tts", response_model=None)
async def tts(req: TTSRequest):
    from app import tts_service

    if not tts_service.tts_available():
        raise HTTPException(status_code=503, detail="语音服务暂不可用")
    try:
        audio = await tts_service.text_to_speech(req.text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("TTS 合成失败（降级）：%s", exc)
        raise HTTPException(status_code=503, detail="语音合成失败") from exc
    return Response(content=audio, media_type="audio/mpeg")


# ── P0 用户档案聚合（记忆模块：只读展示，零写库、零合规豁免） ──
_PROFILE_MEAL_TAGS = ("早餐", "午餐", "晚餐")


def _build_user_profile(user_id: str, session_id: str | None = None) -> dict:
    """P0 用户档案聚合（记忆模块·只读）：身份 + 偏好 + 行为 + AI 自动档案。

    全部复用现有数据（users / sessions.allergies+goal_tag / DietLog /
    health_state / taboo_map / users.profile_json），零新表；仅展示层反查，
    不参与合规判定（守卫 2/3：不新增「档案→免免责/免拒药」通道、自由文本不进 LLM）。

    产品决策：不再返回 next_step（目标切换/下一步建议均去掉）；AI 档案
    （profile_json）由对话规则检测自动维护，UI 不可编辑，此处只读聚合展示。
    """
    nickname: str | None = None
    since: str | None = None
    goal_tag: str | None = None
    allergy_ids: list[str] = []
    recent_meals: list[dict] = []
    db = SessionLocal()
    try:
        user = db.get(models.User, user_id)
        if user is not None:
            nickname = user.nickname
            if user.created_at is not None:
                # created_at 存 UTC，本地化到 +8 时区后取自然日
                since = (user.created_at + timedelta(hours=8)).strftime("%Y-%m-%d")
        # 会话画像：优先指定 session，其次该用户最近一条会话。
        # 二审 P1-2：指定 session 但属于其他用户 → 视为未指定（走该用户最近会话），
        # userB 不得读取 userA 的档案画像。
        sess = db.get(models.Session, session_id) if session_id else None
        if sess is not None and sess.user_id != user_id:
            sess = None
        if sess is None:
            sess = (
                db.query(models.Session)
                .filter(models.Session.user_id == user_id)
                .order_by(models.Session.created_at.desc())
                .first()
            )
        if sess is not None:
            goal_tag = sess.goal_tag
            allergy_ids = list(sess.allergies or [])
        # 最近台账（DietLog 最近 5 条，复用 _handle_diet_query 查询模式）
        logs = (
            db.query(models.DietLog)
            .filter(models.DietLog.user_id == user_id)
            .order_by(models.DietLog.created_at.desc(), models.DietLog.id.desc())
            .limit(5)
            .all()
        )
        recent_meals = [{"meal_tag": log.meal_tag, "content": log.content} for log in logs]
    except Exception:  # noqa: BLE001
        logger.exception("用户档案聚合失败（降级为空 profile）")
    finally:
        db.close()

    # 禁忌反查（与合规执行层同一 taboo_map id 体系，绝不另起炉灶）
    taboos = _load_taboos()
    by_id = {t["id"]: t for t in taboos}
    allergy_names = [by_id[a]["name"] for a in allergy_ids if a in by_id]
    excluded_list = excluded_foods(allergy_ids)
    disease_ids = [
        a for a in allergy_ids if a in by_id and by_id[a].get("disclaimer_type") == "disease"
    ]
    # disease_labels：按 disease 类禁忌 name 匹配 _DISEASE_LABELS（高血压→控盐 等）
    disease_labels: list[str] = []
    for did in disease_ids:
        name = by_id[did]["name"]
        for kw, label in _DISEASE_LABELS:
            if kw in name and label not in disease_labels:
                disease_labels.append(label)
                break

    today_summary = health_state.get_today_summary(user_id)
    recorded_tags = {m["meal_tag"] for m in today_summary["meals"]}
    today_missing = [t for t in _PROFILE_MEAL_TAGS if t not in recorded_tags]

    # AI 自动维护档案（users.profile_json）：只读聚合，供前端展示（用户不可编辑）
    ai_profile = user_profile.get_profile(user_id)

    return {
        "identity": {"nickname": nickname, "since": since},
        "preferences": {
            "goal_tag": goal_tag,
            "allergy_ids": allergy_ids,
            "allergy_names": allergy_names,
            "excluded_foods": excluded_list,
            "disease_labels": disease_labels,
        },
        "behavior": {
            "streak": health_state.get_streak(user_id),
            "today_meal_count": today_summary["meal_count"],
            "week_trend": health_state.get_week_trend(user_id),
            "recent_meals": recent_meals,
            "today_missing_meals": today_missing,
        },
        "ai_profile": ai_profile,
    }


@app.get("/api/state", response_model=UnifiedResponse)
def health_state_endpoint(
    user_id: str = Query(min_length=1, max_length=64),
    session_id: str | None = Query(default=None, max_length=64),
) -> UnifiedResponse:
    """健康状态引擎（M2/M3）+ 用户档案聚合（P0 记忆模块）。

    返回：连续坚持天数 + 今日饮食总结 + AI 欢迎语（M18）+ 只读 profile
    （身份/偏好/行为/AI 自动档案 ai_profile）+ taboo_options（前端档案渲染）。
    纯只读聚合，不写库；session_id 可选，缺省取该用户最近会话画像。
    产品决策：不再返回 next_step（目标切换/下一步建议均去掉）。
    """
    return make_response(
        {
            "streak": health_state.get_streak(user_id),
            "today_summary": health_state.get_today_summary(user_id),
            "week_trend": health_state.get_week_trend(user_id),
            "greeting": health_state.build_greeting(user_id),
            "profile": _build_user_profile(user_id, session_id),
            "taboo_options": [
                {
                    "id": t["id"],
                    "name": t["name"],
                    "disclaimer_type": t.get("disclaimer_type", "allergy"),
                }
                for t in _load_taboos()
            ],
        },
        sources=[],
        model="local-rules",
        degraded=True,
    )


# ── 前端静态资源（同源提供 demo UI，免 CORS）──
FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
