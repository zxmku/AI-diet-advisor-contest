"""FastAPI 入口（M3 API 网关）：注册全部 6 个端点 + 全局中间件。

当前为「检索 grounding + 规则合规」实现（local-rules 降级模式）：
- 回答 = BM25 检索素材 A/B 原文块（零编造、带来源标注）；
- 合规层（compliance.py）硬注入医疗免责 + 拒答用药 + 禁忌排除；
- 多轮对话落库（MOD-04），history 真实可读；
- 推荐/快捷基于素材 B 真实方案生成理由（无「接口联调中」占位）；
- 未配置 DeepSeek Key 时以上述规则兜底，配置后可由 app/llm.py 升级为 LLM 合成（M13）。
"""
from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from jieba import lcut
from pathlib import Path

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
    DISCLAIMER_STANDARD,
    detect_allergies,
    excluded_foods,
    is_disease_query,
    is_medication_query,
)
from app.database import SessionLocal, init_db
from app import models
from app.retrieval import SourceChunk, get_retriever
from app import nutrition_lookup
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
    return {
        "status": "ok",
        "llm_key_set": bool(config.DEEPSEEK_API_KEY),
        "llm_model": config.DEEPSEEK_MODEL if config.DEEPSEEK_API_KEY else None,
        "database": "sqlite" if config.DATABASE_URL.startswith("sqlite") else "external",
        # BUG-5 加固：暴露营养速查表就绪状态，使静默降级可被一眼观测。
        "nutrition_table_ready": nutrition_lookup.is_nutrition_table_ready(),
        "nutrition_table_rows": nutrition_lookup.nutrition_table_status()["rows"],
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


def _token_overlap(content: str, query_tokens: set[str]) -> int:
    """回复块优选：统计块中命中的查询词数（同义重叠优先）。"""
    if not query_tokens:
        return 0
    return sum(1 for t in query_tokens if t in content)


# BUG-5：数值问答意图词 & 数值单位（命中则进入二次排序）
_NUMERIC_HINTS = ("千卡", "卡路里", "卡", "kcal", "热量", "蛋白质", "脂肪", "碳水", "克", "g", "多少", "几")
_VALUE_UNITS = ("千卡", "卡路里", "卡", "kcal", "g", "克", "蛋白质", "脂肪", "碳水", "热量")
# 数值问答里需剔除的疑问/停用词，剩下的实词当作候选食材名
_NUM_STOP = {"多少", "几", "吃", "吗", "的", "能", "有", "是", "怎么", "如何", "什么", "可以", "想", "要", "该", "这", "那", "我", "你", "它"}
# 平台/定价类问题优先选自 C 库（平台私有内容），避免被营养块误排（缺陷 A 修复）
_PLATFORM_HINTS = ("会员", "价格", "多少钱", "收费", "费用", "套餐", "订阅", "平台", "开通", "升级", "付费", "多少钱一个月")
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
    """从 DB 读取会话已声明的禁忌（与对话触发、请求声明合并）。"""
    if not session_id:
        return []
    db = SessionLocal()
    try:
        sess = db.get(models.Session, session_id)
        return list(sess.allergies) if sess else []
    except Exception:  # noqa: BLE001
        return []
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


def _session_has_disease(session_id: str | None) -> bool:
    """会话历史中是否曾出现疾病/医疗意图（跨轮延续免责）。"""
    if not session_id:
        return False
    db = SessionLocal()
    try:
        msgs = db.query(models.Message).filter(models.Message.session_id == session_id).all()
        return any(is_disease_query(m.content) for m in msgs)
    except Exception:  # noqa: BLE001
        return False
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
    """多轮对话：检索 grounding 回答 + 合规层（免责/拒药/禁忌）+ 对话落库。"""
    message = validate_message(req.message)
    # 多轮上下文：最近用户消息参与检索；疾病意图跨轮延续免责
    recent = _recent_user_messages(req.session_id, 2)
    ctx_query = " ".join([message] + recent)
    disease = is_disease_query(message) or _session_has_disease(req.session_id)

    chunks = get_retriever().retrieve(ctx_query, top_k=4, current_query=message)

    reply = ""
    intent = "nutrition_qa"
    model_tag = "local-rules"
    degraded = True

    # ── BUG-5 数值问答确定性工具命中（团长指令：数值必须工具计算，禁止模型推理）──
    # 在模糊检索之上插入确定性强校验：命中权威表则直接返回精确值，跳过 _pick_reply_chunk。
    numeric_hit = False
    numeric_miss = False  # 数值意图命中但表中无该食材：诚实告知，不臆测、不回退 BM25
    sources: list = list(chunks)
    is_num, food = nutrition_lookup.is_numeric_lookup_query(message)
    if is_num and food:
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
        reply = (
            f"【营养速查表】表中暂未收录「{food}」的精确营养数据，无法给出确定数值，恕不臆测。\n"
            "您可以换问表中已有食材（如鸡胸肉、糙米、西兰花、三文鱼、鸡蛋等），"
            "或告诉我您想了解的营养维度（热量 / 蛋白质 / 脂肪 / 碳水）。"
        )
        intent = "nutrition_lookup_miss"
        sources = []

    # 禁忌识别：请求声明 + 对话触发 + 会话已声明，三路合并
    allergy_ids = set(req.allergies) | set(detect_allergies(message)) | set(
        _session_allergies(req.user_id, req.session_id)
    )
    excluded = excluded_foods(list(allergy_ids))

    disclaimer = DISCLAIMER_STANDARD if (disease or is_medication_query(message)) else None

    if not numeric_hit and not numeric_miss:
        if not chunks:
            reply = (
                "抱歉，我暂时没有在膳食知识库中找到与您问题直接相关的内容。"
                "您可以换个说法，或告诉我您更关注减脂 / 增肌 / 慢病调理中的哪一类？"
            )
            intent = "chitchat"
        else:
            top = _pick_reply_chunk(chunks, message)
            intent = "platform" if top.source == "C" else "nutrition_qa"
            # 命中检索块 -> 优先尝试 LLM 合成（Key 就绪且非用药类问题时）。
            # 用药类问题由合规层硬拦截，不经 LLM，避免任何用药建议风险。
            used_llm = False
            if llm.is_enabled() and not is_medication_query(message):
                llm_reply = llm.synthesize(
                    message,
                    chunks,
                    history=_recent_user_messages(req.session_id, 3),
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

    if excluded:
        reply += f"\n\n⚠️ 已按您的禁忌排除以下食材：{', '.join(excluded)}"

    _persist_turn(req.user_id, req.session_id, message, reply, [c.model_dump() for c in chunks])
    return make_response(
        {"reply": reply, "intent": intent, "goal_tag": None},
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
    """基于素材 B 检索生成真实推荐（去占位文案），并按禁忌排除食材。"""
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
    plan = {
        "name": base["name"],
        "kcal_range": base["kcal_range"],
        "foods": foods,
        "reason": reason or f"{base['name']}：基于《中国居民膳食指南》与素材 B 方案生成。",
        "source_chapter": chunks[0].chapter if chunks else None,
    }
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
    return make_response(
        {"plans": [plan], "intent": "recommend", "goal_tag": goal},
        sources=get_retriever().retrieve(f"{goal or '减脂'} 饮食计划 营养素目标 推荐食材", top_k=2),
        model="local-rules",
        degraded=True,
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
        sessions = (
            [{"session_id": session_id, "message_count": len(messages)}]
            if session_id and messages
            else []
        )
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
    """新建/切换会话，携带人群标签与禁忌列表（真实写入 sessions 表）。"""
    db = SessionLocal()
    try:
        # 确保 users 表存在对应行：FK 约束开启时，缺失 user 行会触发 IntegrityError → 500
        user = db.get(models.User, req.user_id)
        if user is None:
            user = models.User(id=req.user_id, nickname=None)
            db.add(user)
            db.flush()

        sess = db.get(models.Session, req.session_id) if req.session_id else None
        if sess is None:
            sess = models.Session(
                id=req.session_id or uuid.uuid4().hex,
                user_id=req.user_id,
                goal_tag=req.goal_tag,
                allergies=list(req.allergies),
            )
            db.add(sess)
        else:
            if req.goal_tag is not None:
                sess.goal_tag = req.goal_tag
            if req.allergies:
                sess.allergies = list(req.allergies)
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


# ── 前端静态资源（同源提供 demo UI，免 CORS）──
FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
