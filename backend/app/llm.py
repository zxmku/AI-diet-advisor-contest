"""LLM 合成层（M13）：在检索 grounding 之上调用 DeepSeek 生成自然语言答复。

安全与降级纪律（与本地规则层一致，绝不裸奔）：
- 仅当 config.DEEPSEEK_API_KEY 非空才启用；否则 is_enabled()=False，上层必走 local-rules。
- 所有回答必须严格基于传入的检索块（grounding），禁止模型凭空编造营养/医疗结论；
- 系统提示硬性禁止用药建议，医疗/疾病类问题由上层合规层拦截（不进入此函数）；
- 任何异常（网络/超时/非 200/JSON 解析失败/依赖缺失）一律返回 None，上层无缝降级 local-rules；
- 密钥只从 config 读取，代码内零明文；日志与响应绝不回显密钥。
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence

from app import config
from app.cost_gate import cost_gate
from app.retrieval import SourceChunk
from app.time_context import get_meal_time_context

logger = logging.getLogger("healthpick.llm")

_SYSTEM_PROMPT = (
    "你是「健康优选」AI 智能膳食顾问，服务于有减脂 / 增肌 / 慢病调理需求的人群。"
    "请严格依据下方【参考资料】作答，用简体中文、亲切但专业、简洁（200 字以内）。\n"
    "规则：\n"
    "1. 只引用资料中的事实，不得编造营养数值或医疗结论；资料未覆盖时，明确说「资料未提及」，不要猜测。\n"
    "2. 不提供任何用药建议，不诊断疾病；涉及疾病或用药请引导用户咨询医生。\n"
    "3. 若用户有已声明的食物禁忌，主动提醒避开相关食材。\n"
    "4. 可就膳食搭配、分量、烹饪方式给出实用建议。\n"
    "5. 若用户只是打招呼（你好/嗨/在吗/早上好）或简短闲聊，用 1-2 句简短自然回应（≤30 字），"
    "像真人聊天，不要展开自我介绍、不要罗列能力、不要加引导语；询问你的身份时可一句话说明；"
    "遇到明显超出膳食/营养/健康范畴的请求，礼貌说明你的服务范围并引导回减脂/增肌/调理。\n"
    "6. 用户表达孤独、疲惫、没胃口或情绪低落时，先共情再给轻饮食建议，语气温暖不敷衍。\n"
    "7. 先共情，后建议：当用户表达加班累、嘴馋、想吃炸鸡奶茶、减脂平台期焦虑、控糖焦虑时，"
    "严禁生硬说教或机械拒绝；先给予情绪理解（身体疲倦与多巴胺渴求是正常反应，不必苛责自己），"
    "再给出无痛可落地的替代方案（如想吃炸鸡→非油炸烤鸡胸/无糖饮品，便利店/外卖场景直接给点单建议），"
    "而不是简单否定。"
)


def is_enabled() -> bool:
    """LLM 模式是否可用（取决于 Key 是否已配置）。"""
    return bool(config.DEEPSEEK_API_KEY)


def _grounding_text(chunks: Sequence[SourceChunk], limit: int = 3) -> str:
    """把检索块拼成可读的 grounding 上下文（带来源标注）。"""
    if not chunks:
        return "（当前问题暂无直接相关的参考资料）"
    parts: list[str] = []
    for i, c in enumerate(chunks[:limit], 1):
        head = f"【资料{i}】（来源{c.source} · {c.chapter} · {c.section}）"
        parts.append(f"{head}\n{c.content.strip()}")
    return "\n\n".join(parts)


def _post_json(url: str, headers: dict, payload: dict, timeout: int) -> dict | None:
    """POST JSON 并返回解析后的 dict；优先 httpx，回退 requests，再回退 urllib（标准库）。

    任何异常（含依赖缺失）一律返回 None，交由上层降级。
    """
    try:
        try:
            import httpx

            resp = httpx.post(url, headers=headers, json=payload, timeout=timeout)
            return resp.json()
        except ImportError:
            pass
        try:
            import requests

            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            return resp.json()
        except ImportError:
            pass
        # 标准库兜底：保证即使 httpx/requests 均未安装也不裸奔。
        import urllib.request

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM 请求异常（降级 local-rules）：%s", exc)
        return None


def synthesize(
    user_message: str,
    chunks: Sequence[SourceChunk],
    history: Sequence[str] | None = None,
    *,
    max_tokens: int = 400,
    temperature: float = 0.3,
    session_id: str | None = None,
    user_id: str | None = None,
    excluded_foods: list[str] | None = None,
    state_context: str | None = None,
) -> str | None:
    """基于检索块合成自然语言答复。

    返回模型文本；任何失败返回 None（上层降级 local-rules）。
    history 为最近用户消息（旧→新），用于轻量多轮上下文。
    session_id 用于 M17 成本闸门：会话级限速 + 记账归属（可为 None）。
    user_id 用于 M17 回复缓存隔离（V06）：缓存键并入用户维度，杜绝跨用户串扰。
    excluded_foods 为已声明禁忌食材清单（红线②）：非空时注入系统提示，
    强制模型回答中严禁出现或推荐这些食材；为空时行为与之前完全一致。

    M17 成本闸门（在发起真实 API 调用之前）：
    - 当日预算超限（LLM_DAILY_BUDGET_TOKENS）→ 直接返回 None 自动降级；
    - 回复缓存仅覆盖**无上下文的单轮请求**（history 为空）：同消息 TTL 内命中
      直接返回缓存，零花费；带 history 的多轮请求完全跳过缓存读/写（直通 API），
      杜绝跨会话上下文串扰（DEFECT-1 修复）；
    - V06 修复：缓存键 = hash(user_id/session_id + user_message + excluded_foods
      + state_context)，仅按 user_message 做键会把 A 用户（海鲜过敏）的禁忌回复
      命中给 B 用户（无禁忌）——跨用户串扰；
    - 调用成功后记账（usage 缺失则按消息体/回复长度估算）；仅无上下文的单轮请求写缓存。

    允许传入空 chunks：用于打招呼 / 闲聊 / 身份问询等场景，靠系统提示第 5 条
    约束模型行为——仍以「资料未提及则不臆测」为铁律，绝不裸奔。
    """
    if not is_enabled():
        return None

    # M17 预算闸门：当日已用 token 达到预算即降级，不再发起任何调用。
    if not cost_gate.check_budget():
        logger.info("LLM 当日预算已用尽（tokens >= %s），降级 local-rules", cost_gate.budget_tokens)
        return None

    # M17 回复缓存：仅覆盖无上下文的单轮请求（history 为空）。
    # 缓存键只含 user_message，无法区分多轮上下文；history 非空时若复用缓存，
    # 会把其他会话的上下文带进回复（DEFECT-1 跨会话串扰）。因此 history 非空时
    # 完全跳过缓存读与缓存写（读直通 API，成功后也不写缓存）。
    # V06：键并入 用户维度 + 消息 + 禁忌 + 健康状态 的哈希（sha1），既隔离跨用户
    # 串扰，又避免把用户原始消息明文落进缓存文件。
    use_cache = not history
    if use_cache:
        cache_scope = "|".join(
            [
                user_id or session_id or "_anonymous",
                user_message.strip(),
                "|".join(sorted(excluded_foods or [])),
                state_context or "",
            ]
        )
        cache_key = hashlib.sha1(cache_scope.encode("utf-8")).hexdigest()
    else:
        cache_key = None
    if use_cache:
        cached = cost_gate.cache_get(cache_key)
        if cached is not None:
            return cached

    # M17 会话限速：单会话 60 秒窗口内调用次数达到限额即降级（缓存命中不占额度）。
    if not cost_gate.check_rate(session_id):
        logger.info("LLM 会话限速触发（session=%s），降级 local-rules", session_id)
        return None

    grounding = _grounding_text(chunks)
    # 红线②「禁忌必排除」：用户已声明禁忌时，注入系统提示强制模型不得出现/推荐。
    system_prompt = _SYSTEM_PROMPT
    if excluded_foods:
        system_prompt += (
            "\n用户已声明以下食物禁忌，回答中严禁出现或推荐这些食材"
            "（也不要用别名变体暗示）：" + "、".join(excluded_foods)
        )
    # 时间感知注入：当前时段问候与建议焦点（管道适配器，热插拔）
    _tctx = get_meal_time_context()
    system_prompt += f"\n当前时段：{_tctx['period']}（{_tctx['focus']}）"
    # 成长记忆注入：用户近期饮食状态（昨天/今天/坚持情况），供「结合你的情况」式回答
    if state_context:
        system_prompt += f"\n【用户近期状态】{state_context}"
    messages = [
        {
            "role": "system",
            "content": system_prompt + "\n\n【参考资料】\n" + grounding,
        }
    ]
    for h in (history or [])[-3:]:
        messages.append({"role": "user", "content": h})
    messages.append({"role": "user", "content": user_message})

    headers = {
        "Authorization": f"Bearer {config.DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    url = f"{config.DEEPSEEK_BASE_URL.rstrip('/')}/chat/completions"
    data = _post_json(url, headers, payload, timeout=15)
    if not data:
        return None
    try:
        content = data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        logger.warning("LLM 响应解析失败（降级 local-rules）：%s", exc)
        return None
    # M17 记账 + 缓存：仅当拿到有效回复才记录（失败/空回复不计费、不缓存）。
    if content:
        usage = data.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        if prompt_tokens <= 0 and completion_tokens <= 0:
            # 响应无 usage 字段：按消息体长度*1.3 / 回复字符数粗略估算。
            messages_json = json.dumps(messages, ensure_ascii=False)
            prompt_tokens = int(len(messages_json) * 1.3)
            completion_tokens = len(content)
        cost_gate.record(session_id, prompt_tokens, completion_tokens)
        # 仅无上下文的单轮请求写缓存；带 history 的多轮请求不落缓存（防串扰）。
        if use_cache:
            cost_gate.cache_set(cache_key, content)
    return content or None
