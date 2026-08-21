"""LLM 合成层（M13）：在检索 grounding 之上调用 DeepSeek 生成自然语言答复。

安全与降级纪律（与本地规则层一致，绝不裸奔）：
- 仅当 config.DEEPSEEK_API_KEY 非空才启用；否则 is_enabled()=False，上层必走 local-rules。
- 所有回答必须严格基于传入的检索块（grounding），禁止模型凭空编造营养/医疗结论；
- 系统提示硬性禁止用药建议，医疗/疾病类问题由上层合规层拦截（不进入此函数）；
- 任何异常（网络/超时/非 200/JSON 解析失败/依赖缺失）一律返回 None，上层无缝降级 local-rules；
- 密钥只从 config 读取，代码内零明文；日志与响应绝不回显密钥。
"""
from __future__ import annotations

import json
import logging
from collections.abc import Sequence

from app import config
from app.cost_gate import cost_gate
from app.retrieval import SourceChunk

logger = logging.getLogger("healthpick.llm")

_SYSTEM_PROMPT = (
    "你是「健康优选」AI 智能膳食顾问，服务于有减脂 / 增肌 / 慢病调理需求的人群。"
    "请严格依据下方【参考资料】作答，用简体中文、亲切但专业、简洁（200 字以内）。\n"
    "规则：\n"
    "1. 只引用资料中的事实，不得编造营养数值或医疗结论；资料未覆盖时，明确说「资料未提及」，不要猜测。\n"
    "2. 不提供任何用药建议，不诊断疾病；涉及疾病或用药请引导用户咨询医生。\n"
    "3. 若用户有已声明的食物禁忌，主动提醒避开相关食材。\n"
    "4. 可就膳食搭配、分量、烹饪方式给出实用建议。\n"
    "5. 若用户只是打招呼、闲聊或询问你的身份，自然友好回应即可，无需引用资料；"
    "遇到明显超出膳食/营养/健康范畴的请求，礼貌说明你的服务范围并引导回减脂/增肌/调理。"
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
) -> str | None:
    """基于检索块合成自然语言答复。

    返回模型文本；任何失败返回 None（上层降级 local-rules）。
    history 为最近用户消息（旧→新），用于轻量多轮上下文。
    session_id 用于 M17 成本闸门：会话级限速 + 记账归属（可为 None）。

    M17 成本闸门（在发起真实 API 调用之前）：
    - 当日预算超限（LLM_DAILY_BUDGET_TOKENS）→ 直接返回 None 自动降级；
    - 同消息 TTL 内命中回复缓存（LLM_CACHE_TTL_SECONDS）→ 直接返回缓存，零花费；
    - 调用成功后记账（usage 缺失则按消息体/回复长度估算）并写入缓存。

    允许传入空 chunks：用于打招呼 / 闲聊 / 身份问询等场景，靠系统提示第 5 条
    约束模型行为——仍以「资料未提及则不臆测」为铁律，绝不裸奔。
    """
    if not is_enabled():
        return None

    # M17 预算闸门：当日已用 token 达到预算即降级，不再发起任何调用。
    if not cost_gate.check_budget():
        logger.info("LLM 当日预算已用尽（tokens >= %s），降级 local-rules", cost_gate.budget_tokens)
        return None

    # M17 回复缓存：同消息 TTL 内命中直接返回，零 API 花费。
    cache_key = user_message.strip()
    cached = cost_gate.cache_get(cache_key)
    if cached is not None:
        return cached

    # M17 会话限速：单会话 60 秒窗口内调用次数达到限额即降级（缓存命中不占额度）。
    if not cost_gate.check_rate(session_id):
        logger.info("LLM 会话限速触发（session=%s），降级 local-rules", session_id)
        return None

    grounding = _grounding_text(chunks)
    messages = [
        {
            "role": "system",
            "content": _SYSTEM_PROMPT + "\n\n【参考资料】\n" + grounding,
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
        cost_gate.cache_set(cache_key, content)
    return content or None
