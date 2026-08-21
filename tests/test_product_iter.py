"""P2 产品化迭代测试：个性化记忆贯穿（过敏持久化 + 目标动态更新）+ 过敏反问追问。

覆盖团长核心需求「聊起来 + 记得我」：
- 首次声明过敏 → 回复以追问开头（含「具体」/「哪一类」）+ 末尾确认排除提示；
- 同一会话再问（不再声明）→ 不再追问（只问一次）+ 记忆已持久化仍排除；
- 消息隐含人群目标（血糖偏高）→ session.goal_tag 动态更新为调理类。

复用 conftest 的 temp DB 隔离；红线 R3 正文剔除不回退一并校验。
"""
from __future__ import annotations

from app import models
from app.database import SessionLocal

# 海鲜过敏（seafood_allergy）在 knowledge/taboo_map.json 中声明的排除食材。
SEAFOOD_EXCLUDED = {"虾仁", "三文鱼", "鳕鱼", "鲈鱼", "龙利鱼", "蛤蜊", "鱿鱼"}


def _chat(client, message: str, *, session_id: str, user_id: str, allergies: list[str] | None = None):
    """便捷封装：POST /api/chat。"""
    return client.post(
        "/api/chat",
        json={
            "user_id": user_id,
            "session_id": session_id,
            "message": message,
            "allergies": allergies or [],
        },
    )


def _session_row(session_id: str) -> models.Session | None:
    """直接查 temp DB 中的会话行（断言持久化结果）。"""
    db = SessionLocal()
    try:
        return db.get(models.Session, session_id)
    finally:
        db.close()


def _pre_exclusion_part(reply: str) -> str:
    """取回复中「⚠️ 已按您的禁忌排除」标记之前的部分（即回复正文）。"""
    marker = "⚠️ 已按您的禁忌排除"
    return reply.split(marker, 1)[0]


def test_p2_allergy_followup_once_and_persisted(client):
    """首次声明过敏：追问开头 + 排除提示末尾；次轮不再追问但记忆仍在（只问一次+持久化）。"""
    # 第一轮：声明「我对海鲜过敏」→ 回复以追问开头
    r1 = _chat(client, "我对海鲜过敏", session_id="p2a", user_id="u_p2")
    assert r1.status_code == 200
    reply1 = r1.json()["data"]["reply"]
    assert reply1.startswith("收到，已为您记录海鲜过敏"), f"未以追问开头: {reply1[:50]}"
    assert ("具体" in reply1) or ("哪一类" in reply1)
    # 末尾仍保留确认排除提示
    assert "已按您的禁忌排除以下食材" in reply1

    # 记忆已持久化：sessions.allergies 已写入（后续轮次仍生效）
    sess = _session_row("p2a")
    assert sess is not None
    assert "seafood_allergy" in (sess.allergies or [])

    # 第二轮：不再声明过敏 → 无追问（只问一次），但排除提示仍在（记忆贯穿）
    r2 = _chat(client, "推荐减脂方案", session_id="p2a", user_id="u_p2")
    assert r2.status_code == 200
    reply2 = r2.json()["data"]["reply"]
    assert not reply2.startswith("收到，已为您记录"), "次轮不应再次追问"
    assert "已按您的禁忌排除以下食材" in reply2
    # 红线 R3 不回退：正文（排除提示之前）不得含任一海鲜食材名
    pre = _pre_exclusion_part(reply2)
    assert not any(f in pre for f in SEAFOOD_EXCLUDED), (
        f"回复正文仍含禁忌海鲜: {[f for f in SEAFOOD_EXCLUDED if f in pre]}"
    )


def test_p2_goal_dynamic_update(client):
    """消息隐含人群目标（血糖偏高）→ session.goal_tag 动态更新为调理类。"""
    r = _chat(client, "我血糖偏高，三餐该怎么吃", session_id="p2b", user_id="u_p2")
    assert r.status_code == 200
    sess = _session_row("p2b")
    assert sess is not None
    assert sess.goal_tag == "调理"


# 坚果过敏追问话术原文（compliance.ALLERGY_FOLLOWUP["nut_allergy"] 逐字）：
# 追问里的精确食材名（花生/腰果/杏仁）必须原样保留，绝不能被 R3 剔除循环替换成乱码。
_FOLLOWUP_NUT = (
    "收到，已为您记录坚果过敏。为了更精准地帮您避开，"
    "方便告诉我具体是哪一类坚果吗？比如花生、腰果、杏仁，还是核桃？"
)


def test_p2_allergy_followup_keeps_original_text(client):
    """DEFECT-A 回归：坚果过敏追问原文完整（花生/腰果/杏仁 不被 R3 剔除循环乱码化），
    追问不经剔除循环、排除提示仍在尾部。"""
    r = _chat(client, "我对坚果过敏", session_id="p2d", user_id="u_p2")
    assert r.status_code == 200
    reply = r.json()["data"]["reply"]
    # 追问原文完整且位于回复开头（若被剔除循环乱码化，此精确串必然断裂）
    assert reply.startswith(_FOLLOWUP_NUT), f"追问原文被破坏: {reply[:80]}"
    # 追问段不得出现「（已按禁忌剔除）」乱码（追问文本完全不经剔除循环）
    assert "（已按禁忌剔除）" not in _FOLLOWUP_NUT
    # 排除提示仍在尾部（追问之后）
    assert "已按您的禁忌排除以下食材" in reply
    assert reply.index("已按您的禁忌排除以下食材") > len(_FOLLOWUP_NUT)
    # 红线 R3 正文剔除仍生效：排除提示之前的正文部分（追问之后）中，正文若含禁忌
    # 食材应被替换为「（已按禁忌剔除）」（知识库脂肪/餐盘块含「坚果」「牛油果」）。
    pre = _pre_exclusion_part(reply)
    body_part = pre[len(_FOLLOWUP_NUT):]
    assert "（已按禁忌剔除）" in body_part, f"R3 正文剔除失效: {body_part[:120]}"


def test_p2_request_word_does_not_overwrite_goal(client):
    """DEFECT-B 回归：已有 goal=调理 的会话，用户问「推荐减脂方案」（请求语义）
    → goal_tag 不被覆盖为减脂，仍保持调理。"""
    r_sess = client.post(
        "/api/session",
        json={
            "user_id": "u_p2",
            "session_id": "p2c",
            "action": "new",
            "goal_tag": "调理",
            "allergies": [],
        },
    )
    assert r_sess.status_code == 200
    r = _chat(client, "推荐减脂方案", session_id="p2c", user_id="u_p2")
    assert r.status_code == 200
    sess = _session_row("p2c")
    assert sess is not None
    assert sess.goal_tag == "调理"


def test_p2_geiwo_bangwo_request_does_not_overwrite_goal(client):
    """QA Round2 遗留风险#1 回归：裸「我」不再构成个人声明——
    已有 goal=调理 的会话发「给我推荐减脂方案」「帮我推荐减脂方案」
    （纯请求语义，含「我」子串但不含「我的/本人」或健康状态短语）
    → goal_tag 仍为调理，不被覆盖为减脂。"""
    r_sess = client.post(
        "/api/session",
        json={
            "user_id": "u_p2",
            "session_id": "p2e",
            "action": "new",
            "goal_tag": "调理",
            "allergies": [],
        },
    )
    assert r_sess.status_code == 200
    for msg in ("给我推荐减脂方案", "帮我推荐减脂方案"):
        r = _chat(client, msg, session_id="p2e", user_id="u_p2")
        assert r.status_code == 200
        sess = _session_row("p2e")
        assert sess is not None
        assert sess.goal_tag == "调理", f"{msg} 不应覆盖 goal_tag: {sess.goal_tag}"
