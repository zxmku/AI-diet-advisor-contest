"""V08-V10 + 遗留加固 回归测试（第二轮收尾）。

覆盖：
- V10 裸「药」误报：「山药/药膳/药食同源」不拒药；「吃药期间能喝牛奶吗/我吃了什么药」
  仍拒药（组合词方案不破坏既有拒药）；
- 遗留① 占位符加固：validate_message 剥离 C0 控制符；_redact_excluded 伪造占位符不可命中；
- V08 一餐触发词补全：「晚上吃什么/中午吃什么」在有目标会话中出餐；问候「晚上好」不被劫持；
- V09 领域门控补 GI：「低GI食物有哪些」不再漏判成闲聊。

复用 conftest 的 temp DB 隔离。
"""
from __future__ import annotations

import pytest

REFUSE_MARK = "不提供用药建议"


def _chat(client, message, *, session_id, user_id="u_v10"):
    r = client.post(
        "/api/chat",
        json={"user_id": user_id, "session_id": session_id, "message": message},
    )
    assert r.status_code == 200
    return r.json()


def _new_session(client, session_id, goal_tag="减脂", user_id="u_v10"):
    r = client.post(
        "/api/session",
        json={
            "user_id": user_id,
            "session_id": session_id,
            "action": "new",
            "goal_tag": goal_tag,
            "allergies": [],
        },
    )
    assert r.status_code == 200


# ── V10 裸「药」误报：山药/药膳/药食同源 不拒药 ──────────────
@pytest.mark.parametrize("msg", ["山药怎么吃", "药膳排骨", "药食同源"])
def test_v10_no_refuse_on_yao_false_positive(client, msg):
    """含「药」字但非用药咨询：不得拒药（去掉裸「药」关键词）。"""
    body = _chat(client, msg, session_id="v10_ok", user_id="u_v10")
    reply = body["data"]["reply"]
    assert REFUSE_MARK not in reply, f"「{msg}」被误拒药: {reply[:80]}"
    assert body["data"]["intent"] != "medication_refuse"


@pytest.mark.parametrize(
    "msg",
    ["吃药期间能喝牛奶吗", "我吃了什么药", "减肥药能随便吃吗"],
)
def test_v10_medication_still_refused(client, msg):
    """组合词方案：吃药/什么药/减肥药 等问法必须仍拒药。"""
    body = _chat(client, msg, session_id="v10_ref", user_id="u_v10")
    assert REFUSE_MARK in body["data"]["reply"], f"「{msg}」未拒答用药"


def test_v10_ibuprofen_still_refused(client):
    """既有红线不回退：布洛芬（无「药」字）仍拒药。"""
    body = _chat(client, "布洛芬能吃吗", session_id="v10_ibu", user_id="u_v10")
    assert REFUSE_MARK in body["data"]["reply"]


# ── 遗留① 占位符加固 ────────────────────────────────────
def test_validate_message_strips_c0_controls():
    """validate_message 剥离 C0 控制符（\x00-\x1f，保留 \n）。"""
    from app.middleware.guard import validate_message

    assert validate_message("a\x00b\x1fc") == "abc"
    assert validate_message("吃\x00饭\x1f了") == "吃饭了"
    assert validate_message("换行\n保留") == "换行\n保留"


def test_redact_excluded_fake_placeholder_not_hit():
    """伪造占位符（旧格式 \x00qN\x00 / 新格式 \x1fQN\x1f）不得命中引号占位。"""
    from app.main import _redact_excluded

    # 伪造占位符文本 + 真实引号内容：引号内食材名保持原样、无剔除损坏标记。
    for fake in ("\x00q0\x00", "\x1fQ0\x1f", "\x1fQdeadbeef:0\x1f"):
        out = _redact_excluded(f"「坚果」的热量 {fake}", ["坚果"])
        assert "「坚果」" in out, f"引号内容被破坏: {out!r}"
        assert "（已按禁忌剔除）" not in out, f"引号内食材被误剔除: {out!r}"


def test_chat_with_c0_controls_sane(client):
    """API 级：消息带 C0 控制符被入口剥离，拒药/路由不受影响、回复无控制符残留。"""
    body = _chat(client, "我\x1f吃了\x00布洛芬怎么办", session_id="v10_c0", user_id="u_v10")
    assert REFUSE_MARK in body["data"]["reply"]
    assert "\x1f" not in body["data"]["reply"]
    assert "\x00" not in body["data"]["reply"]


# ── V08 一餐触发词补全 ───────────────────────────────────
@pytest.mark.parametrize("msg", ["晚上吃什么", "中午吃什么", "早上吃什么"])
def test_v08_time_meal_intent(client, msg):
    """有目标会话问「时段+吃什么」→ 直接出餐（不再漏判成闲聊）。"""
    _new_session(client, "v08a", goal_tag="减脂")
    body = _chat(client, msg, session_id="v08a", user_id="u_v10")
    assert body["data"]["intent"] == "meal", f"「{msg}」未出餐: {body['data']['intent']}"
    assert "为你搭配的" in body["data"]["reply"]


def test_v08_greeting_not_hijacked(client):
    """回归防护：裸时段问候「晚上好/中午好」不得被一餐触发词劫持成追问目标。"""
    for i, msg in enumerate(("晚上好", "中午好", "早上好")):
        body = _chat(client, msg, session_id=f"v08b{i}", user_id="u_v10")
        assert body["data"]["intent"] != "meal_goal_ask", f"问候「{msg}」被劫持成追问目标"
        assert body["data"]["intent"] == "chitchat", f"问候「{msg}」未走闲聊: {body['data']['intent']}"


# ── V09 领域门控补 GI ───────────────────────────────────
@pytest.mark.parametrize("msg", ["低GI食物有哪些", "GI值多少算低", "升糖指数是什么"])
def test_v09_gi_not_chitchat(client, msg):
    """GI / 升糖指数 属膳食领域：不得漏判成闲聊。"""
    body = _chat(client, msg, session_id="v09a", user_id="u_v10")
    assert body["data"]["intent"] != "chitchat", f"「{msg}」被误判成闲聊"
