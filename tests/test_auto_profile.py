"""AI 自动写用户档案测试：幂等迁移 + 对话规则检测 + 字段级去重 + API 级写入。

覆盖（对应团队规格）：
- 幂等迁移：init_db 后 users 表有 profile_json 列（老库 ALTER 平滑升级）；
- 检测规则：孕期/昵称/口味/运动量各一条第一人称陈述 → 提取正确值；
  问句/第三人称 → 不提取（非第一人称陈述）；
- 去重：同一字段已有非空值不覆盖（保留先说的），不同字段互不影响；
- API 级：chat() 自动写库，/api/state 的 profile.ai_profile 出现该字段。

复用 conftest 的 temp DB 隔离（session 级 TestClient 触发 init_db）。
"""
from __future__ import annotations

from sqlalchemy import text

from app.database import SessionLocal
from app.user_profile import (
    add_profile_fields,
    detect_user_profile,
    get_profile,
    set_profile_field,
)


def _chat(client, message, *, session_id, user_id):
    return client.post(
        "/api/chat",
        json={
            "user_id": user_id,
            "session_id": session_id,
            "message": message,
            "allergies": [],
        },
    )


# ── 幂等迁移：users 表有 profile_json 列 ─────────────────────────
def test_users_table_has_profile_json_column(client):
    """init_db（conftest 已触发）后 users 表存在 profile_json 列。"""
    db = SessionLocal()
    try:
        rows = db.execute(text("PRAGMA table_info(users)")).fetchall()
        cols = [r[1] for r in rows]
    finally:
        db.close()
    assert "profile_json" in cols, "users 表缺 profile_json 列（迁移未生效）"


# ── 检测规则：第一人称陈述 → 提取正确值 ──────────────────────────
def test_detect_pregnancy():
    assert detect_user_profile("我怀孕了")["pregnancy"] == "孕期"


def test_detect_nickname():
    assert detect_user_profile("我是小明")["nickname"] == "小明"
    assert detect_user_profile("我叫小优")["nickname"] == "小优"
    assert detect_user_profile("可以叫我小优")["nickname"] == "小优"


def test_detect_taste():
    assert detect_user_profile("我喜欢吃辣")["taste"] == "爱吃辣"
    assert detect_user_profile("我不吃香菜")["taste"] == "不吃香菜"


def test_detect_exercise():
    assert detect_user_profile("我每天跑步")["exercise"] == "每天跑步"


def test_detect_meal_style():
    assert detect_user_profile("我平时喜欢吃面食")["meal_style"] == "爱吃面食"
    assert detect_user_profile("我吃素")["meal_style"] == "素食"


def test_question_and_third_person_not_extracted():
    """问句/第三人称不是第一人称陈述 → 不提取。"""
    assert detect_user_profile("你能吃辣吗") == {}
    assert detect_user_profile("你能吃辣吗？") == {}
    assert detect_user_profile("我朋友怀孕了") == {}
    assert detect_user_profile("我怀孕了吗") == {}


def test_self_description_not_nickname():
    """「我是孕妇」是自我描述 → 提取孕期但不把「孕妇」当昵称。"""
    prof = detect_user_profile("我是孕妇")
    assert prof.get("pregnancy") == "孕期"
    assert "nickname" not in prof


# ── 字段级去重：已有非空值不覆盖，不同字段互不影响 ────────────────
# 注：依赖 client 夹具触发 init_db（建表），保证直连 SessionLocal 可用。
def test_set_profile_field_no_overwrite(client):
    """先写 pregnancy=孕期，再写 pregnancy=孕早期 → 保留第一次（返回 False）。"""
    user_id = "u_auto_dedup"
    assert set_profile_field(user_id, "pregnancy", "孕期") is True
    assert set_profile_field(user_id, "pregnancy", "孕早期") is False
    assert get_profile(user_id)["pregnancy"] == "孕期"


def test_set_profile_field_different_fields_independent(client):
    """不同字段互不影响：pregnancy 与 taste 可同时存在。"""
    user_id = "u_auto_dedup2"
    set_profile_field(user_id, "pregnancy", "孕期")
    set_profile_field(user_id, "taste", "爱吃辣")
    prof = get_profile(user_id)
    assert prof["pregnancy"] == "孕期"
    assert prof["taste"] == "爱吃辣"


def test_add_profile_fields_partial_dedup(client):
    """批量写入：已有字段跳过，未写字段写入；返回实际写入集。"""
    user_id = "u_auto_dedup3"
    set_profile_field(user_id, "pregnancy", "孕期")
    written = add_profile_fields(user_id, {"pregnancy": "孕晚期", "exercise": "久坐"})
    assert written == {"exercise": "久坐"}
    prof = get_profile(user_id)
    assert prof["pregnancy"] == "孕期"  # 保留先说的
    assert prof["exercise"] == "久坐"


# ── API 级：chat() 自动写库 + /api/state 展示 ────────────────────
def test_chat_writes_profile_and_state_shows_it(client):
    """POST chat「我怀孕了，我叫小优」→ DB 有值 + /api/state profile.ai_profile 展示。"""
    user = "u_auto_api"
    sid = "auto_api_sid"
    r = _chat(client, "我怀孕了，我叫小优", session_id=sid, user_id=user)
    assert r.status_code == 200

    # DB 有值（写入前判断已有则覆盖与否在此由规则保证）
    prof = get_profile(user)
    assert prof.get("pregnancy") == "孕期"
    assert prof.get("nickname") == "小优"

    # /api/state 的 profile.ai_profile 出现孕期字段（只读展示，用户不可编辑）
    body = client.get(f"/api/state?user_id={user}&session_id={sid}").json()
    ai = body["data"]["profile"].get("ai_profile", {})
    assert ai.get("pregnancy") == "孕期"
    assert ai.get("nickname") == "小优"


def test_chat_profile_no_overwrite_across_turns(client):
    """跨轮：用户先「我怀孕了」后「我是孕早期」→ 保留第一次的「孕期」。"""
    user = "u_auto_api2"
    sid = "auto_api_sid2"
    assert _chat(client, "我怀孕了", session_id=sid, user_id=user).status_code == 200
    assert _chat(client, "我是孕早期", session_id=sid, user_id=user).status_code == 200
    assert get_profile(user).get("pregnancy") == "孕期"
