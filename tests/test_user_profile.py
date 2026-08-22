"""P0 用户档案（记忆模块）测试：/api/state profile 聚合 + 守卫 1/2 校验。

覆盖：
- /api/state 返回 profile 结构（identity/preferences/behavior/next_step 键齐全）+
  taboo_options（前端档案禁忌编辑渲染）；
- 建会话 + 声明海鲜过敏 + 记台账 → profile 聚合出 名字/排除食材/坚持天数/最近台账/
  今日未记录餐次/下一步建议；
- 守卫1：POST /api/session allergies 含未知 id → 白名单过滤（返回与写库均不含未知 id）；
- 守卫2：档案疾病（高血压）+ 问布洛芬 → 仍拒药+免责；档案疾病 + 问推荐 → 引导语出现；
- 红线不破坏：既有 115 用例由全量套件保证。

复用 conftest 的 temp DB 隔离。
"""
from __future__ import annotations

from app import models
from app.database import SessionLocal


def _chat(client, message, *, session_id, user_id, allergies=None):
    return client.post(
        "/api/chat",
        json={
            "user_id": user_id,
            "session_id": session_id,
            "message": message,
            "allergies": allergies or [],
        },
    )


def _state(client, user_id, session_id=None):
    url = "/api/state?user_id=" + user_id
    if session_id:
        url += "&session_id=" + session_id
    r = client.get(url)
    assert r.status_code == 200
    return r.json()


def _session_row(session_id: str) -> models.Session | None:
    db = SessionLocal()
    try:
        return db.get(models.Session, session_id)
    finally:
        db.close()


# ── /api/state profile 结构契约 ─────────────────────────
def test_state_profile_structure(client):
    """/api/state 返回 profile 四段 + taboo_options（前端渲染所需）。"""
    body = _state(client, "u_prof")
    d = body["data"]
    assert "profile" in d, "/api/state 缺 profile 段"
    prof = d["profile"]
    assert set(prof.keys()) >= {"identity", "preferences", "behavior", "next_step"}
    assert set(prof["identity"].keys()) >= {"nickname", "since"}
    assert set(prof["preferences"].keys()) >= {
        "goal_tag", "allergy_ids", "allergy_names", "excluded_foods", "disease_labels",
    }
    assert set(prof["behavior"].keys()) >= {
        "streak", "today_meal_count", "week_trend", "recent_meals", "today_missing_meals",
    }
    assert set(prof["next_step"].keys()) >= {"text", "action"}
    assert isinstance(d["taboo_options"], list)
    ids = {t["id"] for t in d["taboo_options"]}
    assert "seafood_allergy" in ids and "hypertension" in ids


# ── 档案聚合：过敏 + 台账 → 名字/排除/坚持/最近台账 ────────
def test_profile_aggregates_allergy_and_logs(client):
    """建会话（海鲜过敏）+ 记台账 → profile 聚合出 名字/排除食材/坚持/最近台账/建议。"""
    user = "u_prof2"
    sid = "prof2"
    r = client.post(
        "/api/session",
        json={
            "user_id": user,
            "session_id": sid,
            "action": "new",
            "goal_tag": "减脂",
            "allergies": ["seafood_allergy"],
        },
    )
    assert r.status_code == 200
    # 记台账（早餐吃了鸡蛋和牛奶）
    r1 = _chat(client, "帮我记一下：早餐吃了两个鸡蛋和一杯牛奶", session_id=sid, user_id=user)
    assert r1.status_code == 200

    body = _state(client, user, sid)
    prof = body["data"]["profile"]
    # identity
    assert prof["identity"]["nickname"] is None
    assert prof["identity"]["since"] is not None
    # preferences
    assert prof["preferences"]["goal_tag"] == "减脂"
    assert "海鲜过敏" in prof["preferences"]["allergy_names"]
    assert "虾仁" in prof["preferences"]["excluded_foods"]
    assert "三文鱼" in prof["preferences"]["excluded_foods"]
    # behavior
    assert prof["behavior"]["streak"] >= 1
    assert prof["behavior"]["today_meal_count"] >= 1
    assert any("鸡蛋" in m["content"] for m in prof["behavior"]["recent_meals"])
    assert prof["behavior"]["today_missing_meals"], "今日未记录餐次不应为空"
    # next_step：今天有记录且还差餐 → 建议补记
    assert "今天还差" in prof["next_step"]["text"]
    assert prof["next_step"]["action"].startswith("record_")


def test_profile_next_step_build_meal_when_no_log(client):
    """无任何台账 → 下一步建议为「搭一餐」（build_meal）。"""
    body = _state(client, "u_prof_empty")
    step = body["data"]["profile"]["next_step"]
    assert step["action"] == "build_meal"
    assert "搭一餐" in step["text"]


# ── 守卫1：allergies 白名单过滤（未知 id 安全失败） ────────
def test_session_allergies_whitelist_filtered(client):
    """POST /api/session allergies 含未知 id → 过滤（响应与写库均不含未知 id）。"""
    r = client.post(
        "/api/session",
        json={
            "user_id": "u_prof3",
            "session_id": "prof3",
            "action": "new",
            "goal_tag": "减脂",
            "allergies": ["seafood_allergy", "nonexistent_id", "hack_id"],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert "seafood_allergy" in body["data"]["allergies"]
    assert "nonexistent_id" not in body["data"]["allergies"]
    assert "hack_id" not in body["data"]["allergies"]
    # 写库同样过滤（双保险）
    sess = _session_row("prof3")
    assert sess is not None
    assert "nonexistent_id" not in (sess.allergies or [])
    assert "hack_id" not in (sess.allergies or [])


def test_session_allergies_unknown_only_keeps_existing(client):
    """仅含未知 id 的更新 → 保持既有禁忌不变（安全失败，不误清真实数据）。"""
    r0 = client.post(
        "/api/session",
        json={
            "user_id": "u_prof3b",
            "session_id": "prof3b",
            "action": "new",
            "goal_tag": "调理",
            "allergies": ["hypertension"],
        },
    )
    assert r0.status_code == 200
    r1 = client.post(
        "/api/session",
        json={
            "user_id": "u_prof3b",
            "session_id": "prof3b",
            "action": "switch",
            "allergies": ["nonexistent_id"],
        },
    )
    assert r1.status_code == 200
    assert "hypertension" in r1.json()["data"]["allergies"]
    assert "nonexistent_id" not in r1.json()["data"]["allergies"]


# ── 守卫2：档案疾病 ≠ 免免责/免拒药；只做引导语差异化 ──────
def test_profile_disease_does_not_bypass_medication_refuse(client):
    """档案有高血压（disease 类）+ 问布洛芬 → 仍拒药+免责（拒药以消息关键词为准）。"""
    user = "u_prof4"
    sid = "prof4"
    r = client.post(
        "/api/session",
        json={
            "user_id": user,
            "session_id": sid,
            "action": "new",
            "goal_tag": "调理",
            "allergies": ["hypertension"],
        },
    )
    assert r.status_code == 200
    r1 = _chat(client, "布洛芬能吃吗", session_id=sid, user_id=user)
    assert r1.status_code == 200
    body = r1.json()
    assert "不提供用药建议" in body["data"]["reply"]
    assert body.get("disclaimer") and "不构成医疗建议" in body["disclaimer"]


def test_profile_disease_guide_appears(client):
    """档案有高血压 + 问推荐（消息无病名）→ 跨轮免责 + 引导语出现（结合您的控盐需求）。"""
    user = "u_prof5"
    sid = "prof5"
    r = client.post(
        "/api/session",
        json={
            "user_id": user,
            "session_id": sid,
            "action": "new",
            "goal_tag": "调理",
            "allergies": ["hypertension"],
        },
    )
    assert r.status_code == 200
    # disease_labels 反查：高血压 → 控盐
    body = _state(client, user, sid)
    assert "控盐" in body["data"]["profile"]["preferences"]["disease_labels"]
    # 对话：无病名消息 → 引导语出现（档案驱动）+ 免责延续
    r1 = _chat(client, "推荐减脂方案", session_id=sid, user_id=user)
    assert r1.status_code == 200
    body1 = r1.json()
    assert "结合您的控盐需求" in body1["data"]["reply"]
    assert body1.get("disclaimer") and "不构成医疗建议" in body1["disclaimer"]
