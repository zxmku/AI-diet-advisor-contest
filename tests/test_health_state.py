"""健康状态引擎（M2 成长记忆 / M3 状态引擎 / M9 今日总结 / M10 Streak / M18 欢迎语）测试。

复用 conftest temp DB 隔离；LLM 路径 mock，不触网。
"""
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from fastapi.testclient import TestClient  # noqa: E402

from app import config, health_state, llm as llm_mod, models  # noqa: E402
from app.database import SessionLocal, init_db  # noqa: E402

init_db()  # 幂等：确保 diet_logs 表存在（temp DB）


def _seed(user_id: str, local_day: date, count: int, meal_tag: str = "早餐"):
    """在本地自然日 local_day 造 count 条饮食记录（created_at 存 UTC=本地-8h）。"""
    db = SessionLocal()
    try:
        u = db.get(models.User, user_id)
        if u is None:
            u = models.User(id=user_id, nickname=None)
            db.add(u)
            db.flush()
        s = models.Session(
            id=f"s_{user_id}_{local_day}", user_id=user_id, goal_tag=None, allergies=[]
        )
        db.add(s)
        db.flush()
        for i in range(count):
            db.add(
                models.DietLog(
                    session_id=s.id,
                    user_id=user_id,
                    meal_tag=meal_tag,
                    content=f"{local_day} 记录{i + 1}",
                    created_at=datetime(local_day.year, local_day.month, local_day.day, 4, 0)  # 本地 12:00
                    + timedelta(minutes=i),
                )
            )
        db.commit()
    finally:
        db.close()


def test_streak_continuous_and_gap():
    uid = "hs_streak"
    today = date.today()
    _seed(uid, today, 1)
    _seed(uid, today - timedelta(days=1), 1)
    _seed(uid, today - timedelta(days=2), 1)
    assert health_state.get_streak(uid) >= 3, "连续 3 天应得 streak>=3"

    uid2 = "hs_gap"
    _seed(uid2, today, 1)
    _seed(uid2, today - timedelta(days=3), 1)  # 前天断档
    assert health_state.get_streak(uid2) == 1, "断档后 streak 只算今天"


def test_today_summary_and_context():
    uid = "hs_today"
    today = date.today()
    _seed(uid, today, 2, meal_tag="午餐")
    t = health_state.get_today_summary(uid)
    assert t["meal_count"] == 2 and t["meals"][0]["meal_tag"] == "午餐"

    ctx = health_state.build_state_context(uid)
    assert "今日已记录 2 餐" in ctx
    assert "午餐" in ctx


def test_greeting_reflects_streak():
    uid = "hs_greet"
    today = date.today()
    _seed(uid, today, 1)
    _seed(uid, today - timedelta(days=1), 1)
    g = health_state.build_greeting(uid)
    assert "连续坚持" in g


def test_api_state():
    uid = "hs_api"
    today = date.today()
    _seed(uid, today, 1)
    from app.main import app

    with TestClient(app) as c:
        r = c.get(f"/api/state?user_id={uid}")
    assert r.status_code == 200
    d = r.json()["data"]
    assert "streak" in d and "greeting" in d and "today_summary" in d
    assert d["streak"] >= 1


def test_llm_state_context_injected():
    captured = {}

    def fake_post(url, headers, payload, timeout):
        captured["sys"] = payload["messages"][0]["content"]
        return {"choices": [{"message": {"content": "结合你的情况"}}]}

    llm_mod._post_json = fake_post
    config.DEEPSEEK_API_KEY = "dummy"
    out = llm_mod.synthesize(
        "今天吃什么",
        [],
        history=[],
        state_context="用户已连续坚持饮食记录 2 天；今日已记录 1 餐：早餐：鸡蛋和牛奶",
    )
    assert out == "结合你的情况"
    assert "【用户近期状态】" in captured["sys"]
    assert "连续坚持" in captured["sys"]
    config.DEEPSEEK_API_KEY = ""
