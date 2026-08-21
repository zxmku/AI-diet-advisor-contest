"""健康状态引擎（Health State Engine）· Growing Memory 适配器（管道适配器 · 热插拔）。

基于饮食台账（DietLog）生成：今日饮食总结 / 连续坚持天数（Streak）/ 昨日概况 / AI 欢迎语，
供 LLM 上下文注入（成长记忆：结合昨天与今天）与前端 /api/state 展示。
纯读取组装，零副作用；异常一律降级返回空/0，不打断主流程。
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

from app import models
from app.database import SessionLocal

_UTC8 = timedelta(hours=8)
MEAL_TAGS = ("早餐", "午餐", "晚餐")


def _local_date(dt: datetime) -> date:
    """DietLog.created_at 存 UTC，转本地（UTC+8）自然日。"""
    return (dt + _UTC8).date()


def _fetch_logs(user_id: str) -> list:
    db = SessionLocal()
    try:
        return (
            db.query(models.DietLog)
            .filter(models.DietLog.user_id == user_id)
            .order_by(models.DietLog.created_at.desc())
            .limit(100)
            .all()
        )
    except Exception:  # noqa: BLE001
        return []
    finally:
        db.close()


def get_streak(user_id: str, today: date | None = None) -> int:
    """连续坚持天数：按本地自然日去重，从今天（或昨天）往前连续计数。"""
    today = today or date.today()
    days = {_local_date(r.created_at) for r in _fetch_logs(user_id)}
    d = today if today in days else today - timedelta(days=1)
    streak = 0
    while d in days:
        streak += 1
        d -= timedelta(days=1)
    return streak


def _day_summary(user_id: str, day: date) -> dict:
    rows = [r for r in _fetch_logs(user_id) if _local_date(r.created_at) == day]
    meals = [{"meal_tag": r.meal_tag, "content": r.content} for r in rows]
    return {"date": day.isoformat(), "meal_count": len(meals), "meals": meals}


def get_today_summary(user_id: str, today: date | None = None) -> dict:
    return _day_summary(user_id, today or date.today())


def get_yesterday_summary(user_id: str, today: date | None = None) -> dict:
    today = today or date.today()
    return _day_summary(user_id, today - timedelta(days=1))


def build_state_context(user_id: str, today: date | None = None) -> str:
    """组装可注入 LLM 的近期状态文本（成长记忆：昨天 + 今天 + 坚持情况）。"""
    today = today or date.today()
    parts = []
    streak = get_streak(user_id, today)
    if streak > 0:
        parts.append(f"用户已连续坚持饮食记录 {streak} 天")
    t = get_today_summary(user_id, today)
    if t["meal_count"]:
        meals_txt = "；".join(f"{m['meal_tag']}：{m['content']}" for m in t["meals"])
        parts.append(f"今日已记录 {t['meal_count']} 餐：{meals_txt}")
        missing = [tag for tag in MEAL_TAGS if tag not in {m["meal_tag"] for m in t["meals"]}]
        if missing:
            parts.append("今日未记录：" + "、".join(missing))
    else:
        parts.append("今日尚未记录任何饮食")
    y = get_yesterday_summary(user_id, today)
    if y["meal_count"]:
        parts.append(f"昨日记录 {y['meal_count']} 餐")
    return "；".join(parts) if parts else "暂无近期饮食记录"


def build_greeting(user_id: str, today: date | None = None) -> str:
    """AI 欢迎语（M18）：结合坚持情况与今日记录主动问候，服务饮食决策。"""
    today = today or date.today()
    streak = get_streak(user_id, today)
    t = get_today_summary(user_id, today)
    if streak >= 2:
        return f"早上好！你已连续坚持 {streak} 天，真棒 💪 今天打算怎么吃？告诉我吃了什么，或直接让我帮你搭一餐。"
    if t["meal_count"] == 0:
        return "新的一天！今天吃了什么？我帮你记着，也帮你搭好每一餐。"
    return "回来啦！今天继续把每一餐记好，我来帮你盯营养。"
