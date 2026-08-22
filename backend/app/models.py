"""ORM 模型：users / sessions / messages / plans_cache / diet_logs 五表（全局数据契约 2.3）。

- sessions.allergies 存禁忌 id 的 JSON 数组，每次请求必须加载进合规拦截，不因轮次丢失；
- messages.sources 存引用溯源 JSON 数组，与统一响应格式的 sources 结构一致；
- diet_logs（P5）存用户饮食台账（记住吃了什么），meal_tag ∈ 早餐/午餐/晚餐/加餐/正餐；
- users.profile_json 存 AI 自动维护的用户档案（JSON 字符串：孕期/口味/主食风格/运动量/
  昵称等），UI 不可编辑，由对话规则检测写入，字段级去重（已有不覆盖）。
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def _utcnow() -> datetime:
    """UTC 当前时间（带时区）。"""
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    """生成 UUID 字符串主键。"""
    return uuid.uuid4().hex


class User(Base):
    """用户身份：匿名 ID（前端 localStorage）或昵称。

    profile_json 存 AI 自动维护的用户档案 JSON 字符串（如
    ``{"pregnancy":"孕期","taste":"爱吃辣","exercise":"每天跑步","nickname":"小优"}``），
    由 user_profile.set_profile_field 维护（字段级去重，已有不覆盖）。
    老库通过 database.init_db 的幂等 ALTER 补列升级。
    """

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_uuid)
    nickname: Mapped[str | None] = mapped_column(String(64), nullable=True)
    profile_json: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    sessions: Mapped[list["Session"]] = relationship(back_populates="user")


class Session(Base):
    """会话与用户画像：goal_tag（减脂/增肌/调理）+ allergies（禁忌 id JSON 数组）。"""

    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_new_uuid)
    user_id: Mapped[str] = mapped_column(String(64), ForeignKey("users.id"), index=True)
    goal_tag: Mapped[str | None] = mapped_column(String(16), nullable=True)
    allergies: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    user: Mapped["User"] = relationship(back_populates="sessions")
    messages: Mapped[list["Message"]] = relationship(back_populates="session")


class Message(Base):
    """对话历史与溯源：role ∈ {user, assistant}，sources 为引用溯源 JSON 数组。"""

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("sessions.id"), index=True
    )
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    sources: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    session: Mapped["Session"] = relationship(back_populates="messages")


class PlansCache(Base):
    """推荐结果缓存（可选）：记录某会话推荐过的方案类型与时间。"""

    __tablename__ = "plans_cache"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("sessions.id"), index=True
    )
    plan_type: Mapped[str] = mapped_column(String(32))
    recommended_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class DietLog(Base):
    """P5 饮食台账：记住用户每餐吃了什么（「记得你吃了什么」核心）。

    - meal_tag ∈ {早餐, 午餐, 晚餐, 加餐, 正餐}（按消息含早/午/晚/加餐判定，默认正餐）；
    - content 存消息原文（不加工），热量估算仅展示用、不落库——避免把估算值当台账事实。
    """

    __tablename__ = "diet_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("sessions.id"), index=True
    )
    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.id"), index=True
    )
    meal_tag: Mapped[str] = mapped_column(String(16), default="正餐")
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
