"""统一 API 响应模型与请求模型（全局数据契约 2.1 / 2.6）。

所有端点返回统一格式：``{data, sources, disclaimer?, meta}``。
- ``sources`` 无引用时为空数组，不得省略字段；
- ``disclaimer`` 仅在触发免责时出现，不触发时可缺省（None 不序列化）；
- ``meta.model`` 前端固定展示（模型来源可见）；降级时 model="local-rules" 且 degraded=true。
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

# ── 统一响应（契约 2.1）──

# 意图枚举：营养问答 / 方案推荐 / 平台咨询 / 闲聊
IntentType = Literal["nutrition_qa", "recommend", "platform", "chitchat"]
# 人群标签枚举：减脂 / 增肌 / 调理
GoalTag = Literal["减脂", "增肌", "调理"]


class SourceChunk(BaseModel):
    """检索块（契约 2.2）：source/chapter/section/content 四字段必需，score 检索时附加。"""

    source: Literal["A", "B", "C"]
    chapter: str
    section: str
    content: str
    score: float | None = None


class Meta(BaseModel):
    """响应元信息：模型来源标注（前端可见）+ 降级标志 + 时间戳 + 请求 ID。"""

    model: str = "local-rules"
    degraded: bool = False
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex)


class UnifiedResponse(BaseModel):
    """统一 API 响应格式（契约 2.1），全部 6 个端点共用。"""

    data: dict[str, Any]
    sources: list[SourceChunk] = Field(default_factory=list)
    disclaimer: str | None = None
    meta: Meta = Field(default_factory=Meta)


def make_response(
    data: dict[str, Any],
    *,
    sources: list[SourceChunk] | None = None,
    disclaimer: str | None = None,
    model: str = "local-rules",
    degraded: bool = False,
) -> UnifiedResponse:
    """构造统一响应。Stub 与真实现都经此出口，保证契约一致。"""
    return UnifiedResponse(
        data=data,
        sources=sources or [],
        disclaimer=disclaimer,
        meta=Meta(model=model, degraded=degraded),
    )


# ── 请求模型（契约 2.6）──


class ChatRequest(BaseModel):
    """POST /api/chat 请求体。"""

    user_id: str = Field(min_length=1, max_length=64)
    session_id: str | None = Field(default=None, max_length=64)
    message: str = ""
    allergies: list[str] = Field(default_factory=list)


class RecommendRequest(BaseModel):
    """POST /api/recommend 请求体。"""

    user_id: str = Field(min_length=1, max_length=64)
    session_id: str = Field(min_length=1, max_length=64)
    goal_tag: GoalTag | None = None
    allergies: list[str] = Field(default_factory=list)


class QuickRequest(BaseModel):
    """POST /api/quick 请求体：四个快捷操作。"""

    user_id: str = Field(min_length=1, max_length=64)
    session_id: str | None = Field(default=None, max_length=64)
    action: Literal["lose_fat", "gain_muscle", "control_sugar", "today_meal"]
    allergies: list[str] = Field(default_factory=list)


class SessionRequest(BaseModel):
    """POST /api/session 请求体：新建/切换会话，可携带画像与禁忌。"""

    user_id: str = Field(min_length=1, max_length=64)
    action: Literal["new", "switch"]
    session_id: str | None = Field(default=None, max_length=64)
    goal_tag: GoalTag | None = None
    allergies: list[str] = Field(default_factory=list)


class UserCreateRequest(BaseModel):
    """POST /api/user 请求体：创建用户身份（昵称可选）。"""

    nickname: str | None = Field(default=None, max_length=64)
