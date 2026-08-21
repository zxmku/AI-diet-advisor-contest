"""网关守卫（M3 横切面）：输入校验 / 限流 / 安全响应头 / 全局异常兜底。

蓝图 8.4 三套指定提示语（边界兜底，不可改动）：
- 空输入        → 「请输入您的问题」
- 超长（>500字）→ 「输入内容过长，请精简后重试」
- 系统异常      → 「服务繁忙，请稍后重试」

限流按 session/用户级（蓝图 8.4 [v2]，非纯 IP），避免同公司 IP 多人误杀。
"""
from __future__ import annotations

import logging
import threading
import time
import traceback
from collections import defaultdict

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import MAX_MESSAGE_LEN, RATE_LIMIT_PER_MIN
from app.secret_redaction import redact

logger = logging.getLogger("healthpick.guard")

# ── 蓝图 8.4 三套指定提示语（边界兜底文案，全项目统一引用此处常量）──
MSG_EMPTY_INPUT = "请输入您的问题"
MSG_INPUT_TOO_LONG = "输入内容过长，请精简后重试"
MSG_SERVICE_BUSY = "服务繁忙，请稍后重试"
MSG_RATE_LIMITED = "请求过于频繁，请稍后再试"


class InputViolation(Exception):
    """输入边界违规：携带对用户展示的指定提示语。"""

    def __init__(self, user_message: str) -> None:
        super().__init__(user_message)
        self.user_message = user_message


def validate_message(text: str | None) -> str:
    """校验用户消息：去空白后判空与超长（蓝图 8.4）。

    Returns:
        去首尾空白后的消息文本。

    Raises:
        InputViolation: 空输入或超长，携带指定提示语。
    """
    cleaned = (text or "").strip()
    if not cleaned:
        raise InputViolation(MSG_EMPTY_INPUT)
    if len(cleaned) > MAX_MESSAGE_LEN:
        raise InputViolation(MSG_INPUT_TOO_LONG)
    return cleaned


# ── 安全响应头中间件（原生 ASGI，置于最外层）──
_SEC_HEADERS: list[tuple[bytes, bytes]] = [
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"deny"),
    (b"referrer-policy", b"strict-origin-when-cross-origin"),
]


class SecurityHeadersMiddleware:
    """为所有 HTTP 响应追加安全响应头（已存在则不覆盖）。"""

    def __init__(self, app) -> None:  # noqa: ANN001
        self.app = app

    async def __call__(self, scope, receive, send) -> None:  # noqa: ANN001
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        async def _send(message) -> None:  # noqa: ANN001
            if message["type"] == "http.response.start":
                headers = message.setdefault("headers", [])
                have = {h[0].lower() for h in headers}
                for key, value in _SEC_HEADERS:
                    if key not in have:
                        headers.append((key, value))
            await send(message)

        await self.app(scope, receive, _send)


# ── 限流中间件（session/用户级，蓝图 8.4 [v2]）──
_rate_lock = threading.Lock()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """按 session/用户限流：POST /api/ 端点每分钟上限（默认 60，可调防评审误杀）。

    限流键优先级：请求体外的 X-Session-Id 头 → X-User-Id 头 → 客户端 IP。
    桶数量封顶，防内存无限增长；超限返回 429 + 统一风格错误体。
    """

    _MAX_BUCKETS = 4096

    def __init__(self, app) -> None:  # noqa: ANN001
        super().__init__(app)
        self._buckets: dict[str, list] = defaultdict(lambda: [0.0, 0])

    def _client_key(self, request: Request) -> str:
        """限流键：优先 session/user 标识（蓝图要求非纯 IP），兜底客户端 IP。"""
        session_id = request.headers.get("X-Session-Id", "").strip()
        if session_id:
            return f"sid:{session_id}"
        user_id = request.headers.get("X-User-Id", "").strip()
        if user_id:
            return f"uid:{user_id}"
        fwd = request.headers.get("X-Forwarded-For", "")
        ip = fwd.split(",")[0].strip() or (
            request.client.host if request.client else "unknown"
        )
        return f"ip:{ip}"

    async def dispatch(self, request: Request, call_next):  # noqa: ANN001
        if request.method == "POST" and request.url.path.startswith("/api/"):
            key = self._client_key(request)
            now = time.time()
            with _rate_lock:
                if len(self._buckets) > self._MAX_BUCKETS:
                    self._buckets.pop(next(iter(self._buckets)), None)
                bucket = self._buckets[key]
                if now - bucket[0] > 60:
                    bucket[0], bucket[1] = now, 0
                bucket[1] += 1
                if bucket[1] > RATE_LIMIT_PER_MIN:
                    from app.api.schemas import make_response

                    payload = make_response(
                        {"reply": MSG_RATE_LIMITED, "intent": "chitchat", "goal_tag": None},
                        model="local-rules",
                    )
                    return JSONResponse(status_code=429, content=payload.model_dump())
        return await call_next(request)


def register_exception_handlers(app: FastAPI) -> None:
    """注册全局异常兜底：任何未捕获异常统一返回「服务繁忙，请稍后重试」。

    细节只进日志（先脱敏），不外泄栈/路径；对外响应保持统一响应格式。
    """

    @app.exception_handler(InputViolation)
    async def _input_violation_handler(
        _request: Request, exc: InputViolation
    ) -> JSONResponse:
        # 空/超长输入：HTTP 200 + 统一响应格式，reply 即指定提示语，前端直接展示
        from app.api.schemas import make_response

        payload = make_response(
            {"reply": exc.user_message, "intent": "chitchat", "goal_tag": None},
            model="local-rules",
        )
        return JSONResponse(status_code=200, content=payload.model_dump())

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        logger.warning("请求参数校验失败: %s", redact(str(exc)))
        from app.api.schemas import make_response

        payload = make_response(
            {"reply": "请求参数不完整或格式错误", "intent": "chitchat", "goal_tag": None},
            model="local-rules",
        )
        return JSONResponse(status_code=400, content=payload.model_dump())

    @app.exception_handler(Exception)
    async def _unhandled_handler(_request: Request, exc: Exception) -> JSONResponse:
        # 系统异常：日志记细节（消息与堆栈均脱敏，密钥永不进日志），
        # 对外仅返回指定提示语（蓝图 8.4）
        logger.error(
            "未捕获异常: %s\n%s", redact(str(exc)), redact(traceback.format_exc())
        )
        from app.api.schemas import make_response

        payload = make_response(
            {"reply": MSG_SERVICE_BUSY, "intent": "chitchat", "goal_tag": None},
            model="local-rules",
            degraded=True,
        )
        return JSONResponse(status_code=500, content=payload.model_dump())
