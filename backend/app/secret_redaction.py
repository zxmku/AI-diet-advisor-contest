"""密钥脱敏层：日志与导出边界防 Key 泄露。

设计原则：
- 只在日志/响应等「导出边界」脱敏，不修改业务内部数据；
- 覆盖常见密钥形态：sk- 系列（DeepSeek/OpenAI 等）、AWS AKIA、GitHub ghp_、
  Slack xox、Bearer token、api_key= 赋值；
- 长度阈值避免误伤普通 hex/uuid 文本；
- 红线：真实 API Key 永不入库、永不进日志（先脱敏）。
"""
from __future__ import annotations

import re

# 密钥形态（长度阈值避免误伤普通 hex/uuid）
_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"sk-or-v1-[A-Za-z0-9]{20,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._\-]{20,}"),
    re.compile(r"(?i)api[_-]?key[\"']?\s*[:=]\s*[\"']?[A-Za-z0-9_\-]{16,}[\"']?"),
]

_MASK = "[REDACTED]"


def redact(text: str) -> str:
    """脱敏单个字符串。非字符串原样返回。"""
    if not text or not isinstance(text, str):
        return text
    out = text
    for pattern in _PATTERNS:
        out = pattern.sub(_MASK, out)
    return out


def redact_obj(obj):
    """递归脱敏 dict/list/tuple/str；其他类型原样返回。用于导出边界包裹整个对象。"""
    if isinstance(obj, str):
        return redact(obj)
    if isinstance(obj, dict):
        return {k: redact_obj(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(redact_obj(v) for v in obj)
    return obj


def contains_secret(text: str) -> bool:
    """是否含已知密钥形态（用于写入前 fail-closed 断言/审计）。仅判字符串。"""
    if not text or not isinstance(text, str):
        return False
    return any(p.search(text) for p in _PATTERNS)
