"""集中配置：.env 加载 + 环境变量解析 + 密钥纪律。

纪律：
- 敏感配置（DEEPSEEK_API_KEY / AUDIT_HMAC_KEY 等）只从环境变量读取，代码内零明文；
- 所有配置项均有安全默认值，零配置即可本地运行（纯规则降级模式）；
- 数值型环境变量解析失败时 fail-fast，给出清晰报错而非隐性跑偏；
- 日志输出前须经 secret_redaction 脱敏，密钥永不进日志。
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger("healthpick.config")


class ConfigError(Exception):
    """配置错误：环境变量误配时 fail-fast，给出可读报错。"""


def _load_dotenv(path: str | None = None) -> bool:
    """零依赖 .env 加载器。

    把 .env 中的 ``KEY=VALUE`` 注入 os.environ（仅当该键尚未存在于环境中，
    即真实环境变量优先于 .env 文件）。搜索顺序：显式 path → 后端目录 .env →
    当前工作目录 .env。.env 不入版本库，仓库只留 .env.example。
    """
    candidates: list[Path] = []
    if path:
        candidates.append(Path(path))
    backend_dir = Path(__file__).resolve().parent.parent  # backend/
    candidates.append(backend_dir / ".env")
    candidates.append(Path.cwd() / ".env")
    for cand in candidates:
        if cand.is_file():
            try:
                with open(cand, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        key, _, value = line.partition("=")
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key and key not in os.environ:
                            os.environ[key] = value
                return True
            except OSError:
                return False
    return False


# import 时先加载 .env（若存在），再读取下方配置项。
_load_dotenv()


def _as_int(env_name: str, default: str) -> int:
    """读取整型环境变量，解析失败 fail-fast。"""
    raw = os.environ.get(env_name, default)
    try:
        return int(raw)
    except (TypeError, ValueError) as e:
        raise ConfigError(
            f"环境变量 {env_name} 必须为整数，当前值无法解析: {raw!r}"
        ) from e


# ── 运行环境 ──
ENV: str = os.environ.get("HEALTHPICK_ENV", "dev").lower()  # dev / prod
HOST: str = os.environ.get("HEALTHPICK_HOST", "127.0.0.1")
PORT: int = _as_int("HEALTHPICK_PORT", "8000")

# ── 数据库（M14：默认本地 SQLite，公网可切 Postgres，零配置可跑）──
DATABASE_URL: str = os.environ.get(
    "DATABASE_URL", f"sqlite:///{(Path(__file__).resolve().parent.parent / 'healthpick.db')}"
)

# ── LLM（M13/M17：DeepSeek 动态调用；缺 Key 时上层自动降级本地规则，永不裸奔）──
DEEPSEEK_API_KEY: str = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL: str = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL: str = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

# ── LLM 成本闸门（M17：预算/缓存/限速，防止接真模型后无上限烧 token）──
LLM_DAILY_BUDGET_TOKENS: int = _as_int("LLM_DAILY_BUDGET_TOKENS", "1000000")
LLM_CACHE_TTL_SECONDS: int = _as_int("LLM_CACHE_TTL_SECONDS", "86400")
LLM_RATE_LIMIT_PER_MIN: int = _as_int("LLM_RATE_LIMIT_PER_MIN", "10")

# ── 审计账本 HMAC（缺省时禁用签名仅保留哈希链，记 warning）──
AUDIT_HMAC_KEY: str = os.environ.get("AUDIT_HMAC_KEY", "")
if not AUDIT_HMAC_KEY:
    logger.warning("AUDIT_HMAC_KEY 未设置：审计账本降级为仅哈希链（无 HMAC 签名）")

# ── 网关边界（蓝图 8.4：空输入/超长/限流）──
MAX_MESSAGE_LEN: int = _as_int("HEALTHPICK_MAX_MESSAGE_LEN", "500")
RATE_LIMIT_PER_MIN: int = _as_int("HEALTHPICK_RATE_LIMIT_PER_MIN", "60")

# ── CORS 白名单（逗号分隔；默认本地前端端口，prod 禁用 *）──
_cors_raw = os.environ.get("HEALTHPICK_CORS_ORIGINS", "")
if _cors_raw:
    CORS_ORIGINS: list[str] = [o.strip() for o in _cors_raw.split(",") if o.strip()]
else:
    CORS_ORIGINS = [
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ]
if CORS_ORIGINS == ["*"] and ENV == "prod":
    logger.warning("prod 环境 CORS 为 * 不安全，请用 HEALTHPICK_CORS_ORIGINS 配置白名单")

# ── 知识库目录（MOD-01 产出的结构化 JSON，相对项目根）──
KNOWLEDGE_DIR: Path = Path(
    os.environ.get(
        "HEALTHPICK_KNOWLEDGE_DIR",
        str(Path(__file__).resolve().parent.parent.parent / "knowledge"),
    )
)
