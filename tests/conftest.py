"""pytest 根配置：环境隔离 + 全局 TestClient 夹具（红线自动化测试 / 蓝图第十一节）。

核心纪律（最重要）：
1. 在 import 任何 ``app.*`` 之前，先把 ``DATABASE_URL`` 指向 tempfile 临时库、
   ``DEEPSEEK_API_KEY`` 置空、限流阈值调高 —— 保证测试**绝不污染线上
   backend/healthpick.db**，也不触发任何真实 LLM 调用（始终 local-rules 降级模式）；
2. 再将 ``backend`` 目录插入 ``sys.path`` 顶部，之后才 import ``app.main``；
3. 提供 session 级 TestClient：with 块触发 lifespan（建表 + 检索索引预热），
   测试数据全部落在临时库中，会话结束自动清理临时目录。
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

# ── 项目根（tests/ 的上一级）与后端目录 ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"

# ── 1. 环境隔离：必须在 import app.* 之前设置 ──
_TMP_DIR = Path(tempfile.mkdtemp(prefix="healthpick_test_"))
_TMP_DB = _TMP_DIR / "test_healthpick.db"

os.environ["DATABASE_URL"] = f"sqlite:///{_TMP_DB.as_posix()}"
os.environ["DEEPSEEK_API_KEY"] = ""
# 限流阈值调高，避免整套用例共用同一 IP 桶时被误杀（不影响被测逻辑）。
os.environ["HEALTHPICK_RATE_LIMIT_PER_MIN"] = "100000"
# cost_gate 账本/缓存同样隔离到临时目录（否则测试会写真实 backend/data/*.json，污染运行时数据）
os.environ["HEALTHPICK_DATA_DIR"] = str(_TMP_DIR / "data")
# 保证测试以 dev 环境运行（docs/openapi 可见，与生产差异最小）。
os.environ.setdefault("HEALTHPICK_ENV", "dev")

# ── 2. 插入 backend 到 sys.path，随后导入被测应用 ──
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi.testclient import TestClient  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture(scope="session")
def client() -> TestClient:
    """全局 TestClient：with 块触发 lifespan（建表 + BM25 索引预热）。"""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session", autouse=True)
def _cleanup_temp_db() -> None:
    """会话结束清理临时目录，不留任何测试脏数据。"""
    yield
    shutil.rmtree(_TMP_DIR, ignore_errors=True)
