"""本地开发启动器：从 deploy/.env 注入 DEEPSEEK_API_KEY 后启动 uvicorn。

用途：让本地演示实例恢复 LLM 能力（Key 只从 deploy/.env 读取，不落命令行、不进日志）。
用法：python dev_run.py  （在 backend 目录下执行）
"""
import os
import re
import sys
from pathlib import Path

# 只读取 deploy/.env 中的 DEEPSEEK_API_KEY，注入进程环境
env_path = Path(__file__).resolve().parent.parent / "deploy" / ".env"
if env_path.is_file():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value

if os.environ.get("DEEPSEEK_API_KEY"):
    print("[dev_run] DEEPSEEK_API_KEY 已注入（来自 deploy/.env）")
else:
    print("[dev_run] 未找到 DEEPSEEK_API_KEY，服务将以本地规则模式运行")

# 启动 uvicorn（同进程，避免 os.execv 在 Windows 下的引号问题）
from uvicorn import run  # noqa: E402

if __name__ == "__main__":
    run("app.main:app", host="127.0.0.1", port=8137)
