"""M17 成本闸门（cost gate）：为 LLM 调用提供 预算 / 限速 / 回复缓存 三层保护。

背景：接入真实 DeepSeek 模型后，若不加限制，单日 token 消耗与费用可能失控。
本模块在 ``llm.synthesize()`` 发起真实 API 调用之前拦截：

1. 预算闸门：当日已用 token 达到 ``LLM_DAILY_BUDGET_TOKENS`` 后自动降级
   （``check_budget()`` 返回 False，上层返回 None 无缝回退 local-rules）；
2. 回复缓存：相同用户消息（``strip()`` 后原文）在 TTL 内直接命中，
   零 API 花费（``LLM_CACHE_TTL_SECONDS``）；
3. 会话限速：单会话 60 秒窗口内调用次数达到 ``LLM_RATE_LIMIT_PER_MIN`` 后降级
   （进程内内存字典，不落盘）。

账本与缓存均为运行时数据（``backend/data/``），不入版本库（.gitignore 已排除）；
所有文件读写异常一律捕获（非致命），与项目「降级不崩」哲学一致——
闸门自身故障绝不影响主流程，最坏情况是闸门暂时失效（放行）。
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

from app import config

logger = logging.getLogger("healthpick.cost_gate")

_DATA_DIR = Path(
    os.environ.get("HEALTHPICK_DATA_DIR") or (Path(__file__).resolve().parent.parent / "data")
)  # backend/data；可用 HEALTHPICK_DATA_DIR 覆盖（测试隔离/部署自定义）
_LEDGER_PATH = _DATA_DIR / "cost_ledger.json"
_CACHE_PATH = _DATA_DIR / "llm_cache.json"

# deepseek-chat 近似单价（美元 / 每百万 token）：输入 0.0005，输出 0.002。
_PROMPT_PRICE_PER_1M = 0.0005
_COMPLETION_PRICE_PER_1M = 0.002

_RATE_WINDOW_SECONDS = 60


def _today() -> str:
    """本地日期 YYYY-MM-DD（账本按本地日切分）。"""
    return datetime.now().strftime("%Y-%m-%d")


class CostGate:
    """成本闸门单例：预算账本（磁盘）+ 回复缓存（磁盘）+ 会话限速（内存）。"""

    def __init__(self) -> None:
        self.budget_tokens: int = max(0, int(config.LLM_DAILY_BUDGET_TOKENS or 0))
        self.cache_ttl_seconds: int = max(0, int(config.LLM_CACHE_TTL_SECONDS or 0))
        self.rate_limit_per_min: int = max(0, int(config.LLM_RATE_LIMIT_PER_MIN or 0))
        # 会话 -> 最近调用时间戳列表（monotonic），进程内内存，60 秒窗口滑动清理。
        self._rate: dict[str, list[float]] = {}
        # 缓存命中计数（进程内累计，供 /health 观测）。
        self._cache_hits: int = 0
        self._lock = threading.RLock()

    # ── 账本读写 ──

    def _load_json(self, path: Path, default):
        """读 JSON，文件缺失返回 default；任何异常抛给调用方统一兜底。"""
        if not path.is_file():
            return default
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _save_json_atomic(self, path: Path, data) -> None:
        """原子写 JSON：先写同目录临时文件再 os.replace，避免并发/中断写坏账本。"""
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=str(_DATA_DIR), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2)
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def _load_ledger(self) -> dict:
        """读取当日账本；异常由调用方捕获（非致命）。"""
        data = self._load_json(_LEDGER_PATH, {})
        return data if isinstance(data, dict) else {}

    def _load_cache(self) -> dict:
        """读取回复缓存；异常由调用方捕获（非致命）。"""
        data = self._load_json(_CACHE_PATH, {})
        return data if isinstance(data, dict) else {}

    # ── 预算闸门 ──

    def check_budget(self) -> bool:
        """当日已用 token < 预算才 True；读取失败按放行（不因闸门故障误伤主流程）。"""
        try:
            ledger = self._load_ledger()
            day = ledger.get(_today(), {})
            tokens_used = int(day.get("tokens", 0) or 0)
            return tokens_used < self.budget_tokens
        except Exception as exc:  # noqa: BLE001
            logger.warning("成本账本读取失败（按放行处理）: %s", exc)
            return True

    # ── 会话限速（进程内内存）──

    def check_rate(self, session_id: str | None) -> bool:
        """该 session 最近 60 秒内 LLM 调用次数 < 限额才 True；顺带清理过期记录。"""
        key = session_id or "_anonymous"
        now = time.monotonic()
        with self._lock:
            stamps = [
                t for t in self._rate.get(key, [])
                if now - t < _RATE_WINDOW_SECONDS
            ]
            self._rate[key] = stamps
            return len(stamps) < self.rate_limit_per_min

    def _record_rate(self, session_id: str | None) -> None:
        """记录一次成功调用时间戳，并清理该会话过期记录。"""
        key = session_id or "_anonymous"
        now = time.monotonic()
        with self._lock:
            stamps = [
                t for t in self._rate.get(key, [])
                if now - t < _RATE_WINDOW_SECONDS
            ]
            stamps.append(now)
            self._rate[key] = stamps

    # ── 记账 ──

    def record(
        self,
        session_id: str | None,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        """累加当日账本 tokens/calls，并按 deepseek-chat 近似价估算 cost（美元）。

        任何文件异常仅记 warning，不影响主流程。
        """
        prompt_tokens = max(0, int(prompt_tokens or 0))
        completion_tokens = max(0, int(completion_tokens or 0))
        try:
            with self._lock:
                ledger = self._load_ledger()
                day_key = _today()
                day = ledger.get(day_key, {"tokens": 0, "calls": 0, "cost": 0.0})
                day["tokens"] = int(day.get("tokens", 0) or 0) + prompt_tokens + completion_tokens
                day["calls"] = int(day.get("calls", 0) or 0) + 1
                day["cost"] = float(day.get("cost", 0.0) or 0.0) + (
                    prompt_tokens * _PROMPT_PRICE_PER_1M / 1e6
                    + completion_tokens * _COMPLETION_PRICE_PER_1M / 1e6
                )
                ledger[day_key] = day
                self._save_json_atomic(_LEDGER_PATH, ledger)
        except Exception as exc:  # noqa: BLE001
            logger.warning("成本账本写入失败（非致命）: %s", exc)
        self._record_rate(session_id)

    # ── 回复缓存 ──

    def cache_get(self, key: str) -> str | None:
        """TTL 内命中返回缓存回复；过期/超 TTL 忽略并清理该键；异常返回 None。"""
        key = (key or "").strip()
        if not key:
            return None
        try:
            with self._lock:
                cache = self._load_cache()
                entry = cache.get(key)
                if not isinstance(entry, dict):
                    return None
                ts = float(entry.get("ts", 0) or 0)
                if self.cache_ttl_seconds > 0 and time.time() - ts > self.cache_ttl_seconds:
                    cache.pop(key, None)
                    self._save_json_atomic(_CACHE_PATH, cache)
                    return None
                reply = entry.get("reply")
                if isinstance(reply, str):
                    self._cache_hits += 1
                    return reply
                return None
        except Exception as exc:  # noqa: BLE001
            logger.warning("回复缓存读取失败（非致命）: %s", exc)
            return None

    def cache_set(self, key: str, reply: str) -> None:
        """写入回复缓存；异常仅记 warning。"""
        key = (key or "").strip()
        reply = (reply or "").strip()
        if not key or not reply:
            return
        try:
            with self._lock:
                cache = self._load_cache()
                cache[key] = {"reply": reply, "ts": time.time()}
                self._save_json_atomic(_CACHE_PATH, cache)
        except Exception as exc:  # noqa: BLE001
            logger.warning("回复缓存写入失败（非致命）: %s", exc)

    # ── 状态观测 ──

    def status(self) -> dict:
        """今日成本与闸门状态，供 /health 展示；异常时返回全零兜底。"""
        try:
            ledger = self._load_ledger()
            day = ledger.get(_today(), {})
            tokens_used = int(day.get("tokens", 0) or 0)
            calls = int(day.get("calls", 0) or 0)
            cost = float(day.get("cost", 0.0) or 0.0)
        except Exception as exc:  # noqa: BLE001
            logger.warning("成本账本读取失败（status 兜底全零）: %s", exc)
            tokens_used, calls, cost = 0, 0, 0.0
        return {
            "date": _today(),
            "tokens_used": tokens_used,
            "calls": calls,
            "cost_usd": round(cost, 6),
            "budget_tokens": self.budget_tokens,
            "budget_exceeded": tokens_used >= self.budget_tokens,
            "cache_hits": self._cache_hits,
            "rate_limit_per_min": self.rate_limit_per_min,
        }


# 模块级单例：全进程共享同一账本/缓存/限速状态。
cost_gate = CostGate()
