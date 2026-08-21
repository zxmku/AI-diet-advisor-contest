"""R8 M17 成本闸门单元测试：mock app.llm._post_json，绝不触网。

验证三层保护（对应蓝图 M17 / 项目 cost_gate.py）：
- 预算=0 → synthesize 直接返回 None（自动降级 local-rules）；
- 同消息且无 history 两次 → 第一次真实走 API、第二次缓存命中（零 API 花费）；
- 限速=2 → 同一会话第三条调用返回 None（会话级限速）。

数据隔离：每个用例把 cost_gate 的数据目录重定向到 pytest tmp_path，
并清零限速桶/缓存命中计数，避免污染 backend/data 与跨用例串扰。
"""
from __future__ import annotations

import pytest

import app.config as config
import app.cost_gate as cost_gate_module
import app.llm as llm
from app.cost_gate import cost_gate


@pytest.fixture(autouse=True)
def _isolated_gate(tmp_path, monkeypatch):
    """每个用例独立闸门状态：假 Key + 临时数据目录 + 清零限速/缓存计数。

    注意：_DATA_DIR/_LEDGER_PATH/_CACHE_PATH 是 cost_gate 模块级常量
    （文件读写经模块全局查找），须 monkeypatch 在模块上；
    budget_tokens 等是单例实例属性，直接 patch 实例。
    """
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(cost_gate_module, "_DATA_DIR", tmp_path)
    monkeypatch.setattr(cost_gate_module, "_LEDGER_PATH", tmp_path / "cost_ledger.json")
    monkeypatch.setattr(cost_gate_module, "_CACHE_PATH", tmp_path / "llm_cache.json")
    cost_gate._rate.clear()
    cost_gate._cache_hits = 0
    yield cost_gate


def _fake_post_json(reply: str = "mock-reply"):
    """构造假 _post_json：记录每次真实调用的最后一条用户消息。"""
    calls: list[str] = []

    def _post(url: str, headers: dict, payload: dict, timeout: int) -> dict:
        calls.append(payload["messages"][-1]["content"])
        return {
            "choices": [{"message": {"content": reply}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

    _post.calls = calls  # type: ignore[attr-defined]
    return _post


def test_m17_budget_zero_returns_none(monkeypatch, _isolated_gate):
    """预算=0：当日 token 已达上限，synthesize 返回 None（不触网）。"""
    monkeypatch.setattr(_isolated_gate, "budget_tokens", 0)
    monkeypatch.setattr(_isolated_gate, "rate_limit_per_min", 10)
    monkeypatch.setattr(llm, "_post_json", _fake_post_json(reply="不应被调用"))
    assert llm.is_enabled()  # 确保返回 None 源于预算闸门，而非 Key 缺失
    assert llm.synthesize("你好", []) is None


def test_m17_cache_hit_on_second_call(monkeypatch, _isolated_gate):
    """同消息且无 history：第一次走 API，第二次缓存命中（仅 1 次触网）。"""
    monkeypatch.setattr(_isolated_gate, "budget_tokens", 1_000_000)
    monkeypatch.setattr(_isolated_gate, "rate_limit_per_min", 10)
    fake = _fake_post_json(reply="cached-reply")
    monkeypatch.setattr(llm, "_post_json", fake)
    assert llm.is_enabled()
    assert llm.synthesize("鸡胸肉多少千卡", []) == "cached-reply"
    assert llm.synthesize("鸡胸肉多少千卡", []) == "cached-reply"
    assert len(fake.calls) == 1  # 第二次命中缓存，不再发起 API 调用
    assert _isolated_gate._cache_hits >= 1


def test_m17_rate_limit_third_call_returns_none(monkeypatch, _isolated_gate):
    """限速=2：同会话第三条调用触发限速，返回 None（降级 local-rules）。"""
    monkeypatch.setattr(_isolated_gate, "budget_tokens", 1_000_000)
    monkeypatch.setattr(_isolated_gate, "rate_limit_per_min", 2)
    fake = _fake_post_json(reply="ok")
    monkeypatch.setattr(llm, "_post_json", fake)
    assert llm.is_enabled()
    assert llm.synthesize("消息一", []) == "ok"
    assert llm.synthesize("消息二", []) == "ok"
    assert llm.synthesize("消息三", []) is None  # 第三条触发会话限速
