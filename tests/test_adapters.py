"""P-迭代 适配器测试：时间感知 / 情绪共情规则 / 语音输出 TTS / 采购清单。

复用 tests/conftest.py 的 temp DB 隔离；LLM/TTS 均 mock，不触网。
"""
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from fastapi.testclient import TestClient  # noqa: E402

from app import config, llm as llm_mod, tts_service  # noqa: E402


# ── 1. 时间感知引擎 ──────────────────────────────────────────────
def test_time_context_period_mapping():
    from app.time_context import get_meal_time_context

    cases = [
        (datetime(2026, 8, 22, 8, 0), "早餐时段"),
        (datetime(2026, 8, 22, 12, 30), "午餐时段"),
        (datetime(2026, 8, 22, 18, 0), "晚餐时段"),
        (datetime(2026, 8, 22, 23, 30), "非正餐/夜宵时段"),
    ]
    for dt, expect in cases:
        ctx = get_meal_time_context(dt)
        assert ctx["period"] == expect, f"{dt.hour}时 → {ctx['period']} != {expect}"
        assert ctx["greeting"] and ctx["focus"]


def test_llm_prompt_has_time_and_empathy_rules():
    """系统提示包含：时间感知注入 + 情绪共情规则（先共情后建议）。"""
    captured = {}

    def fake_post(url, headers, payload, timeout):
        captured["sys"] = payload["messages"][0]["content"]
        return {"choices": [{"message": {"content": "好的"}}]}

    llm_mod._post_json = fake_post
    config.DEEPSEEK_API_KEY = "dummy"
    out = llm_mod.synthesize("好累，想吃炸鸡", [], history=[])
    assert out == "好的"
    sys_text = captured["sys"]
    assert "当前时段" in sys_text, "时间感知未注入系统提示"
    assert "先共情，后建议" in sys_text, "情绪共情规则未注入"
    assert "参考资料" in sys_text
    config.DEEPSEEK_API_KEY = ""


# ── 2. 语音输出 TTS 适配器 ───────────────────────────────────────
def test_tts_clean_text():
    assert "来源" in tts_service._clean_text("【资料】**鸡胸肉** [来源速查表]")
    assert "已剔除" in tts_service._clean_text("（已按禁忌剔除）")


def test_tts_unavailable_returns_503(monkeypatch):
    monkeypatch.setattr(tts_service, "_EDGE_AVAILABLE", False)
    from app.main import app

    with TestClient(app) as c:
        r = c.post("/api/tts", json={"text": "你好"})
    assert r.status_code == 503


def test_tts_route_returns_audio(monkeypatch):
    class FakeCommunicate:
        def __init__(self, text, voice):
            self.text, self.voice = text, voice

        async def stream(self):
            yield {"type": "audio", "data": b"\xff\xf3FAKE_MP3"}
            yield {"type": "WordBoundary", "offset": 0, "length": 2}

    class FakeEdge:
        Communicate = FakeCommunicate

    monkeypatch.setattr(tts_service, "edge_tts", FakeEdge)
    monkeypatch.setattr(tts_service, "_EDGE_AVAILABLE", True)
    from app.main import app

    with TestClient(app) as c:
        r = c.post("/api/tts", json={"text": "你好"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/mpeg")
    assert b"FAKE_MP3" in r.content


# ── 3. 采购清单（P3 一餐派生）────────────────────────────────────
def test_meal_shopping_list(monkeypatch):
    from app.main import app

    with TestClient(app) as c:
        r = c.post(
            "/api/chat",
            json={"user_id": "u", "session_id": "s", "message": "今晚减脂餐"},
        )
    assert r.status_code == 200
    meal = r.json()["data"].get("meal")
    assert meal is not None
    sl = meal.get("shopping_list")
    assert sl and len(sl) >= 3, "采购清单缺失或过短"
    assert all(x.get("food") for x in sl)
    foods = [x["food"] for x in sl]
    assert any("鸡胸肉" in f for f in foods), "采购清单应含主蛋白"
