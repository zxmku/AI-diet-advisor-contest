"""V01-V06 审计红线回归：逻辑层审计发现的 6 项红线级漏洞。

覆盖：
- V01 P5 分支绕过用药硬拦截：「我吃了布洛芬怎么办」须拒药而非记台账；
  「一个人吃饭可以吃布洛芬吗」须拒药而非陪伴；
- V02 禁忌漏识别（同族替换泄露）：「我对鱼过敏 / 我对大豆过敏 / 吃鸡蛋起疹」
  须识别出禁忌并给出排除提示；
- V03 拒药词表盲区：诺和泰 / ozempic / 优甲乐 / 止疼片 须拒药；
- V04 裸「坚果」误报 + R3 剔除损坏引号内容：坚果热量问不误判过敏、
  回复引号内容完整不出现「（已按禁忌剔除）」损坏；「坚果过敏」仍识别；
- V05 疾病免责词表盲区：脂肪肝 / 血脂高 / 贫血 须带免责声明；
- V06 LLM 缓存跨用户串扰：不同用户同消息不得命中彼此缓存（mock LLM）。

复用 conftest 的 temp DB 隔离；V06 用例把 cost_gate 数据目录重定向到 tmp_path。
"""
from __future__ import annotations

import pytest

import app.config as config
import app.cost_gate as cost_gate_module
import app.llm as llm
from app.cost_gate import cost_gate

REFUSE_MARK = "不提供用药建议"
DIET_RECORD_MARK = "已帮你记下"
ALLERGY_MARK = "已按您的禁忌排除"
REDACT_MARK = "（已按禁忌剔除）"


def _chat(client, message, *, session_id, user_id, allergies=None):
    return client.post(
        "/api/chat",
        json={
            "user_id": user_id,
            "session_id": session_id,
            "message": message,
            "allergies": allergies or [],
        },
    )


# ── V01 P5 分支绕过用药硬拦截 ─────────────────────────────
def test_v01_diet_record_not_hijack_medication(client):
    """「我吃了布洛芬怎么办」必须拒药，不得被记成饮食台账。"""
    r = _chat(client, "我吃了布洛芬怎么办", session_id="v01a", user_id="u_v01")
    assert r.status_code == 200
    body = r.json()
    reply = body["data"]["reply"]
    assert REFUSE_MARK in reply, f"未拒答用药: {reply[:80]}"
    assert DIET_RECORD_MARK not in reply, "用药问题被误记成饮食台账"
    assert body["data"]["intent"] != "diet_record"


def test_v01_companion_not_hijack_medication(client):
    """「一个人吃饭可以吃布洛芬吗」必须拒药，不得走情感陪伴。"""
    r = _chat(client, "一个人吃饭可以吃布洛芬吗", session_id="v01b", user_id="u_v01")
    assert r.status_code == 200
    body = r.json()
    assert REFUSE_MARK in body["data"]["reply"], f"未拒答用药: {body['data']['reply'][:80]}"
    assert body["data"]["intent"] != "companion"


# ── V02 禁忌漏识别（同族替换泄露）──────────────────────────
def test_v02_fish_allergy_detected(client):
    """「我对鱼过敏」→ 识别鱼类禁忌，排除提示出现且列出鱼类。"""
    r = _chat(client, "我对鱼过敏", session_id="v02a", user_id="u_v02")
    assert r.status_code == 200
    reply = r.json()["data"]["reply"]
    assert ALLERGY_MARK in reply, f"未识别鱼类禁忌: {reply[:80]}"
    assert "三文鱼" in reply and "鳕鱼" in reply


def test_v02_soy_allergy_excludes_tofu_soymilk(client):
    """「我对大豆过敏」→ 排除提示出现且含豆腐/豆浆。"""
    r = _chat(client, "我对大豆过敏", session_id="v02b", user_id="u_v02")
    assert r.status_code == 200
    reply = r.json()["data"]["reply"]
    assert ALLERGY_MARK in reply, f"未识别大豆禁忌: {reply[:80]}"
    assert "豆腐" in reply and "豆浆" in reply


def test_v02_egg_symptom_detected(client):
    """「吃鸡蛋起疹」→ 症状式识别鸡蛋禁忌，排除提示出现且含鸡蛋。"""
    r = _chat(client, "吃鸡蛋起疹", session_id="v02c", user_id="u_v02")
    assert r.status_code == 200
    reply = r.json()["data"]["reply"]
    assert ALLERGY_MARK in reply, f"未识别鸡蛋禁忌: {reply[:80]}"
    assert "鸡蛋" in reply


# ── V03 拒药词表盲区 ─────────────────────────────────────
@pytest.mark.parametrize(
    "msg",
    ["诺和泰减肥怎么样", "ozempic 减肥", "优甲乐能吃吗", "止疼片能随便吃吗"],
)
def test_v03_medication_keywords_refuse(client, msg):
    """V03 新增拒药词：GLP-1（诺和泰/ozempic）、处方药（优甲乐）、口语（止疼片）须拒药。"""
    r = _chat(client, msg, session_id="v03a", user_id="u_v03")
    assert r.status_code == 200
    assert REFUSE_MARK in r.json()["data"]["reply"], f"「{msg}」未拒答用药"


# ── V04 裸「坚果」误报 + R3 剔除损坏引号内容 ───────────────
def test_v04_bare_nut_not_false_allergy(client):
    """「坚果的热量是多少」→ 不误判过敏（无已为您记录），回复引号内容完整无剔除损坏。"""
    r = _chat(client, "坚果的热量是多少", session_id="v04a", user_id="u_v04")
    assert r.status_code == 200
    reply = r.json()["data"]["reply"]
    assert "已为您记录" not in reply, f"裸「坚果」被误判成过敏: {reply[:80]}"
    assert REDACT_MARK not in reply, f"回复出现剔除损坏标记: {reply[:80]}"
    assert "「坚果」" in reply, f"引号内容被破坏: {reply[:80]}"


def test_v04_nut_allergy_still_detected(client):
    """「坚果过敏」→ 仍识别坚果禁忌（移除裸词不破坏既有识别）。"""
    r = _chat(client, "坚果过敏", session_id="v04b", user_id="u_v04")
    assert r.status_code == 200
    reply = r.json()["data"]["reply"]
    assert ALLERGY_MARK in reply, f"坚果过敏未被识别: {reply[:80]}"


# ── V05 疾病免责词表盲区 ─────────────────────────────────
@pytest.mark.parametrize(
    "msg",
    ["我有脂肪肝怎么办", "我血脂高怎么办", "我贫血怎么吃"],
)
def test_v05_disease_keywords_have_disclaimer(client, msg):
    """V05 新增疾病词：脂肪肝/血脂高/贫血 须带标准免责声明。"""
    r = _chat(client, msg, session_id="v05a", user_id="u_v05")
    assert r.status_code == 200
    body = r.json()
    assert body.get("disclaimer"), f"「{msg}」未携带免责声明"
    assert "不构成医疗建议" in body["disclaimer"]


# ── V06 LLM 缓存跨用户串扰 ───────────────────────────────
@pytest.fixture
def _isolated_gate(tmp_path, monkeypatch):
    """独立闸门状态：假 Key + 临时数据目录 + 清零限速/缓存计数（同 test_cost_gate）。"""
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


def test_v06_cache_isolated_across_users(monkeypatch, _isolated_gate):
    """同消息：userA（海鲜过敏）与 userB（无禁忌）不得命中彼此缓存 → 2 次真实调用。"""
    monkeypatch.setattr(_isolated_gate, "budget_tokens", 1_000_000)
    monkeypatch.setattr(_isolated_gate, "rate_limit_per_min", 100)
    fake = _fake_post_json(reply="mock-reply")
    monkeypatch.setattr(llm, "_post_json", fake)
    assert llm.is_enabled()

    assert (
        llm.synthesize(
            "给我推荐一份晚餐",
            [],
            session_id="sess_A",
            user_id="user_A",
            excluded_foods=["虾仁", "三文鱼", "鳕鱼"],
        )
        == "mock-reply"
    )
    assert (
        llm.synthesize(
            "给我推荐一份晚餐",
            [],
            session_id="sess_B",
            user_id="user_B",
            excluded_foods=None,
        )
        == "mock-reply"
    )
    assert len(fake.calls) == 2, "userB 命中了 userA 的缓存（跨用户串扰未修复）"


def test_v06_cache_still_hits_same_user(monkeypatch, _isolated_gate):
    """同用户同消息（同禁忌状态）→ 缓存仍生效（第二次命中，1 次真实调用）。"""
    monkeypatch.setattr(_isolated_gate, "budget_tokens", 1_000_000)
    monkeypatch.setattr(_isolated_gate, "rate_limit_per_min", 100)
    fake = _fake_post_json(reply="mock-reply")
    monkeypatch.setattr(llm, "_post_json", fake)
    assert llm.is_enabled()

    for _ in range(2):
        assert (
            llm.synthesize(
                "给我推荐一份晚餐",
                [],
                session_id="sess_A",
                user_id="user_A",
                excluded_foods=["虾仁", "三文鱼", "鳕鱼"],
            )
            == "mock-reply"
        )
    assert len(fake.calls) == 1, "同用户缓存未命中（缓存功能回归）"
