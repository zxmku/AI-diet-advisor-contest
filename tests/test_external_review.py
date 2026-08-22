"""外部强模型 Agent 审查修复回归测试（2026-08-22 夜间）。

覆盖外部审查抓出的 3 项已修复问题 + 1 项评估不修（BM25 阈值）：
1. 时区 Bug：Docker/Linux 容器默认 UTC，time_context 必须按 UTC+8 计算时段
   （北京 21:03 = UTC 13:03，此前误判「午餐时段」）。
2. 多轮历史丢 Assistant：传给 LLM 的历史必须含 user+assistant 轮次，
   避免「谢谢」时模型看到两个连续 user 消息而复读上一轮方案。
3. 闲聊/客套分支清空 sources：chitchat 回复绝不外挂上一轮检索的来源分块。
4. （评估项）BM25 绝对阈值 1.2 实测无效（凑分查询 2.7~3.0 > 1.2，正常查询贴线 1.22），
   不修；现象由第 3 项（sources 清空）覆盖——不为此加断言，防止过度约束检索行为。
"""
from __future__ import annotations

from datetime import datetime, timezone

from app.main import _recent_chat_turns, _recent_user_messages
from app.time_context import get_meal_time_context


# ── 1. 时区：UTC+8 换算 ────────────────────────────────────────────────
def test_timezone_utc_container_evening():
    """容器 UTC 13:03 = 北京 21:03 → 必须判「非正餐/夜宵时段」，不得误判午餐。"""
    utc = datetime(2026, 8, 22, 13, 3, tzinfo=timezone.utc)
    assert get_meal_time_context(utc)["period"] == "非正餐/夜宵时段"


def test_timezone_utc_container_noon():
    """容器 UTC 04:00 = 北京 12:00 → 午餐时段。"""
    utc = datetime(2026, 8, 22, 4, 0, tzinfo=timezone.utc)
    assert get_meal_time_context(utc)["period"] == "午餐时段"


def test_timezone_naive_local_still_works():
    """naive datetime 视为 +8 本地时间，原有行为不变（早餐/晚餐边界）。"""
    assert get_meal_time_context(datetime(2026, 8, 22, 7, 30))["period"] == "早餐时段"
    assert get_meal_time_context(datetime(2026, 8, 22, 18, 30))["period"] == "晚餐时段"


# ── 2. 多轮历史：LLM 对话轮次含 assistant ──────────────────────────────
def test_recent_chat_turns_includes_assistant(client):
    """_recent_chat_turns 返回的轮次必须同时含 user 与 assistant，且旧→新。"""
    sid = "ext_chat_turns"
    for msg in ("我平时吸烟，吃饭上有什么忌口吗", "非常好。谢谢你的建议"):
        client.post("/api/chat", json={"user_id": "u", "session_id": sid, "message": msg})
    turns = _recent_chat_turns(sid, 4)
    roles = [t["role"] for t in turns]
    assert roles == ["user", "assistant", "user", "assistant"], f"应含完整轮次，实际 {roles}"
    # 检索增强用的 _recent_user_messages 仍只取 user（两函数职责分离）
    assert len(_recent_user_messages(sid, 4)) == 2


def test_llm_synthesize_accepts_dict_history(client, monkeypatch):
    """llm.synthesize 支持带 role 的 dict 历史（兼容纯字符串旧调用）。"""
    import app.llm as llm_mod
    from app.llm import synthesize

    captured: dict = {}

    def fake_post_json(url, headers, payload, timeout=15):
        captured["payload"] = payload
        return {"choices": [{"message": {"content": "不客气呀～"}}]}

    monkeypatch.setattr(llm_mod, "_post_json", fake_post_json)
    monkeypatch.setattr(llm_mod.config, "DEEPSEEK_API_KEY", "sk-test-fake")  # 让 is_enabled()=True
    # dict 历史（user+assistant 交错）+ 字符串历史兼容
    hist_dict = [
        {"role": "user", "content": "我吸烟有忌口吗"},
        {"role": "assistant", "content": "资料未提及，但建议多喝水"},
    ]
    out = synthesize("谢谢", [], history=hist_dict, user_id="u", session_id="s")
    assert out == "不客气呀～"
    msgs = captured["payload"]["messages"]
    roles = [m["role"] for m in msgs[1:4]]
    assert roles == ["user", "assistant", "user"], f"LLM 应收到完整轮次，实际 {roles}"
    # 字符串历史仍兼容
    synthesize("你好", [], history=["旧消息"], user_id="u2", session_id="s2")


# ── 3. 闲聊/客套清空 sources ───────────────────────────────────────────
def test_chitchat_thanks_has_no_sources(client):
    """「谢谢」走闲聊分支：sources 必须为空，绝不外挂上一轮检索分块。"""
    sid = "ext_thanks"
    client.post("/api/chat", json={"user_id": "u", "session_id": sid, "message": "控糖怎么吃"})
    d = client.post("/api/chat", json={
        "user_id": "u", "session_id": sid, "message": "非常好。谢谢"}).json()["data"]
    assert d["intent"] in ("chitchat", "medication_refuse")
    assert d.get("sources") in (None, [], [None]), f"闲聊不应挂来源，实际 {d.get('sources')}"


def test_chitchat_offdomain_has_no_sources(client):
    """非膳食问法（世界首富是谁）走闲聊：sources 为空。"""
    d = client.post("/api/chat", json={
        "user_id": "u", "session_id": "ext_off", "message": "世界首富是谁"}).json()["data"]
    assert d["intent"] == "chitchat"
    assert d.get("sources") in (None, [], [None])


# ═══ 暴风雪压力测试回归（2026-08-22 夜，100 条攻击式指令实测挖出的 4 缺陷）═══

def test_numeric_measure_word_stripped(client):
    """「鸡胸肉每100克有多少千卡」→ 量词剥离 → 命中 165（此前提取成「鸡胸肉每100」诚实 miss）。"""
    d = client.post("/api/chat", json={
        "user_id": "u", "session_id": "bz_measure", "message": "鸡胸肉每100克有多少千卡和蛋白质？"}).json()["data"]
    assert d["intent"] == "nutrition_lookup", f"应数值命中，实际 {d['intent']}"
    assert "165" in str(d.get("reply") or ""), "应含 165"


def test_medication_not_hijacked_by_numeric(client):
    """「头孢克肟吃完后几天不能喝酒」→ 拒药优先于数值分支（此前走 numeric miss 漏拒药）。"""
    d = client.post("/api/chat", json={
        "user_id": "u", "session_id": "bz_cef", "message": "头孢克肟吃完后几天不能喝酒？"}).json()["data"]
    assert d["intent"] == "medication_refuse", f"应拒药，实际 {d['intent']}"
    assert "不提供用药建议" in str(d.get("reply") or "")


def test_offdomain_not_numeric_miss(client):
    """「世界首富马斯克今天吃什么」→ 非膳食走闲聊（此前数值 miss 答「暂未收录「世界首富马斯」」）。"""
    d = client.post("/api/chat", json={
        "user_id": "u", "session_id": "bz_musk", "message": "世界首富马斯克今天吃什么？"}).json()["data"]
    assert d["intent"] == "chitchat", f"应走闲聊，实际 {d['intent']}"
    assert "暂未收录" not in str(d.get("reply") or "")


def test_negation_normal_sugar_not_guide(client):
    """「我没有糖尿病，血糖一切正常」→ 不触发控糖引导语（此前「血糖一切正常」漏剥离）。"""
    d = client.post("/api/chat", json={
        "user_id": "u", "session_id": "bz_neg", "message": "我没有糖尿病，血糖一切正常，给我推荐个减脂食谱"}).json()["data"]
    assert "结合您的控糖需求" not in str(d.get("reply") or ""), "否定句不得贴控糖引导"
    assert "血糖" not in str(d.get("reply") or "")[:20], "回复开头不应提血糖"


# ═══ 暴风雪压力测试第二波回归（QA 复核发现：点单路由/疾病截获/人设问）═══

def test_decision_not_hijacked_by_numeric_words(client):
    """「吉野家怎么点能少摄入点热量」→ 点单决策优先（此前含指标词被数值 miss 截获）。"""
    d = client.post("/api/chat", json={
        "user_id": "u", "session_id": "bz_gy", "message": "在吉野家吃牛肉饭，怎么点能少摄入点热量？"}).json()["data"]
    assert d["intent"] == "decision", f"应点单决策，实际 {d['intent']}"
    assert "暂未收录" not in str(d.get("reply") or "")


def test_disease_not_numeric_miss(client):
    """「有严重的脂肪肝…早餐吃什么」→ 疾病问法不得被数值提取成「严重」报 miss。"""
    d = client.post("/api/chat", json={
        "user_id": "u", "session_id": "bz_fat", "message": "有严重的脂肪肝，平时早餐吃什么能改善？"}).json()["data"]
    assert "暂未收录" not in str(d.get("reply") or ""), "疾病问法不得报暂未收录"
    assert d["intent"] not in ("nutrition_lookup_miss",)


def test_identity_question_not_meal_ask(client):
    """「你到底是谁？能帮我做些什么？」→ 人设问自我介绍，不得被「帮我做」截获成一餐追问。"""
    d = client.post("/api/chat", json={
        "user_id": "u", "session_id": "bz_who", "message": "你到底是谁？能帮我做些什么？"}).json()["data"]
    assert d["intent"] == "chitchat", f"应自我介绍，实际 {d['intent']}"
    assert "膳食顾问" in str(d.get("reply") or "")


def test_disease_meal_trigger_not_plan_template(client):
    """「有严重的脂肪肝…早餐吃什么能改善」→ 疾病问法不得落一餐方案模板（local-rules 下
    曾降级成增肌方案）；应走疾病路径（脂肪肝/控油/高纤维相关建议）。"""
    d = client.post("/api/chat", json={
        "user_id": "u", "session_id": "bz_fat2", "message": "有严重的脂肪肝，平时早餐吃什么能改善？"}).json()["data"]
    r = str(d.get("reply") or "")
    assert "增肌" not in r and "热量盈余" not in r, "疾病问法不得给增肌方案"
    assert ("脂肪肝" in r) or ("控油" in r) or ("控糖" in r), "应给疾病针对性建议"
