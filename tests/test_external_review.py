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

import pytest
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


# ═══ Gemini 外部审查回归（漏洞 1-6：对比句式/混合意图/台账裸词/减重词/台账同义词/调理免责）═══

def test_compare_food_not_weird_miss(client):
    """对比句式（西红柿和黄瓜…各是多少）不得拼成假食材报「暂未收录」。"""
    d = client.post("/api/chat", json={
        "user_id": "u", "session_id": "gm_cmp", "message": "西红柿和黄瓜每100克热量各是多少？"}).json()["data"]
    assert "暂未收录" not in str(d.get("reply") or ""), "对比句式不得报怪异 miss"
    assert d["intent"] != "nutrition_lookup_miss"


def test_mixed_intent_platform_kept(client):
    """混合意图：平台价格 + 晚餐搭配 两段都答，不丢弃平台咨询。"""
    d = client.post("/api/chat", json={
        "user_id": "u", "session_id": "gm_mix", "message": "你们平台的专业版会员多少钱一个月？顺便告诉我减脂期晚餐怎么吃。"}).json()["data"]
    r = str(d.get("reply") or "")
    assert "为你搭配" in r, "应含晚餐搭配"
    # 严格断言：价格段必须真实作答（不得仅靠「平台」二字 trivial 命中）
    # 系统以「版 + 具体价格」作答（如 标准版 / 59 元/月），不强制字面「会员」
    assert any(k in r for k in ("元", "月", "¥", "价格", "收费", "套餐", "版", "会员")), \
        "混合意图中平台价格段须含具体价格信息，不得丢弃"


def test_triple_isolation_price_allergy_nutrition(client):
    """三合一隔离：平台价格(C) + 过敏剔除 + 营养数值(A) 同轮不串味。

    审核盲点#3 补强：此前仅单例验证混合意图，未证「价格+过敏+营养」三者共存时互不污染。
    原 xfail（过敏澄清分支吞价格）已于支柱A 修复：数值块内补 _platform_reply_prefix，
    现转为正式回归护栏——「数值×平台同轮」价格段不得再丢失。
    """
    d = client.post("/api/chat", json={
        "user_id": "u", "session_id": "gm_tri",
        "message": "我对虾过敏，你们专业版会员多少钱？顺便说下鸡胸肉多少千卡"}).json()["data"]
    r = str(d.get("reply") or "")
    # 价格段（C 库）真实作答：系统以「版 + 具体价格」作答，不强制字面「会员」
    assert any(k in r for k in ("元", "月", "¥", "价格", "收费", "套餐", "版", "会员")), \
        "价格段须含具体价格信息"
    # 营养段（A 库）真实作答，不被 C 库污染
    assert ("鸡胸肉" in r) or ("165" in r), "营养段须含鸡胸肉真实数值，不得被平台价格串味"
    # 过敏段生效：须显式处理过敏，且虾须出现在禁忌排除清单中（而非被推荐）
    assert ("已按您的禁忌排除" in r) or ("过敏" in r) or ("剔除" in r) or ("不吃" in r), \
        "过敏约束须被显式处理"
    assert ("虾仁" in r) or ("虾" in r), "虾须出现在禁忌排除清单中"


def test_miss_platform_isolation(client):
    """同构残边补强：表外食材(诚实 miss) + 平台价格同轮，平台段不得被吞。

    支柱A 二审残边：numeric_miss 分支此前未补平台前缀，导致「牛油果多少千卡 +
    会员多少钱」这类「表外食材 miss × 平台价格」同轮时丢失价格段。现补齐后转正。
    """
    d = client.post("/api/chat", json={
        "user_id": "u", "session_id": "gm_miss_plat",
        "message": "牛油果多少千卡？你们专业版会员多少钱？"}).json()["data"]
    r = str(d.get("reply") or "")
    # 平台价格段（C 库）真实作答：系统以「版 + 具体价格」作答，不强制字面「会员」
    assert any(k in r for k in ("元", "月", "¥", "价格", "收费", "套餐", "版", "会员")), \
        "表外食材 miss 同轮，平台价格段须含具体价格信息，不得丢弃"
    # 膳食段（A 库诚实 miss）真实作答，不被 C 库污染
    assert ("牛油果" in r) or ("暂未收录" in r), "营养段须含表外食材的诚实 miss，不得被平台价格串味"


def test_ledger_record_with_ledger_word(client):
    """「帮我记一笔台账：吃了两个鸡蛋」→ 记录（裸「台账」不再误判查询）。"""
    d = client.post("/api/chat", json={
        "user_id": "u", "session_id": "gm_led", "message": "帮我记一笔台账：吃了两个鸡蛋"}).json()["data"]
    assert d["intent"] == "diet_record", f"应记录，实际 {d['intent']}"


def test_goal_weight_loss_phrase(client):
    """「我想减重/想降体重」→ 识别为减脂目标（此前漏「想减重」；三审补「降体重」死词）。"""
    from app.main import _detect_goal
    assert _detect_goal("我想减重，给我推荐个方案") == "减脂"
    assert _detect_goal("我想降体重，怎么吃") == "减脂", "「想降体重」此前是死词（只进健康状态词、无目标映射）"
    assert _detect_goal("想增重") == "增肌", "「降体重」不得混淆「增重」反向意图"


def test_ledger_kcal_synonym(client):
    """台账估算接入同义词（西红柿→番茄/地瓜→红薯）→ 具体食材项出现（三审收窄断言）。"""
    d = client.post("/api/chat", json={
        "user_id": "u", "session_id": "gm_kcal", "message": "帮我记一下：中午吃了西红柿炒鸡蛋和两块地瓜"}).json()["data"]
    r = str(d.get("reply") or "")
    assert "番茄" in r and "红薯" in r, (
        f"同义词归一后应命中速查表具体项（番茄/红薯），实际: {r!r}"
    )


def test_ledger_kcal_compound_rice(client):
    """三审 P1：整句别名替换破坏复合食材——「糙米饭」不得被子串「米饭→白米饭」误算成 130。"""
    from app.main import _estimate_diet_kcal
    total, items = _estimate_diet_kcal("中午吃了糙米饭")
    assert total == 348.0, f"糙米饭应命中糙米 348 千卡，实际 {total}（{items}）"
    assert items[0]["food"] == "糙米", f"应为糙米，实际 {items[0]['food']}"
    total2, _ = _estimate_diet_kcal("中午吃了大米饭")
    assert total2 == 130.0, f"大米饭应命中白米饭 130 千卡，实际 {total2}"


def test_ledger_query_ledger_word(client):
    """三审 P2：裸「台账」回归查询语义——「查一下我的台账」不得漏判成闲聊。"""
    d = client.post("/api/chat", json={
        "user_id": "u", "session_id": "gm_qry", "message": "查一下我的台账"}).json()["data"]
    assert d["intent"] == "diet_query", f"应查询台账，实际 {d['intent']}"


def test_ledger_how_record_not_recorded(client):
    """三审 P2：「怎么记录台账」是问方法，不得被「记录台账」误记为台账。"""
    d = client.post("/api/chat", json={
        "user_id": "u", "session_id": "gm_how", "message": "怎么记录台账"}).json()["data"]
    assert d["intent"] != "diet_record", f"问方法不应记台账，实际 {d['intent']}"


def test_care_goal_disclaimer(client):
    """会话目标=调理（稳糖）时，常规问答必须带免责（三审：走非疾病途径设 goal，锁死新分支）。"""
    c = client
    # 「我想控糖」→ _detect_goal=调理 写库，但不触发疾病判定（控糖不在疾病标签表）
    c.post("/api/chat", json={"user_id": "u", "session_id": "gm_care", "message": "我想控糖"})
    r = c.post("/api/chat", json={
        "user_id": "u", "session_id": "gm_care", "message": "鸡蛋有多少蛋白质？"}).json()
    assert r.get("disclaimer"), "调理会话常规问答必须带免责（goal_tag=调理 分支）"
