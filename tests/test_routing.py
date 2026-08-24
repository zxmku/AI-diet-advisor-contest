"""领域路由门控测试：膳食顾问需求 vs 闲聊/非膳食。

核心诉求：加入闲聊/陪聊后，回复逻辑必须区分两种模式——
①膳食顾问（命中红线需求）→ RAG 检索 ABC 整理回复，语气像陪伴伙伴；
②闲聊/非膳食 → 不走知识库，自然回应（不能硬套知识库）。

否则「大龙虾吃什么」「世界首富是谁」会被 BM25 检索到低相关块后硬套成膳食内容。
"""
from __future__ import annotations


def _chat(client, message, *, session_id, user_id="u", allergies=None):
    r = client.post(
        "/api/chat",
        json={
            "user_id": user_id,
            "session_id": session_id,
            "message": message,
            "allergies": allergies or [],
        },
    )
    assert r.status_code == 200
    return r.json()["data"]


# ── 非膳食领域 → 必须走 chitchat（绝不硬套膳食知识块）─────────────────
def test_non_dietary_lobster_feeding(client):
    d = _chat(client, "大龙虾吃什么", session_id="r_lobster")
    assert d["intent"] == "chitchat", f"动物习性问题不得硬套膳食内容: {d['intent']}"


def test_non_dietary_world_richest(client):
    d = _chat(client, "世界首富是谁", session_id="r_rich")
    assert d["intent"] == "chitchat"


def test_non_dietary_arithmetic(client):
    d = _chat(client, "1加1等于几", session_id="r_math")
    assert d["intent"] == "chitchat"


def test_non_dietary_emotion(client):
    d = _chat(client, "我想你了", session_id="r_miss")
    assert d["intent"] == "chitchat"


def test_non_dietary_movie(client):
    d = _chat(client, "推荐一部电影", session_id="r_movie")
    assert d["intent"] == "chitchat"


# ── 膳食领域（含口语化诉求）→ 不得误判为闲聊 ────────────────────────
def test_dietary_colloquial_not_misrouted(client):
    for i, msg in enumerate(("我想减肚子", "吃什么能长肌肉", "怎么吃才能瘦", "最近总饿怎么办")):
        d = _chat(client, msg, session_id=f"r_col{i}")
        assert d["intent"] != "chitchat", f"口语化膳食需求不得误判为闲聊: {msg} → {d['intent']}"


# ── M11 画像感知引导语：疾病用户回复差异化 ──────────────────────────
def test_disease_profile_guide_first_round(client):
    """糖尿病用户首轮即带「结合您的控糖需求」引导（顾问感）。"""
    d = _chat(client, "糖尿病能吃燕麦吗", session_id="m11_dm")
    assert "结合您的控糖需求" in d["reply"], f"糖尿病用户应带控糖引导: {d['reply'][:50]}"
    assert d["intent"] != "chitchat"


def test_disease_profile_guide_cross_round(client):
    """声明痛风后，次轮任意问（消息无病名）仍带「结合您的低嘌呤需求」（跨轮延续）。"""
    _chat(client, "我有痛风", session_id="m11_gout")
    d = _chat(client, "推荐减脂方案", session_id="m11_gout")
    assert "结合您的低嘌呤需求" in d["reply"], f"痛风跨轮应带低嘌呤引导: {d['reply'][:50]}"


def test_no_disease_no_guide(client):
    """普通用户（无疾病）不得误加「结合您的」引导。"""
    for i, msg in enumerate(("鸡胸肉多少千卡", "你好")):
        d = _chat(client, msg, session_id=f"m11_no{i}")
        assert "结合您的" not in d["reply"], f"无疾病用户不应带引导: {msg}"


def test_no_disease_no_guide_multiround(client):
    """二审 P0-5 回归：普通增肌用户次轮不得被误判为「控糖用户」。

    修复前 _session_has_disease/_profile_guide 扫全部消息（含 assistant 回复），
    assistant 素材块原文含「血糖/糖尿病」→ 普通用户次轮被贴「结合您的控糖需求」
    + 过度免责（产品记忆错乱）。修复后只扫 user 消息，模型输出永不当用户事实。
    """
    _chat(client, "我想增肌", session_id="m11_clean", user_id="u_clean")
    d = _chat(client, "我在便利店准备点餐", session_id="m11_clean", user_id="u_clean")
    assert "结合您的控糖需求" not in d["reply"], (
        f"普通增肌用户次轮被误贴控糖引导: {d['reply'][:60]}"
    )
    # 疾病用户跨轮延续不受影响（正向对照）
    _chat(client, "我有糖尿病", session_id="m11_dm2", user_id="u_dm")
    d2 = _chat(client, "推荐减脂方案", session_id="m11_dm2", user_id="u_dm")
    assert "结合您的控糖需求" in d2["reply"], "糖尿病跨轮引导应保留"


def test_disease_three_high_meal_has_disclaimer(client):
    """二审 P0-4（红线②）回归：三高/糖高人群要一餐必须带免责。

    修复前「三高」「糖高」不在 _DISEASE_KEYWORDS → 出稳糖餐却 disclaimer=None。
    """
    for i, msg in enumerate(("三高人群午餐", "我糖高早餐吃什么")):
        r = client.post(
            "/api/chat",
            json={"user_id": "u_th", "session_id": f"m11_th{i}", "message": msg},
        )
        assert r.status_code == 200
        body = r.json()
        assert body.get("disclaimer"), f"「{msg}」出餐未带免责"
        assert "不构成医疗建议" in body["disclaimer"]


def test_quick_control_sugar_has_disclaimer(client):
    """二审 P0-4（红线②）回归：/api/quick control_sugar 必须带免责。

    修复前 quick 端点漏 disclaimer（recommend(调理) 有、quick 无，两入口不一致）。
    """
    r = client.post(
        "/api/quick",
        json={"user_id": "u_q", "session_id": "q_cs", "action": "control_sugar"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body.get("disclaimer"), "quick(control_sugar) 未带免责"
    assert "不构成医疗建议" in body["disclaimer"]
    # 对照：减脂快捷不带免责
    r2 = client.post(
        "/api/quick",
        json={"user_id": "u_q", "session_id": "q_lf", "action": "lose_fat"},
    )
    assert r2.status_code == 200
    assert r2.json().get("disclaimer") is None


def test_cannot_eat_shrimp_next_round_excluded(client):
    """二审 P0 新增：否定式禁忌声明「我不能吃虾」→ 次轮推荐不得含虾。"""
    r1 = _chat(client, "我不能吃虾", session_id="m11_shrimp", user_id="u_sh")
    assert "已为您记录" in r1["reply"], f"「我不能吃虾」应识别为禁忌: {r1['reply'][:60]}"
    r2 = _chat(client, "推荐减脂方案", session_id="m11_shrimp", user_id="u_sh")
    assert "虾仁" not in r2["reply"], f"次轮推荐仍含虾: {r2['reply'][:80]}"


def test_glp1_refuse_with_disclaimer(client):
    """二审 P0 新增：GLP-1 别称必须拒药 + 免责。"""
    r = client.post(
        "/api/chat",
        json={"user_id": "u_g", "session_id": "m11_glp", "message": "GLP-1针副作用"},
    )
    assert r.status_code == 200
    body = r.json()
    assert "不提供用药建议" in body["data"]["reply"], "GLP-1 应拒药"
    assert body.get("disclaimer"), "GLP-1 应带免责"


# ── 2026-08-24 合规修复：第三人称健康声明不得写库/贴前缀 ─────────────
def test_third_party_disease_not_own_profile(client):
    """「我妈妈有糖尿病」不得贴「您的控糖需求」前缀（编造用户健康事实）。"""
    d = _chat(client, "我妈妈有糖尿病，怎么给她做饭", session_id="r_tp1")
    assert "您的控糖" not in d["reply"], f"第三人称不得贴本人前缀: {d['reply'][:60]}"
    assert "结合您" not in d["reply"], f"第三人称不得贴'结合您'前缀: {d['reply'][:60]}"


def test_third_party_disease_not_persisted(client):
    """第三人称健康声明不写库：下一轮普通问题不触发「您的」前缀（无持久污染）。"""
    _chat(client, "我妈妈有糖尿病", session_id="r_tp2")
    d = _chat(client, "今天中午吃什么好", session_id="r_tp2")
    assert "您的控糖" not in d["reply"], f"第三人称污染画像: {d['reply'][:60]}"


def test_first_person_disease_still_guided(client):
    """第一人称「我有糖尿病」仍正常贴「结合您的控糖需求」前缀（对照回归）。"""
    d = _chat(client, "我有糖尿病，平时吃什么好", session_id="r_fp1")
    assert "您的控糖" in d["reply"], f"第一人称必须保留画像引导: {d['reply'][:60]}"


# ── 2026-08-24 场景词路由：场景化点餐不劫持成一餐 ───────────────────
def test_scene_meal_not_hijacked_convenience(client):
    """便利店买的早餐 → 不得走 meal（自己做一餐），应走点单决策。"""
    d = _chat(client, "便利店买的早餐能吃吗", session_id="r_sc1")
    assert d["intent"] == "decision", f"便利店场景应走点单决策: {d['intent']}"


def test_scene_meal_not_hijacked_hotel(client):
    """酒店自助早餐 → 不得走 meal，给场景化指导（检索/决策均可）。"""
    d = _chat(client, "酒店自助早餐琳琅满目怎么拿", session_id="r_sc2")
    assert d["intent"] != "meal", f"酒店自助不得判成自己做一餐: {d['intent']}"


def test_normal_meal_intent_kept(client):
    """对照：无场景词的「今晚减脂餐」仍正常走一餐生成。"""
    d = _chat(client, "今晚减脂餐吃什么", session_id="r_sc3")
    assert d["intent"] == "meal", f"正常一餐意图不得受影响: {d['intent']}"


# ── 2026-08-24 H1：第三人称过敏不得记入用户禁忌清单 ─────────────────
def test_third_party_allergy_not_own_excluded(client):
    """「我妈妈对花生过敏」不得把坚果类排除到用户本人。"""
    _chat(client, "我妈妈对花生过敏", session_id="r_tp_a1")
    d = _chat(client, "今天中午吃什么好", session_id="r_tp_a1")
    # 用户本人未被排除花生/坚果：回复不应出现「您的禁忌/过敏清单」
    assert "您的禁忌" not in d["reply"] and "您的过敏" not in d["reply"], \
        f"第三人称过敏不得记入用户禁忌: {d['reply'][:80]}"


def test_first_person_allergy_still_excluded(client):
    """对照：本人「我对花生过敏」仍正常排除花生（多轮拦截）。"""
    _chat(client, "我对花生过敏", session_id="r_fp_a1")
    d = _chat(client, "那我现在能吃花生酱吗", session_id="r_fp_a1")
    assert d["intent"] == "allergy_block", f"本人过敏仍必须拦截: {d['intent']}"
