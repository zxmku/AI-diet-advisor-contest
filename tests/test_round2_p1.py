"""波次2（P1 七项 + 修复A）回归测试。

覆盖二审报告/QA 复验中的 P1 级代码逻辑漏洞：
- 修复A/P1-6：症状正则负向守卫（吃鱼油难受/吃蛋白粉过敏/吃虾片过敏 不得误记禁忌）
- P1-1：C 库词表漂移（升级专业版/专业版怎么买/怎么开通你们的服务 → platform + C 源）
- P1-2：跨用户会话串号（userB 借 userA 的 session_id 不得继承画像/读档案）
- P1-3：禁忌否定式识别补全（不能吃海产品/对花生不耐受/海鲜不能吃/鸡蛋碰都不能碰）
- P1-4：负面表述误判（我血糖正常/我没有糖尿病 不得贴控糖引导）
- P1-5：用药针剂类词表（打降糖针/减肥针 → 拒药+免责）
- P1-7：今晚天气劫持（今晚天气怎么样 → 非一餐；今晚吃什么仍一餐）

全部通过 conftest 的临时 DB 隔离；不污染生产库。
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
    return r.json()


# ── 修复A/P1-6：症状正则负向守卫（红线②边界）────────────────────
def test_symptom_fish_oil_not_fish_allergy(client):
    """「我吃鱼油难受」不得误记 fish_allergy（鱼油是补剂不是鱼）。"""
    body = _chat(client, "我吃鱼油难受", session_id="s2_fishoil", user_id="u_f")
    reply = body["data"]["reply"]
    assert "已为您记录" not in reply, f"吃鱼油难受被误记过敏: {reply[:60]}"
    assert "已按您的禁忌排除" not in reply, f"吃鱼油难受被误记禁忌: {reply[:60]}"


def test_symptom_fish_allergy_positive_control(client):
    """对照：真「我吃鱼过敏」必须识别 fish_allergy。"""
    body = _chat(client, "我吃鱼过敏", session_id="s2_fishpos", user_id="u_f")
    reply = body["data"]["reply"]
    assert "已为您记录" in reply or "已按您的禁忌排除" in reply, (
        f"吃鱼过敏未识别: {reply[:60]}"
    )


def test_symptom_protein_powder_not_egg_allergy(client):
    """「我吃蛋白粉过敏」不得误记 egg_allergy（蛋白粉多为乳清/大豆，非鸡蛋）。"""
    body = _chat(client, "我吃蛋白粉过敏", session_id="s2_pp", user_id="u_e")
    reply = body["data"]["reply"]
    assert "已为您记录" not in reply, f"吃蛋白粉过敏被误记蛋过敏: {reply[:60]}"


def test_symptom_shrimp_chips_not_seafood_allergy(client):
    """「我吃虾片过敏」不得误记 seafood_allergy（虾片非明确海鲜制品）。"""
    body = _chat(client, "我吃虾片过敏", session_id="s2_chips", user_id="u_s")
    reply = body["data"]["reply"]
    assert "已为您记录" not in reply, f"吃虾片过敏被误记海鲜过敏: {reply[:60]}"


# ── P1-1：C 库词表漂移（红线①：素材C绝不混入/也绝不丢失）──────────
def test_platform_hints_unified(client):
    """升级专业版/专业版怎么买/怎么开通你们的服务 → platform + sources 含 C。"""
    for i, msg in enumerate(("升级专业版", "专业版怎么买", "怎么开通你们的服务")):
        body = _chat(client, msg, session_id=f"s2_plat{i}", user_id="u_p")
        assert body["data"]["intent"] == "platform", (
            f"「{msg}」未走平台意图: {body['data']['intent']}"
        )
        assert any(s["source"] == "C" for s in body["sources"]), (
            f"「{msg}」sources 未含 C 库: {[s['source'] for s in body['sources']]}"
        )


# ── P1-2：跨用户会话串号 ──────────────────────────────────────
def test_session_cross_user_no_inherit(client):
    """userB 借用 userA 的 session_id：不得继承 A 的目标/禁忌、不得读 A 档案。"""
    # userA 建会话（增肌）+ 声明蛋过敏
    r = client.post(
        "/api/session",
        json={"user_id": "userA", "session_id": "sA", "action": "new", "goal_tag": "增肌"},
    )
    assert r.status_code == 200
    _chat(client, "我鸡蛋过敏", session_id="sA", user_id="userA")
    # userB 借用 sA 点单：目标应为默认减脂（非 A 的增肌），且不继承蛋过敏
    body = _chat(client, "我在便利店准备点餐", session_id="sA", user_id="userB")
    d = body["data"]
    assert d["intent"] == "decision"
    # 关键：不按 A 的增肌画像出餐（归属校验生效 → 默认减脂）
    goal = d.get("decision", {}).get("goal", "")
    assert goal != "增肌", f"userB 借用了 A 的增肌画像: {goal}"
    items = d.get("decision", {}).get("items", [])
    joined = "".join(items)
    assert "白煮蛋" not in joined, f"userB 点单仍含白煮蛋（疑似继承 A 的增肌模板）: {joined}"
    # userB /api/state 不得读到 A 的档案（蛋过敏画像）
    r_state = client.get("/api/state", params={"user_id": "userB", "session_id": "sA"})
    assert r_state.status_code == 200
    profile = r_state.json()["data"]["profile"]
    assert "egg_allergy" not in profile["preferences"]["allergy_ids"], (
        f"userB 读到 A 的蛋过敏档案: {profile['preferences']['allergy_ids']}"
    )


def test_session_switch_cross_user_rejected(client):
    """userB 用 action=switch 改写 userA 的会话画像 → 拒绝（返回错误）。"""
    r = client.post(
        "/api/session",
        json={"user_id": "userA", "session_id": "sS", "action": "new", "goal_tag": "减脂"},
    )
    assert r.status_code == 200
    r2 = client.post(
        "/api/session",
        json={"user_id": "userB", "session_id": "sS", "action": "switch", "goal_tag": "调理"},
    )
    assert r2.status_code == 200
    assert "error" in r2.json()["data"], "userB 改写 userA 会话应被拒绝"
    # A 的画像未被篡改
    r3 = client.post(
        "/api/session",
        json={"user_id": "userA", "session_id": "sS", "action": "switch"},
    )
    assert r3.status_code == 200
    assert r3.json()["data"]["goal_tag"] == "减脂"


# ── P1-3：禁忌否定式识别补全 ─────────────────────────────────
def test_avoid_statement_expanded(client):
    """不能吃海产品/对花生不耐受/海鲜不能吃/鸡蛋碰都不能碰 → 识别并次轮排除。"""
    cases = [
        ("不能吃海产品", "seafood_allergy", "虾仁"),
        ("对花生不耐受", "nut_allergy", "花生"),
    ]
    for i, (msg, _, banned) in enumerate(cases):
        body = _chat(client, msg, session_id=f"s2_av{i}", user_id="u_av")
        assert "已为您记录" in body["data"]["reply"], f"「{msg}」未识别为禁忌"
        # 次轮推荐不得含该禁忌食材
        body2 = _chat(client, "推荐减脂方案", session_id=f"s2_av{i}", user_id="u_av")
        assert banned not in body2["data"]["reply"], (
            f"「{msg}」次轮推荐仍含 {banned}: {body2['data']['reply'][:80]}"
        )
    # 语序 2：食物在前（海鲜不能吃 / 鸡蛋碰都不能碰）
    for i, msg in enumerate(("海鲜不能吃", "鸡蛋碰都不能碰")):
        body = _chat(client, msg, session_id=f"s2_avb{i}", user_id="u_av2")
        reply = body["data"]["reply"]
        # egg 无追问话术（ALLERGY_FOLLOWUP 未配置）→ 用排除提示判定识别成功
        assert ("已为您记录" in reply) or ("已按您的禁忌排除" in reply), (
            f"「{msg}」未识别为禁忌: {reply[:80]}"
        )


# ── P1-4：负面表述误判 ──────────────────────────────────────
def test_negative_disease_no_guide(client):
    """「我血糖正常，帮我推荐减脂方案」不得贴控糖引导/写调理目标。"""
    body = _chat(client, "我血糖正常，帮我推荐减脂方案", session_id="s2_neg", user_id="u_n")
    reply = body["data"]["reply"]
    assert "结合您的控糖需求" not in reply, f"血糖正常被贴控糖引导: {reply[:60]}"
    # 不得把画像写成调理
    from app.database import SessionLocal
    from app import models

    db = SessionLocal()
    try:
        sess = db.get(models.Session, "s2_neg")
    finally:
        db.close()
    if sess is not None:
        assert sess.goal_tag != "调理", f"血糖正常被写成调理目标: {sess.goal_tag}"


def test_negative_disease_no_disclaimer(client):
    """「我没有糖尿病」不得被判为患病（控糖引导/免责不应出现）。

    注：检索块正文可能含「稳糖」标题（BM25 召回），但不带引导语/免责即不构成
    「画像错乱」——判据是引导语与免责（产品记忆与合规触发的标志）。
    """
    body = _chat(client, "我没有糖尿病", session_id="s2_neg2", user_id="u_n")
    reply = body["data"]["reply"]
    assert "结合您的控糖需求" not in reply, f"否认糖尿病仍被贴控糖引导: {reply[:60]}"
    # 否认患病 → 不触发免责（免责是「患病/用药」的安全冗余，非否认场景）
    assert body.get("disclaimer") is None, f"否认糖尿病仍带免责: {body.get('disclaimer')}"


# ── P1-5：用药针剂类词表 ────────────────────────────────────
def test_medication_injection_keywords(client):
    """打降糖针/减肥针 → 拒药 + 免责。"""
    for i, msg in enumerate(("打降糖针要注意什么", "减肥针安全吗")):
        body = _chat(client, msg, session_id=f"s2_inj{i}", user_id="u_m")
        assert "不提供用药建议" in body["data"]["reply"], f"「{msg}」未拒药"
        assert body.get("disclaimer"), f"「{msg}」未带免责"


# ── P1-7：今晚天气劫持 ──────────────────────────────────────
def test_tonight_weather_not_meal_hijack(client):
    """「今晚天气怎么样」不得被劫持成一餐追问；「今晚吃什么」仍走一餐。"""
    body = _chat(client, "今晚天气怎么样", session_id="s2_wx", user_id="u_t")
    assert body["data"]["intent"] != "meal_goal_ask", (
        f"天气问被劫持成一餐: {body['data']['intent']}"
    )
    body2 = _chat(client, "今晚吃什么", session_id="s2_meal", user_id="u_t")
    assert body2["data"]["intent"] in ("meal", "meal_goal_ask"), (
        f"今晚吃什么应走一餐: {body2['data']['intent']}"
    )


# ── 二审残余1：禁忌层否定剥离（P1-4 补漏）──────────────────────
def test_negated_disease_no_taboo_persisted(client):
    """「我没有高血压」不得误记 hypertension 禁忌（QA 复验残余1）。

    免责/引导层已剥离否定，但 taboo trigger_keywords 扫描此前未剥离 →
    「我没有高血压」仍落库 hypertension、后续排除酱油/腌制品。
    """
    from app.database import SessionLocal
    from app import models

    body = _chat(client, "我没有高血压", session_id="s2_ht", user_id="u_h")
    reply = body["data"]["reply"]
    assert "已为您记录" not in reply, f"没有高血压被误记禁忌: {reply[:60]}"
    assert "已按您的禁忌排除" not in reply, f"没有高血压被误记禁忌: {reply[:60]}"
    db = SessionLocal()
    try:
        sess = db.get(models.Session, "s2_ht")
        allergies = list(sess.allergies) if sess else []
    finally:
        db.close()
    assert "hypertension" not in allergies, f"「我没有高血压」落库禁忌: {allergies}"


def test_negated_blood_sugar_normal_no_taboo(client):
    """「我血糖正常」不得触发任何禁忌画像。"""
    body = _chat(client, "我血糖正常", session_id="s2_bsn", user_id="u_h")
    reply = body["data"]["reply"]
    assert "已按您的禁忌排除" not in reply, f"血糖正常被误记禁忌: {reply[:60]}"


def test_positive_hypertension_still_detected(client):
    """对照：真「我有高血压」必须识别 hypertension（触发词正常路径不被剥离误伤）。"""
    from app.database import SessionLocal
    from app import models

    body = _chat(client, "我有高血压", session_id="s2_hp", user_id="u_h")
    reply = body["data"]["reply"]
    assert ("已为您记录" in reply) or ("已按您的禁忌排除" in reply), (
        f"我有高血压未识别: {reply[:60]}"
    )
    db = SessionLocal()
    try:
        sess = db.get(models.Session, "s2_hp")
        allergies = list(sess.allergies) if sess else []
    finally:
        db.close()
    assert "hypertension" in allergies, f"「我有高血压」未落库: {allergies}"


# ── 二审残余2：多字食材否定式识别（P1-3 整词补漏）──────────────
def test_avoid_multi_char_food_detected(client):
    """不吃螃蟹/不吃三文鱼/不吃花生（多字食材）→ 识别对应禁忌。"""
    from app.database import SessionLocal
    from app import models

    cases = [
        ("不吃螃蟹", "seafood_allergy"),
        ("不吃三文鱼", "fish_allergy"),
        ("不吃花生", "nut_allergy"),
    ]
    for i, (msg, aid) in enumerate(cases):
        body = _chat(client, msg, session_id=f"s2_mc{i}", user_id="u_mc")
        reply = body["data"]["reply"]
        assert ("已为您记录" in reply) or ("已按您的禁忌排除" in reply), (
            f"「{msg}」未识别为禁忌: {reply[:60]}"
        )
        db = SessionLocal()
        try:
            sess = db.get(models.Session, f"s2_mc{i}")
            allergies = list(sess.allergies) if sess else []
        finally:
            db.close()
        assert aid in allergies, f"「{msg}」未落库 {aid}: {allergies}"
        # 次轮推荐不得含该禁忌食材
        body2 = _chat(client, "推荐减脂方案", session_id=f"s2_mc{i}", user_id="u_mc")
        banned = {"seafood_allergy": "虾仁", "fish_allergy": "鳕鱼", "nut_allergy": "花生"}[aid]
        assert banned not in body2["data"]["reply"], (
            f"「{msg}」次轮推荐仍含 {banned}: {body2['data']['reply'][:80]}"
        )


def test_avoid_fish_oil_negative_control(client):
    """对照：「我不吃鱼油」不得误记 fish_allergy（鱼油是补剂不是鱼）。"""
    body = _chat(client, "我不吃鱼油", session_id="s2_fo", user_id="u_mc")
    reply = body["data"]["reply"]
    assert "已为您记录" not in reply, f"不吃鱼油被误记鱼过敏: {reply[:60]}"
    assert "已按您的禁忌排除" not in reply, f"不吃鱼油被误记禁忌: {reply[:60]}"
