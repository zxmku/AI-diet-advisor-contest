"""P2/P3 产品化迭代测试：个性化记忆贯穿（过敏持久化 + 目标动态更新）+ 过敏反问追问
+ 确定性一餐生成（meal 结构化 + 热量可核对 + 禁忌排除 + 无目标追问）。

覆盖产品核心需求「聊起来 + 记得我」：
- 首次声明过敏 → 回复以追问开头（含「具体」/「哪一类」）+ 末尾确认排除提示；
- 同一会话再问（不再声明）→ 不再追问（只问一次）+ 记忆已持久化仍排除；
- 消息隐含人群目标（血糖偏高）→ session.goal_tag 动态更新为调理类。

P3「从问题到一餐」：
- 「今晚减脂餐」→ data.meal 结构化存在；total_kcal 与速查表值×克数一致（工具计算）；
- 海鲜过敏会话要晚餐 → 一餐 items/swaps 均无海鲜 + 排除提示仍在；
- 无目标会话问「吃什么好」→ 先追问目标（不直接出餐）。

复用 conftest 的 temp DB 隔离；红线 R3 正文剔除不回退一并校验。
"""
from __future__ import annotations

from app import config
from app import llm
from app import models
from app import nutrition_lookup
from app.cost_gate import cost_gate
from app.database import SessionLocal

# 海鲜过敏（seafood_allergy）在 knowledge/taboo_map.json 中声明的排除食材。
SEAFOOD_EXCLUDED = {"虾仁", "三文鱼", "鳕鱼", "鲈鱼", "龙利鱼", "蛤蜊", "鱿鱼"}


def _chat(client, message: str, *, session_id: str, user_id: str, allergies: list[str] | None = None):
    """便捷封装：POST /api/chat。"""
    return client.post(
        "/api/chat",
        json={
            "user_id": user_id,
            "session_id": session_id,
            "message": message,
            "allergies": allergies or [],
        },
    )


def _session_row(session_id: str) -> models.Session | None:
    """直接查 temp DB 中的会话行（断言持久化结果）。"""
    db = SessionLocal()
    try:
        return db.get(models.Session, session_id)
    finally:
        db.close()


def _pre_exclusion_part(reply: str) -> str:
    """取回复中「⚠️ 已按您的禁忌排除」标记之前的部分（即回复正文）。"""
    marker = "⚠️ 已按您的禁忌排除"
    return reply.split(marker, 1)[0]


def test_p2_allergy_followup_once_and_persisted(client):
    """首次声明过敏：追问开头 + 排除提示末尾；次轮不再追问但记忆仍在（只问一次+持久化）。"""
    # 第一轮：声明「我对海鲜过敏」→ 回复以追问开头
    r1 = _chat(client, "我对海鲜过敏", session_id="p2a", user_id="u_p2")
    assert r1.status_code == 200
    reply1 = r1.json()["data"]["reply"]
    assert reply1.startswith("收到，已为您记录海鲜过敏"), f"未以追问开头: {reply1[:50]}"
    assert ("具体" in reply1) or ("哪一类" in reply1)
    # 末尾仍保留确认排除提示
    assert "已按您的禁忌排除以下食材" in reply1

    # 记忆已持久化：sessions.allergies 已写入（后续轮次仍生效）
    sess = _session_row("p2a")
    assert sess is not None
    assert "seafood_allergy" in (sess.allergies or [])

    # 第二轮：不再声明过敏 → 无追问（只问一次）；记忆已持久化（见上 71 行），
    # 本回复正文无海鲜、无剔除动作 → 二审 P0-3 修复③下排除提示不强制出现。
    r2 = _chat(client, "推荐减脂方案", session_id="p2a", user_id="u_p2")
    assert r2.status_code == 200
    reply2 = r2.json()["data"]["reply"]
    assert not reply2.startswith("收到，已为您记录"), "次轮不应再次追问"
    # 红线 R3 不回退：正文（排除提示之前）不得含任一海鲜食材名
    pre = _pre_exclusion_part(reply2)
    assert not any(f in pre for f in SEAFOOD_EXCLUDED), (
        f"回复正文仍含禁忌海鲜: {[f for f in SEAFOOD_EXCLUDED if f in pre]}"
    )


def test_p2_goal_dynamic_update(client):
    """消息隐含人群目标（血糖偏高）→ session.goal_tag 动态更新为调理类。"""
    r = _chat(client, "我血糖偏高，三餐该怎么吃", session_id="p2b", user_id="u_p2")
    assert r.status_code == 200
    sess = _session_row("p2b")
    assert sess is not None
    assert sess.goal_tag == "调理"


# 坚果过敏追问话术原文（compliance.ALLERGY_FOLLOWUP["nut_allergy"] 逐字）：
# 追问里的精确食材名（花生/腰果/杏仁）必须原样保留，绝不能被 R3 剔除循环替换成乱码。
_FOLLOWUP_NUT = (
    "收到，已为您记录坚果过敏。为了更精准地帮您避开，"
    "方便告诉我具体是哪一类坚果吗？比如花生、腰果、杏仁，还是核桃？"
)


def test_p2_allergy_followup_keeps_original_text(client):
    """DEFECT-A 回归：坚果过敏追问原文完整（花生/腰果/杏仁 不被 R3 剔除循环乱码化），
    追问不经剔除循环、排除提示仍在尾部。"""
    r = _chat(client, "我对坚果过敏", session_id="p2d", user_id="u_p2")
    assert r.status_code == 200
    reply = r.json()["data"]["reply"]
    # 追问原文完整且位于回复开头（若被剔除循环乱码化，此精确串必然断裂）
    assert reply.startswith(_FOLLOWUP_NUT), f"追问原文被破坏: {reply[:80]}"
    # 追问段不得出现「（已按禁忌剔除）」乱码（追问文本完全不经剔除循环）
    assert "（已按禁忌剔除）" not in _FOLLOWUP_NUT
    # 排除提示仍在尾部（追问之后）
    assert "已按您的禁忌排除以下食材" in reply
    assert reply.index("已按您的禁忌排除以下食材") > len(_FOLLOWUP_NUT)
    # 红线 R3 正文剔除仍生效：排除提示之前的正文部分（追问之后）中，正文若含禁忌
    # 食材应被替换为「（已按禁忌剔除）」（知识库脂肪/餐盘块含「坚果」「牛油果」）。
    pre = _pre_exclusion_part(reply)
    body_part = pre[len(_FOLLOWUP_NUT):]
    assert "（已按禁忌剔除）" in body_part, f"R3 正文剔除失效: {body_part[:120]}"


def test_p2_request_word_does_not_overwrite_goal(client):
    """DEFECT-B 回归：已有 goal=调理 的会话，用户问「推荐减脂方案」（请求语义）
    → goal_tag 不被覆盖为减脂，仍保持调理。"""
    r_sess = client.post(
        "/api/session",
        json={
            "user_id": "u_p2",
            "session_id": "p2c",
            "action": "new",
            "goal_tag": "调理",
            "allergies": [],
        },
    )
    assert r_sess.status_code == 200
    r = _chat(client, "推荐减脂方案", session_id="p2c", user_id="u_p2")
    assert r.status_code == 200
    sess = _session_row("p2c")
    assert sess is not None
    assert sess.goal_tag == "调理"


def test_p2_geiwo_bangwo_request_does_not_overwrite_goal(client):
    """QA Round2 遗留风险#1 回归：裸「我」不再构成个人声明——
    已有 goal=调理 的会话发「给我推荐减脂方案」「帮我推荐减脂方案」
    （纯请求语义，含「我」子串但不含「我的/本人」或健康状态短语）
    → goal_tag 仍为调理，不被覆盖为减脂。"""
    r_sess = client.post(
        "/api/session",
        json={
            "user_id": "u_p2",
            "session_id": "p2e",
            "action": "new",
            "goal_tag": "调理",
            "allergies": [],
        },
    )
    assert r_sess.status_code == 200
    for msg in ("给我推荐减脂方案", "帮我推荐减脂方案"):
        r = _chat(client, msg, session_id="p2e", user_id="u_p2")
        assert r.status_code == 200
        sess = _session_row("p2e")
        assert sess is not None
        assert sess.goal_tag == "调理", f"{msg} 不应覆盖 goal_tag: {sess.goal_tag}"


# ═══ P3 确定性一餐生成 ═══


def _nutrition_kcal(food: str, grams: int) -> float:
    """按速查表核对热量：表值 × 份量/100（与后端 _build_meal 同一算法，工具计算）。"""
    row = nutrition_lookup.lookup(food)
    assert row is not None, f"速查表未收录 {food}"
    assert row.get("kcal") is not None, f"{food} 速查表无 kcal 值"
    return round(float(row["kcal"]) * grams / 100, 1)


def test_p3_meal_structured_and_kcal_traceable(client):
    """「今晚减脂餐」→ data.meal 结构化存在；total_kcal 与速查表值×克数一致；
    items 含主蛋白/主食碳水/蔬菜/好脂肪；sources 无 C 库；回复为自然引言。"""
    r = _chat(client, "今晚减脂餐", session_id="p3a", user_id="u_p3")
    assert r.status_code == 200
    body = r.json()
    assert body["data"]["intent"] == "meal"
    meal = body["data"]["meal"]
    assert meal is not None
    # 结构契约：name/items/total_kcal/macros/method/swaps/sources 齐全
    assert "name" in meal and "items" in meal and "total_kcal" in meal
    assert "macros" in meal and "method" in meal and "swaps" in meal and "sources" in meal
    cats = {it["category"] for it in meal["items"]}
    assert {"主蛋白", "主食碳水", "蔬菜", "好脂肪"} <= cats, f"缺槽位: {cats}"
    # 热量可核对：抽查鸡胸肉（150g）/糙米（100g）/西兰花（200g）
    by_food = {it["food"]: it for it in meal["items"]}
    for food, grams in (("鸡胸肉（去皮）", 150), ("糙米", 100), ("西兰花", 200)):
        item = by_food.get(food)
        assert item is not None, f"一餐缺 {food}"
        assert item["kcal"] == _nutrition_kcal(food, grams), (
            f"{food} 热量与速查表不一致: {item['kcal']} != {_nutrition_kcal(food, grams)}"
        )
    # total_kcal = 全部可算 item kcal 之和（橄榄油表外不计入）
    computed = sum(it["kcal"] for it in meal["items"] if it["kcal"] is not None)
    assert abs(meal["total_kcal"] - computed) < 0.01, (
        f"total_kcal 不可核对: {meal['total_kcal']} != {computed}"
    )
    # 红线：sources 绝不混 C 库
    assert meal["sources"], "一餐缺少来源标注"
    assert all(s["source"] != "C" for s in meal["sources"])
    # 回复为自然引言（如「为你搭配的…」）
    assert "为你搭配的" in body["data"]["reply"]
    assert "千卡" in body["data"]["reply"]


def test_p3_meal_allergy_exclusion_and_followup(client):
    """海鲜过敏声明 + 晚餐 → 首轮追问；次轮出餐 items/swaps 均无海鲜 + 排除提示仍在。"""
    # 会话先定目标（无目标时一餐会先追问目标，无法直接出餐）
    r_sess = client.post(
        "/api/session",
        json={
            "user_id": "u_p3",
            "session_id": "p3b",
            "action": "new",
            "goal_tag": "减脂",
            "allergies": [],
        },
    )
    assert r_sess.status_code == 200
    # 首轮声明过敏 → 追问（复用 P2 追问风格）
    r1 = _chat(client, "我对海鲜过敏", session_id="p3b", user_id="u_p3")
    assert r1.status_code == 200
    assert r1.json()["data"]["reply"].startswith("收到，已为您记录海鲜过敏")
    # 次轮要晚餐 → 出餐且餐内无海鲜（items 与同族替换全部剔除）
    r2 = _chat(client, "给我晚餐", session_id="p3b", user_id="u_p3")
    assert r2.status_code == 200
    body2 = r2.json()
    assert body2["data"]["intent"] == "meal"
    meal = body2["data"]["meal"]
    assert meal is not None
    item_foods = {it["food"] for it in meal["items"]}
    assert not (SEAFOOD_EXCLUDED & item_foods), f"餐内仍含海鲜: {item_foods & SEAFOOD_EXCLUDED}"
    swap_foods = set(meal["swaps"]["protein"]) | set(meal["swaps"]["carbs"])
    assert not (SEAFOOD_EXCLUDED & swap_foods), f"替换项仍含海鲜: {swap_foods & SEAFOOD_EXCLUDED}"
    # 排除提示保留在回复尾部（追问/引言之后）
    assert "已按您的禁忌排除以下食材" in body2["data"]["reply"]


def test_p3_meal_goal_ask_when_no_goal(client):
    """无目标会话问「吃什么好」→ 先追问目标（不直接出餐，data 无 meal）。"""
    r = _chat(client, "吃什么好", session_id="p3c", user_id="u_p3")
    assert r.status_code == 200
    body = r.json()
    assert body["data"]["intent"] == "meal_goal_ask"
    reply = body["data"]["reply"]
    assert "减脂" in reply and "增肌" in reply and "控糖" in reply
    assert "meal" not in body["data"]


def test_p3_r1_numeric_lookup_not_hijacked_by_meal(client):
    """R1 回归：一餐意图不劫持数值问答——
    「晚餐鸡胸肉多少千卡」须走数值路径返回速查表精确值 165，而不是被劫持成一餐/追问目标。"""
    r = _chat(client, "晚餐鸡胸肉多少千卡", session_id="p3r1", user_id="u_p3")
    assert r.status_code == 200
    body = r.json()
    assert body["data"]["intent"] == "nutrition_lookup"
    assert "165" in body["data"]["reply"], f"未返回速查表精确值: {body['data']['reply'][:80]}"
    assert "meal" not in body["data"]


def test_p3_r2_medication_refuse_priority_over_meal(client):
    """R2 回归：用药硬拦截优先于一餐——
    「吃布洛芬后晚餐吃什么好」须先拒答用药（含免责），不进一餐。"""
    r = _chat(client, "吃布洛芬后晚餐吃什么好", session_id="p3r2", user_id="u_p3")
    assert r.status_code == 200
    body = r.json()
    assert "不提供用药建议" in body["data"]["reply"], f"未拒答用药: {body['data']['reply'][:80]}"
    assert body.get("disclaimer") and "不构成医疗建议" in body["disclaimer"]
    assert "meal" not in body["data"]


# ═══ P5a 饮食管理（记住饮食）═══


def _diet_logs(user_id: str) -> list[models.DietLog]:
    """直接查 temp DB 中的饮食台账（断言持久化结果，最新在前）。"""
    db = SessionLocal()
    try:
        return (
            db.query(models.DietLog)
            .filter(models.DietLog.user_id == user_id)
            .order_by(models.DietLog.created_at.desc(), models.DietLog.id.desc())
            .all()
        )
    finally:
        db.close()


def test_p5_diet_record_then_query(client):
    """P5a：记录「早餐吃了两个鸡蛋和一杯牛奶」→ DietLog 落库 + 回复确认（含速查表热量估算）；
    再问「我最近吃了什么」→ 汇总含该条（记住饮食闭环）。"""
    r1 = _chat(
        client,
        "帮我记一下：早餐吃了两个鸡蛋和一杯牛奶",
        session_id="p5a",
        user_id="u_p5a",
    )
    assert r1.status_code == 200
    reply1 = r1.json()["data"]["reply"]
    assert "记下" in reply1 and "早餐" in reply1, f"记录确认缺失: {reply1[:80]}"
    # 鸡蛋/牛奶均在速查表（155/65 千卡每 100g）→ 回复给出「约 X 千卡」合计
    assert "千卡" in reply1 and "约" in reply1, f"热量估算缺失: {reply1[:120]}"

    # 台账已落库：meal_tag=早餐，content 为消息原文
    logs = _diet_logs("u_p5a")
    assert len(logs) == 1, f"DietLog 未落库: {len(logs)}"
    assert logs[0].meal_tag == "早餐"
    assert "鸡蛋" in logs[0].content and "牛奶" in logs[0].content

    # 查询闭环：最近吃了什么 → 汇总含该条
    r2 = _chat(client, "我最近吃了什么", session_id="p5a", user_id="u_p5a")
    assert r2.status_code == 200
    reply2 = r2.json()["data"]["reply"]
    assert "你最近记了" in reply2, f"查询汇总缺失: {reply2[:80]}"
    assert "早餐" in reply2 and "鸡蛋" in reply2 and "牛奶" in reply2


def test_p5_companion_empathy(client):
    """P5b：一个人吃饭好孤独 → 回复非空、先共情/给轻建议、不 500。"""
    r = _chat(client, "一个人吃饭好孤独", session_id="p5b", user_id="u_p5b")
    assert r.status_code == 200
    reply = r.json()["data"]["reply"]
    assert reply, "陪伴回复为空"
    assert (
        ("我陪你" in reply)
        or ("好好吃" in reply)
        or ("简单快手" in reply)
        or ("一个人" in reply)
    ), f"回复缺共情或轻建议: {reply[:100]}"


def test_p5_companion_model_labeling(client, monkeypatch):
    """进阶2 模型来源标注诚实性：陪伴分支 LLM 成功 → meta.model=DEEPSEEK_MODEL 且 degraded=False；
    LLM 失败降级 → meta.model=local-rules 且 degraded=True（不误标）。"""
    # 隔离成本闸门：测试不得读写 backend/data 账本与缓存（QA 实测数据），保持确定性。
    monkeypatch.setattr(cost_gate, "check_budget", lambda: True)
    monkeypatch.setattr(cost_gate, "check_rate", lambda session_id=None: True)
    monkeypatch.setattr(cost_gate, "record", lambda *args, **kwargs: None)
    monkeypatch.setattr(cost_gate, "cache_get", lambda key: None)
    monkeypatch.setattr(cost_gate, "cache_set", lambda key, reply: None)

    monkeypatch.setattr(llm, "is_enabled", lambda: True)
    fake_reply = "一个人吃饭也要好好吃，我陪你🍚 先吃口热乎的，别饿着。"
    fake_resp = {
        "choices": [{"message": {"content": fake_reply}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 10},
    }
    monkeypatch.setattr(llm, "_post_json", lambda *args, **kwargs: fake_resp)

    # LLM 成功：回复为模型文本，meta 标注 DEEPSEEK_MODEL / degraded=False
    r1 = _chat(client, "一个人吃饭好孤独", session_id="p5m1", user_id="u_p5m")
    assert r1.status_code == 200
    body1 = r1.json()
    assert body1["data"]["intent"] == "companion"
    assert body1["data"]["reply"] == fake_reply
    assert body1["meta"]["model"] == config.DEEPSEEK_MODEL
    assert body1["meta"]["degraded"] is False

    # LLM 失败（_post_json 返回 None）：降级本地模板 → local-rules / degraded=True
    monkeypatch.setattr(llm, "_post_json", lambda *args, **kwargs: None)
    r2 = _chat(client, "加班好累没胃口", session_id="p5m2", user_id="u_p5m")
    assert r2.status_code == 200
    body2 = r2.json()
    assert body2["data"]["intent"] == "companion"
    assert body2["meta"]["model"] == "local-rules"
    assert body2["meta"]["degraded"] is True
    assert "我陪你" in body2["data"]["reply"]


def test_p5_companion_mood_caught(client):
    """支柱B 加固：口语情绪（心情好差/被骂）须进陪伴分支，降级态也接住情绪，
    不得落 _chitchat 兜底的『帮不上忙』。"""
    r = _chat(client, "今天心情好差，工作被骂了，好想喝全糖奶茶吃炸鸡暴饮暴食",
              session_id="p5mood", user_id="u_p5mood")
    assert r.status_code == 200
    body = r.json()
    assert body["data"]["intent"] == "companion", f"情绪未进陪伴: {body['data']['reply'][:80]}"
    assert body["data"]["reply"], "陪伴回复为空"
    assert "帮不上忙" not in body["data"]["reply"], "情绪场景被打成『帮不上忙』"


def test_p5_companion_explicit_request(client):
    """支柱B 加固：明确的陪伴请求（陪我聊聊天）须进 companion。"""
    r = _chat(client, "陪我聊聊天呗", session_id="p5chat", user_id="u_p5chat")
    assert r.status_code == 200
    body = r.json()
    assert body["data"]["intent"] == "companion", f"陪伴请求未进 companion: {body['data']['reply'][:80]}"
    assert body["data"]["reply"]


def test_p5_companion_substring_safe(client):
    """支柱B 加固（C3 子串安全）：裸『聊天』不得误触发陪伴——
    『我的聊天记录在哪里』不应进 companion（只含裸『聊天』，未加进触发词）。"""
    r = _chat(client, "我的聊天记录在哪里", session_id="p5sub", user_id="u_p5sub")
    assert r.status_code == 200
    body = r.json()
    assert body["data"]["intent"] != "companion", f"裸『聊天』误触发陪伴: {body['data']['reply'][:80]}"
