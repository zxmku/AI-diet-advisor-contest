"""V12 修复回归测试：锁死两项真实缺陷的修复，防回退。

缺陷一（C柱评分被吞）：评分触发词扩为强/弱两级，且评分可叠加于记录/查询/陪伴意图，
  多意图不再丢分、陪伴路径绝不进 LLM 编造分数。
缺陷二（拒药漏网 + 保健品软路径）：补全痛风降尿酸处方药（非布司他等）与中成药降脂
  （血脂康）硬闸门；保健品（辅酶Q10）在「替代处方药」语境走硬拒药，在选购/辨别语境
  追加安全提醒（非拒药），在食物营养语境零追加（不误伤正常营养问）。

复用 conftest 的 temp DB 隔离，绝不污染运行时 DB。
"""
from __future__ import annotations


def _chat(client, message: str, *, session_id: str, user_id: str, allergies: list[str] | None = None):
    return client.post(
        "/api/chat",
        json={
            "user_id": user_id,
            "session_id": session_id,
            "message": message,
            "allergies": allergies or [],
        },
    )


# ───────── 缺陷一：C柱评分派发 ─────────

def test_score_synonym_dagefen(client):
    """同义触发「打个分」也能命中评分（此前仅「打分/评分」命中，漏「打个分」）。"""
    _chat(client, "早餐吃了一个鸡蛋", session_id="v12_s1", user_id="u_v12_s1")
    r = _chat(client, "今天的饮食打个分，我70kg", session_id="v12_s1b", user_id="u_v12_s1")
    assert r.status_code == 200
    body = r.json()
    assert body["data"]["intent"] == "diet_score", body["data"]["reply"]
    assert "分（百分制）" in body["data"]["reply"], f"缺总分结构: {body['data']['reply']}"


def test_score_multi_record(client):
    """「记早餐+打分」多意图 → 先落库再算分，评分意图不被吞（intent=diet_record_score）。"""
    r = _chat(
        client,
        "帮我记早餐两个鸡蛋，然后给我打个分，我70kg",
        session_id="v12_mr", user_id="u_v12_mr",
    )
    assert r.status_code == 200
    body = r.json()
    assert body["data"]["intent"] == "diet_record_score", body["data"]["reply"]
    # 含本餐 + 纯规则评分块（never LLM 编分）
    assert "分（百分制）" in body["data"]["reply"], f"未出评分块: {body['data']['reply']}"
    assert body["meta"]["model"] == "local-rules"


def test_score_multi_companion(client):
    """「陪我聊天+打分」多意图 → 本地陪伴回复 + 纯规则评分块，绝不进 LLM 编「85分」。"""
    # 先为该用户记一顿，保证评分有数据
    _chat(client, "午餐吃了200克鸡胸肉和150克糙米", session_id="v12_mc_1", user_id="u_v12_mc")
    r = _chat(
        client,
        "陪我聊天，顺便给我今天的饮食打个分，我70kg",
        session_id="v12_mc_2", user_id="u_v12_mc",
    )
    assert r.status_code == 200
    body = r.json()
    assert body["data"]["intent"] == "companion_score", body["data"]["reply"]
    reply = body["data"]["reply"]
    assert "分（百分制）" in reply, f"未出评分块: {reply}"
    # 关键：陪伴路径不得让 LLM 自由编造分数（如「85分」）
    assert "85分" not in reply, f"陪伴路径 LLM 编造分数: {reply}"
    assert body["meta"]["model"] == "local-rules"


def test_score_weak_context(client):
    """弱触发「考核」+ 饮食语境也能命中评分（此前完全漏捕）。"""
    _chat(client, "晚餐吃了100克三文鱼", session_id="v12_w1", user_id="u_v12_syn")
    r = _chat(client, "考核一下我今天的饮食，我70kg", session_id="v12_w2", user_id="u_v12_syn")
    assert r.status_code == 200
    body = r.json()
    assert body["data"]["intent"] == "diet_score", body["data"]["reply"]
    assert "分（百分制）" in body["data"]["reply"]


# ───────── 缺陷二：拒药真药补全 + 保健品软路径 ─────────

def test_refuse_gout_drug_feibusi(client):
    """痛风降尿酸处方药「非布司他」此前漏拒药 → 现在必须硬拒（medication_refuse）。"""
    r = _chat(client, "我有痛风，非布司他能降尿酸吗？", session_id="v12_g1", user_id="u_v12_g1")
    assert r.status_code == 200
    body = r.json()
    assert body["data"]["intent"] == "medication_refuse", body["data"]["reply"]
    assert "不提供用药建议" in body["data"]["reply"]


def test_refuse_tcm_lipid_xuezhi(client):
    """中成药降脂「血脂康」此前漏拒药 → 现在必须硬拒（medication_refuse）。"""
    r = _chat(
        client,
        "降脂中成药血脂康胶囊能和深海鱼油一起吃吗？",
        session_id="v12_t1", user_id="u_v12_t1",
    )
    assert r.status_code == 200
    body = r.json()
    assert body["data"]["intent"] == "medication_refuse", body["data"]["reply"]
    assert "不提供用药建议" in body["data"]["reply"]


def test_supplement_replace_drug_refuse(client):
    """保健品「辅酶Q10 代替降压药」→ 医疗安全红线，硬拒药（非软路径）。"""
    r = _chat(
        client,
        "辅酶Q10能不能代替降压药来降血压？",
        session_id="v12_sr", user_id="u_v12_sr",
    )
    assert r.status_code == 200
    body = r.json()
    assert body["data"]["intent"] == "medication_refuse", body["data"]["reply"]
    reply = body["data"]["reply"]
    assert "不能擅自替代" in reply, f"缺替代药警告: {reply}"
    assert "遵医嘱" in reply, f"缺遵医嘱引导: {reply}"


def test_supplement_buy_advisory_not_refuse(client):
    """保健品选购语境「想买辅酶Q10推荐牌子」→ 不拒药，追加安全提醒（软路径）。"""
    r = _chat(
        client,
        "我想买辅酶Q10推荐个牌子",
        session_id="v12_sb", user_id="u_v12_sb",
    )
    assert r.status_code == 200
    body = r.json()
    assert body["data"]["intent"] != "medication_refuse", f"不应拒药: {body['data']['reply']}"
    assert "保健品（膳食补充剂）不是药品" in body["data"]["reply"], f"缺安全提醒: {body['data']['reply']}"


def test_supplement_food_context_no_advisory(client):
    """保健品食物语境「哪些食物含辅酶Q10」→ 正常营养作答，零追加（不误伤）。"""
    r = _chat(
        client,
        "哪些食物含有辅酶Q10？",
        session_id="v12_sf", user_id="u_v12_sf",
    )
    assert r.status_code == 200
    body = r.json()
    assert body["data"]["intent"] != "medication_refuse", f"不应拒药: {body['data']['reply']}"
    assert "保健品（膳食补充剂）不是药品" not in body["data"]["reply"], f"食物语境误加提醒: {body['data']['reply']}"
