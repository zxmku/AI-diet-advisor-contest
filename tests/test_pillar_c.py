"""支柱C 考核评分（饮食评估打分）测试。

覆盖 P5c 能力：基于该 user_id 当日 DietLog 台账 + 营养速查表真实值，
给当天饮食打结构化分数（热量 40% + 蛋白 35% + 餐次 25%）。

护栏（方案第八节 + 增强）：
- basic：记早/午/晚三餐后评分 → 进 diet_score、含总分与三维、含「约」标注、local-rules；
- no_log：无当日记录请求评分 → 引导先记账，不报错，仍进 diet_score；
- tuning_disclaimer：goal_tag=调理 会话评分 → 带标准免责（disclaimer 字段，红线②）；
- isolation：评分请求零牵连会员/企业/用药（不与 C 库/拒药相互污染）；
- no_weight_fallback：未提供体重 → 蛋白维度 N/A 不崩，总分仍落在 0-100。

复用 conftest 的 temp DB 隔离（红线自动化测试纪律：绝不污染运行时 DB）。
"""
from __future__ import annotations

import re


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


def test_diet_score_basic(client):
    """记早/午/晚三餐后请求评分 → 进 diet_score、含总分与三维、含「约」标注、local-rules。

    跨会话记账（不同 session_id 同一 user_id）也应被按 user_id 汇总评分。
    """
    for msg, sid in (
        ("早餐吃了一个鸡蛋和一杯无糖豆浆", "c_basic_1"),
        ("午餐吃了200克鸡胸肉、150克糙米和100克西兰花", "c_basic_2"),
        ("晚餐吃了200克西兰花和100克三文鱼", "c_basic_3"),
    ):
        r = _chat(client, msg, session_id=sid, user_id="u_c_basic")
        assert r.status_code == 200, f"记账失败: {msg}"

    r = _chat(client, "给我打个分，我70kg", session_id="c_basic_score", user_id="u_c_basic")
    assert r.status_code == 200
    body = r.json()
    assert body["data"]["intent"] == "diet_score", body["data"]["reply"]
    reply = body["data"]["reply"]
    assert "分（百分制）" in reply, f"缺总分结构: {reply}"
    assert "热量" in reply and "蛋白质" in reply and "餐次" in reply, f"缺三维: {reply}"
    assert "约" in reply, f"缺约计标注: {reply}"
    # 纯规则路径：零 AI 味、红线⑤安全
    assert body["meta"]["model"] == "local-rules"
    assert body["meta"]["degraded"] is True


def test_diet_score_no_log(client):
    """无当日记录请求评分 → 引导先记账，不报错，仍进 diet_score 分支。"""
    r = _chat(client, "给我打个分，我70kg", session_id="c_nolog", user_id="u_c_nolog")
    assert r.status_code == 200
    body = r.json()
    assert body["data"]["intent"] == "diet_score", body["data"]["reply"]
    reply = body["data"]["reply"]
    assert ("还没记饮食" in reply) or ("告诉我你吃了什么" in reply), f"未引导先记账: {reply}"


def test_diet_score_tuning_disclaimer(client):
    """goal_tag=调理 会话请求评分 → 带标准免责（disclaimer 字段，红线②）。

    复用会话级 disease 判定（goal==调理 即 disease=True），disclaimer 由 chat 外层
    统一注入（与疾病/慢病路径一致），评分分支无需手写免责文本。
    """
    r_sess = client.post(
        "/api/session",
        json={
            "user_id": "u_c_disc",
            "session_id": "c_disc",
            "action": "new",
            "goal_tag": "调理",
            "allergies": [],
        },
    )
    assert r_sess.status_code == 200
    # 记一顿再评分（也可不记，仅验证免责注入）
    _chat(client, "早餐吃了一个鸡蛋", session_id="c_disc", user_id="u_c_disc")
    r = _chat(client, "给我打个分，我70kg", session_id="c_disc", user_id="u_c_disc")
    assert r.status_code == 200
    body = r.json()
    assert body["data"]["intent"] == "diet_score", body["data"]["reply"]
    disc = body.get("disclaimer")
    assert disc and "不构成医疗建议" in disc, f"调理会话评分未带标准免责: {disc}"


def test_diet_score_isolation(client):
    """评分请求不含会员/企业/用药词，确认零牵连（不与 C 库/拒药相互污染）。

    用「今天吃得怎么样」触发评分且未记账 → intent 必为 diet_score，不得被平台/
    拒药逻辑劫持成 membership_refuse / medication_refuse，reply 不得含会员/企业/用药。
    """
    r = _chat(client, "今天吃得怎么样，我70kg", session_id="c_iso", user_id="u_c_iso")
    assert r.status_code == 200
    body = r.json()
    assert body["data"]["intent"] == "diet_score", f"评分未独立命中: {body['data']['reply']}"
    reply = body["data"]["reply"]
    assert "会员" not in reply, f"评分牵连平台: {reply}"
    assert "企业" not in reply, f"评分牵连企业: {reply}"
    assert "不提供用药建议" not in reply, f"评分被拒药劫持: {reply}"
    assert body["meta"]["model"] == "local-rules"


def test_diet_score_no_weight_fallback(client):
    """评分消息未提供体重 → 蛋白维度 N/A（不崩、不编造），总分仍落在 0-100。"""
    _chat(client, "早餐吃了一个鸡蛋", session_id="c_nw_1", user_id="u_c_nw")
    r = _chat(client, "给我打个分", session_id="c_nw_score", user_id="u_c_nw")
    assert r.status_code == 200
    body = r.json()
    assert body["data"]["intent"] == "diet_score", body["data"]["reply"]
    reply = body["data"]["reply"]
    assert "暂未提供体重" in reply, f"未标注缺体重: {reply}"
    m = re.search(r"打 (\d+) 分", reply)
    assert m, f"未解析到总分: {reply}"
    score = int(m.group(1))
    assert 0 <= score <= 100, f"总分越界: {score}"
