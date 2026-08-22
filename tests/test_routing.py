"""领域路由门控测试：膳食顾问需求 vs 闲聊/非膳食。

团长核心诉求：加入闲聊/陪聊后，回复逻辑必须区分两种模式——
①膳食顾问（命中红线需求）→ RAG 检索 ABC 整理回复，语气像陪伴伙伴；
②闲聊/非膳食 → 不走知识库，自然回应（不能硬套知识库）。

否则「大龙虾吃什么」「世界首富是谁」会被 BM25 检索到低相关块后硬套成膳食内容。
"""
from __future__ import annotations


def _chat(client, message, *, session_id):
    r = client.post(
        "/api/chat",
        json={"user_id": "u", "session_id": session_id, "message": message},
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
