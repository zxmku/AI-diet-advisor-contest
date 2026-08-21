"""MOD-02 检索路由：BM25 + jieba 轻量 RAG（蓝图 M3 智能层·检索域）。

设计（落实「不造轮子」与「C 库物理隔离」双红线）：
- A/B 库合并建 BM25 索引（营养基础 + 个性化方案同域检索）；
- C 库（平台/价格/企业 SaaS）独立索引，仅当 query 显式命中平台意图时才检索，
  绝不在普通营养问答里混入 C 内容；
- 纯本地 rank_bm25 + jieba，不调用任何向量库 / embedding 服务；
- 检索只回「原文块 + 来源」，回答由上层（本模块 compose 或后续 LLM 闸门）生成，
  保证零改写、零编造。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from jieba import lcut
from rank_bm25 import BM25Okapi

from app.api.schemas import SourceChunk
from app.config import KNOWLEDGE_DIR

logger = logging.getLogger("healthpick.retrieval")

# 平台意图触发词：命中才允许检索隔离的 C 库
PLATFORM_HINTS = (
    "会员", "价格", "多少钱", "费用", "企业", "SaaS", "API", "案例",
    "白皮书", "平台", "套餐", "订阅", "定价",
)


def _load_chunks(source_id: str, filename: str) -> list[dict]:
    path = KNOWLEDGE_DIR / filename
    if not path.is_file():
        logger.warning("知识库文件缺失: %s", path)
        return []
    try:
        lib = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("知识库解析失败 %s: %s", path, exc)
        return []
    return lib.get("chunks", [])


def _tokenize(text: str) -> list[str]:
    """jieba 中文分词，丢弃空白 token。"""
    return [w for w in lcut(text) if w.strip()]


class Retriever:
    """BM25 检索器：懒加载知识库并建索引，retrieve 返回 SourceChunk 列表。"""

    def __init__(self) -> None:
        self.a_chunks = _load_chunks("A", "core_nutrition_A.json")
        self.b_chunks = _load_chunks("B", "core_plans_B.json")
        self.c_chunks = _load_chunks("C", "platform_C.json")

        # A+B 同域合并索引（营养 + 方案）
        self.ab_chunks = self.a_chunks + self.b_chunks
        self.ab_corpus = [_tokenize(c["content"]) for c in self.ab_chunks]
        self.bm25 = BM25Okapi(self.ab_corpus) if self.ab_corpus else None

        # C 独立索引（隔离域）
        self.c_corpus = [_tokenize(c["content"]) for c in self.c_chunks]
        self.bm25_c = BM25Okapi(self.c_corpus) if self.c_corpus else None

        logger.info(
            "检索索引就绪: A=%d B=%d C=%d (隔离)",
            len(self.a_chunks), len(self.b_chunks), len(self.c_chunks),
        )

    @staticmethod
    def _is_platform_query(query: str) -> bool:
        return any(hint in query for hint in PLATFORM_HINTS)

    def retrieve(self, query: str, top_k: int = 3, current_query: str | None = None) -> list[SourceChunk]:
        """检索与 query 相关的知识块。

        A/B 始终检索；C 仅当 query 命中平台意图时追加（隔离）。
        返回按相关度降序的 SourceChunk 列表（含 score）。
        """
        results: list[SourceChunk] = []
        q_tokens = _tokenize(query)
        if self.bm25 and q_tokens:
            scores = self.bm25.get_scores(q_tokens)
            top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            for i in top_idx:
                if scores[i] <= 0:
                    continue
                c = self.ab_chunks[i]
                results.append(
                    SourceChunk(
                        source=c["source"],
                        chapter=c["chapter"],
                        section=c["section"],
                        content=c["content"],
                        score=round(float(scores[i]), 3),
                    )
                )

        # 隔离域：仅平台意图触达 C 库（闸门只看本轮 current_query，杜绝多轮拼接越红线）
        if self._is_platform_query(current_query or query) and self.bm25_c and q_tokens:
            scores = self.bm25_c.get_scores(q_tokens)
            top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            for i in top_idx:
                if scores[i] <= 0:
                    continue
                c = self.c_chunks[i]
                results.append(
                    SourceChunk(
                        source="C",
                        chapter=c["chapter"],
                        section=c["section"],
                        content=c["content"],
                        score=round(float(scores[i]), 3),
                    )
                )
        return results


_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    """进程内单例检索器（首次调用建索引）。"""
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever
