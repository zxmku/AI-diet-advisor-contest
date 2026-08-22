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

from jieba import lcut
from rank_bm25 import BM25Okapi

from app.api.schemas import SourceChunk
from app.config import KNOWLEDGE_DIR

logger = logging.getLogger("healthpick.retrieval")

# 平台意图触发词：命中才允许检索隔离的 C 库。
# 二审 P1-1（红线①）：与 main._PLATFORM_HINTS 合并为单一公共常量（此前两处漂移，
# 「怎么开通/升级专业版/收费」等问法 main 判平台、retrieval 闸门不检索 C 库 →
# C 素材形同虚设）。main.py 引用本常量，禁止另立词表。
PLATFORM_HINTS = (
    "会员", "价格", "多少钱", "费用", "企业", "SaaS", "API", "案例",
    "白皮书", "平台", "套餐", "订阅", "定价",
    "收费", "开通", "升级", "付费", "多少钱一个月",
    # 二审 P1-1 补：版本/购买词（C 库 2.1 有 免费版/标准版/专业版 内容，
    # 「专业版怎么买」「怎么升级专业版」须命中 C 库）
    "专业版", "免费版", "标准版", "购买", "续费",
)


def _load_synonyms() -> dict[str, list[str]]:
    """从 knowledge/synonyms.json 构建「词 → 扩展词列表」映射（文本级查询扩展）。

    「降糖」→ [控糖, 稳糖]、「鸡脯肉」→ [鸡胸肉]：提升 BM25 对标准表述知识块的命中。
    失败时返回空映射，同义词扩展停用（不影响基础检索）。
    """
    path = KNOWLEDGE_DIR / "synonyms.json"
    mapping: dict[str, list[str]] = {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        for group in data.get("synonyms", []):
            canon = group.get("canonical", "")
            aliases = group.get("aliases", [])
            if not canon:
                continue
            all_terms = list(dict.fromkeys([canon] + aliases))  # 去重保序
            for term in all_terms:
                for t in all_terms:
                    if t != term:
                        mapping.setdefault(term, []).append(t)
    except (OSError, json.JSONDecodeError):
        logger.warning("synonyms.json 加载失败，同义词扩展停用")
    return mapping


_SYNONYMS: dict[str, list[str]] = _load_synonyms()


def _expand_query(query: str) -> str:
    """文本级查询扩展：query 中出现同义词（≥2 字）时，追加其标准词家族。

    设计要点：在**原始文本**上做子串匹配（而非分词后 token 匹配）——jieba 会把
    「鸡脯肉」切成「鸡脯/肉」，token 级扩展匹配不上，文本级不受分词影响。
    """
    if not _SYNONYMS or not query:
        return query
    extra: list[str] = []
    for term, expansions in _SYNONYMS.items():
        if len(term) >= 2 and term in query:
            extra.extend(expansions)
    if not extra:
        return query
    return query + " " + " ".join(dict.fromkeys(extra))


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
        # 同义词查询扩展只作用于 A/B 知识域；C 库闸门仍用原始 current_query/query，
        # 避免「会员」等平台词被同义词追加干扰闸门判断。
        q_tokens = _tokenize(_expand_query(query))
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
