"""增强检索器 —— 融合多路召回 + 重排序"""
from typing import List

from langchain_core.documents import Document

from backend.config import settings
from backend.knowledge_base.vector_store import vector_store


class EnhancedRetriever:
    """带 MMR 和关键词增强的检索器"""

    def retrieve(self, query: str, k: int = None) -> List[Document]:
        """
        多策略检索:
        1. 相似度检索（基础）
        2. MMR 检索（多样性）
        3. 合并去重
        """
        k = k or settings.RETRIEVER_K

        # 策略1：语义候选 + 关键词/分类重排，优先保留与岗位和交接直接相关的资料。
        hybrid_results = vector_store.hybrid_search_with_relevance_scores(query, k=k)
        sim_docs = [doc for doc, _score in hybrid_results]

        # 策略2：MMR 检索（更关注多样性）
        mmr_docs = vector_store.max_marginal_relevance_search(
            query, k=k, fetch_k=k * 2
        )

        # 合并去重（按 page_content 去重）
        seen_contents = set()
        merged = []
        for doc in sim_docs + mmr_docs:
            if doc.page_content not in seen_contents:
                seen_contents.add(doc.page_content)
                merged.append(doc)

        return merged[:k]

    def retrieve_with_context(self, query: str, k: int = None) -> str:
        """检索并格式化为上下文文本"""
        docs = self.retrieve(query, k=k)
        if not docs:
            return ""

        contexts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source_file", "未知")
            contexts.append(f"[{i}] 来源: {source}\n{doc.page_content}")

        return "\n\n---\n\n".join(contexts)


# 全局单例
retriever = EnhancedRetriever()
