"""
知识关系图谱模块 —— 建立知识块之间的关联网络

实现：
1. 基于内容相似度的知识关联发现
2. 文档-知识块-标签的网状索引
3. 知识溯源：从决策结果追溯到依据文档
4. 可视化知识关系数据输出
"""
import re
import json
import hashlib
import logging
import math
from collections import defaultdict
from typing import Any, List, Dict, Optional, Set, Tuple
from datetime import datetime
from pathlib import Path

from langchain_core.documents import Document

from backend.knowledge_base.vector_store import vector_store
from backend.knowledge_base.project_context import knowledge_category_label

logger = logging.getLogger(__name__)

AUTO_EDGE_LIMIT_PER_NODE = 4
AUTO_EDGE_MIN_SCORE = 0.42
AUTO_RELATIONS = {"related", "strong_related", "semantic_related", "topic_related", "document_sequence"}


class KnowledgeGraph:
    """知识关系图谱"""

    def __init__(self):
        # 知识节点：{node_id: {content, source, category, tags, refs}}
        self._nodes: Dict[str, dict] = {}
        # 边：{(from_id, to_id): 可解释关系元数据}
        self._edges: Dict[Tuple[str, str], dict] = {}
        # 标签索引：{tag: [node_ids]}
        self._tag_index: Dict[str, List[str]] = defaultdict(list)
        # 关键词索引
        self._keyword_index: Dict[str, Set[str]] = defaultdict(set)

    @staticmethod
    def _is_current_metadata(metadata: dict) -> bool:
        """Fail closed when an index snapshot contains a non-current version."""
        if metadata.get("is_current_version") is False:
            return False
        allowed = {"active", "published", "effective", "review_due"}
        for key in ("status", "asset_status", "version_status"):
            value = str(metadata.get(key) or "").strip().lower()
            if value and value not in allowed:
                return False
        return True

    def _generated_node_id(self, chunk_id: str, metadata: dict, fallback_index: int) -> str:
        """Prefer an asset-stable node identity while retaining chunk evidence."""
        chunk_index = metadata.get("chunk_index")
        if chunk_index is None:
            chunk_index = metadata.get("page")
        if chunk_index is None:
            chunk_index = fallback_index
        asset_id = str(metadata.get("asset_id") or "").strip()
        version_id = str(metadata.get("version_id") or "").strip()
        if asset_id:
            return self._bounded_generated_id("asset", asset_id, chunk_index)
        if version_id:
            return self._bounded_generated_id("version", version_id, chunk_index)
        return str(chunk_id)

    def _bounded_generated_id(self, prefix: str, identity: str, chunk_index: Any) -> str:
        """Keep node IDs readable and collision resistant within the UI limit."""
        identity_token = self._string_id(identity)
        if len(str(identity).strip()) > 48:
            digest = hashlib.sha256(str(identity).encode("utf-8")).hexdigest()[:12]
            identity_token = f"{identity_token[:34]}__{digest}"
        chunk_token = self._string_id(chunk_index)
        if len(chunk_token) > 16:
            chunk_token = hashlib.sha256(chunk_token.encode("utf-8")).hexdigest()[:12]
        return self._string_id(f"{prefix}__{identity_token}__chunk__{chunk_token}")

    def build_graph(self):
        """从当前向量库构建可解释的混合关联图谱。"""
        try:
            all_data = vector_store.get(include_vectors=True)
        except Exception as e:
            logger.exception("构建知识图谱失败（向量索引不可用）")
            raise RuntimeError("知识图谱构建失败：向量索引不可用") from e

        metadatas = all_data.get("metadatas", [])
        documents = all_data.get("documents", [])
        ids = all_data.get("ids", [])
        vectors = all_data.get("vectors", [])
        ids, documents, metadatas, vectors = self._expand_legacy_chunks(
            ids, documents, metadatas, vectors,
        )
        saved_graph = self._load_saved_graph()

        # 清空重建
        self._nodes.clear()
        self._edges.clear()
        self._tag_index.clear()
        self._keyword_index.clear()

        imported_graph = self._load_imported_graph()
        if not all_data or not ids:
            self.save_graph_data(self._merge_for_rebuild({"nodes": [], "edges": []}, imported_graph, saved_graph))
            return

        recognized_graph_sources = set()
        for index, content in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) else {}
            source = str(metadata.get("source_file") or "")
            if source.lower().endswith(".json"):
                candidate = self.graph_from_json_content(content, source, metadata)
                if candidate:
                    imported_graph = self._merge_graphs(imported_graph, candidate)
                    recognized_graph_sources.add(source)
        if imported_graph.get("nodes"):
            self._save_imported_graph(imported_graph)

        # 1. 构建节点
        for i, doc_id in enumerate(ids):
            content = documents[i] if i < len(documents) else ""
            meta = metadatas[i] if i < len(metadatas) else {}
            if not self._is_current_metadata(meta):
                continue
            if meta.get("source_file") in recognized_graph_sources:
                continue

            # 提取关键词作为标签
            tags = self._extract_keywords(content, top_k=5)
            source = meta.get("source_file", "unknown")
            page = meta.get("page")
            location = ""
            if isinstance(page, int) or (isinstance(page, str) and page.isdigit()):
                location = f"第 {int(page) + 1} 页"

            node_id = self._generated_node_id(str(doc_id), meta, i)
            if node_id in self._nodes:
                node_id = self._string_id(f"{node_id}__{doc_id}")
            category_code = meta.get("category") or (meta.get("categories") or ["项目知识"])[0]
            node = {
                "id": node_id,
                "chunk_id": str(meta.get("chunk_id") or doc_id),
                "document_id": str(meta.get("document_id") or ""),
                "source_id": str(meta.get("source_id") or ""),
                "asset_id": str(meta.get("asset_id") or ""),
                "version_id": str(meta.get("version_id") or ""),
                "project_id": str(meta.get("project_id") or ""),
                "access_policy_id": str(meta.get("access_policy_id") or ""),
                "visibility": str(meta.get("visibility") or ""),
                "owner_user_id": str(meta.get("owner_user_id") or ""),
                "acl_revision": meta.get("acl_revision", 0),
                "is_current_version": meta.get("is_current_version", True),
                "content": content[:200],  # 存储摘要
                "source": source,
                "stored_file": str(meta.get("stored_file") or ""),
                "topic": self._derive_topic_label(content, source, tags),
                "category": knowledge_category_label(category_code),
                "category_code": str(category_code or ""),
                "chunk_index": meta.get("chunk_index", i),
                "location": location,
                "tags": tags,
                "created_at": datetime.now().isoformat(),
                "ref_count": 0,
                "vector": vectors[i] if i < len(vectors) else [],
            }
            self._nodes[node_id] = node

            # 更新标签索引
            for tag in tags:
                self._tag_index[tag].append(node_id)
            # 更新关键词索引
            for word in self._tokenize(content):
                self._keyword_index[word].add(node_id)

        # 2. 混合关联：语义向量 + 共同标签 + 分类语境，且限制每个节点的自动连边数。
        self._build_auto_edges()
        self._build_document_sequence_edges()

        logger.info("知识图谱构建完成: %d 节点, %d 边",
                     len(self._nodes), len(self._edges))
        merged = self._merge_for_rebuild(self._generated_graph_data(), imported_graph, saved_graph)
        self.save_graph_data(merged)

    def _expand_legacy_chunks(self, ids, documents, metadatas, vectors):
        """Expand historical one-file-one-chunk records for graph projection.

        The authoritative parent chunk ID is retained on every graph segment so
        ACL and lifecycle filtering continue to use SQLite-owned identities.
        """
        expanded=[]
        pending_embeddings=[]
        for index,document_id in enumerate(ids):
            content=documents[index] if index<len(documents) else ""
            metadata=dict(metadatas[index] if index<len(metadatas) else {})
            original_vector=vectors[index] if index<len(vectors) else []
            pieces=vector_store.split_documents([
                Document(page_content=content,metadata=metadata)
            ])
            if len(pieces)<=1:
                expanded.append((str(document_id),content,metadata,original_vector))
                continue
            parent_chunk_id=str(metadata.get("chunk_id") or document_id)
            for segment_index,piece in enumerate(pieces):
                segment_meta=dict(piece.metadata or {})
                segment_meta["chunk_id"]=parent_chunk_id
                segment_meta["chunk_index"]=f"{metadata.get('chunk_index', index)}.{segment_index}"
                segment_meta["graph_segment_index"]=segment_index
                expanded.append((
                    f"{document_id}__segment__{segment_index}",
                    piece.page_content,
                    segment_meta,
                    None,
                ))
                pending_embeddings.append(len(expanded)-1)
        if pending_embeddings:
            embedded=vector_store.embeddings.embed_documents([
                expanded[position][1] for position in pending_embeddings
            ])
            for position,embedding in zip(pending_embeddings,embedded):
                item=expanded[position]
                expanded[position]=(item[0],item[1],item[2],embedding)
        if not expanded:
            return [],[],[],[]
        return tuple([item[column] for item in expanded] for column in range(4))

    def _build_auto_edges(self):
        """为每个节点选择少量高质量的自动关联，避免画布成为稠密关系网。"""
        candidates: Dict[str, List[Tuple[float, str, dict]]] = defaultdict(list)
        node_ids = list(self._nodes.keys())

        for index, source_id in enumerate(node_ids):
            for target_id in node_ids[index + 1:]:
                details = self._score_relation_pair(source_id, target_id)
                if not details:
                    continue
                score = details["confidence"]
                candidates[source_id].append((score, target_id, details))
                candidates[target_id].append((score, source_id, details))

        selected_pairs: Dict[Tuple[str, str], dict] = {}
        for source_id, node_candidates in candidates.items():
            for _, target_id, details in sorted(node_candidates, key=lambda item: item[0], reverse=True)[:AUTO_EDGE_LIMIT_PER_NODE]:
                pair = tuple(sorted((source_id, target_id)))
                previous = selected_pairs.get(pair)
                if not previous or details["confidence"] > previous["confidence"]:
                    selected_pairs[pair] = details

        for (source_id, target_id), details in selected_pairs.items():
            self._edges[(source_id, target_id)] = details
            self._nodes[source_id]["ref_count"] += 1
            self._nodes[target_id]["ref_count"] += 1

    def _build_document_sequence_edges(self):
        """Connect adjacent topics from the same document with explicit evidence."""
        by_document=defaultdict(list)
        for node_id,node in self._nodes.items():
            document_key=str(node.get("document_id") or node.get("source") or "").strip()
            if document_key:
                by_document[document_key].append((node.get("chunk_index"),node_id))

        def order_key(item):
            value=str(item[0] if item[0] is not None else "")
            parts=[]
            for token in value.split("."):
                try:
                    parts.append((0,int(token)))
                except ValueError:
                    parts.append((1,token))
            return parts

        for items in by_document.values():
            ordered=[node_id for _,node_id in sorted(items,key=order_key)]
            for source_id,target_id in zip(ordered,ordered[1:]):
                pair=tuple(sorted((source_id,target_id)))
                source=self._nodes[source_id]
                target=self._nodes[target_id]
                previous=self._edges.get(pair)
                if previous:
                    source["ref_count"]=max(0,source["ref_count"]-1)
                    target["ref_count"]=max(0,target["ref_count"]-1)
                edge={
                    "id":f"sequence__{source_id}__{target_id}",
                    "source":source_id,"target":target_id,
                    "from":source_id,"to":target_id,
                    "label":"同文档相邻主题","relation":"document_sequence",
                    "width":2,"confidence":0.92,
                    "evidence":f"同一文档「{source.get('source') or '知识资料'}」中的相邻知识片段",
                    "method":"document_structure","auto_generated":True,"protected":False,
                }
                self._edges[pair]=edge
                source["ref_count"]+=1
                target["ref_count"]+=1

    def _score_relation_pair(self, source_id: str, target_id: str) -> dict:
        """计算两个知识主题的关联，并留下可面向用户解释的证据。"""
        source = self._nodes[source_id]
        target = self._nodes[target_id]
        semantic = self._cosine_similarity(source.get("vector", []), target.get("vector", []))
        tags_a = set(source.get("tags", []))
        tags_b = set(target.get("tags", []))
        shared_tags = sorted(tags_a & tags_b)
        tag_score = len(shared_tags) / len(tags_a | tags_b) if tags_a or tags_b else 0.0
        same_category = bool(source.get("category") and source.get("category") == target.get("category"))
        same_source = bool(source.get("source") and source.get("source") == target.get("source"))

        confidence = 0.66 * semantic + 0.26 * tag_score + (0.08 if same_category else 0.0)
        # 同一文件的页/块不应仅因邻近文本而挤满画布，除非证据非常强。
        if same_source:
            confidence -= 0.18

        enough_evidence = (
            (semantic >= 0.42 and confidence >= AUTO_EDGE_MIN_SCORE)
            or (tag_score >= 0.34 and semantic >= 0.18)
        )
        if same_source:
            enough_evidence = enough_evidence and semantic >= 0.62 and tag_score >= 0.40
        if not enough_evidence:
            return {}

        confidence = max(0.0, min(0.99, confidence))
        if confidence >= 0.72:
            relation, label, width = "strong_related", "强语义关联", 4
        elif tag_score >= 0.50 and semantic < 0.52:
            relation, label, width = "topic_related", "主题关联", 2
        else:
            relation, label, width = "semantic_related", "语义关联", 3

        evidence = [f"语义相似度 {semantic:.0%}"]
        if shared_tags:
            evidence.append(f"共同标签：{'、'.join(shared_tags[:3])}")
        if same_category:
            evidence.append(f"同属「{knowledge_category_label(source.get('category'))}」")
        if same_source:
            evidence.append("同一来源的高置信主题")

        return {
            "id": f"auto__{source_id}__{target_id}",
            "source": source_id,
            "target": target_id,
            "from": source_id,
            "to": target_id,
            "label": label,
            "relation": relation,
            "width": width,
            "confidence": round(confidence, 2),
            "evidence": "；".join(evidence),
            "method": "hybrid_auto",
            "auto_generated": True,
            "protected": False,
        }

    def _cosine_similarity(self, vector_a: Any, vector_b: Any) -> float:
        if not isinstance(vector_a, list) or not isinstance(vector_b, list) or len(vector_a) != len(vector_b) or not vector_a:
            return 0.0
        try:
            dot = sum(float(a) * float(b) for a, b in zip(vector_a, vector_b))
            norm_a = math.sqrt(sum(float(a) ** 2 for a in vector_a))
            norm_b = math.sqrt(sum(float(b) ** 2 for b in vector_b))
        except (TypeError, ValueError):
            return 0.0
        if norm_a < 1e-9 or norm_b < 1e-9:
            return 0.0
        return max(0.0, min(1.0, dot / (norm_a * norm_b)))

    def get_related_nodes(self, node_id: str, max_depth: int = 2) -> List[dict]:
        """获取与指定节点相关的知识链（用于溯源分析）"""
        graph = self.get_graph_data()
        nodes_by_id = {node.get("id"): node for node in graph.get("nodes", [])}
        edges = graph.get("edges", [])
        visited = set()
        results = []
        queue = [(node_id, 0, None)]

        while queue and len(results) < 20:
            current, depth, via_edge = queue.pop(0)
            if current in visited or depth > max_depth:
                continue
            visited.add(current)

            node = nodes_by_id.get(current)
            if node and depth > 0:  # 不包含自身
                results.append({
                    **node,
                    "depth": depth,
                    "relation": (via_edge or {}).get("label") or "关联",
                    "evidence": (via_edge or {}).get("evidence") or "",
                })

            # BFS 查找相邻节点
            for edge in edges:
                f = edge.get("source") or edge.get("from")
                t = edge.get("target") or edge.get("to")
                if f == current and t not in visited:
                    queue.append((t, depth + 1, edge))
                elif t == current and f not in visited:
                    queue.append((f, depth + 1, edge))

        return results

    def trace_decision(self, query: str) -> dict:
        """
        决策溯源：从查询内容追溯到相关文档和决策依据
        返回完整的决策链路
        """
        # 1. 从向量库检索相关文档
        docs = vector_store.similarity_search(query, k=5)

        trace = {
            "query": query,
            "decisions": [],
            "source_documents": [],
            "related_knowledge": [],
            "trace_path": [],
        }

        graph = self.get_graph_data()
        graph_nodes = graph.get("nodes", [])
        for doc in docs:
            source = doc.metadata.get("source_file", "未知")
            trace["source_documents"].append({
                "source": source,
                "content": doc.page_content[:300],
            })

            # 在知识图谱中查找关联节点
            for node in graph_nodes:
                node_id = str(node.get("id") or "")
                if node_id and self._source_matches(node.get("source"), source):
                    related = self.get_related_nodes(node_id, max_depth=1)
                    for r in related:
                        trace["related_knowledge"].append({
                            "source": r.get("source", "未知"),
                            "content": r.get("title") or r.get("content") or "",
                            "relation": r.get("relation") or "关联",
                            "evidence": r.get("evidence") or "",
                        })
                    break

        # 构建溯源路径
        trace["trace_path"] = [
            "📌 问题 → 知识库检索 → 匹配相关文档 → 关联知识发现 → 决策链路",
            f"找到 {len(trace['source_documents'])} 篇相关文档",
            f"发现 {len(trace['related_knowledge'])} 条关联知识",
        ]

        return trace

    def get_graph_data(self) -> dict:
        """获取图谱数据，并按 SQLite 当前有效来源执行读取时失效控制。"""
        saved = self._load_saved_graph()
        generated = self._generated_graph_data()
        imported = self._load_imported_graph()
        if generated.get("nodes"):
            graph = self._merge_for_rebuild(generated, imported, saved)
        elif saved.get("nodes"):
            # After a process restart the in-memory generated graph is empty.
            # The persisted projection remains the best available snapshot and
            # is filtered against current SQLite assets below.
            graph = saved
        elif imported.get("nodes"):
            graph = imported
        else:
            return {"nodes": [], "edges": [], "starter": False, "updated_at": datetime.now().isoformat()}
        return self._filter_inactive_sources(graph)

    def _filter_inactive_sources(self, graph: dict) -> dict:
        """Fail closed against current authorized assets and explicit policies.

        Human-readable source names are presentation metadata, never ACL keys.
        Generated nodes are authorized by chunk ID; imported/manual nodes must
        bind to an authorized asset/version or carry an enforceable policy.
        """
        from backend.storage import get_repository

        repository = get_repository()
        active_chunk_ids = repository.active_chunk_ids()
        documents = [item for item in repository.list_documents() if isinstance(item, dict)]
        allowed_assets = {
            str(item.get("asset_id")) for item in documents if item.get("asset_id")
        }
        allowed_versions = {
            (str(item.get("asset_id")), str(item.get("version_id") or item.get("current_version_id")))
            for item in documents
            if item.get("asset_id") and (item.get("version_id") or item.get("current_version_id"))
        }

        normalized = self._normalize_graph(graph)
        remaining_nodes = []
        for node in normalized.get("nodes", []):
            origin = str(node.get("origin") or "").lower()
            if origin == "generated":
                chunk_id = str(node.get("chunk_id") or node.get("id") or "")
                if chunk_id in active_chunk_ids and self._is_current_metadata(node):
                    remaining_nodes.append(node)
                continue
            if self._node_binding_authorized(node, allowed_assets, allowed_versions, repository):
                remaining_nodes.append(node)

        remaining_ids = {str(node.get("id")) for node in remaining_nodes}
        remaining_edges = [
            edge for edge in normalized.get("edges", [])
            if str(edge.get("source") or edge.get("from") or "") in remaining_ids
            and str(edge.get("target") or edge.get("to") or "") in remaining_ids
        ]
        return self._normalize_graph({"nodes": remaining_nodes, "edges": remaining_edges})

    def _node_binding_authorized(
        self, node: dict, allowed_assets: Set[str],
        allowed_versions: Set[Tuple[str, str]], repository: Any,
    ) -> bool:
        asset_id = str(node.get("asset_id") or "").strip()
        version_id = str(node.get("version_id") or "").strip()
        if asset_id:
            if asset_id not in allowed_assets:
                return False
            return not version_id or (asset_id, version_id) in allowed_versions

        policy_id = str(node.get("access_policy_id") or "").strip()
        if not policy_id:
            return False
        try:
            from backend.auth_context import get_current_identity

            identity = get_current_identity()
            return bool(repository._asset_visible({
                "project_id": str(node.get("project_id") or repository.project_id),
                "visibility": "restricted",
                "access_policy_id": policy_id,
                "owner_user_id": str(node.get("owner_user_id") or ""),
            }, identity))
        except Exception:
            logger.warning("图谱节点策略校验失败，已拒绝节点 %s", node.get("id"))
            return False

    def save_graph_data(self, graph: dict) -> dict:
        """保存前端编辑后的图谱快照"""
        normalized = self._normalize_graph(graph)
        from backend.storage import get_repository

        get_repository().save_graph_edits(normalized)
        return normalized

    def import_graph_data(self, graph: dict) -> dict:
        """持久化导入图谱，并和自动生成的文档图谱合并展示。"""
        defaults = {
            key: graph.get(key) for key in (
                "asset_id", "version_id", "project_id", "access_policy_id",
                "visibility", "owner_user_id", "acl_revision",
            ) if graph.get(key) is not None
        } if isinstance(graph, dict) else {}
        raw = dict(graph or {})
        raw["nodes"] = [
            {**defaults, **item} for item in raw.get("nodes", []) if isinstance(item, dict)
        ]
        imported = self._normalize_graph(raw)
        for node in imported.get("nodes", []):
            node["origin"] = "imported"
        for edge in imported.get("edges", []):
            edge.update({"method": "imported", "auto_generated": False, "protected": True})
        self._save_imported_graph(imported)
        merged = self._merge_for_rebuild(self._generated_graph_data(), imported, self._load_saved_graph())
        return self.save_graph_data(merged)

    def remove_source(self, source_name: str) -> dict:
        """删除某份来源文档产生的节点，并同步清理保存快照和 JSON 导入图谱。"""
        source_name = str(source_name or "").strip()
        if not source_name:
            return {"nodes": 0, "edges": 0, "imported_nodes": 0}

        saved_graph = self._load_saved_graph() or {"nodes": [], "edges": []}
        imported_graph = self._load_imported_graph()
        cleaned_saved, saved_counts = self._remove_source_from_graph(saved_graph, source_name)
        cleaned_imported, imported_counts = self._remove_source_from_graph(imported_graph, source_name)

        self.save_graph_data(cleaned_saved)
        if imported_graph.get("nodes") or imported_counts["nodes"]:
            self._save_imported_graph(cleaned_imported)

        from backend.storage import get_repository

        get_repository().remove_graph_source(source_name)

        removed_memory_ids = {
            node_id
            for node_id, node in self._nodes.items()
            if self._node_matches_source(node, source_name)
        }
        if removed_memory_ids:
            self._nodes = {
                node_id: node for node_id, node in self._nodes.items()
                if node_id not in removed_memory_ids
            }
            self._edges = {
                pair: edge for pair, edge in self._edges.items()
                if pair[0] not in removed_memory_ids and pair[1] not in removed_memory_ids
            }

        return {
            "nodes": saved_counts["nodes"],
            "edges": saved_counts["edges"],
            "imported_nodes": imported_counts["nodes"],
        }

    def _remove_source_from_graph(self, graph: dict, source_name: str) -> Tuple[dict, dict]:
        nodes = [node for node in graph.get("nodes", []) if isinstance(node, dict)]
        removed_ids = {
            str(node.get("id"))
            for node in nodes
            if node.get("id") and self._node_matches_source(node, source_name)
        }
        remaining_nodes = [node for node in nodes if str(node.get("id")) not in removed_ids]
        raw_edges = [edge for edge in graph.get("edges", []) if isinstance(edge, dict)]
        remaining_edges = [
            edge for edge in raw_edges
            if str(edge.get("source") or edge.get("from") or "") not in removed_ids
            and str(edge.get("target") or edge.get("to") or "") not in removed_ids
        ]
        return self._normalize_graph({"nodes": remaining_nodes, "edges": remaining_edges}), {
            "nodes": len(nodes) - len(remaining_nodes),
            "edges": len(raw_edges) - len(remaining_edges),
        }

    def _source_matches(self, node_source: Any, requested_source: str) -> bool:
        node_name = Path(str(node_source or "")).name.strip()
        requested_name = Path(str(requested_source or "")).name.strip()
        if not node_name or not requested_name:
            return False
        return (
            node_name == requested_name
            or node_name.endswith(f"_{requested_name}")
            or requested_name.endswith(f"_{node_name}")
        )

    def _node_matches_source(self, node: dict, requested_source: str) -> bool:
        stored_file = str(node.get("stored_file") or "").strip()
        if stored_file:
            return self._source_matches(stored_file, requested_source)
        return self._source_matches(node.get("source"), requested_source)

    def _load_imported_graph(self) -> dict:
        from backend.storage import get_repository

        graph = get_repository().load_graph_edits()
        imported_nodes = [node for node in graph.get("nodes", []) if node.get("origin") == "imported"]
        imported_ids = {str(node.get("id")) for node in imported_nodes}
        imported_edges = [
            edge for edge in graph.get("edges", [])
            if edge.get("method") == "imported"
            or edge.get("origin") == "imported"
            or str(edge.get("source") or edge.get("from") or "") in imported_ids
            or str(edge.get("target") or edge.get("to") or "") in imported_ids
        ]
        return self._filter_inactive_sources(
            self._normalize_graph({"nodes": imported_nodes, "edges": imported_edges})
        )

    def _save_imported_graph(self, graph: dict):
        normalized = self._normalize_graph(graph)
        for node in normalized.get("nodes", []):
            node["origin"] = "imported"
        for edge in normalized.get("edges", []):
            edge.update({"method": "imported", "auto_generated": False, "protected": True})
        from backend.storage import get_repository

        get_repository().replace_graph_origin("imported", normalized)

    def _merge_graphs(self, generated: dict, imported: dict) -> dict:
        """合并自动图谱和 JSON 图谱，保留导入关系的语义。"""
        generated = generated or {"nodes": [], "edges": []}
        imported = imported or {"nodes": [], "edges": []}
        return self._normalize_graph({
            "nodes": [*(generated.get("nodes", []) or []), *(imported.get("nodes", []) or [])],
            "edges": [*(generated.get("edges", []) or []), *(imported.get("edges", []) or [])],
        })

    def _merge_for_rebuild(self, generated: dict, imported: dict, saved: dict) -> dict:
        """重建时以自动图谱为基准，保留人工节点、人工关系和已固定布局。"""
        merged = self._merge_graphs(generated, imported)
        saved = saved or {"nodes": [], "edges": []}
        saved_nodes = {str(node.get("id")): node for node in saved.get("nodes", []) if isinstance(node, dict) and node.get("id")}
        base_nodes = {str(node.get("id")): node for node in merged.get("nodes", []) if isinstance(node, dict) and node.get("id")}

        for node_id, node in base_nodes.items():
            previous = saved_nodes.get(node_id)
            if not previous:
                continue
            for key in ("x", "y", "locked"):
                if key in previous:
                    node[key] = previous[key]
            if previous.get("manual_override"):
                for key in ("label", "topic", "group", "type", "title", "size", "tags"):
                    if key in previous:
                        node[key] = previous[key]
                node["manual_override"] = True

        # 手工新增的节点没有自动来源，必须继续保留在后续重建中。
        for node_id, previous in saved_nodes.items():
            if node_id not in base_nodes and self._is_manual_node(previous):
                node = dict(previous)
                node["origin"] = "manual"
                base_nodes[node_id] = node

        base_edges = list(merged.get("edges", []))
        base_edge_ids = {str(edge.get("id")) for edge in base_edges if isinstance(edge, dict)}
        for edge in saved.get("edges", []):
            if not isinstance(edge, dict) or not self._is_protected_edge(edge):
                continue
            source = str(edge.get("source") or edge.get("from") or "")
            target = str(edge.get("target") or edge.get("to") or "")
            if source not in base_nodes or target not in base_nodes or source == target:
                continue
            preserved = dict(edge)
            preserved.update({
                "source": source,
                "target": target,
                "from": source,
                "to": target,
                "auto_generated": False,
                "protected": True,
                "method": preserved.get("method") or "manual",
            })
            edge_id = str(preserved.get("id") or f"manual__{source}__{target}__{preserved.get('label') or '关联'}")
            preserved["id"] = edge_id
            if edge_id not in base_edge_ids:
                base_edges.append(preserved)
                base_edge_ids.add(edge_id)

        return self._normalize_graph({"nodes": list(base_nodes.values()), "edges": base_edges})

    def _is_manual_node(self, node: dict) -> bool:
        origin = str(node.get("origin") or "").lower()
        return origin == "manual" or (not node.get("source") and origin != "imported")

    def _is_protected_edge(self, edge: dict) -> bool:
        if edge.get("protected") or edge.get("auto_generated") is False:
            return True
        method = str(edge.get("method") or "").lower()
        if method in {"manual", "imported"}:
            return True
        # 兼容旧快照：原有自动边只有 related/strong_related 两种内部关系。
        relation = str(edge.get("relation") or edge.get("label") or "").lower()
        return relation not in AUTO_RELATIONS

    def graph_from_json_content(
        self, content: str, source_name: str = "uploaded.json",
        source_metadata: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """识别常见 JSON 图谱格式，返回可保存的图谱数据；无法识别时返回空 dict。"""
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return {}
        if not isinstance(payload, dict):
            return {}

        source_metadata = source_metadata or {}
        binding_defaults = {
            key: source_metadata.get(key) or payload.get(key)
            for key in (
                "asset_id", "version_id", "project_id", "access_policy_id",
                "visibility", "owner_user_id", "acl_revision",
            )
            if source_metadata.get(key) is not None or payload.get(key) is not None
        }

        raw_nodes = payload.get("nodes") or payload.get("vertices") or payload.get("items") or []
        raw_edges = payload.get("edges") or payload.get("links") or payload.get("relations") or []
        node_sections = []
        edge_sections = []
        if isinstance(raw_nodes, dict):
            for section, items in raw_nodes.items():
                if isinstance(items, list):
                    node_sections.extend((section, item) for item in items)
        elif isinstance(raw_nodes, list):
            node_sections = [("项目知识", item) for item in raw_nodes]
        if isinstance(raw_edges, dict):
            for section, items in raw_edges.items():
                if isinstance(items, list):
                    edge_sections.extend((section, item) for item in items)
        elif isinstance(raw_edges, list):
            edge_sections = [("关联", item) for item in raw_edges]
        if not node_sections:
            return {}

        nodes = []
        known_ids = set()
        for index, (section, item) in enumerate(node_sections):
            if not isinstance(item, dict):
                item = {"label": str(item)}
            node_id = self._string_id(
                item.get("id") or item.get("key") or item.get("uid") or item.get("name") or item.get("label") or f"json_node_{index + 1}"
            )
            if not node_id or node_id in known_ids:
                node_id = f"{node_id or 'json_node'}_{index + 1}"
            known_ids.add(node_id)
            label = str(item.get("label") or item.get("name") or item.get("title") or node_id)
            nodes.append({
                **binding_defaults,
                "id": node_id,
                "label": label,
                "topic": label,
                "group": str(item.get("group") or item.get("category") or item.get("type") or section or source_name.rsplit(".", 1)[-1]),
                "type": str(item.get("type") or item.get("kind") or "JSON 节点"),
                "title": str(item.get("title") or item.get("description") or item.get("content") or ""),
                "source": source_name,
                "location": str(item.get("location") or item.get("page") or ""),
                "tags": item.get("tags", []) if isinstance(item.get("tags"), list) else [],
                "size": self._safe_int(item.get("size") or item.get("value"), 18),
                "x": item.get("x"),
                "y": item.get("y"),
                "locked": bool(item.get("locked", False)),
                "origin": "imported",
                "manual_override": bool(item.get("manual_override", False)),
                **{
                    key: item.get(key) for key in binding_defaults if item.get(key) is not None
                },
            })

        edges = []
        for index, (section, item) in enumerate(edge_sections):
            if not isinstance(item, dict):
                continue
            source = self._edge_endpoint(item.get("source") or item.get("from") or item.get("source_id") or item.get("start"))
            target = self._edge_endpoint(item.get("target") or item.get("to") or item.get("target_id") or item.get("end"))
            if not source or not target:
                continue
            label = str(item.get("label") or item.get("relation") or item.get("type") or section or "关联")
            edges.append({
                "id": str(item.get("id") or f"{source}__{target}__{index + 1}"),
                "source": source,
                "target": target,
                "from": source,
                "to": target,
                "label": label,
                "relation": str(item.get("relation") or label),
                "width": self._safe_int(item.get("width") or item.get("weight"), 2),
                "confidence": self._safe_float(item.get("confidence") or item.get("score"), 0.9),
                "evidence": str(item.get("evidence") or item.get("reason") or f"从 {source_name} 导入"),
                "method": "imported",
                "auto_generated": False,
                "protected": True,
            })

        normalized = self._normalize_graph({"nodes": nodes, "edges": edges})
        return normalized if normalized.get("nodes") else {}

    def _string_id(self, value) -> str:
        return re.sub(r"\s+", "_", str(value).strip())[:80]

    def _edge_endpoint(self, value) -> str:
        if isinstance(value, dict):
            value = value.get("id") or value.get("key") or value.get("name") or value.get("label")
        return self._string_id(value) if value is not None else ""

    def _safe_int(self, value, fallback: int) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return fallback

    def _safe_float(self, value, fallback: float) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return fallback

    def _derive_topic_label(self, content: str, source: str, tags: List[str]) -> str:
        """从知识块中提取适合画布展示的短主题，避免重复显示文件名。"""
        source_name = Path(source).stem.replace("_", " ")
        source_key = self._topic_key(source_name)
        text = (content or "").replace("\x01", " ").replace("\u3000", " ")
        text = re.sub(r"[\t\r]+", " ", text)
        candidates = []

        # Markdown 标题和文档中的编号章节通常是最准确的主题来源。
        candidates.extend(re.findall(r"^\s*#{1,6}\s+(.+?)\s*$", text, flags=re.MULTILINE))
        candidates.extend(re.findall(r"^\s*(?:第[一二三四五六七八九十百\d]+[章节、.]|[一二三四五六七八九十]+[、.])\s*(.+?)\s*$", text, flags=re.MULTILINE))
        candidates.extend(re.findall(r"【([^】]{2,40})】", text))

        # PDF 经常没有规范标题，取第一条可读的业务语句作为兜底。
        candidates.extend(text.split("\n"))

        ignored = {"功能", "变更", "变更内容", "概要", "概述", "名词定义", "目录", "版本", "日期", "状态", "创建"}
        for raw_candidate in candidates:
            candidate = re.sub(r"^\s*(?:[-*]|\d+[.、)])\s*", "", raw_candidate)
            candidate = re.sub(r"^(?:第[一二三四五六七八九十百\d]+[章节、.]|[一二三四五六七八九十]+[、.])\s*", "", candidate)
            candidate = re.sub(r"[*`_#]", "", candidate)
            candidate = re.sub(r"\s+", " ", candidate).strip(" ：:;；，,。.")
            candidate = re.sub(r"^(?:如果|当前|本次|需要|建议)", "", candidate)
            candidate = re.sub(r"(?:时)?提示$", "", candidate).strip()
            candidate_key = self._topic_key(candidate)
            if not self._is_useful_topic(candidate, candidate_key, source_key, ignored):
                continue
            return candidate[:28]

        tag_topics = [str(tag).strip() for tag in tags if len(str(tag).strip()) >= 2]
        if tag_topics:
            return " / ".join(tag_topics[:2])[:28]
        return "未命名知识主题"

    def _topic_key(self, value: str) -> str:
        return re.sub(r"[^\w\u4e00-\u9fff]", "", (value or "").lower())

    def _is_useful_topic(self, candidate: str, candidate_key: str, source_key: str, ignored: Set[str]) -> bool:
        if len(candidate) < 3 or len(candidate) > 36 or candidate in ignored:
            return False
        if not re.search(r"[A-Za-z\u4e00-\u9fff]", candidate):
            return False
        if re.fullmatch(r"[\d\s./:-]+", candidate) or re.search(r"\b\d{4}[-/]?\d{1,2}[-/]?\d{0,2}\b", candidate):
            return False
        if candidate_key and source_key and (candidate_key == source_key or (len(candidate_key) >= 8 and candidate_key in source_key)):
            return False
        return True

    def _is_source_label(self, label: str, source: str) -> bool:
        """识别旧版按文件名生成的标签，以便读取旧图谱时自动迁移。"""
        label_key = self._topic_key(Path(label).stem)
        source_key = self._topic_key(Path(source).stem)
        return bool(label_key and source_key and (label_key == source_key or (len(label_key) >= 8 and source_key.startswith(label_key))))

    def _generated_graph_data(self) -> dict:
        """从内存节点生成可编辑图谱数据"""
        nodes = []
        edges = []
        topic_counts = defaultdict(int)

        for node_id, node in self._nodes.items():
            topic = node.get("topic") or self._derive_topic_label(node.get("content", ""), node.get("source", ""), node.get("tags", []))
            topic_counts[topic] += 1
            label = topic if topic_counts[topic] == 1 else f"{topic} · {topic_counts[topic]}"
            nodes.append({
                "id": node_id,
                "label": label,
                "topic": topic,
                "group": node.get("category") or "项目知识",
                "type": "知识主题",
                "size": min(30, 10 + node["ref_count"] * 3),
                "title": node["content"][:100],
                "source": node["source"],
                "stored_file": node.get("stored_file", ""),
                "chunk_id": node.get("chunk_id", ""),
                "document_id": node.get("document_id", ""),
                "source_id": node.get("source_id", ""),
                "asset_id": node.get("asset_id", ""),
                "version_id": node.get("version_id", ""),
                "project_id": node.get("project_id", ""),
                "access_policy_id": node.get("access_policy_id", ""),
                "visibility": node.get("visibility", ""),
                "owner_user_id": node.get("owner_user_id", ""),
                "acl_revision": node.get("acl_revision", 0),
                "is_current_version": node.get("is_current_version", True),
                "location": node.get("location", ""),
                "tags": node.get("tags", []),
                "locked": False,
                "origin": "generated",
                "manual_override": False,
            })

        for (source, target), edge in self._edges.items():
            edges.append({
                **edge,
                "id": edge.get("id") or f"auto__{source}__{target}",
                "from": source,
                "to": target,
                "source": source,
                "target": target,
            })

        return {"nodes": nodes[:100], "edges": edges[:200]}

    def _load_saved_graph(self) -> dict:
        try:
            from backend.storage import get_repository

            data = get_repository().load_graph_edits()
            if data.get("nodes") or data.get("edges"):
                return self._normalize_graph(data)
            return {}
        except Exception as e:
            logger.warning("读取知识图谱编辑快照失败: %s", e)
            return {}

    def _normalize_graph(self, graph: dict) -> dict:
        raw_nodes = graph.get("nodes", []) if isinstance(graph, dict) else []
        raw_edges = graph.get("edges", []) if isinstance(graph, dict) else []
        nodes = []
        seen_nodes = set()
        seen_labels = defaultdict(int)

        for item in raw_nodes:
            if not isinstance(item, dict):
                continue
            node_id = str(item.get("id") or "").strip()
            label = str(item.get("label") or item.get("name") or "").strip()
            if not node_id or not label or node_id in seen_nodes:
                continue
            source = str(item.get("source") or "").strip()
            tags = [str(tag)[:24] for tag in item.get("tags", [])[:8]] if isinstance(item.get("tags"), list) else []
            node_type = str(item.get("type") or item.get("group") or "知识节点")[:30]
            topic = str(item.get("topic") or label).strip()
            origin = str(item.get("origin") or "").strip().lower()
            if origin not in {"generated", "imported", "manual"}:
                origin = "manual" if not source else ("imported" if source.lower().endswith(".json") else "generated")
            if not item.get("topic") and node_type in {"知识块", "文档节点"} and self._is_source_label(label, source):
                topic = self._derive_topic_label(str(item.get("title") or item.get("content") or ""), source, tags)
                label = topic
            seen_labels[label] += 1
            if seen_labels[label] > 1:
                label = f"{label} · {seen_labels[label]}"
            seen_nodes.add(node_id)
            normalized_node = {
                "id": node_id,
                "label": label[:60],
                "topic": topic[:60],
                "group": str(item.get("group") or item.get("type") or "项目知识")[:30],
                "type": node_type,
                "size": max(8, min(44, self._safe_int(item.get("size"), 16))),
                "title": str(item.get("title") or item.get("content") or "")[:500],
                "source": source[:200],
                "stored_file": str(item.get("stored_file") or "")[:200],
                "chunk_id": str(item.get("chunk_id") or "")[:160],
                "document_id": str(item.get("document_id") or "")[:160],
                "source_id": str(item.get("source_id") or "")[:160],
                "asset_id": str(item.get("asset_id") or "")[:160],
                "version_id": str(item.get("version_id") or "")[:160],
                "project_id": str(item.get("project_id") or "")[:160],
                "access_policy_id": str(item.get("access_policy_id") or "")[:160],
                "visibility": str(item.get("visibility") or "")[:30],
                "owner_user_id": str(item.get("owner_user_id") or "")[:160],
                "acl_revision": max(0, self._safe_int(item.get("acl_revision"), 0)),
                "is_current_version": item.get("is_current_version", True) is not False,
                "location": str(item.get("location") or "")[:80],
                "tags": tags,
                "locked": bool(item.get("locked", False)),
                "origin": origin,
                "manual_override": bool(item.get("manual_override", False)),
            }
            for axis in ("x", "y"):
                try:
                    if item.get(axis) is not None:
                        normalized_node[axis] = round(float(item.get(axis)), 2)
                except (TypeError, ValueError):
                    pass
            nodes.append(normalized_node)

        edges = []
        seen_edges = set()
        for item in raw_edges:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source") or item.get("from") or "").strip()
            target = str(item.get("target") or item.get("to") or "").strip()
            if source not in seen_nodes or target not in seen_nodes or source == target:
                continue
            label = str(item.get("label") or item.get("relation") or "关联").strip()[:40]
            edge_id = str(item.get("id") or f"{source}__{target}__{label}").strip()
            if edge_id in seen_edges:
                continue
            relation = str(item.get("relation") or label)[:40]
            raw_auto_generated = item.get("auto_generated")
            auto_generated = bool(raw_auto_generated) if raw_auto_generated is not None else relation in AUTO_RELATIONS
            method = str(item.get("method") or ("hybrid_auto" if auto_generated else "manual"))[:30]
            seen_edges.add(edge_id)
            edges.append({
                "id": edge_id,
                "from": source,
                "to": target,
                "source": source,
                "target": target,
                "label": label,
                "relation": relation,
                "width": max(1, min(6, self._safe_int(item.get("width"), 1))),
                "confidence": self._safe_float(item.get("confidence"), 0.65 if auto_generated else 0.9),
                "evidence": str(item.get("evidence") or "")[:240],
                "method": method,
                "auto_generated": auto_generated,
                "protected": bool(item.get("protected", not auto_generated)),
            })

        return {
            "nodes": nodes[:300],
            "edges": edges[:600],
            "updated_at": datetime.now().isoformat(),
        }

    def _compute_similarity(self, n1: str, n2: str) -> float:
        """计算两个节点的内容相似度（基于共同标签）"""
        tags1 = set(self._nodes.get(n1, {}).get("tags", []))
        tags2 = set(self._nodes.get(n2, {}).get("tags", []))

        if not tags1 or not tags2:
            return 0.0

        intersection = tags1 & tags2
        union = tags1 | tags2

        return len(intersection) / len(union) if union else 0.0

    def _extract_keywords(self, text: str, top_k: int = 5) -> List[str]:
        """从文本中提取关键词"""
        # 简单高频词提取
        words = self._tokenize(text)
        word_freq = defaultdict(int)
        for w in words:
            if len(w) > 1:  # 过滤单字
                word_freq[w] += 1

        sorted_words = sorted(word_freq.items(), key=lambda x: -x[1])
        return [w for w, _ in sorted_words[:top_k]]

    def _tokenize(self, text: str) -> List[str]:
        """简易分词（中文按字分割 + 英文按空格）"""
        if not text:
            return []

        # 提取中文词汇（2-4字组合）
        chinese_chars = re.findall(r"[\u4e00-\u9fff]+", text)
        result = []

        for char_seq in chinese_chars:
            # 使用双字词
            for i in range(len(char_seq) - 1):
                result.append(char_seq[i:i+2])
            # 也加入整词（<=4字）
            if len(char_seq) <= 4:
                result.append(char_seq)
            else:
                result.append(char_seq[:4])

        # 英文词
        english_words = re.findall(r"[a-zA-Z_]\w+", text)
        result.extend([w.lower() for w in english_words])

        return result


# 全局单例
knowledge_graph = KnowledgeGraph()
