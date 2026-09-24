import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

from backend.config import settings
from backend.knowledge_base.knowledge_graph import KnowledgeGraph
from backend.knowledge_base.project_context import ProjectContext
from backend.knowledge_base.readiness import calculate_role_readiness
from backend.knowledge_base.vector_store import VectorStore


class OrthogonalEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[0.0, 1.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


class FakeRepository:
    def __init__(self, chunks=None, documents=None, active_ids=None):
        self._chunks = list(chunks or [])
        self._documents = list(documents or [])
        self._active_ids = set(active_ids or [item.get("chunk_id") for item in self._chunks])
        self.project_id = "project-default"

    def document_chunks(self):
        return list(self._chunks)

    def list_documents(self):
        return list(self._documents)

    def active_chunk_ids(self, chunk_ids=()):
        requested = set(chunk_ids or self._active_ids)
        return requested & self._active_ids

    def active_document_sources(self):
        return [item.get("source_file") for item in self._documents if item.get("source_file")]

    def upsert_document_chunks(self, documents, vector_ids=()):
        return {
            "document_id": "doc-new",
            "chunk_ids": [f"chunk-new-{index}" for index, _ in enumerate(documents)],
        }

    def _asset_visible(self, asset, identity):
        return asset.get("access_policy_id") == "policy-allowed"

    def project_name(self):
        return "AI 赋能项目"


class AssetAIIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.previous_chroma = settings.CHROMA_PERSIST_DIR
        settings.CHROMA_PERSIST_DIR = str(Path(self.temp.name) / "vectors")

    def tearDown(self):
        settings.CHROMA_PERSIST_DIR = self.previous_chroma
        self.temp.cleanup()

    def test_vector_metadata_is_replaced_by_authoritative_asset_version(self):
        chunk = {
            "chunk_id": "chunk-current",
            "document_id": "doc-current",
            "chunk_index": 0,
            "text": "当前版本的项目发布流程",
            "metadata": {"source_file": "流程.md", "version_id": "version-stale"},
            "version_id": "version-current",
            "is_current_version": True,
        }
        parent = {
            "document_id": "doc-current",
            "source_file": "流程.md",
            "stored_file": "流程-v2.md",
            "source_id": "source-process",
            "asset_id": "asset-process",
            "version_id": "version-current",
            "status": "active",
            "authority_level": "official",
            "visibility": "project",
        }
        repository = FakeRepository([chunk], [parent], {"chunk-current"})
        store = VectorStore()
        store._loaded = True
        store._embeddings = OrthogonalEmbeddings()
        store._saved_index_fingerprint = store._current_index_fingerprint()
        store._ids = ["chunk-current"]
        store._texts = ["旧索引正文"]
        store._metas = [{"source_file": "流程.md", "version_id": "version-stale"}]
        store._vecs = [np.array([0.0, 1.0], dtype=np.float32)]

        with patch("backend.storage.get_repository", return_value=repository):
            result = store.reconcile_with_storage()
            metadata = store.get()["metadatas"][0]

        self.assertTrue(result["changed"])
        self.assertEqual(metadata["asset_id"], "asset-process")
        self.assertEqual(metadata["version_id"], "version-current")
        self.assertEqual(metadata["chunk_id"], "chunk-current")
        self.assertEqual(metadata["authority_level"], "official")

    def test_vector_consistency_detects_truncated_derived_index(self):
        chunks = [
            {"chunk_id": "chunk-a", "document_id": "doc-a", "text": "正文 A", "metadata": {}},
            {"chunk_id": "chunk-b", "document_id": "doc-b", "text": "正文 B", "metadata": {}},
        ]
        repository = FakeRepository(chunks=chunks, active_ids={"chunk-a", "chunk-b"})
        store = VectorStore()
        store._loaded = True
        store._embeddings = OrthogonalEmbeddings()
        store._saved_index_fingerprint = store._current_index_fingerprint()
        store._ids = ["chunk-a"]
        store._texts = []
        store._metas = [{}]
        store._vecs = [np.array([0.0, 1.0], dtype=np.float32)]

        with patch("backend.storage.get_repository", return_value=repository):
            before = store.consistency_status()
            repaired = store.reconcile_with_storage()
            after = store.consistency_status()

        self.assertFalse(before["consistent"])
        self.assertFalse(before["shape_valid"])
        self.assertEqual(before["missing_chunks"], 1)
        self.assertTrue(repaired["changed"])
        self.assertTrue(after["consistent"])
        self.assertEqual(after["indexed_chunks"], 2)

    def test_all_retrieval_paths_return_empty_when_evidence_is_below_threshold(self):
        repository = FakeRepository(active_ids={"unrelated"})
        store = VectorStore()
        store._loaded = True
        store._embeddings = OrthogonalEmbeddings()
        store._saved_index_fingerprint = store._current_index_fingerprint()
        store._ids = ["unrelated"]
        store._texts = ["完全无关的历史内容"]
        store._metas = [{"chunk_id": "unrelated", "source_file": "历史.md"}]
        store._vecs = [np.array([0.0, 1.0], dtype=np.float32)]

        with (
            patch("backend.storage.get_repository", return_value=repository),
            patch.object(store, "reconcile_with_storage", return_value={"changed": False}),
        ):
            self.assertEqual(store.similarity_search("部署流程", k=3), [])
            self.assertEqual(store.similarity_search_with_relevance_scores("部署流程", k=3), [])
            self.assertEqual(store.hybrid_search_with_relevance_scores("部署流程", k=3), [])
            self.assertEqual(store.max_marginal_relevance_search("部署流程", k=3), [])

    def test_add_documents_raises_and_restores_memory_when_index_persistence_fails(self):
        repository = FakeRepository()
        store = VectorStore()
        store._loaded = True
        store._embeddings = OrthogonalEmbeddings()
        store._ids = ["existing"]
        store._texts = ["既有知识"]
        store._metas = [{"document_id": "doc-existing", "chunk_id": "existing"}]
        store._vecs = [np.array([1.0, 0.0], dtype=np.float32)]

        with (
            patch("backend.storage.get_repository", return_value=repository),
            patch.object(store, "_save", side_effect=OSError("disk full")),
        ):
            with self.assertRaisesRegex(OSError, "disk full"):
                store.add_documents([Document(
                    page_content="新知识",
                    metadata={"source_file": "新知识.md", "stored_file": "新知识.md"},
                )])

        self.assertEqual(store._ids, ["existing"])
        self.assertEqual(store._texts, ["既有知识"])

    def test_hybrid_search_rejects_zero_vector_similarity_even_on_exact_keyword_match(self):
        repository = FakeRepository(
            documents=[{
                "document_id": "doc-zero", "source_file": "部署.md", "stored_file": "部署.md",
                "status": "active", "version_id": "version-current",
            }],
            active_ids={"chunk-zero"},
        )
        store = VectorStore()
        store._loaded = True
        store._embeddings = OrthogonalEmbeddings()
        store._ids = ["chunk-zero"]
        store._texts = ["部署流程"]
        store._metas = [{
            "document_id": "doc-zero", "chunk_id": "chunk-zero", "source_file": "部署.md",
        }]
        store._vecs = [np.array([0.0, 1.0], dtype=np.float32)]

        with patch("backend.storage.get_repository", return_value=repository):
            self.assertEqual(
                store.hybrid_search_with_relevance_scores("部署流程", k=3, score_threshold=0.0),
                [],
            )

    def test_online_retrieval_does_not_reconcile_and_refreshes_authoritative_metadata(self):
        repository = FakeRepository(
            documents=[{
                "document_id": "doc-live", "source_file": "当前.md", "stored_file": "当前.md",
                "status": "active", "asset_id": "asset-live", "version_id": "version-live",
                "acl_revision": 4,
            }],
            active_ids={"chunk-live"},
        )
        store = VectorStore()
        store._loaded = True
        store._embeddings = OrthogonalEmbeddings()
        store._ids = ["chunk-live"]
        store._texts = ["部署流程"]
        store._metas = [{
            "document_id": "doc-live", "chunk_id": "chunk-live",
            "source_file": "当前.md", "version_id": "version-stale", "acl_revision": 1,
        }]
        store._vecs = [np.array([1.0, 0.0], dtype=np.float32)]

        with (
            patch("backend.storage.get_repository", return_value=repository),
            patch.object(store, "reconcile_with_storage", side_effect=AssertionError("online rebuild")),
        ):
            results = store.similarity_search("部署流程", k=1, score_threshold=0.1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].metadata["version_id"], "version-live")
        self.assertEqual(results[0].metadata["acl_revision"], 4)

    def test_graph_uses_asset_stable_ids_skips_old_versions_and_preserves_manual_edge(self):
        graph = KnowledgeGraph()
        current_metadata = {
            "chunk_id": "chunk-v1",
            "chunk_index": 0,
            "source_file": "架构.md",
            "asset_id": "asset-architecture",
            "version_id": "version-v1",
            "is_current_version": True,
            "status": "active",
            "category": "proj_architecture",
        }
        old_metadata = {
            **current_metadata,
            "chunk_id": "chunk-old",
            "chunk_index": 1,
            "version_id": "version-old",
            "is_current_version": False,
        }
        stable_id = graph._generated_node_id("chunk-v1", current_metadata, 0)
        saved = {
            "nodes": [
                {"id": stable_id, "label": "架构主题", "source": "架构.md", "origin": "generated", "x": 12},
                {"id": "manual-risk", "label": "人工风险", "origin": "manual"},
            ],
            "edges": [{
                "id": "manual-link", "source": stable_id, "target": "manual-risk",
                "label": "影响", "method": "manual", "auto_generated": False, "protected": True,
            }],
        }
        first_payload = {
            "ids": ["chunk-v1", "chunk-old"],
            "documents": ["# 部署架构\n当前网关部署方案", "旧版架构"],
            "metadatas": [current_metadata, old_metadata],
            "vectors": [[1.0, 0.0], [0.9, 0.1]],
        }
        captured = {}

        def save(value):
            captured["graph"] = value
            return value

        with (
            patch("backend.knowledge_base.knowledge_graph.vector_store.get", return_value=first_payload),
            patch.object(graph, "_load_saved_graph", return_value=saved),
            patch.object(graph, "_load_imported_graph", return_value={}),
            patch.object(graph, "save_graph_data", side_effect=save),
        ):
            graph.build_graph()

        first = captured["graph"]
        self.assertIn(stable_id, {item["id"] for item in first["nodes"]})
        self.assertNotIn("chunk-old", {item.get("chunk_id") for item in first["nodes"]})
        self.assertIn("manual-link", {item["id"] for item in first["edges"]})

        next_metadata = {**current_metadata, "chunk_id": "chunk-v2", "version_id": "version-v2"}
        next_payload = {
            "ids": ["chunk-v2"],
            "documents": ["# 部署架构\n新版网关部署方案"],
            "metadatas": [next_metadata],
            "vectors": [[1.0, 0.0]],
        }
        with (
            patch("backend.knowledge_base.knowledge_graph.vector_store.get", return_value=next_payload),
            patch.object(graph, "_load_saved_graph", return_value=first),
            patch.object(graph, "_load_imported_graph", return_value={}),
            patch.object(graph, "save_graph_data", side_effect=save),
        ):
            graph.build_graph()

        rebuilt = captured["graph"]
        stable_node = next(item for item in rebuilt["nodes"] if item["id"] == stable_id)
        self.assertEqual(stable_node["version_id"], "version-v2")
        self.assertEqual(stable_node["x"], 12)
        self.assertIn("manual-link", {item["id"] for item in rebuilt["edges"]})

    def test_project_context_excludes_noncurrent_and_expired_assets(self):
        repository = FakeRepository(documents=[
            {
                "document_id": "current", "source_file": "当前.md", "stored_file": "当前.md",
                "status": "active", "is_current_version": True,
            },
            {
                "document_id": "old", "source_file": "旧版.md", "stored_file": "旧版.md",
                "status": "active", "is_current_version": False,
            },
            {
                "document_id": "expired", "source_file": "过期.md", "stored_file": "过期.md",
                "status": "active", "is_current_version": True, "valid_until": "2025-01-01T00:00:00+00:00",
            },
        ])
        context = ProjectContext()
        with patch("backend.storage.get_repository", return_value=repository):
            documents = context.list_documents()
        self.assertEqual([item["document_id"] for item in documents], ["current"])

    def test_project_context_does_not_reuse_cache_after_acl_or_lifecycle_revocation(self):
        repository = FakeRepository(documents=[{
            "document_id": "visible", "source_file": "可见.md", "stored_file": "可见.md",
            "status": "active", "is_current_version": True, "acl_revision": 1,
        }])
        context = ProjectContext()
        with patch("backend.storage.get_repository", return_value=repository):
            self.assertEqual(len(context.list_documents()), 1)
            repository._documents = []
            self.assertEqual(context.list_documents(), [])

    def test_graph_requires_asset_version_or_explicit_policy_for_non_generated_nodes(self):
        repository = FakeRepository(
            documents=[{
                "document_id": "doc-current", "source_file": "同名.json", "stored_file": "同名.json",
                "asset_id": "asset-current", "version_id": "version-current", "status": "active",
            }],
            active_ids={"chunk-current"},
        )
        graph = KnowledgeGraph()
        payload = {
            "nodes": [
                {"id": "generated", "label": "自动节点", "origin": "generated", "chunk_id": "chunk-current"},
                {"id": "filename-only", "label": "仅同名", "origin": "imported", "source": "同名.json"},
                {"id": "bound", "label": "当前绑定", "origin": "imported", "asset_id": "asset-current", "version_id": "version-current"},
                {"id": "stale", "label": "旧版绑定", "origin": "imported", "asset_id": "asset-current", "version_id": "version-old"},
                {"id": "policy", "label": "策略节点", "origin": "manual", "access_policy_id": "policy-allowed"},
                {"id": "denied", "label": "拒绝节点", "origin": "manual", "access_policy_id": "policy-denied"},
            ],
            "edges": [
                {"id": "allowed-edge", "source": "bound", "target": "policy", "label": "关联"},
                {"id": "denied-edge", "source": "bound", "target": "denied", "label": "关联"},
            ],
        }
        identity = SimpleNamespace(project_id="project-default", user_id="user-1", role="member", is_system=False)
        with (
            patch("backend.storage.get_repository", return_value=repository),
            patch("backend.auth_context.get_current_identity", return_value=identity),
        ):
            result = graph._filter_inactive_sources(payload)

        self.assertEqual(
            {item["id"] for item in result["nodes"]},
            {"generated", "bound", "policy"},
        )
        self.assertEqual([item["id"] for item in result["edges"]], ["allowed-edge"])

    def test_json_graph_inherits_authoritative_asset_binding(self):
        graph = KnowledgeGraph()
        result = graph.graph_from_json_content(
            '{"nodes":[{"id":"n1","label":"节点一"}],"edges":[]}',
            "graph.json",
            {"asset_id": "asset-json", "version_id": "version-json", "access_policy_id": "policy-json"},
        )
        self.assertEqual(result["nodes"][0]["asset_id"], "asset-json")
        self.assertEqual(result["nodes"][0]["version_id"], "version-json")
        self.assertEqual(result["nodes"][0]["access_policy_id"], "policy-json")

    def test_role_readiness_has_explainable_weighted_dimensions(self):
        template = {
            "role": "开发工程师",
            "topics": [
                {"topic_id": "architecture", "label": "技术架构", "weight": 2},
                {"topic_id": "release", "label": "发布流程", "weight": 1},
            ],
        }
        assets = [{
            "asset_id": "asset-architecture",
            "version_id": "version-2",
            "title": "系统架构说明",
            "topics": ["architecture"],
            "status": "active",
            "is_current_version": True,
            "authority_score": 0.8,
            "freshness_score": 0.4,
            "traceability_score": 1.0,
            "applicable_roles": ["开发工程师"],
        }, {
            "asset_id": "asset-revoked",
            "topics": ["release"],
            "status": "revoked",
            "is_current_version": True,
            "authority_score": 1.0,
            "freshness_score": 1.0,
            "traceability_score": 1.0,
        }]

        result = calculate_role_readiness(
            template,
            assets,
            now=datetime(2026, 8, 22, tzinfo=timezone.utc),
        )

        self.assertEqual(result["readiness"], 56.0)
        self.assertEqual(result["dimensions"]["topic_coverage"]["score"], 66.7)
        self.assertEqual(result["dimensions"]["authority"]["score"], 53.3)
        self.assertEqual(result["dimensions"]["freshness"]["score"], 26.7)
        self.assertEqual(result["dimensions"]["traceability"]["score"], 66.7)
        self.assertEqual(result["input_asset_count"], 2)
        self.assertEqual(result["eligible_asset_count"], 1)
        self.assertEqual(result["excluded_asset_count"], 1)
        self.assertEqual(result["gaps"], [{"topic_id": "release", "label": "发布流程", "weight": 1.0}])
        evidence = result["topics"][0]["evidence"][0]
        self.assertEqual(evidence["asset_id"], "asset-architecture")
        self.assertEqual(evidence["version_id"], "version-2")

    def test_role_readiness_accepts_sqlite_asset_shape(self):
        result = calculate_role_readiness(
            {"role": "开发工程师", "topics": [{"topic_id": "架构", "weight": 1}]},
            [{
                "asset_id": "asset-sqlite",
                "current_version_id": "version-current",
                "primary_source_id": "source-primary",
                "title": "架构说明",
                "categories": ["架构"],
                "status": "active",
                "authority_level": 80,
                "review_due_at": "2027-01-01T00:00:00+00:00",
                "external_key": "feishu:message:1",
                "applicable_roles": [{"role": "开发工程师", "requirement_level": "required", "weight": 100}],
            }],
            now=datetime(2026, 8, 22, tzinfo=timezone.utc),
        )

        self.assertEqual(result["readiness"], 96.0)
        self.assertEqual(result["dimensions"]["authority"]["score"], 80.0)
        evidence = result["topics"][0]["evidence"][0]
        self.assertEqual(evidence["version_id"], "version-current")
        self.assertEqual(evidence["source_id"], "source-primary")


if __name__ == "__main__":
    unittest.main()
