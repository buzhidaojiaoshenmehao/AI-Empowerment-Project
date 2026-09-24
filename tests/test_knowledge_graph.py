import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document

from backend.config import settings
from backend.knowledge_base.knowledge_graph import KnowledgeGraph
from backend.knowledge_base.vector_store import LocalEmbeddings, VectorStore, vector_store
from backend.storage import get_repository, reset_repository_for_tests


class KnowledgeGraphRelationTest(unittest.TestCase):
    def test_legacy_long_chunk_expands_to_chinese_topic_nodes_with_structure_edges(self):
        long_content = "\n\n".join(
            f"## 主题 {index}\n这是第 {index} 个项目知识主题，包含流程、责任人和验收要求。"
            for index in range(1, 30)
        )
        payload = {
            "ids": ["legacy_chunk"],
            "documents": [long_content],
            "metadatas": [{
                "chunk_id": "legacy_chunk",
                "document_id": "document_architecture",
                "source_file": "架构说明.md",
                "category": "eef_standard",
                "asset_id": "asset_architecture",
                "chunk_index": 0,
            }],
            "vectors": [[1.0, 0.0, 0.0]],
        }
        graph = KnowledgeGraph()
        with (
            patch.object(vector_store, "get", return_value=payload),
            patch.object(vector_store, "_embeddings", LocalEmbeddings()),
            patch.object(graph, "_filter_inactive_sources", side_effect=lambda value: value),
            patch.object(graph, "_load_saved_graph", return_value={}),
            patch.object(graph, "_load_imported_graph", return_value={}),
            patch.object(graph, "save_graph_data", side_effect=lambda value: value),
        ):
            graph.build_graph()
            result = graph.get_graph_data()

        self.assertGreater(len(result["nodes"]), 2)
        self.assertEqual({node["group"] for node in result["nodes"]}, {"行业标准与法规"})
        self.assertTrue(any(edge["relation"] == "document_sequence" for edge in result["edges"]))
        self.assertTrue(any("相邻知识片段" in edge["evidence"] for edge in result["edges"]))

    def test_empty_graph_does_not_show_demo_business_nodes(self):
        graph = KnowledgeGraph()
        with (
            patch.object(graph, "_load_saved_graph", return_value={}),
            patch.object(graph, "_load_imported_graph", return_value={}),
        ):
            result = graph.get_graph_data()
        self.assertEqual(result["nodes"], [])
        self.assertEqual(result["edges"], [])
        self.assertFalse(result["starter"])

    def test_graph_build_propagates_vector_failures(self):
        graph = KnowledgeGraph()
        with patch.object(vector_store, "get", side_effect=RuntimeError("index unavailable")):
            with self.assertRaisesRegex(RuntimeError, "向量索引不可用"):
                graph.build_graph()

    def test_hybrid_relations_are_explainable_and_manual_edges_survive_rebuild(self):
        payload = {
            "ids": ["api_design", "api_spec", "acceptance"],
            "documents": [
                "接口鉴权方案。接口鉴权流程需要在联调前完成，记录接口风险。",
                "接口鉴权规范。接口鉴权流程由网关统一校验，记录接口风险。",
                "项目验收报告。验收范围、验收结论和交付清单。",
            ],
            "metadatas": [
                {"source_file": "接口方案.md", "category": "架构"},
                {"source_file": "接口规范.md", "category": "架构"},
                {"source_file": "验收报告.md", "category": "验收"},
            ],
            "vectors": [[1.0, 0.0, 0.0], [0.97, 0.12, 0.0], [0.0, 0.0, 1.0]],
        }

        previous_directory = os.getcwd()
        with tempfile.TemporaryDirectory() as temp_directory:
            os.chdir(temp_directory)
            try:
                graph = KnowledgeGraph()
                with (
                    patch.object(vector_store, "get", return_value=payload),
                    patch.object(graph, "_filter_inactive_sources", side_effect=lambda value: value),
                ):
                    graph.build_graph()
                    generated = graph.get_graph_data()

                    automatic_edges = [edge for edge in generated["edges"] if edge["auto_generated"]]
                    self.assertTrue(automatic_edges)
                    self.assertTrue(any("语义相似度" in edge["evidence"] for edge in automatic_edges))
                    self.assertTrue(all(0 <= edge["confidence"] <= 1 for edge in automatic_edges))

                    generated["edges"].append({
                        "id": "manual_api_to_acceptance",
                        "source": "api_design",
                        "target": "acceptance",
                        "label": "依赖",
                        "relation": "依赖",
                        "width": 3,
                        "confidence": 0.95,
                        "evidence": "架构方案是验收范围的人工确认依据",
                        "method": "manual",
                        "auto_generated": False,
                        "protected": True,
                    })
                    graph.save_graph_data(generated)
                    graph.build_graph()

                    rebuilt = graph.get_graph_data()
                    preserved = next(edge for edge in rebuilt["edges"] if edge["id"] == "manual_api_to_acceptance")
                    self.assertEqual(preserved["label"], "依赖")
                    self.assertTrue(preserved["protected"])
            finally:
                os.chdir(previous_directory)

    def test_remove_source_cleans_saved_and_imported_graphs(self):
        previous_directory = os.getcwd()
        with tempfile.TemporaryDirectory() as temp_directory:
            os.chdir(temp_directory)
            try:
                graph = KnowledgeGraph()
                with patch.object(graph, "_filter_inactive_sources", side_effect=lambda value: value):
                    graph.import_graph_data({
                        "nodes": [
                            {"id": "json_a", "label": "JSON 节点", "source": "knowledge_graph.json"},
                            {"id": "json_b", "label": "JSON 关联节点", "source": "knowledge_graph.json"},
                        ],
                        "edges": [
                            {"id": "json_relation", "source": "json_a", "target": "json_b", "label": "引用"},
                        ],
                    })
                    saved = graph.get_graph_data()
                    saved["nodes"].append({"id": "other", "label": "其他节点", "source": "其他资料.md", "origin": "manual"})
                    saved["nodes"].append({"id": "manual", "label": "人工节点", "origin": "manual"})
                    saved["edges"].append({
                        "id": "manual_to_other", "source": "manual", "target": "other", "label": "依赖",
                        "method": "manual", "auto_generated": False, "protected": True,
                    })
                    graph.save_graph_data(saved)

                    result = graph.remove_source("knowledge_graph.json")

                    cleaned = graph.get_graph_data()
                self.assertEqual(result["nodes"], 2)
                self.assertEqual(result["imported_nodes"], 2)
                self.assertEqual({node["id"] for node in cleaned["nodes"]}, {"other", "manual"})
                self.assertEqual([edge["id"] for edge in cleaned["edges"]], ["manual_to_other"])
                self.assertFalse(graph._load_imported_graph()["nodes"])
            finally:
                os.chdir(previous_directory)

    def test_revoked_source_is_hidden_when_graph_rebuild_fails(self):
        original_database = settings.DATABASE_PATH
        original_chroma = settings.CHROMA_PERSIST_DIR
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings.DATABASE_PATH = str(root / "data" / "app.db")
            settings.CHROMA_PERSIST_DIR = str(root / "chroma_db")
            reset_repository_for_tests()
            try:
                store = VectorStore()
                store._embeddings = LocalEmbeddings()
                store.add_documents([Document(
                    page_content="仍然有效的架构知识",
                    metadata={"source_file": "有效.md", "stored_file": "stored_有效.md"},
                )])
                store.add_documents([Document(
                    page_content="撤销后不可展示的敏感决策",
                    metadata={"source_file": "撤销.md", "stored_file": "stored_撤销.md"},
                )])
                graph = KnowledgeGraph()
                with patch.object(vector_store, "get", return_value=store.get(include_vectors=True)):
                    graph.build_graph()

                get_repository().delete_document("撤销.md")
                with patch.object(graph, "build_graph", side_effect=RuntimeError("projection failed")):
                    with self.assertRaisesRegex(RuntimeError, "projection failed"):
                        graph.build_graph()

                visible = graph.get_graph_data()
                self.assertTrue(any(node.get("source") == "有效.md" for node in visible["nodes"]))
                self.assertFalse(any(node.get("source") == "撤销.md" for node in visible["nodes"]))

                with patch.object(vector_store, "similarity_search", return_value=[Document(
                    page_content="仍然有效的架构知识",
                    metadata={"source_file": "有效.md"},
                )]):
                    trace = graph.trace_decision("架构知识")
                self.assertFalse(any(
                    item.get("source") == "撤销.md"
                    for item in trace.get("related_knowledge", [])
                ))
            finally:
                settings.DATABASE_PATH = original_database
                settings.CHROMA_PERSIST_DIR = original_chroma
                reset_repository_for_tests()


if __name__ == "__main__":
    unittest.main()
