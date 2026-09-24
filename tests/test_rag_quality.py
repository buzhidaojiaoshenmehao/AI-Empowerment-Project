import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from langchain_core.documents import Document

from backend.auth import ROLE_PERMISSIONS
from backend.auth_context import Identity, reset_current_identity, set_current_identity
from backend.auth_middleware import required_permission
from backend.knowledge_assets import KnowledgeAssetService
from backend.knowledge_base.vector_store import LocalEmbeddings
from backend.knowledge_tasks import KnowledgeTaskService
from backend.rag_quality import RAGQualityService
from backend.storage.database import Database
from backend.storage.repository import StorageRepository


class QueryRetriever:
    def __init__(self, document):
        self.document = document

    def retrieve(self, query, k=4):
        return [] if "不存在" in query else [self.document]


class RAGQualityWorkflowTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repository = StorageRepository(Database(Path(self.temp.name) / "app.db"))
        member = self.repository.provision_project_member(
            "rag-admin@example.com", "RAG 管理员", "project_admin",
        )
        self.identity = Identity(
            user_id=member["user_id"], email=member["email"], display_name=member["display_name"],
            organization_id="org_default", project_id=member["project_id"],
            project_name=member["project_name"], role=member["role"],
            permissions=ROLE_PERMISSIONS[member["role"]], session_id="test",
        )
        self.identity_token = set_current_identity(self.identity)
        asset_service = KnowledgeAssetService(lambda: self.repository)
        source_document = SimpleNamespace(page_content="部署步骤包括备份、发布、健康检查和回滚验证。", metadata={})
        prepared = asset_service.prepare_documents(
            [source_document], source_type="manual_upload", external_key="deploy.md",
            title="部署手册", actor=self.identity.user_id, stored_file="deploy-v1.md",
            categories=["项目规范"],
        )
        stored = self.repository.upsert_document_chunks([source_document], ["vec-deploy-v1"])
        asset_service.record_projection(prepared["version_id"], "vector", succeeded=True)
        asset_service.publish(prepared, actor=self.identity.user_id)
        source_document.metadata.update({
            "asset_id": prepared["asset_id"], "version_id": prepared["version_id"],
            "document_id": stored["document_id"], "chunk_id": stored["chunk_ids"][0],
            "source_file": "deploy.md",
        })
        self.document = Document(
            page_content=source_document.page_content, metadata=dict(source_document.metadata),
        )
        self.service = RAGQualityService(
            lambda: self.repository, QueryRetriever(self.document),
        )
        self.tasks = KnowledgeTaskService(lambda: self.repository)

    def tearDown(self):
        reset_current_identity(self.identity_token)
        self.temp.cleanup()

    def test_schema_permission_and_authoritative_citation_contract(self):
        required = {
            "rag_evaluation_sets", "rag_evaluation_cases", "rag_evaluation_runs",
            "rag_evaluation_results", "rag_answer_feedback",
        }
        with self.repository.database.transaction() as connection:
            tables = {str(row[0]) for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )}
        self.assertTrue(required.issubset(tables))
        self.assertEqual(required_permission("/api/rag/feedback", "POST"), "knowledge.read")
        self.assertEqual(required_permission("/api/rag/evaluation-sets", "GET"), "knowledge.read")
        self.assertEqual(required_permission("/api/rag/evaluation-sets", "POST"), "knowledge.manage")

        evidence = self.service.retrieve_evidence("部署流程是什么", self.identity)
        self.assertTrue(evidence["sufficient"])
        self.assertEqual(evidence["citation_validity_rate"], 1.0)
        self.assertTrue(evidence["citations"][0]["valid"])
        self.assertEqual(evidence["citations"][0]["asset_id"], self.document.metadata["asset_id"])

        invalid = self.service.validate_citations([{
            **evidence["citations"][0], "chunk_id": "removed-chunk",
        }], self.identity)
        self.assertFalse(invalid[0]["valid"])
        self.assertIn("inactive_chunk", invalid[0]["invalid_reasons"])
        invalid_document = Document(
            page_content=self.document.page_content,
            metadata={**self.document.metadata, "chunk_id": "removed-chunk"},
        )
        invalid_evidence = RAGQualityService(
            lambda: self.repository, QueryRetriever(invalid_document),
        ).retrieve_evidence("部署流程是什么", self.identity)
        self.assertTrue(invalid_evidence["no_answer"])
        self.assertEqual(invalid_evidence["documents"], [])

    def test_versioned_evaluation_run_covers_answer_and_no_answer_cases(self):
        evaluation_set = self.service.create_evaluation_set({
            "name": "项目真实问题", "version": 1, "description": "部署策略基线",
        }, self.identity)
        evaluation_set_id = evaluation_set["evaluation_set_id"]
        self.service.add_evaluation_case(evaluation_set_id, {
            "question": "部署流程是什么", "scenario": "process",
            "expected_points": ["部署步骤"], "allowed_source_files": ["deploy.md"],
        }, self.identity)
        self.service.add_evaluation_case(evaluation_set_id, {
            "question": "不存在的客户合同金额是多少", "scenario": "no_answer",
            "expect_no_answer": True,
        }, self.identity)

        run = self.service.run_evaluation(evaluation_set_id, self.identity)

        self.assertEqual(run["metrics"]["case_count"], 2)
        self.assertEqual(run["metrics"]["passed_count"], 2)
        self.assertEqual(run["metrics"]["citation_validity_rate"], 1.0)
        self.assertEqual(run["metrics"]["permission_leak_count"], 0)
        self.assertEqual(run["metrics"]["no_answer_accuracy"], 1.0)
        detail = self.service.get_evaluation_set(evaluation_set_id)
        self.assertEqual(detail["case_count"], 2)
        self.assertFalse(detail["ready"])
        self.assertEqual(detail["runs"][0]["strategy"]["evaluation_contract"], "retrieval_evidence_v1")
        self.assertEqual(
            self.repository.list_audit_events(action="rag.evaluation_set_created")["total"],
            1,
        )
        run_audit = self.repository.list_audit_events(action="rag.evaluation_run_completed")
        self.assertEqual(run_audit["total"], 1)
        self.assertEqual(run_audit["items"][0]["object_id"], run["evaluation_run_id"])
        self.assertEqual(run_audit["items"][0]["detail"]["case_count"], 2)
        self.assertEqual(run_audit["items"][0]["detail"]["permission_leak_count"], 0)

    def test_pdf_compatibility_characters_match_normal_user_queries(self):
        embeddings = LocalEmbeddings()
        self.assertEqual(
            embeddings.embed_query("使用云端地图数据"),
            embeddings.embed_query("使⽤云端地图数据"),
        )
        self.assertTrue(self.service._point_matches(
            "联网时使用云端地图数据",
            "联⽹时使⽤云端地图数据进行描画",
        ))

    def test_negative_feedback_is_idempotent_and_creates_knowledge_task(self):
        payload = {
            "answer_id": "raga-demo", "question": "部署流程是什么", "helpful": False,
            "reason": "knowledge_outdated", "note": "回滚步骤已经变化",
            "citations": self.service.citations_for_documents([self.document], self.identity),
        }
        first = self.service.record_feedback(payload, self.identity, self.tasks)
        second = self.service.record_feedback(payload, self.identity, self.tasks)

        self.assertEqual(first["feedback"]["feedback_id"], second["feedback"]["feedback_id"])
        self.assertEqual(first["knowledge_task"]["task_id"], second["knowledge_task"]["task_id"])
        task = self.tasks.get(first["knowledge_task"]["task_id"], self.identity)
        self.assertEqual(task["task_type"], "qa_feedback")
        self.assertEqual(task["linked_asset_id"], self.document.metadata["asset_id"])
        self.assertEqual(task["occurrence_count"], 2)


if __name__ == "__main__":
    unittest.main()
