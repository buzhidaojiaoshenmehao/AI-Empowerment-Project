import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from backend.auth import AuthService, AuthenticationError, ROLE_PERMISSIONS
from backend.config import settings
from backend.auth_context import Identity, reset_current_identity, set_current_identity
from backend.auth_middleware import required_permission, successful_access_action
from backend.knowledge_base.knowledge_graph import KnowledgeGraph
from backend.knowledge_base.vector_store import LocalEmbeddings, VectorStore
from backend.storage.database import Database
from backend.storage.repository import StorageRepository


class IdentityAndAccessTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repository = StorageRepository(Database(Path(self.temp.name) / "app.db"))
        self.auth = AuthService(lambda: self.repository)
        self.demo_mode = patch.object(settings, "DEMO_SIMPLE_AUTH", False)
        self.demo_mode.start()

    def tearDown(self):
        self.demo_mode.stop()
        self.temp.cleanup()

    def _member(self, email, name, role="project_member"):
        return self.repository.provision_project_member(email, name, role)

    @staticmethod
    def _identity(member):
        role = str(member["role"])
        return Identity(
            user_id=str(member["user_id"]), email=str(member["email"]),
            display_name=str(member["display_name"]), organization_id="org_default",
            project_id=str(member["project_id"]), project_name=str(member["project_name"]),
            role=role, permissions=ROLE_PERMISSIONS[role], session_id="test",
        )

    def _document(self, stored_file, text, **metadata):
        meta = {
            "source_file": stored_file, "stored_file": stored_file,
            "category": "测试", "categories": ["测试"], **metadata,
        }
        return self.repository.upsert_document_chunks(
            [SimpleNamespace(page_content=text, metadata=meta)], [f"vector-{stored_file}"],
        )

    def test_invitation_activation_login_and_revocation(self):
        member = self._member("member@example.com", "成员")
        invitation = self.auth.create_invitation(member["user_id"])
        activated = self.auth.activate(invitation["token"], "Strong-password-2026")
        self.assertEqual(activated["email"], "member@example.com")

        result = self.auth.login("MEMBER@example.com", "Strong-password-2026")
        self.assertEqual(result["identity"].role, "project_member")
        self.assertTrue(self.auth.validate_csrf(result["session_token"], result["csrf_token"]))

        self.repository.update_project_membership(member["user_id"], status="revoked")
        self.assertIsNone(self.auth.identity_from_token(result["session_token"]))

    def test_demo_mode_activates_members_with_email_prefix_password(self):
        member = self._member("demo-member@example.com", "演示成员")
        with patch.object(settings, "DEMO_SIMPLE_AUTH", True):
            self.assertEqual(self.auth.ensure_demo_accounts(), 1)
            self.assertEqual(self.auth.ensure_demo_accounts(), 0)
            result = self.auth.login("demo-member@example.com", "demo-member")
            self.assertEqual(result["identity"].user_id, member["user_id"])
            with self.assertRaisesRegex(AuthenticationError, "无需激活"):
                self.auth.activate("unused", "Strong-password-2026")

    def test_repeated_bad_password_locks_account(self):
        member = self._member("locked@example.com", "锁定测试")
        invitation = self.auth.create_invitation(member["user_id"])
        self.auth.activate(invitation["token"], "Strong-password-2026")
        for _ in range(5):
            with self.assertRaises(AuthenticationError):
                self.auth.login("locked@example.com", "wrong-password")
        with self.assertRaisesRegex(AuthenticationError, "临时锁定"):
            self.auth.login("locked@example.com", "Strong-password-2026")

    def test_role_matrix_protects_admin_and_publish_operations(self):
        self.assertEqual(required_permission("/api/settings/llm", "POST"), "settings.manage")
        self.assertEqual(required_permission("/api/settings/knowledge-library/reset", "POST"), "settings.manage")
        self.assertEqual(required_permission("/api/governance/policy", "GET"), "settings.manage")
        self.assertEqual(required_permission("/api/ops/overview", "GET"), "settings.manage")
        self.assertEqual(required_permission("/api/ops/capacity/validate", "POST"), "settings.manage")
        self.assertEqual(required_permission("/api/documents/uploads", "POST"), "knowledge.upload")
        self.assertEqual(required_permission("/api/governance/audit", "GET"), "audit.read")
        self.assertEqual(required_permission("/api/projects/current/members/abc", "PATCH"), "settings.manage")
        self.assertEqual(required_permission("/api/documents/file.md/access", "PATCH"), "knowledge.manage")
        self.assertEqual(required_permission("/api/knowledge/assets", "GET"), "knowledge.read")
        self.assertEqual(required_permission("/api/knowledge/assets/asset_1", "PATCH"), "knowledge.manage")
        self.assertEqual(
            required_permission("/api/knowledge/assets/asset_1/transition", "POST"),
            "knowledge.manage",
        )
        self.assertEqual(required_permission("/api/resignation/case/accept", "POST"), "handover.accept")
        self.assertEqual(required_permission("/api/resignation/case/cancel", "POST"), "handover.submit")
        self.assertEqual(
            required_permission("/api/resignation/case/items/item/draft", "PUT"), "handover.submit",
        )
        self.assertEqual(
            required_permission("/api/resignation/case/items/item/attachments", "POST"), "handover.submit",
        )
        self.assertEqual(
            required_permission("/api/resignation/case/items/item/actions", "POST"), "handover.accept",
        )
        self.assertEqual(
            required_permission("/api/resignation/case/risks/risk/actions", "POST"), "handover.accept",
        )
        self.assertEqual(required_permission("/api/resignation/case/risks", "POST"), "handover.accept")
        self.assertEqual(
            required_permission("/api/resignation/case/inventory/refresh", "POST"), "handover.accept",
        )
        self.assertEqual(
            required_permission("/api/resignation/case/inventory/item/actions", "POST"), "handover.accept",
        )
        self.assertEqual(required_permission("/api/resignation/case/complete", "POST"), "handover.manage")
        self.assertEqual(required_permission("/api/knowledge/tasks", "GET"), "task.read")
        self.assertEqual(required_permission("/api/knowledge/tasks/kt_1/actions", "POST"), "task.read")
        self.assertEqual(required_permission("/api/onboarding/templates", "GET"), "onboarding.read")
        self.assertEqual(required_permission("/api/onboarding/templates/ort_1", "PUT"), "onboarding.manage")
        self.assertEqual(required_permission("/api/onboarding/plans", "POST"), "onboarding.manage")
        self.assertEqual(
            required_permission("/api/onboarding/plans/plan_1/items/item_1", "PATCH"),
            "onboarding.progress",
        )
        self.assertIn("onboarding.progress", ROLE_PERMISSIONS["new_member"])
        self.assertIn("handover.accept", ROLE_PERMISSIONS["new_member"])
        self.assertNotIn("onboarding.manage", ROLE_PERMISSIONS["new_member"])
        self.assertNotIn("settings.manage", ROLE_PERMISSIONS["project_member"])

    def test_successful_access_audit_scope_excludes_request_content(self):
        self.assertEqual(successful_access_action("/api/chat", "POST"), "access.knowledge_answer")
        self.assertEqual(
            successful_access_action("/api/documents/preview/demo.md", "GET"),
            "access.document_preview",
        )
        self.assertIsNone(successful_access_action("/api/health", "GET"))
        self.assertIn("task.read", ROLE_PERMISSIONS["project_member"])
        self.assertNotIn("task.manage", ROLE_PERMISSIONS["project_member"])
        self.assertIn("knowledge.manage", ROLE_PERMISSIONS["knowledge_operator"])
        self.assertIn("task.manage", ROLE_PERMISSIONS["knowledge_operator"])

    def test_document_scope_is_inherited_by_retrieval_and_graph(self):
        alice = self._member("alice@example.com", "Alice")
        bob = self._member("bob@example.com", "Bob")
        alice_identity = self._identity(alice)
        bob_identity = self._identity(bob)

        public = self._document("public.md", "项目公共发布流程")
        private = self._document(
            "private.md", "董事会私密预算 secret alpha",
            visibility="private", owner_user_id=alice["user_id"],
        )
        policy = self.repository.create_access_policy(
            "Alice 专属", user_ids=[alice["user_id"]], created_by="test",
        )
        restricted = self._document(
            "restricted.md", "受限架构决策",
            visibility="restricted", access_policy_id=policy["access_policy_id"],
        )

        token = set_current_identity(bob_identity)
        try:
            self.assertEqual(self.repository.accessible_document_ids(), {public["document_id"]})

            store = VectorStore()
            store._loaded = True
            store._embeddings = LocalEmbeddings()
            chunks = self.repository.document_chunks()
            store._ids = [item["chunk_id"] for item in chunks]
            store._texts = [item["text"] for item in chunks]
            store._metas = [item["metadata"] for item in chunks]
            store._vecs = [np.array(store.embeddings.embed_query(item["text"]), dtype=np.float32) for item in chunks]
            with patch("backend.storage.get_repository", return_value=self.repository):
                results = store.similarity_search("secret alpha", k=5, score_threshold=0.0)
                self.assertEqual([item.metadata["stored_file"] for item in results], ["public.md"])

                graph = KnowledgeGraph()._filter_inactive_sources({
                    "nodes": [
                        {
                            "id": item["chunk_id"], "label": item["metadata"]["source_file"],
                            "source": item["metadata"]["source_file"], "origin": "generated",
                        }
                        for item in chunks
                    ],
                    "edges": [],
                })
                self.assertEqual([node["source"] for node in graph["nodes"]], ["public.md"])
        finally:
            reset_current_identity(token)

        token = set_current_identity(alice_identity)
        try:
            self.assertEqual(
                self.repository.accessible_document_ids(),
                {public["document_id"], private["document_id"], restricted["document_id"]},
            )
        finally:
            reset_current_identity(token)

    def test_management_role_does_not_bypass_private_or_restricted_source_acl(self):
        owner = self._member("owner@example.com", "Owner")
        operator = self._member(
            "operator@example.com", "Operator", role="knowledge_operator",
        )
        admin = self._member(
            "admin@example.com", "Admin", role="project_admin",
        )
        private = self._document(
            "private-governance.md", "仅来源责任人可见",
            visibility="private", owner_user_id=owner["user_id"],
        )
        policy = self.repository.create_access_policy(
            "Owner only", user_ids=[owner["user_id"]], created_by="test",
        )
        restricted = self._document(
            "restricted-governance.md", "仅策略成员可见",
            visibility="restricted", access_policy_id=policy["access_policy_id"],
        )

        expected_denied = set()
        for member in (operator, admin):
            identity = self._identity(member)
            self.assertEqual(
                self.repository.accessible_document_ids(identity), expected_denied,
            )
            token = set_current_identity(identity)
            try:
                self.assertEqual(self.repository.list_documents(), [])
                self.assertIsNone(
                    self.repository.get_document(private["stored_file"]),
                )
            finally:
                reset_current_identity(token)

        owner_identity = self._identity(owner)
        self.assertEqual(
            self.repository.accessible_document_ids(owner_identity),
            {private["document_id"], restricted["document_id"]},
        )


if __name__ == "__main__":
    unittest.main()
