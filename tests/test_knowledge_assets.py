import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException
from backend.auth import ROLE_PERMISSIONS
from backend.auth_context import Identity
from backend.knowledge_assets import KnowledgeAssetService, ProjectionRepairBusyError
from backend.storage.database import Database
from backend.storage.repository import StorageRepository


class KnowledgeAssetLifecycleTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repository = StorageRepository(Database(Path(self.temp.name) / "app.db"))
        self.service = KnowledgeAssetService(lambda: self.repository)

    def tearDown(self):
        self.temp.cleanup()

    def _publish(self, text: str, stored_file: str, **kwargs):
        document = SimpleNamespace(page_content=text, metadata={})
        prepared = self.service.prepare_documents(
            [document], source_type=kwargs.pop("source_type", "manual_upload"),
            external_key=kwargs.pop("external_key", "guide.md"), title=kwargs.pop("title", "guide.md"),
            actor=kwargs.pop("actor", "tester"), stored_file=stored_file,
            categories=kwargs.pop("categories", ["项目规范"]), **kwargs,
        )
        if prepared.get("duplicate"):
            return prepared, None
        stored = self.repository.upsert_document_chunks([document], [f"vector-{stored_file}"])
        self.service.record_projection(prepared["version_id"], "vector", succeeded=True)
        asset = self.service.publish(prepared, actor="tester")
        return prepared, {"asset": asset, "stored": stored}

    @staticmethod
    def _identity(member):
        return Identity(
            user_id=member["user_id"], email=member["email"], display_name=member["display_name"],
            organization_id="org_default", project_id=member["project_id"],
            project_name=member["project_name"], role=member["role"],
            permissions=ROLE_PERMISSIONS[member["role"]], session_id="test",
        )

    def test_migration_v4_and_backfill_are_idempotent(self):
        tables = {
            "knowledge_sources", "knowledge_assets", "asset_versions", "asset_version_documents",
            "asset_categories", "asset_applicable_roles", "asset_projection_states", "domain_outbox",
            "knowledge_asset_aliases",
        }
        with self.repository.database.transaction() as connection:
            existing = {str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertTrue(tables.issubset(existing))

        legacy = SimpleNamespace(
            page_content="legacy body",
            metadata={"source_file": "legacy.md", "stored_file": "legacy.md", "categories": ["项目规范"]},
        )
        stored = self.repository.upsert_document_chunks([legacy], ["legacy-vector"])
        first = self.repository.backfill_knowledge_assets(force=True)
        second = self.repository.backfill_knowledge_assets(force=True)

        self.assertEqual(first["documents"], 1)
        self.assertEqual(second["documents"], 0)
        with self.repository.database.transaction() as connection:
            self.assertEqual(connection.execute(
                "SELECT COUNT(*) FROM asset_version_documents WHERE document_id=?", (stored["document_id"],)
            ).fetchone()[0], 1)
            self.assertEqual(connection.execute(
                "SELECT COUNT(*) FROM asset_versions WHERE status='active'"
            ).fetchone()[0], 1)

    def test_unmatched_legacy_feishu_asset_is_registered_fail_closed(self):
        self.repository.save_feishu_workspace({
            "groups": [], "messages": [], "candidates": [], "audit": [],
            "assets": [{
                "asset_id": "fa_legacy_only", "candidate_id": "fc_legacy_only",
                "status": "published", "stored_file": "missing_projection.md",
                "source_file": "历史飞书知识.md",
            }],
            "diagnostics": {}, "_revision": 0,
        })

        result = self.repository.backfill_knowledge_assets(force=True)
        asset_id = self.repository.resolve_asset_id("fa_legacy_only")
        asset = self.repository.get_asset(asset_id)

        self.assertEqual(result["assets"], 1)
        self.assertEqual(asset["status"], "draft")
        self.assertTrue(asset["metadata"]["repair_required"])
        self.assertEqual(self.repository.list_assets(), [])

    def test_backfill_skips_targeted_deleted_legacy_records(self):
        legacy = SimpleNamespace(
            page_content="deleted body",
            metadata={"source_file": "deleted.md", "stored_file": "deleted.md"},
        )
        stored = self.repository.upsert_document_chunks([legacy], ["deleted-vector"])
        self.repository.save_feishu_workspace({
            "groups": [], "messages": [], "candidates": [], "audit": [],
            "assets": [{
                "asset_id": "fa_deleted", "candidate_id": "fc_deleted",
                "status": "reverted", "stored_file": "",
                "targeted_deleted": True,
            }],
            "diagnostics": {}, "_revision": 0,
        })
        with self.repository.database.transaction(write=True) as connection:
            connection.execute(
                "UPDATE documents SET status='deleted',metadata_json=? WHERE document_id=?",
                ('{"targeted_deleted":true}', stored["document_id"]),
            )

        result = self.repository.backfill_knowledge_assets(force=True)

        self.assertEqual(result, {"documents": 0, "assets": 0, "versions": 0, "aliases": 0})
        self.assertEqual(self.repository.list_assets(include_inactive=True), [])

    def test_deleted_assets_require_explicit_tombstone_query(self):
        prepared, _ = self._publish("deleted body", "deleted-list.md")
        with self.repository.database.transaction(write=True) as connection:
            connection.execute(
                "UPDATE knowledge_assets SET status='deleted' WHERE asset_id=?",
                (prepared["asset_id"],),
            )

        self.assertEqual(self.repository.list_assets(include_inactive=True), [])
        tombstones = self.repository.list_assets(
            include_inactive=True, include_deleted=True,
        )
        self.assertEqual([item["asset_id"] for item in tombstones], [prepared["asset_id"]])

    def test_permanently_deleted_source_can_be_reuploaded_with_fresh_identity(self):
        original, _ = self._publish(
            "original body", "original-guide.md",
            external_key="reusable-guide.md", title="reusable-guide.md",
        )
        with self.repository.database.transaction(write=True) as connection:
            connection.execute(
                "UPDATE knowledge_sources SET external_key=?,display_name='[已删除来源]',status='deleted',"
                "metadata_json=? WHERE source_id=?",
                (f"deleted:{original['source_id']}", '{"targeted_deleted":true}', original["source_id"]),
            )
            connection.execute(
                "UPDATE knowledge_assets SET asset_key=?,title='[已删除资产]',status='deleted',metadata_json=? "
                "WHERE asset_id=?",
                (f"deleted:{original['asset_id']}", '{"targeted_deleted":true}', original["asset_id"]),
            )

        replacement = SimpleNamespace(page_content="replacement body", metadata={})
        prepared = self.service.prepare_documents(
            [replacement], source_type="manual_upload", external_key="reusable-guide.md",
            title="reusable-guide.md", actor="tester", stored_file="replacement-guide.md",
            categories=["项目规范"],
        )

        self.assertNotEqual(prepared["source_id"], original["source_id"])
        self.assertNotEqual(prepared["asset_id"], original["asset_id"])
        with self.repository.database.transaction() as connection:
            old_source = connection.execute(
                "SELECT status,external_key FROM knowledge_sources WHERE source_id=?",
                (original["source_id"],),
            ).fetchone()
            new_source = connection.execute(
                "SELECT status,external_key FROM knowledge_sources WHERE source_id=?",
                (prepared["source_id"],),
            ).fetchone()
        self.assertEqual(old_source["status"], "deleted")
        self.assertEqual(old_source["external_key"], f"deleted:{original['source_id']}")
        self.assertEqual(new_source["status"], "active")
        self.assertEqual(new_source["external_key"], "reusable-guide.md")

    def test_asset_api_excludes_deleted_records_unless_explicitly_requested(self):
        from backend import main

        identity = SimpleNamespace(can=lambda permission: permission == "knowledge.manage")
        processing = {
            "failed_version_count": 0,
            "repair_required_count": 0,
            "processing_count": 0,
        }
        with (
            patch.object(main, "_require_knowledge_permission", return_value=identity),
            patch.object(main.knowledge_asset_service, "list_assets", return_value=[]) as list_assets,
            patch.object(
                main,
                "get_repository",
                return_value=SimpleNamespace(summarize_asset_processing=lambda asset_ids: processing),
            ),
        ):
            result = asyncio.run(main.list_knowledge_assets(
                include_inactive=True, include_deleted=False,
            ))

        self.assertEqual(result["assets"], [])
        list_assets.assert_called_once_with(
            identity, include_inactive=True, include_deleted=False,
        )

    def test_vector_is_hard_publish_gate(self):
        document = SimpleNamespace(page_content="not embedded", metadata={})
        prepared = self.service.prepare_documents(
            [document], source_type="manual_upload", external_key="blocked.md", title="blocked.md",
            actor="tester", stored_file="v1_blocked.md",
        )
        self.repository.upsert_document_chunks([document], ["blocked-vector"])
        with self.assertRaisesRegex(ValueError, "向量索引"):
            self.service.publish(prepared, actor="tester")
        self.assertEqual(self.repository.active_chunk_ids(), set())

    def test_new_asset_uses_project_default_sensitivity(self):
        self.repository.update_data_governance_policy(
            {"default_sensitivity": "confidential"}, actor_user_id="admin",
        )
        prepared, result = self._publish("governed body", "governed.md")

        self.assertIsNotNone(result)
        asset = self.repository.get_asset(prepared["asset_id"])
        source = self.repository.get_source(prepared["source_id"])
        self.assertEqual(asset["sensitivity_level"], "confidential")
        self.assertEqual(source["sensitivity_level"], "confidential")

    def test_duplicate_publish_is_idempotent(self):
        first, result = self._publish("same body", "v1_guide.md")
        duplicate, duplicate_result = self._publish("same body", "v2_guide.md")

        self.assertIsNotNone(result)
        self.assertIsNone(duplicate_result)
        self.assertTrue(duplicate["duplicate"])
        self.assertEqual(first["version_id"], duplicate["version_id"])
        with self.repository.database.transaction() as connection:
            self.assertEqual(connection.execute(
                "SELECT COUNT(*) FROM asset_versions WHERE asset_id=?", (first["asset_id"],)
            ).fetchone()[0], 1)
            self.assertEqual(connection.execute(
                "SELECT COUNT(*) FROM domain_outbox WHERE aggregate_id=? AND event_type='knowledge.asset_activated'",
                (first["asset_id"],),
            ).fetchone()[0], 1)

    def test_new_version_switch_keeps_history_and_hides_old_chunks(self):
        first, first_result = self._publish("old body", "v1_guide.md")
        first_chunk = first_result["stored"]["chunk_ids"][0]
        second, second_result = self._publish("new body", "v2_guide.md")
        second_chunk = second_result["stored"]["chunk_ids"][0]

        versions = self.repository.list_asset_versions(first["asset_id"])
        self.assertEqual([item["status"] for item in versions], ["active", "superseded"])
        self.assertEqual(self.repository.active_chunk_ids([first_chunk, second_chunk]), {second_chunk})
        self.assertEqual(len(versions[1]["documents"]), 1)
        self.assertEqual(versions[1]["documents"][0]["status"], "superseded")
        self.assertEqual(second_result["asset"]["current_version_id"], second["version_id"])

    def test_asset_list_includes_current_version_source_documents_and_projections(self):
        prepared, result = self._publish("list body", "list-v1.md", external_key="list.md", title="list.md")

        assets = self.repository.list_assets()

        self.assertEqual(len(assets), 1)
        asset = assets[0]
        self.assertEqual(asset["asset_id"], prepared["asset_id"])
        self.assertEqual(asset["source"]["source_id"], prepared["source_id"])
        self.assertEqual(asset["current_version"]["version_id"], prepared["version_id"])
        self.assertEqual(asset["current_version"]["documents"][0]["document_id"], result["stored"]["document_id"])
        self.assertEqual(asset["projections"]["vector"]["status"], "ready")

    def test_revoke_is_fail_closed_before_projection_cleanup(self):
        prepared, result = self._publish("sensitive body", "v1_sensitive.md")
        chunk_id = result["stored"]["chunk_ids"][0]
        self.assertEqual(self.repository.active_chunk_ids([chunk_id]), {chunk_id})

        revoked = self.service.transition(prepared["asset_id"], "revoke", "operator", "合规撤销")

        self.assertEqual(revoked["status"], "revoked")
        self.assertEqual(self.repository.active_chunk_ids([chunk_id]), set())
        self.assertEqual(self.repository.list_documents(), [])
        versions = self.repository.list_asset_versions(prepared["asset_id"])
        self.assertEqual(versions[0]["status"], "revoked")
        with self.repository.database.transaction() as connection:
            self.assertEqual(connection.execute(
                "SELECT COUNT(*) FROM audit_events WHERE object_id=? AND action='knowledge.asset_revoked'",
                (prepared["asset_id"],),
            ).fetchone()[0], 1)

    def test_restricted_asset_permissions_apply_to_asset_and_retrieval(self):
        alice = self.repository.provision_project_member("alice@example.com", "Alice", "project_member")
        bob = self.repository.provision_project_member("bob@example.com", "Bob", "project_member")
        policy = self.repository.create_access_policy("Alice only", user_ids=[alice["user_id"]])
        prepared, result = self._publish(
            "restricted body", "v1_restricted.md", external_key="restricted.md", title="restricted.md",
            visibility="restricted", access_policy_id=policy["access_policy_id"], owner_user_id=alice["user_id"],
        )
        chunk_id = result["stored"]["chunk_ids"][0]

        self.assertEqual(len(self.repository.list_assets(self._identity(alice))), 1)
        self.assertEqual(self.repository.list_assets(self._identity(bob)), [])
        self.assertEqual(self.repository.active_chunk_ids([chunk_id]), {chunk_id})
        self.assertEqual(self.repository.accessible_document_ids(self._identity(bob)), set())
        self.assertEqual(self.repository.accessible_document_ids(self._identity(alice)), {result["stored"]["document_id"]})

    def test_restricted_source_acl_cannot_be_widened_to_bob_policy_or_private_owner(self):
        alice = self.repository.provision_project_member("alice@example.com", "Alice", "project_member")
        bob = self.repository.provision_project_member("bob@example.com", "Bob", "project_member")
        alice_policy = self.repository.create_access_policy("Alice only", user_ids=[alice["user_id"]])
        bob_policy = self.repository.create_access_policy("Bob only", user_ids=[bob["user_id"]])
        prepared, result = self._publish(
            "alice source", "alice-v1.md", external_key="alice-source.md", title="alice-source.md",
            visibility="restricted", access_policy_id=alice_policy["access_policy_id"],
            owner_user_id=alice["user_id"],
        )

        for acl in (
            {"visibility": "restricted", "access_policy_id": bob_policy["access_policy_id"]},
            {"visibility": "private", "owner_user_id": bob["user_id"]},
        ):
            with self.assertRaisesRegex(ValueError, "不得宽于"):
                self.service.prepare_documents(
                    [SimpleNamespace(page_content="widened", metadata={})],
                    source_type="manual_upload", external_key="alice-source.md", title="alice-source.md",
                    actor="tester", stored_file="alice-v2.md", asset_id=prepared["asset_id"],
                    source_id=prepared["source_id"], **acl,
                )

        # Reproduce a corrupt legacy row where asset/document allow Bob while
        # the authoritative source remains Alice-only. Every read path must
        # still apply the source intersection and deny Bob.
        with self.repository.database.transaction(write=True) as connection:
            connection.execute(
                "UPDATE knowledge_assets SET access_policy_id=? WHERE asset_id=?",
                (bob_policy["access_policy_id"], prepared["asset_id"]),
            )
            connection.execute(
                "UPDATE documents SET access_policy_id=? WHERE document_id=?",
                (bob_policy["access_policy_id"], result["stored"]["document_id"]),
            )
        bob_identity = self._identity(bob)
        self.assertIsNone(self.repository.get_asset(prepared["asset_id"], identity=bob_identity))
        self.assertEqual(self.repository.list_assets(identity=bob_identity), [])
        self.assertEqual(self.repository.accessible_document_ids(bob_identity), set())

    def test_historical_version_preview_requires_intersected_asset_permission(self):
        from backend import main

        alice = self.repository.provision_project_member("alice@example.com", "Alice", "project_member")
        bob = self.repository.provision_project_member("bob@example.com", "Bob", "project_member")
        policy = self.repository.create_access_policy("Alice preview", user_ids=[alice["user_id"]])
        first, _ = self._publish(
            "historical body", "history-v1.md", external_key="history.md", title="history.md",
            visibility="restricted", access_policy_id=policy["access_policy_id"],
            owner_user_id=alice["user_id"],
        )
        self._publish(
            "current body", "history-v2.md", external_key="history.md", title="history.md",
            visibility="restricted", access_policy_id=policy["access_policy_id"],
            owner_user_id=alice["user_id"],
        )
        upload_dir = Path(self.temp.name) / "uploads"
        upload_dir.mkdir()
        (upload_dir / "history-v1.md").write_text("historical body", encoding="utf-8")
        with (
            patch.object(main, "UPLOAD_DIR", upload_dir),
            patch.object(main, "knowledge_asset_service", self.service),
            patch.object(main, "get_current_identity", return_value=self._identity(alice)),
        ):
            preview = asyncio.run(main.preview_knowledge_asset_version(first["asset_id"], first["version_id"]))
        self.assertIn("historical body", preview["content"])

        with (
            patch.object(main, "UPLOAD_DIR", upload_dir),
            patch.object(main, "knowledge_asset_service", self.service),
            patch.object(main, "get_current_identity", return_value=self._identity(bob)),
        ):
            with self.assertRaises(HTTPException) as denied:
                asyncio.run(main.preview_knowledge_asset_version(first["asset_id"], first["version_id"]))
        self.assertEqual(denied.exception.status_code, 404)

    def test_full_state_machine_rejects_invalid_transitions_and_requires_revoke_reason(self):
        document = SimpleNamespace(page_content="review body", metadata={})
        prepared = self.service.prepare_documents(
            [document], source_type="manual_upload", external_key="review.md", title="review.md",
            actor="tester", stored_file="review-v1.md",
        )
        with self.assertRaisesRegex(ValueError, "没有可审核的 ready 版本"):
            self.service.transition(prepared["asset_id"], "submit_review", "reviewer")
        self.repository.upsert_document_chunks([document], ["review-vector"])
        self.service.record_projection(prepared["version_id"], "vector", succeeded=True)
        for projection_type in ("graph", "context", "readiness"):
            self.repository.set_asset_projection(prepared["version_id"], projection_type, "repair_required")
        self.repository.mark_asset_version_ready(prepared["version_id"])
        self.assertEqual(self.service.transition(prepared["asset_id"], "submit_review", "reviewer")["status"], "pending_review")
        with self.assertRaisesRegex(ValueError, "不允许"):
            self.service.transition(prepared["asset_id"], "mark_review_due", "reviewer")
        self.assertEqual(self.service.transition(prepared["asset_id"], "return_to_draft", "reviewer")["status"], "draft")
        with self.assertRaisesRegex(ValueError, "撤销原因"):
            self.service.transition(prepared["asset_id"], "revoke", "reviewer")

        self.service.publish(prepared, actor="reviewer")
        self.assertEqual(self.service.transition(prepared["asset_id"], "mark_review_due", "reviewer")["status"], "review_due")
        self.assertEqual(self.service.transition(prepared["asset_id"], "confirm_valid", "reviewer")["status"], "active")
        self.assertEqual(self.service.transition(prepared["asset_id"], "expire", "reviewer")["status"], "expired")

    def test_preparing_version_does_not_change_live_metadata_until_publish(self):
        first, first_result = self._publish(
            "old body", "old.md", categories=["旧分类"], summary="旧摘要",
            applicable_roles=["开发工程师"],
        )
        original = self.repository.get_asset_detail(first["asset_id"])
        document = SimpleNamespace(page_content="new body", metadata={})
        second = self.service.prepare_documents(
            [document], source_type="manual_upload", external_key="guide.md", title="新标题.md",
            actor="tester", stored_file="new.md", asset_id=first["asset_id"], source_id=first["source_id"],
            categories=["新分类"], applicable_roles=["测试工程师"], summary="新摘要",
            visibility="restricted", access_policy_id="policy_next",
        )
        during = self.repository.get_asset_detail(first["asset_id"])
        self.assertEqual(during["title"], original["title"])
        self.assertEqual(during["summary"], "旧摘要")
        self.assertEqual(during["categories"], ["旧分类"])
        self.assertEqual(during["visibility"], "project")

        self.repository.upsert_document_chunks([document], ["new-vector"])
        self.service.record_projection(second["version_id"], "vector", succeeded=True)
        published = self.service.publish(second, actor="tester")
        detail = self.repository.get_asset_detail(first["asset_id"])
        self.assertEqual(published["title"], "新标题.md")
        self.assertEqual(detail["summary"], "新摘要")
        self.assertEqual(detail["categories"], ["新分类"])
        self.assertEqual(detail["visibility"], "restricted")
        self.assertEqual(detail["source"]["visibility"], "restricted")
        self.assertEqual(len(detail["documents"]), 1)
        self.assertIn("vector", detail["projections"])
        self.assertEqual(first_result["asset"]["current_version_id"], first["version_id"])

    def test_patch_rejects_unknown_fields_and_acl_wider_than_source(self):
        prepared, _ = self._publish("body", "acl.md", external_key="acl.md", title="acl.md")
        with self.assertRaisesRegex(ValueError, "不支持的资产字段"):
            self.service.patch_metadata(prepared["asset_id"], {"unexpected": True}, "operator")
        with self.assertRaisesRegex(ValueError, "不得宽于"):
            self.service.patch_metadata(prepared["asset_id"], {"visibility": "organization"}, "operator")

    def test_new_version_cannot_widen_source_acl(self):
        prepared, _ = self._publish(
            "restricted", "restricted-v1.md", external_key="restricted-source.md",
            title="restricted-source.md", visibility="restricted",
            access_policy_id="policy_restricted",
        )
        document = SimpleNamespace(page_content="wider version", metadata={})
        with self.assertRaisesRegex(ValueError, "不得宽于"):
            self.service.prepare_documents(
                [document], source_type="manual_upload", external_key="restricted-source.md",
                title="restricted-source.md", actor="tester", stored_file="restricted-v2.md",
                asset_id=prepared["asset_id"], source_id=prepared["source_id"],
                visibility="project",
            )

    def test_mark_review_due_records_time_and_invalidates_readiness_projection(self):
        prepared, _ = self._publish("body", "review-due.md", external_key="review-due.md")
        self.repository.set_asset_projection(prepared["version_id"], "readiness", "ready")

        result = self.service.transition(prepared["asset_id"], "mark_review_due", "operator")
        projection = self.repository.asset_projection_status(prepared["asset_id"])

        self.assertTrue(result["review_due_at"])
        self.assertEqual(projection["projections"]["readiness"]["status"], "repair_required")

    def test_projection_repair_is_idempotent(self):
        prepared, _ = self._publish("body", "repair.md", external_key="repair.md", title="repair.md")
        self.repository.set_asset_projection(prepared["version_id"], "graph", "ready")
        self.repository.set_asset_projection(prepared["version_id"], "context", "ready")
        self.repository.set_asset_projection(prepared["version_id"], "readiness", "repair_required", "stale")
        first = self.service.repair_projections(prepared["asset_id"], actor="operator")
        second = self.service.repair_projections(prepared["asset_id"], actor="operator")
        self.assertTrue(first["success"])
        self.assertEqual(first["results"]["readiness"]["count"], 1)
        self.assertEqual(second["results"], {})
        status = self.repository.asset_projection_status(prepared["asset_id"])
        self.assertEqual(status["status"], "healthy")

    def test_force_projection_repair_rebuilds_vector_before_graph(self):
        prepared, _ = self._publish("ordered repair", "ordered.md")
        calls = []

        with (
            patch(
                "backend.knowledge_base.vector_store.vector_store.reconcile_with_storage",
                side_effect=lambda: calls.append("vector") or {"changed": True},
            ),
            patch(
                "backend.knowledge_base.knowledge_graph.knowledge_graph.build_graph",
                side_effect=lambda: calls.append("graph") or {"nodes": 1},
            ),
            patch(
                "backend.knowledge_base.project_context.project_context.reload",
                side_effect=lambda: calls.append("context") or {"documents": 1},
            ),
            patch.object(
                self.repository, "readiness_assets",
                side_effect=lambda identity=None: calls.append("readiness") or [],
            ),
        ):
            result = self.service.repair_projections(
                prepared["asset_id"], actor="operator", force=True,
            )

        self.assertTrue(result["success"])
        self.assertEqual(calls, ["vector", "graph", "context", "readiness"])

    def test_projection_repair_rejects_duplicate_process_execution(self):
        self.assertTrue(KnowledgeAssetService._projection_repair_lock.acquire(blocking=False))
        try:
            with self.assertRaisesRegex(ProjectionRepairBusyError, "正在运行"):
                self.service.repair_projections(actor="operator")
        finally:
            KnowledgeAssetService._projection_repair_lock.release()

    def test_asset_summary_splits_failed_repair_and_processing_counts(self):
        prepared, _ = self._publish("active", "summary-v1.md", external_key="summary.md")
        self.repository.set_asset_projection(
            prepared["version_id"], "context", "repair_required", "context unavailable",
        )
        failed = self.service.prepare_documents(
            [SimpleNamespace(page_content="failed version", metadata={})],
            source_type="manual_upload", external_key="summary.md", title="summary.md",
            actor="tester", stored_file="summary-failed.md",
            asset_id=prepared["asset_id"], source_id=prepared["source_id"],
        )
        self.service.fail(failed, "index failed")
        self.service.prepare_documents(
            [SimpleNamespace(page_content="processing version", metadata={})],
            source_type="manual_upload", external_key="summary.md", title="summary.md",
            actor="tester", stored_file="summary-processing.md",
            asset_id=prepared["asset_id"], source_id=prepared["source_id"],
        )

        summary = self.repository.summarize_asset_processing([prepared["asset_id"]])

        self.assertEqual(summary["failed_version_count"], 1)
        self.assertEqual(summary["repair_required_count"], 1)
        self.assertEqual(summary["processing_count"], 1)

    def test_external_identity_only_maps_active_project_member(self):
        member = self.repository.provision_project_member(
            "member@example.com", "Member", "project_member", status="active",
        )
        mapped = self.repository.resolve_external_identity(
            "feishu", "ou_member", verified_email="member@example.com",
        )
        self.assertEqual(mapped["user_id"], member["user_id"])
        self.assertEqual(self.repository.resolve_external_identity("feishu", "ou_unknown"), None)

    def test_onboarding_endpoint_contract_uses_four_dimension_readiness(self):
        from backend import main

        self._publish(
            "architecture", "architecture.md", external_key="architecture.md", title="architecture.md",
            categories=["proj_architecture"], applicable_roles=["开发工程师"],
        )
        with (
            patch.object(main, "get_repository", return_value=self.repository),
            patch.object(main, "get_current_identity", return_value=None),
        ):
            guide = main._onboarding_guide("开发工程师")
        self.assertEqual(
            guide["formula"], "主题覆盖 50% + 权威性 20% + 新鲜度 20% + 可追溯性 10%",
        )
        self.assertEqual(set(guide["dimensions"]), {"topic_coverage", "authority", "freshness", "traceability"})
        self.assertGreater(guide["readiness"], 0)
        self.assertEqual(guide["recommended_documents"][0]["asset_id"], self.repository.list_assets()[0]["asset_id"])


if __name__ == "__main__":
    unittest.main()
