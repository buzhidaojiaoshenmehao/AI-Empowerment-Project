import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from backend.config import settings
from backend.feishu_knowledge import FeishuKnowledgeService
from backend.feishu_workspace import FeishuWorkspace
from backend.storage.database import Database
from backend.storage.repository import StorageRepository


class FeishuWorkspaceTest(unittest.TestCase):
    def test_group_knowledge_scope_is_validated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = FeishuWorkspace(StorageRepository(Database(Path(temp_dir) / "app.db")))
            with self.assertRaisesRegex(ValueError, "无效的知识可见范围"):
                workspace.upsert_group({"chat_id": "oc_invalid", "visibility": "public"})
            with self.assertRaisesRegex(ValueError, "必须绑定访问策略"):
                workspace.upsert_group({"chat_id": "oc_restricted", "visibility": "restricted"})

            restricted = workspace.upsert_group({
                "chat_id": "oc_restricted",
                "visibility": "restricted",
                "access_policy_id": "policy_project_leads",
            })
            self.assertEqual(restricted["visibility"], "restricted")
            self.assertEqual(restricted["access_policy_id"], "policy_project_leads")

            project = workspace.upsert_group({
                "chat_id": "oc_restricted",
                "visibility": "project",
                "access_policy_id": "policy_should_be_removed",
            })
            self.assertEqual(project["visibility"], "project")
            self.assertEqual(project["access_policy_id"], "")

    def test_group_rules_and_message_review_are_persisted(self):
        original_dir = settings.CHROMA_PERSIST_DIR
        with tempfile.TemporaryDirectory() as temp_dir:
            settings.CHROMA_PERSIST_DIR = temp_dir
            try:
                workspace = FeishuWorkspace(StorageRepository(Database(Path(temp_dir) / "app.db")))
                initial_revision = workspace.revision()
                group = workspace.upsert_group({
                    "chat_id": "oc_project",
                    "name": "项目研发群",
                    "collection_mode": "review",
                    "default_category": "proj_communication",
                    "retention_days": 90,
                })
                self.assertEqual(group["name"], "项目研发群")
                self.assertGreater(workspace.revision(), initial_revision)
                workspace.upsert_group({
                    "chat_id": "oc_direct",
                    "name": "机器人单聊",
                    "chat_type": "p2p",
                })
                self.assertEqual(workspace.summary()["group_count"], 1)
                self.assertEqual(workspace.summary()["conversation_count"], 2)

                added = workspace.upsert_messages("oc_project", [{
                    "message_id": "om_001",
                    "msg_type": "text",
                    "body": {"content": '{"text":"发布风险需要在今天确认"}'},
                    "sender": {"id": "ou_member"},
                    "create_time": str(int(datetime.now().timestamp() * 1000)),
                }])
                self.assertEqual(added, 1)
                self.assertEqual(workspace.summary()["pending_count"], 1)

                filtered = workspace.upsert_messages("oc_project", [{
                    "message_id": "om_bot",
                    "msg_type": "text",
                    "body": {"content": '{"text":"机器人回复"}'},
                    "sender": {"id": "cli_bot", "sender_type": "app"},
                }, {
                    "message_id": "om_system",
                    "msg_type": "system",
                    "body": {"content": '{"text":"成员入群"}'},
                }])
                self.assertEqual(filtered, 0)

                reviewed = workspace.set_review_status("om_001", "approved", "飞书群聊_项目研发群.md")
                self.assertEqual(reviewed["review_status"], "approved")
                summary = workspace.summary()
                self.assertEqual(summary["approved_count"], 1)
                self.assertEqual(summary["today_received"], 1)
                self.assertEqual(summary["today_evaluated_count"], 1)
                self.assertEqual(summary["today_valuable_count"], 1)
                self.assertEqual(summary["today_candidate_count"], 1)
                self.assertEqual(summary["today_published_count"], 1)
                self.assertEqual(summary["published_message_count"], 1)
                self.assertEqual(summary["conversion_rate"], 100.0)
                self.assertEqual(summary["today_conversion_rate"], 100.0)
                self.assertEqual(summary["today_automation_rate"], 100.0)
                diagnostics = workspace.record_event("im.message.receive_v1", "replied", "om_001")
                self.assertEqual(diagnostics["last_action"], "replied")
                self.assertEqual(workspace.diagnostics()["last_message_id"], "om_001")
            finally:
                settings.CHROMA_PERSIST_DIR = original_dir

    def test_four_collection_modes_apply_without_reviewing_normal_chat(self):
        original_dir = settings.CHROMA_PERSIST_DIR
        with tempfile.TemporaryDirectory() as temp_dir:
            settings.CHROMA_PERSIST_DIR = temp_dir
            try:
                workspace = FeishuWorkspace(StorageRepository(Database(Path(temp_dir) / "app.db")))
                auto = workspace.upsert_group({"chat_id": "oc_auto", "name": "内部项目群"})
                review = workspace.upsert_group({"chat_id": "oc_review", "collection_mode": "review"})
                archive = workspace.upsert_group({"chat_id": "oc_archive", "collection_mode": "archive_only"})
                off = workspace.upsert_group({"chat_id": "oc_off", "collection_mode": "off"})
                external = workspace.upsert_group({"chat_id": "oc_external", "external": True})

                self.assertEqual(auto["collection_mode"], "auto")
                self.assertEqual(review["collection_mode"], "review")
                self.assertEqual(archive["collection_mode"], "archive_only")
                self.assertEqual(off["collection_mode"], "off")
                self.assertEqual(external["collection_mode"], "archive_only")
                self.assertEqual(workspace.summary()["group_count"], 4)

                workspace.upsert_messages("oc_auto", [self._message("om_low", "收到")])
                workspace.upsert_messages("oc_auto", [self._message("om_auto", "结论：发布方案已确认，主要风险是回滚流程需要补充。")])
                workspace.upsert_messages("oc_review", [self._message("om_review", "结论：发布方案已确认，主要风险是回滚流程需要补充。")])
                workspace.upsert_messages("oc_archive", [self._message("om_archive", "结论：发布方案已确认，主要风险是回滚流程需要补充。")])
                ignored = workspace.upsert_messages("oc_off", [self._message("om_off", "结论：发布方案已确认。")])

                self.assertEqual(workspace.get_message("om_low")["content_status"], "excluded")
                self.assertEqual(workspace.get_message("om_auto")["content_status"], "candidate")
                self.assertEqual(workspace.get_message("om_review")["content_status"], "review_required")
                self.assertEqual(workspace.get_message("om_archive")["content_status"], "archived")
                self.assertEqual(ignored, 0)
                self.assertIsNone(workspace.get_message("om_off"))
            finally:
                settings.CHROMA_PERSIST_DIR = original_dir

    def test_sensitive_content_requires_review_and_reasons_are_explainable(self):
        original_dir = settings.CHROMA_PERSIST_DIR
        with tempfile.TemporaryDirectory() as temp_dir:
            settings.CHROMA_PERSIST_DIR = temp_dir
            try:
                workspace = FeishuWorkspace(StorageRepository(Database(Path(temp_dir) / "app.db")))
                workspace.upsert_group({"chat_id": "oc_auto", "collection_mode": "auto"})
                workspace.upsert_messages("oc_auto", [self._message("om_secret", "接口 token: abc123，请写入部署流程。")])
                message = workspace.get_message("om_secret")
                candidates = workspace.list_candidates(status="review_required")
                self.assertEqual(message["content_status"], "review_required")
                self.assertIn("敏感信息", message["reason"])
                self.assertEqual(len(candidates), 1)
                self.assertIn("敏感信息", candidates[0]["reasons"][0])
            finally:
                settings.CHROMA_PERSIST_DIR = original_dir

    def test_thread_aggregation_and_duplicate_delivery_are_idempotent(self):
        original_dir = settings.CHROMA_PERSIST_DIR
        with tempfile.TemporaryDirectory() as temp_dir:
            settings.CHROMA_PERSIST_DIR = temp_dir
            try:
                workspace = FeishuWorkspace(StorageRepository(Database(Path(temp_dir) / "app.db")))
                workspace.upsert_group({"chat_id": "oc_thread", "collection_mode": "auto"})
                messages = [
                    self._message("om_1", "决策结论：发布方案采用灰度流程，负责人今天确认。", root_id="om_root"),
                    self._message("om_2", "风险结论：回滚流程需要补充，并在验收前完成。", root_id="om_root"),
                ]
                self.assertEqual(workspace.upsert_messages("oc_thread", messages), 2)
                self.assertEqual(workspace.upsert_messages("oc_thread", messages), 0)
                candidates = workspace.list_candidates()
                self.assertEqual(len(candidates), 1)
                self.assertEqual(candidates[0]["message_count"], 2)
                self.assertEqual(set(candidates[0]["source_message_ids"]), {"om_1", "om_2"})
                self.assertEqual(len(workspace.list_items()), 2)
            finally:
                settings.CHROMA_PERSIST_DIR = original_dir

    def test_content_center_queries_separate_current_history_and_paginate_messages(self):
        original_dir = settings.CHROMA_PERSIST_DIR
        with tempfile.TemporaryDirectory() as temp_dir:
            settings.CHROMA_PERSIST_DIR = temp_dir
            try:
                workspace = FeishuWorkspace(StorageRepository(Database(Path(temp_dir) / "app.db")))
                workspace.upsert_group({"chat_id": "oc_archive", "collection_mode": "archive_only"})
                workspace.upsert_messages("oc_archive", [
                    self._message(f"om_archive_{index:02d}", f"项目归档消息 {index}")
                    for index in range(25)
                ])

                second_page = workspace.query_items(
                    chat_id="oc_archive", page=2, page_size=10,
                )
                self.assertEqual(second_page["total"], 25)
                self.assertEqual(second_page["page"], 2)
                self.assertEqual(second_page["pages"], 3)
                self.assertEqual(len(second_page["items"]), 10)

                workspace.upsert_group({"chat_id": "oc_assets", "collection_mode": "review"})
                workspace.upsert_messages("oc_assets", [
                    self._message("om_current", "决策结论：当前知识保持有效。", root_id="root_current"),
                    self._message("om_history", "风险结论：该知识需要撤销。", root_id="root_history"),
                ])
                candidates = workspace.list_candidates(chat_id="oc_assets")
                for candidate in candidates:
                    workspace.publish_candidate(candidate["candidate_id"], {
                        "source_file": f"{candidate['candidate_id']}.md",
                        "category": "proj_communication",
                    })
                reverted_asset = next(
                    asset for asset in workspace.list_assets()
                    if asset["candidate_id"] == next(
                        item["candidate_id"] for item in candidates if "撤销" in item["title"]
                    )
                )
                workspace.revert_asset(reverted_asset["asset_id"])

                current = workspace.query_assets(statuses=("published",))
                history = workspace.query_assets(statuses=("reverted",))
                self.assertEqual(current["total"], 1)
                self.assertEqual(history["total"], 1)
                self.assertEqual(workspace.summary()["reverted_count"], 1)
                hidden_history = workspace.query_items(excluded_statuses=("reverted",))
                self.assertNotIn("reverted", {item["content_status"] for item in hidden_history["items"]})
            finally:
                settings.CHROMA_PERSIST_DIR = original_dir

    def test_candidate_batch_state_changes_are_audited(self):
        original_dir = settings.CHROMA_PERSIST_DIR
        with tempfile.TemporaryDirectory() as temp_dir:
            settings.CHROMA_PERSIST_DIR = temp_dir
            try:
                workspace = FeishuWorkspace(StorageRepository(Database(Path(temp_dir) / "app.db")))
                workspace.upsert_group({"chat_id": "oc_review", "collection_mode": "review"})
                workspace.upsert_messages("oc_review", [self._message("om_review", "项目风险需要在发布前确认回滚方案。")])
                candidate_id = workspace.list_candidates()[0]["candidate_id"]
                workspace.update_candidate(candidate_id, "category", category="proj_risk")
                self.assertEqual(workspace.get_candidate(candidate_id)["category"], "proj_risk")
                workspace.update_candidate(candidate_id, "exclude")
                self.assertEqual(workspace.get_candidate(candidate_id)["status"], "excluded")
                self.assertEqual(workspace.get_message("om_review")["content_status"], "excluded")
                actions = [item["action"] for item in workspace.list_audit(limit=50)]
                self.assertIn("candidate_category", actions)
                self.assertIn("candidate_exclude", actions)
            finally:
                settings.CHROMA_PERSIST_DIR = original_dir

    def test_messages_outside_group_retention_are_not_kept(self):
        original_dir = settings.CHROMA_PERSIST_DIR
        with tempfile.TemporaryDirectory() as temp_dir:
            settings.CHROMA_PERSIST_DIR = temp_dir
            try:
                workspace = FeishuWorkspace(StorageRepository(Database(Path(temp_dir) / "app.db")))
                workspace.upsert_group({"chat_id": "oc_short", "name": "短留存群", "retention_days": 7})
                expired_time = int((datetime.now().timestamp() - 10 * 24 * 60 * 60) * 1000)
                workspace.upsert_messages("oc_short", [{
                    "message_id": "om_expired",
                    "msg_type": "text",
                    "body": {"content": '{"text":"过期消息"}'},
                    "create_time": str(expired_time),
                }])
                self.assertEqual(workspace.summary()["message_count"], 0)
            finally:
                settings.CHROMA_PERSIST_DIR = original_dir

    def test_legacy_image_is_resynced_and_reevaluated(self):
        original_dir = settings.CHROMA_PERSIST_DIR
        with tempfile.TemporaryDirectory() as temp_dir:
            settings.CHROMA_PERSIST_DIR = temp_dir
            try:
                workspace = FeishuWorkspace(StorageRepository(Database(Path(temp_dir) / "app.db")))
                workspace.upsert_group({"chat_id": "oc_image", "collection_mode": "auto"})
                data = workspace.load()
                data["version"] = 3
                data["messages"].append({
                    "message_id": "om_legacy_image",
                    "chat_id": "oc_image",
                    "sender": "ou_member",
                    "message_type": "image",
                    "content": "[暂不支持直接预览的消息]",
                    "content_status": "excluded",
                    "review_status": "ignored",
                    "reason": "内容无法解析",
                    "create_time": str(int(datetime.now().timestamp() * 1000)),
                })
                workspace.save(data)

                legacy = workspace.get_message("om_legacy_image")
                self.assertEqual(legacy["extraction_status"], "legacy_pending")
                self.assertNotIn("om_legacy_image", workspace.existing_message_ids())

                added = workspace.upsert_messages("oc_image", [{
                    "message_id": "om_legacy_image",
                    "msg_type": "image",
                    "body": {"content": json.dumps({
                        "text": "风险结论：上线前必须完成回滚演练并确认负责人。",
                        "image_key": "img_legacy",
                        "image_keys": ["img_legacy"],
                        "resource_count": 1,
                        "resource_files": [{"stored_file": "legacy.png", "content_type": "image/png"}],
                        "extraction_status": "completed",
                        "extraction_method": "feishu_ocr",
                    }, ensure_ascii=False)},
                    "sender": {"id": "ou_member"},
                    "create_time": legacy["create_time"],
                }])

                refreshed = workspace.get_message("om_legacy_image")
                self.assertEqual(added, 0)
                self.assertEqual(refreshed["extraction_status"], "completed")
                self.assertEqual(refreshed["content_status"], "candidate")
                self.assertIn("om_legacy_image", workspace.existing_message_ids())
                self.assertEqual(len(workspace.list_candidates()), 1)
            finally:
                settings.CHROMA_PERSIST_DIR = original_dir

    def test_published_legacy_post_refresh_reuses_original_candidate(self):
        original_dir = settings.CHROMA_PERSIST_DIR
        with tempfile.TemporaryDirectory() as temp_dir:
            settings.CHROMA_PERSIST_DIR = temp_dir
            try:
                workspace = FeishuWorkspace(StorageRepository(Database(Path(temp_dir) / "app.db")))
                workspace.upsert_group({"chat_id": "oc_post", "collection_mode": "auto"})
                data = workspace.load()
                data["messages"].append({
                    "message_id": "om_legacy_post",
                    "chat_id": "oc_post",
                    "sender": "ou_member",
                    "message_type": "post",
                    "content": "[暂不支持直接预览的消息]",
                    "content_status": "published",
                    "review_status": "approved",
                    "reason": "由旧版工作台迁移",
                    "candidate_id": "fc_original",
                    "asset_id": "fa_original",
                    "create_time": str(int(datetime.now().timestamp() * 1000)),
                })
                data["candidates"].append({
                    "candidate_id": "fc_original",
                    "aggregation_key": "legacy:om_legacy_post",
                    "chat_id": "oc_post",
                    "title": "旧版知识",
                    "source_message_ids": ["om_legacy_post"],
                    "message_count": 1,
                    "category": "proj_communication",
                    "confidence": 1.0,
                    "reasons": ["由旧版工作台迁移"],
                    "status": "published",
                    "asset_id": "fa_original",
                })
                data["assets"].append({
                    "asset_id": "fa_original",
                    "candidate_id": "fc_original",
                    "source_message_ids": ["om_legacy_post"],
                    "status": "published",
                })
                workspace.save(data)

                legacy = workspace.get_message("om_legacy_post")
                workspace.upsert_messages("oc_post", [{
                    "message_id": "om_legacy_post",
                    "msg_type": "post",
                    "body": {"content": json.dumps({
                        "title": "发布复盘",
                        "text": "# 发布复盘\n\n结论：灰度发布方案有效，回滚流程需要保留。",
                        "extraction_status": "completed",
                        "extraction_method": "rich_text_parser",
                    }, ensure_ascii=False)},
                    "sender": {"id": "ou_member"},
                    "create_time": legacy["create_time"],
                }])

                refreshed = workspace.get_message("om_legacy_post")
                candidates = workspace.list_candidates()
                self.assertEqual(refreshed["candidate_id"], "fc_original")
                self.assertEqual(refreshed["asset_id"], "fa_original")
                self.assertEqual(len(candidates), 1)
                self.assertEqual(candidates[0]["candidate_id"], "fc_original")
                self.assertTrue(candidates[0]["update_existing_asset"])
            finally:
                settings.CHROMA_PERSIST_DIR = original_dir

    @staticmethod
    def _message(message_id, text, root_id=""):
        return {
            "message_id": message_id,
            "msg_type": "text",
            "body": {"content": json.dumps({"text": text}, ensure_ascii=False)},
            "sender": {"id": "ou_member"},
            "create_time": str(int(datetime.now().timestamp() * 1000)),
            "root_id": root_id,
        }


class FeishuKnowledgeServiceTest(unittest.TestCase):
    def test_workspace_registration_failure_keeps_authoritative_asset_and_is_repairable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = StorageRepository(Database(Path(temp_dir) / "app.db"))
            workspace = FeishuWorkspace(repository)
            workspace.upsert_group({"chat_id": "oc_partial", "name": "项目群", "collection_mode": "review"})
            workspace.upsert_messages("oc_partial", [FeishuWorkspaceTest._message(
                "om_partial", "决策结论：采用分阶段发布。",
            )])
            candidate_id = workspace.list_candidates()[0]["candidate_id"]
            service = FeishuKnowledgeService(
                workspace=workspace,
                upload_dir=Path(temp_dir) / "uploads",
                document_loader=lambda _path: [SimpleNamespace(page_content="知识正文", metadata={})],
                vector_store=_FakeVectorStore(),
                project_context=_FakeProjectContext(),
                knowledge_graph=_FakeGraph(),
            )

            with patch.object(workspace, "publish_candidate", side_effect=RuntimeError("legacy unavailable")):
                result = service.ingest_candidate(candidate_id, actor="operator")

            self.assertTrue(result["success"])
            self.assertTrue(result["partial_success"])
            self.assertFalse(result["compatibility_registered"])
            self.assertTrue(result["repair_required"])
            self.assertEqual(result["authoritative_asset"]["status"], "active")
            self.assertEqual(repository.list_assets()[0]["asset_id"], result["authoritative_asset"]["asset_id"])
            self.assertEqual(workspace.get_candidate(candidate_id)["status"], "failed")

    def test_batch_actions_support_category_exclude_and_retry(self):
        original_dir = settings.CHROMA_PERSIST_DIR
        with tempfile.TemporaryDirectory() as temp_dir:
            settings.CHROMA_PERSIST_DIR = temp_dir
            try:
                workspace = FeishuWorkspace(StorageRepository(Database(Path(temp_dir) / "app.db")))
                workspace.upsert_group({"chat_id": "oc_review", "collection_mode": "review"})
                workspace.upsert_messages("oc_review", [
                    FeishuWorkspaceTest._message("om_batch_1", "决策结论：采用灰度发布流程。", root_id="root_1"),
                    FeishuWorkspaceTest._message("om_batch_2", "风险结论：回滚步骤需要补充。", root_id="root_2"),
                ])
                candidates = workspace.list_candidates()
                self.assertEqual(len(candidates), 2)
                candidate_ids = [item["candidate_id"] for item in candidates]
                service = FeishuKnowledgeService(
                    workspace=workspace,
                    upload_dir=Path(temp_dir) / "uploads",
                    document_loader=lambda _path: [SimpleNamespace(page_content="知识正文", metadata={})],
                    vector_store=_FakeVectorStore(),
                    project_context=_FakeProjectContext(),
                    knowledge_graph=_FakeGraph(),
                )

                categorized = service.batch_action(candidate_ids, "category", category="proj_risk")
                self.assertTrue(categorized["success"])
                self.assertTrue(all(workspace.get_candidate(item)["category"] == "proj_risk" for item in candidate_ids))

                excluded = service.batch_action([candidate_ids[0]], "exclude")
                self.assertTrue(excluded["success"])
                self.assertEqual(workspace.get_candidate(candidate_ids[0])["status"], "excluded")

                retried = service.batch_action([candidate_ids[1], "missing_candidate"], "retry")
                self.assertFalse(retried["success"])
                self.assertEqual(len(retried["succeeded"]), 1)
                self.assertEqual(len(retried["failed"]), 1)
                self.assertEqual(workspace.get_candidate(candidate_ids[1])["status"], "published")
            finally:
                settings.CHROMA_PERSIST_DIR = original_dir

    def test_publish_and_revert_remove_derived_knowledge_but_keep_archive(self):
        original_dir = settings.CHROMA_PERSIST_DIR
        with tempfile.TemporaryDirectory() as temp_dir:
            settings.CHROMA_PERSIST_DIR = temp_dir
            try:
                workspace = FeishuWorkspace(StorageRepository(Database(Path(temp_dir) / "app.db")))
                workspace.upsert_group({"chat_id": "oc_review", "name": "项目研发群", "collection_mode": "review"})
                workspace.upsert_messages("oc_review", [FeishuWorkspaceTest._message(
                    "om_publish", "决策结论：发布方案采用灰度流程，风险由项目经理跟踪。"
                )])
                candidate_id = workspace.list_candidates()[0]["candidate_id"]

                vector_store = _FakeVectorStore()
                project_context = _FakeProjectContext()
                knowledge_graph = _FakeGraph()
                upload_dir = Path(temp_dir) / "uploads"
                service = FeishuKnowledgeService(
                    workspace=workspace,
                    upload_dir=upload_dir,
                    document_loader=lambda _path: [SimpleNamespace(page_content="知识正文", metadata={})],
                    vector_store=vector_store,
                    project_context=project_context,
                    knowledge_graph=knowledge_graph,
                )
                result = service.ingest_candidate(candidate_id, actor="测试管理员")
                asset = result["asset"]
                self.assertEqual(workspace.get_candidate(candidate_id)["status"], "published")
                self.assertEqual(workspace.get_message("om_publish")["content_status"], "published")
                self.assertTrue((upload_dir / asset["stored_file"]).exists())

                reverted = service.revert_asset(
                    asset["asset_id"], reason="测试撤销", actor="user_reviewer",
                )
                self.assertTrue(reverted["success"])
                self.assertEqual(workspace.get_asset(asset["asset_id"])["status"], "reverted")
                self.assertEqual(workspace.get_message("om_publish")["content_status"], "reverted")
                self.assertIsNotNone(workspace.get_message("om_publish"))
                self.assertTrue((upload_dir / asset["stored_file"]).exists())
                self.assertGreaterEqual(vector_store.deleted, 1)
                self.assertGreaterEqual(knowledge_graph.builds, 2)
                audits = workspace.list_audit(limit=50)
                self.assertIn("asset_reverted", [item["action"] for item in audits])
                reverted_audit = next(item for item in audits if item["action"] == "asset_reverted")
                self.assertEqual(reverted_audit["actor"], "user_reviewer")
            finally:
                settings.CHROMA_PERSIST_DIR = original_dir

    def test_incremental_publish_failure_keeps_previous_knowledge_available(self):
        original_dir = settings.CHROMA_PERSIST_DIR
        with tempfile.TemporaryDirectory() as temp_dir:
            settings.CHROMA_PERSIST_DIR = temp_dir
            try:
                workspace = FeishuWorkspace(StorageRepository(Database(Path(temp_dir) / "app.db")))
                workspace.upsert_group({"chat_id": "oc_auto", "name": "项目研发群", "collection_mode": "auto"})
                workspace.upsert_messages("oc_auto", [FeishuWorkspaceTest._message(
                    "om_first", "决策结论：发布方案采用灰度流程。", root_id="root_release"
                )])
                candidate_id = workspace.list_candidates()[0]["candidate_id"]
                vector_store = _FakeVectorStore()
                project_context = _FakeProjectContext()
                upload_dir = Path(temp_dir) / "uploads"
                service = FeishuKnowledgeService(
                    workspace=workspace,
                    upload_dir=upload_dir,
                    document_loader=lambda _path: [SimpleNamespace(page_content="知识正文", metadata={})],
                    vector_store=vector_store,
                    project_context=project_context,
                    knowledge_graph=_FakeGraph(),
                )

                first = service.ingest_candidate(candidate_id)
                original_asset = first["asset"]
                original_stored_file = original_asset["stored_file"]
                self.assertIn(original_stored_file, vector_store.stored_files)

                workspace.upsert_messages("oc_auto", [FeishuWorkspaceTest._message(
                    "om_second", "风险结论：回滚步骤需要在验收前补充。", root_id="root_release"
                )])
                vector_store.fail_next_add = True
                with self.assertRaisesRegex(RuntimeError, "模拟向量写入失败"):
                    service.ingest_candidate(candidate_id)

                active_asset = workspace.get_asset(original_asset["asset_id"])
                self.assertEqual(active_asset["status"], "published")
                self.assertEqual(active_asset["stored_file"], original_stored_file)
                self.assertEqual(vector_store.stored_files, [original_stored_file])
                self.assertTrue((upload_dir / original_stored_file).exists())
                self.assertEqual(workspace.get_message("om_first")["content_status"], "published")
                self.assertEqual(workspace.get_message("om_second")["content_status"], "failed")
                self.assertEqual(workspace.get_candidate(candidate_id)["status"], "failed")
            finally:
                settings.CHROMA_PERSIST_DIR = original_dir

    def test_batch_revert_supports_multiple_assets_and_reports_failures(self):
        original_dir = settings.CHROMA_PERSIST_DIR
        with tempfile.TemporaryDirectory() as temp_dir:
            settings.CHROMA_PERSIST_DIR = temp_dir
            try:
                workspace = FeishuWorkspace(StorageRepository(Database(Path(temp_dir) / "app.db")))
                workspace.upsert_group({"chat_id": "oc_review", "name": "项目研发群", "collection_mode": "review"})
                workspace.upsert_messages("oc_review", [
                    FeishuWorkspaceTest._message("om_asset_1", "决策结论：采用灰度发布。", root_id="root_1"),
                    FeishuWorkspaceTest._message("om_asset_2", "风险结论：验收前完成回滚演练。", root_id="root_2"),
                ])
                service = FeishuKnowledgeService(
                    workspace=workspace,
                    upload_dir=Path(temp_dir) / "uploads",
                    document_loader=lambda _path: [SimpleNamespace(page_content="知识正文", metadata={})],
                    vector_store=_FakeVectorStore(),
                    project_context=_FakeProjectContext(),
                    knowledge_graph=_FakeGraph(),
                )
                asset_ids = [
                    service.ingest_candidate(candidate["candidate_id"])["asset"]["asset_id"]
                    for candidate in workspace.list_candidates()
                ]

                result = service.batch_revert(asset_ids, reason="测试批量撤销")
                self.assertTrue(result["success"])
                self.assertEqual(len(result["succeeded"]), 2)
                self.assertTrue(all(workspace.get_asset(asset_id)["status"] == "reverted" for asset_id in asset_ids))

                missing = service.batch_revert(["missing_asset"], reason="测试批量撤销")
                self.assertFalse(missing["success"])
                self.assertEqual(len(missing["failed"]), 1)
            finally:
                settings.CHROMA_PERSIST_DIR = original_dir

    def test_revert_stays_fail_closed_when_retrieval_projection_refresh_fails(self):
        original_dir = settings.CHROMA_PERSIST_DIR
        with tempfile.TemporaryDirectory() as temp_dir:
            settings.CHROMA_PERSIST_DIR = temp_dir
            try:
                workspace = FeishuWorkspace(StorageRepository(Database(Path(temp_dir) / "app.db")))
                workspace.upsert_group({"chat_id": "oc_review", "name": "项目研发群", "collection_mode": "review"})
                workspace.upsert_messages("oc_review", [FeishuWorkspaceTest._message(
                    "om_revert_failure", "决策结论：采用灰度发布。", root_id="root_failure"
                )])
                vector_store = _FakeVectorStore()
                upload_dir = Path(temp_dir) / "uploads"
                service = FeishuKnowledgeService(
                    workspace=workspace,
                    upload_dir=upload_dir,
                    document_loader=lambda _path: [SimpleNamespace(page_content="知识正文", metadata={})],
                    vector_store=vector_store,
                    project_context=_FakeProjectContext(),
                    knowledge_graph=_FakeGraph(),
                )
                candidate_id = workspace.list_candidates()[0]["candidate_id"]
                asset = service.ingest_candidate(candidate_id)["asset"]
                vector_store.fail_next_delete = True

                result = service.revert_asset(asset["asset_id"], reason="测试撤销")

                self.assertTrue(result["success"])
                self.assertTrue(result["partial_success"])
                self.assertFalse(result["vector_synced"])
                self.assertEqual(result["projection_status"], "repair_required")
                self.assertIn("知识检索索引同步待重试", result["sync_warnings"])
                self.assertEqual(workspace.get_asset(asset["asset_id"])["status"], "reverted")
                self.assertEqual(workspace.get_candidate(candidate_id)["status"], "reverted")
                self.assertEqual(workspace.get_message("om_revert_failure")["content_status"], "reverted")
                self.assertTrue((upload_dir / asset["stored_file"]).exists())
                self.assertIn(asset["stored_file"], vector_store.stored_files)
                self.assertEqual(
                    workspace.get_asset(asset["asset_id"])["projection_status"], "repair_required"
                )
                self.assertIn("asset_revert_cleanup_warning", [item["action"] for item in workspace.list_audit(limit=50)])

                retried = service.revert_asset(asset["asset_id"], reason="重试投影同步")
                self.assertTrue(retried["already_reverted"])
                self.assertFalse(retried["partial_success"])
                self.assertEqual(retried["projection_status"], "healthy")
                self.assertEqual(workspace.get_asset(asset["asset_id"])["projection_status"], "healthy")
            finally:
                settings.CHROMA_PERSIST_DIR = original_dir


class _FakeVectorStore:
    def __init__(self):
        self.deleted = 0
        self.stored_files = []
        self.fail_next_add = False
        self.fail_next_delete = False

    def add_documents(self, docs):
        self.stored_files.extend(str(doc.metadata.get("stored_file") or "") for doc in docs)
        if self.fail_next_add:
            self.fail_next_add = False
            raise RuntimeError("模拟向量写入失败")
        return len(docs)

    def delete_documents_by_source(self, filename):
        if self.fail_next_delete:
            self.fail_next_delete = False
            raise RuntimeError("模拟向量删除失败")
        before = len(self.stored_files)
        self.stored_files = [item for item in self.stored_files if item != filename]
        removed = before - len(self.stored_files)
        self.deleted += max(1, removed)
        return removed

    def delete_document(self, filename):
        return self.delete_documents_by_source(filename)

    def reconcile_with_storage(self):
        if self.fail_next_delete:
            self.fail_next_delete = False
            raise RuntimeError("模拟向量删除失败")
        removed = len(self.stored_files)
        self.stored_files = []
        self.deleted += removed
        return {"changed": bool(removed), "chunks": 0, "recreated": 0}


class _FakeProjectContext:
    def __init__(self):
        self.documents = {}

    @staticmethod
    def extract_project_metadata(_content):
        return {}

    def register_document(self, document_id, metadata):
        self.documents[document_id] = metadata

    def unregister_document(self, document_id):
        return self.documents.pop(document_id, None) is not None


class _FakeGraph:
    def __init__(self):
        self.builds = 0

    def build_graph(self):
        self.builds += 1


if __name__ == "__main__":
    unittest.main()
