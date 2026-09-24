import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from backend import main
from backend.config import settings
from backend.feishu_bot import feishu_bot
from backend.storage.database import Database
from backend.storage.repository import StorageRepository


class FeishuPermissionApiTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.original_app_id = settings.FEISHU_APP_ID
        self.original_app_secret = settings.FEISHU_APP_SECRET
        self.original_workspace_repository = main.feishu_workspace._storage_repository
        self.storage_directory = tempfile.TemporaryDirectory()
        main.feishu_workspace._storage_repository = StorageRepository(
            Database(Path(self.storage_directory.name) / "app.db")
        )

    async def asyncTearDown(self):
        settings.FEISHU_APP_ID = self.original_app_id
        settings.FEISHU_APP_SECRET = self.original_app_secret
        main.feishu_workspace._storage_repository = self.original_workspace_repository
        self.storage_directory.cleanup()

    async def test_missing_credentials_are_not_reported_as_verified(self):
        settings.FEISHU_APP_ID = ""
        settings.FEISHU_APP_SECRET = ""

        result = await main.check_feishu_permissions()

        capabilities = {item["key"]: item for item in result["capabilities"]}
        self.assertFalse(result["configured"])
        self.assertFalse(result["verified"])
        self.assertEqual(capabilities["credentials"]["status"], "missing")
        self.assertEqual(capabilities["authentication"]["status"], "unverified")

    async def test_saved_credentials_with_failed_auth_remain_unverified(self):
        settings.FEISHU_APP_ID = "cli_test_app"
        settings.FEISHU_APP_SECRET = "test_secret_not_real"

        with (
            patch.object(feishu_bot, "_get_app_access_token", AsyncMock(side_effect=RuntimeError("模拟鉴权失败"))),
            patch.object(main.feishu_workspace, "diagnostics", return_value={}),
            patch.object(main.feishu_workspace, "record_audit"),
            patch.object(main.feishu_long_connection, "status", return_value={"connected": False}),
        ):
            result = await main.check_feishu_permissions()

        capabilities = {item["key"]: item for item in result["capabilities"]}
        self.assertTrue(result["configured"])
        self.assertFalse(result["verified"])
        self.assertEqual(capabilities["credentials"]["status"], "configured")
        self.assertEqual(capabilities["authentication"]["status"], "failed")
        self.assertEqual(capabilities["chat_read"]["status"], "failed")
        self.assertEqual(capabilities["message_send"]["status"], "unverified")
        self.assertEqual(capabilities["document_read"]["status"], "unverified")

    async def test_archived_image_resource_is_served_without_exposing_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            resource_dir = root / "feishu_resources"
            resource_dir.mkdir()
            (resource_dir / "image.png").write_bytes(b"\x89PNG\r\n\x1a\ncontent")
            message = {
                "message_id": "om_image",
                "resource_files": [{"stored_file": "image.png", "content_type": "image/png"}],
            }
            with (
                patch.object(main, "UPLOAD_DIR", root),
                patch.object(main.feishu_workspace, "get_message", return_value=message),
            ):
                response = await main.get_feishu_message_resource("om_image", 0)

            self.assertEqual(response.media_type, "image/png")
            self.assertEqual(Path(response.path).name, "image.png")

    async def test_document_preview_resolves_feishu_display_name_to_stored_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stored_file = "feishu_candidate_fc_test_revision.md"
            (root / stored_file).write_text("# 飞书知识\n\n发布结论：采用灰度流程。", encoding="utf-8")
            asset = {
                "source_file": "飞书知识_项目研发群_test.md",
                "stored_file": stored_file,
                "status": "published",
            }
            with (
                patch.object(main, "UPLOAD_DIR", root),
                patch.object(main, "_get_documents", return_value=[]),
                patch.object(main.feishu_workspace, "list_assets", return_value=[asset]),
            ):
                result = await main.preview_document(asset["source_file"])

            self.assertTrue(result["success"])
            self.assertEqual(result["stored_file"], stored_file)
            self.assertIn("灰度流程", result["content"])

    async def test_reverted_feishu_asset_cannot_preview_residual_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stored_file = "feishu_revoked_secret.md"
            (root / stored_file).write_text("REVOKED_SECRET_CONTENT", encoding="utf-8")
            asset = {
                "source_file": "已撤销飞书知识.md",
                "stored_file": stored_file,
                "status": "reverted",
            }
            with (
                patch.object(main, "UPLOAD_DIR", root),
                patch.object(main, "_get_documents", return_value=[]),
                patch.object(main.feishu_workspace, "list_assets", return_value=[]),
            ):
                with self.assertRaisesRegex(Exception, "文件不存在或已失效"):
                    await main.preview_document(stored_file)

    async def test_unified_governance_revoke_routes_through_feishu_with_real_actor(self):
        identity = SimpleNamespace(user_id="user_knowledge_operator")
        revoke_result = {
            "success": True,
            "knowledge_asset": {"asset_id": "asset_unified", "status": "revoked"},
            "asset": {"asset_id": "fa_legacy", "status": "reverted"},
        }
        with (
            patch.object(main, "_require_knowledge_permission", return_value=identity),
            patch.object(
                main.feishu_knowledge_service,
                "resolve_workspace_asset_id",
                return_value="fa_legacy",
            ),
            patch.object(
                main.feishu_knowledge_service,
                "revert_asset",
                return_value=revoke_result,
            ) as revert,
        ):
            result = await main.transition_knowledge_asset(
                "asset_unified", {"action": "revoke", "reason": "内容已过期"},
            )

        self.assertEqual(result["asset"]["status"], "revoked")
        self.assertEqual(result["feishu"]["asset"]["status"], "reverted")
        revert.assert_called_once_with(
            "fa_legacy", reason="内容已过期", actor="user_knowledge_operator",
        )

    async def test_mark_review_due_creates_governance_task(self):
        identity = SimpleNamespace(user_id="user_knowledge_operator")
        asset = {"asset_id": "asset_review", "title": "项目规范", "status": "review_due"}
        task = {"task_id": "kt_review", "status": "unassigned", "created": True}
        with (
            patch.object(main, "_require_knowledge_permission", return_value=identity),
            patch.object(main.knowledge_asset_service, "transition", return_value=asset),
            patch.object(
                main.knowledge_task_service, "create_asset_review_task", return_value=task,
            ) as create_task,
        ):
            result = await main.transition_knowledge_asset(
                "asset_review", {"action": "mark_review_due", "reason": "到期复审"},
            )

        self.assertEqual(result["asset"]["status"], "review_due")
        self.assertEqual(result["task_sync"]["task_id"], "kt_review")
        create_task.assert_called_once_with(asset, identity)

    async def test_submit_review_creates_exact_asset_publish_task(self):
        identity = SimpleNamespace(user_id="user_knowledge_operator")
        asset = {
            "asset_id": "asset_publish", "title": "上线规范",
            "status": "pending_review", "lifecycle_revision": 3,
        }
        task = {"task_id": "kt_publish", "status": "unassigned", "created": True}
        with (
            patch.object(main, "_require_knowledge_permission", return_value=identity),
            patch.object(main.knowledge_asset_service, "transition", return_value=asset),
            patch.object(
                main.knowledge_task_service, "create_asset_publish_task", return_value=task,
            ) as create_task,
        ):
            result = await main.transition_knowledge_asset(
                "asset_publish", {"action": "submit_review", "expected_revision": 2},
            )

        self.assertEqual(result["asset"]["status"], "pending_review")
        self.assertEqual(result["task_sync"]["task_id"], "kt_publish")
        create_task.assert_called_once_with(asset, identity)

    async def test_task_notification_policy_update_wakes_scheduler(self):
        identity = SimpleNamespace(user_id="user_admin")
        request = SimpleNamespace(state=SimpleNamespace(identity=identity))
        policy = {"enabled": True, "target_chat_id": "oc_reminder"}
        with (
            patch.object(
                main.knowledge_task_service, "update_notification_policy", return_value=policy,
            ) as update,
            patch.object(main.knowledge_task_reminder_scheduler, "wake") as wake,
        ):
            result = await main.update_knowledge_task_notification_policy(
                request, {"enabled": True, "target_chat_id": "oc_reminder"},
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["policy"], policy)
        update.assert_called_once_with(
            {"enabled": True, "target_chat_id": "oc_reminder"}, identity,
        )
        wake.assert_called_once_with()

    async def test_task_notification_scan_requires_task_manager(self):
        denied_identity = SimpleNamespace(can=lambda permission: False)
        denied_request = SimpleNamespace(state=SimpleNamespace(identity=denied_identity))
        with self.assertRaises(HTTPException) as denied:
            await main.scan_knowledge_task_notifications(denied_request)
        self.assertEqual(denied.exception.status_code, 403)

        manager_identity = SimpleNamespace(can=lambda permission: permission == "task.manage")
        manager_request = SimpleNamespace(state=SimpleNamespace(identity=manager_identity))
        scan_result = {"scheduled_count": 2, "job_count": 1}
        with (
            patch.object(
                main.knowledge_task_service, "schedule_due_notifications", return_value=scan_result,
            ),
            patch.object(main.processing_job_runner, "wake") as wake,
        ):
            result = await main.scan_knowledge_task_notifications(manager_request)
        self.assertEqual(result["message"], "已安排 2 项提醒")
        wake.assert_called_once_with()

    async def test_governance_policy_update_wakes_daily_scheduler(self):
        identity = SimpleNamespace(user_id="user_admin")
        policy = {"daily_backup_enabled": True, "schedule_time": "02:00"}
        with (
            patch.object(main, "get_current_identity", return_value=identity),
            patch.object(main.data_governance_service, "update_policy", return_value=policy) as update,
            patch.object(main.data_governance_scheduler, "wake") as wake,
        ):
            result = await main.update_governance_policy(policy)
        self.assertTrue(result["success"])
        update.assert_called_once_with(policy, identity.user_id)
        wake.assert_called_once_with()

    async def test_knowledge_library_reset_uses_current_actor_and_refuses_active_ingestion(self):
        identity = SimpleNamespace(user_id="user_admin")
        reset_result = {
            "sources_deleted": 2, "assets_deleted": 4, "documents_deleted": 3,
            "history": {
                "processing_jobs": 4, "knowledge_tasks": 5,
                "onboarding_learning_plans": 6, "handover_cases": 7,
            },
            "feishu_history": {
                "messages_deleted": 8, "candidates_deleted": 9, "assets_deleted": 10,
                "resource_file_cleanup": {"removed": ["message-image.png"]},
            },
        }
        repository = SimpleNamespace(
            list_processing_jobs=lambda **_kwargs: {"items": []},
        )
        with (
            patch.object(main, "get_current_identity", return_value=identity),
            patch.object(main, "get_repository", return_value=repository),
            patch.object(main.data_governance_service, "reset_knowledge_library", return_value=reset_result) as reset,
        ):
            result = await main.reset_knowledge_library()
        self.assertTrue(result["success"])
        self.assertIn("已撤销", result["message"])
        self.assertIn("8 条飞书历史消息", result["message"])
        self.assertIn("飞书采集群配置", result["message"])
        self.assertIn("未创建备份", result["message"])
        reset.assert_called_once_with(actor_user_id="user_admin")

        busy_repository = SimpleNamespace(list_processing_jobs=lambda **_kwargs: {
            "items": [{"job_type": "document_ingestion"}],
        })
        with (
            patch.object(main, "get_current_identity", return_value=identity),
            patch.object(main, "get_repository", return_value=busy_repository),
            self.assertRaisesRegex(HTTPException, "进行中的处理任务"),
        ):
            await main.reset_knowledge_library()

    async def test_manual_governance_scan_wakes_processing_runner_for_new_jobs(self):
        scan_result = {
            "enabled": True, "job_count": 1, "jobs": [{"job_id": "job_backup"}],
            "local_date": "2026-08-28", "next_scheduled_at": "2026-08-29T18:00:00+00:00",
        }
        with (
            patch.object(main.data_governance_service, "schedule_daily_operations", return_value=scan_result),
            patch.object(main.processing_job_runner, "wake") as wake,
        ):
            result = await main.scan_governance_schedule()
        self.assertEqual(result["job_count"], 1)
        self.assertIn("新安排 1 个任务", result["message"])
        wake.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
