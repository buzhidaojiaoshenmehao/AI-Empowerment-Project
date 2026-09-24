import asyncio
import json
import unittest
from unittest.mock import AsyncMock, Mock, patch

from backend import main
from backend.auth_context import Identity, get_current_identity
from backend.feishu_bot import FeishuBot, handle_feishu_webhook
from backend.feishu_workspace import evaluate_content, feishu_workspace
from backend.processing_jobs import ProcessingJobError, processing_job_service


def identity():
    return Identity(
        user_id="publisher", email="publisher@example.com", display_name="发布者",
        organization_id="org_default", project_id="project_default", project_name="默认项目",
        role="project_manager", permissions=frozenset({"knowledge.publish", "job.read", "job.manage"}),
    )


class _Context:
    def __init__(self):
        self.events = []

    def report(self, stage, progress, message):
        self.events.append((stage, progress, message))


class FeishuEnrichmentJobTest(unittest.TestCase):
    def test_queued_image_is_not_auto_published_before_ocr(self):
        result = evaluate_content(
            {"message_type": "image", "content": "[暂不支持直接预览的消息]", "extraction_status": "queued"},
            {"collection_mode": "auto", "confidence_threshold": 0.85},
        )
        self.assertEqual(result["content_status"], "review_required")
        self.assertIn("后台识别", result["reason"])

    def test_manual_endpoint_returns_job_without_running_ocr_inline(self):
        service = Mock()
        service.enqueue.return_value = {"job_id": "job_ocr", "status": "queued", "created": True}
        message = {"message_id": "om_image", "message_type": "image", "chat_id": "oc_project"}
        with patch.object(main, "_require_knowledge_permission", return_value=identity()), \
                patch.object(main.feishu_workspace, "get_message", return_value=message), \
                patch.object(main, "processing_job_service", service):
            result = asyncio.run(main.enqueue_feishu_message_enrichment("om_image"))
        self.assertTrue(result["accepted"])
        self.assertEqual(result["job"]["job_id"], "job_ocr")
        self.assertEqual(service.enqueue.call_args.args[0], "feishu_message_enrichment")
        self.assertEqual(service.enqueue.call_args.args[1]["message_id"], "om_image")

    def test_job_rechecks_permission_then_archives_successful_ocr(self):
        message = {
            "message_id": "om_image", "message_type": "image", "chat_id": "oc_project",
            "extraction_status": "queued", "image_key": "img_1",
        }
        updated = {
            **message, "extraction_status": "completed", "extraction_method": "local_rapidocr",
            "resource_count": 1, "content_status": "review_required", "candidate_id": "candidate_1",
        }
        context = _Context()
        with patch.object(main, "_processing_identity", return_value=identity()) as permission_check, \
                patch.object(main.feishu_workspace, "get_message", return_value=message), \
                patch.object(main.feishu_workspace, "get_group", return_value={"collection_mode": "review"}), \
                patch.object(main.feishu_workspace, "replace_enriched_message", return_value=updated), \
                patch.object(main.feishu_bot, "retry_enrich_message", new=AsyncMock(return_value={"message_id": "om_image"})):
            result = asyncio.run(main._job_feishu_message_enrichment(context, {
                "message_id": "om_image", "trigger": "manual", "actor_user_id": "publisher",
            }))
        permission_check.assert_called_once_with("publisher", "knowledge.publish", "FEISHU_ENRICHMENT")
        self.assertTrue(result["_job_commit_on_cancel"])
        self.assertEqual(result["extraction_method"], "local_rapidocr")
        self.assertEqual(result["resource_count"], 1)
        self.assertEqual(
            [event[0] for event in context.events],
            ["feishu_enrich_validate", "feishu_enrich_extract", "feishu_enrich_archive", "feishu_enrich_publish"],
        )
        self.assertIsNone(get_current_identity())

    def test_failed_ocr_remains_retryable_and_keeps_failed_record(self):
        message = {
            "message_id": "om_failed", "message_type": "image", "chat_id": "oc_project",
            "extraction_status": "queued",
        }
        failed = {**message, "extraction_status": "failed", "extraction_error": "OCR 暂时不可用"}
        with patch.object(main, "_processing_identity", return_value=identity()), \
                patch.object(main.feishu_workspace, "get_message", return_value=message), \
                patch.object(main.feishu_workspace, "get_group", return_value={"collection_mode": "review"}), \
                patch.object(main.feishu_workspace, "replace_enriched_message", return_value=failed), \
                patch.object(main.feishu_bot, "retry_enrich_message", new=AsyncMock(return_value={"message_id": "om_failed"})):
            with self.assertRaises(ProcessingJobError) as raised:
                asyncio.run(main._job_feishu_message_enrichment(_Context(), {
                    "message_id": "om_failed", "trigger": "manual", "actor_user_id": "publisher",
                }))
        self.assertTrue(raised.exception.retryable)
        self.assertEqual(raised.exception.code, "FEISHU_ENRICHMENT_FAILED")


class FeishuWebhookAsyncEnrichmentTest(unittest.IsolatedAsyncioTestCase):
    async def test_image_webhook_archives_queued_state_and_returns_job_before_ocr(self):
        payload = {
            "header": {"event_type": "im.message.receive_v1", "event_id": "event_image"},
            "event": {
                "sender": {"sender_id": {"open_id": "ou_member"}},
                "message": {
                    "message_id": "om_async_image", "chat_id": "oc_project", "chat_type": "p2p",
                    "message_type": "image", "content": json.dumps({"image_key": "img_async"}),
                    "mentions": [{"id": "bot"}],
                },
            },
        }
        group = {"chat_id": "oc_project", "collection_mode": "auto", "bot_enabled": True}
        job = {"job_id": "job_async_ocr", "status": "queued", "created": True}
        with patch.object(feishu_workspace, "get_group", return_value=group), \
                patch.object(feishu_workspace, "get_message", side_effect=[None, {"extraction_status": "queued"}]), \
                patch.object(feishu_workspace, "upsert_messages", return_value=1) as archive, \
                patch.object(feishu_workspace, "record_event"), \
                patch.object(processing_job_service, "enqueue", return_value=job) as enqueue, \
                patch.object(FeishuBot, "enrich_messages", new=AsyncMock(side_effect=AssertionError("OCR 不应在 Webhook 请求中执行"))), \
                patch.object(FeishuBot, "reply_text_message", new=AsyncMock(return_value=True)):
            result = await handle_feishu_webhook(payload)
        self.assertEqual(result["action"], "image_enrichment_queued")
        self.assertEqual(result["job_id"], "job_async_ocr")
        self.assertEqual(enqueue.call_args.args[0], "feishu_message_enrichment")
        self.assertEqual(enqueue.call_args.args[1], {"message_id": "om_async_image", "trigger": "webhook"})
        queued_raw = archive.call_args.args[1][0]
        queued_content = json.loads(queued_raw["content"])
        self.assertEqual(queued_content["extraction_status"], "queued")
        self.assertEqual(queued_content["original_content"]["image_key"], "img_async")


if __name__ == "__main__":
    unittest.main()
