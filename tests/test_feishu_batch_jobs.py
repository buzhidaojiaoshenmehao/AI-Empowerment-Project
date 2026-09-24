import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from backend import main
from backend.auth_context import Identity, get_current_identity
from backend.processing_jobs import ProcessingJobCancelled


def identity():
    return Identity(
        user_id="publisher", email="publisher@example.com", display_name="发布者",
        organization_id="org_default", project_id="project_default", project_name="默认项目",
        role="project_manager", permissions=frozenset({"knowledge.publish", "job.read"}),
    )


class _Context:
    def __init__(self, cancel_on_item=0):
        self.events = []
        self.item_reports = 0
        self.cancel_on_item = cancel_on_item

    def report(self, stage, progress, message):
        self.events.append((stage, progress, message))
        if stage == "feishu_batch_item":
            self.item_reports += 1
            if self.cancel_on_item and self.item_reports == self.cancel_on_item:
                raise ProcessingJobCancelled()


class FeishuCandidateBatchJobTest(unittest.TestCase):
    def test_async_endpoint_returns_persistent_job_without_processing_inline(self):
        service = Mock()
        service.enqueue.return_value = {"job_id": "job_batch", "status": "queued", "created": True}
        with patch.object(main, "_require_knowledge_permission", return_value=identity()), \
                patch.object(main, "processing_job_service", service):
            result = asyncio.run(main.enqueue_feishu_candidate_batch({
                "candidate_ids": ["candidate_2", "candidate_1", "candidate_2"],
                "action": "approve",
            }))
        self.assertTrue(result["accepted"])
        self.assertEqual(result["job"]["job_id"], "job_batch")
        self.assertEqual(service.enqueue.call_args.args[0], "feishu_candidate_batch")
        self.assertEqual(
            service.enqueue.call_args.args[1]["candidate_ids"], ["candidate_2", "candidate_1"],
        )

    def test_job_rechecks_permission_and_keeps_per_item_failures(self):
        service = Mock()
        service.batch_action.side_effect = [
            {"succeeded": [{"candidate_id": "candidate_1", "result": {"asset": {"asset_id": "asset_1"}}}], "failed": []},
            {"succeeded": [], "failed": [{"candidate_id": "candidate_2", "error": "知识候选不存在"}]},
            {"succeeded": [{"candidate_id": "candidate_3", "result": {}}], "failed": []},
        ]
        with patch.object(main, "_processing_identity", return_value=identity()) as permission_check, \
                patch.object(main, "feishu_knowledge_service", service):
            result = main._job_feishu_candidate_batch(_Context(), {
                "candidate_ids": ["candidate_1", "candidate_2", "candidate_3"],
                "action": "approve", "actor_user_id": "publisher",
            })
        permission_check.assert_called_once_with("publisher", "knowledge.publish", "FEISHU_BATCH")
        self.assertFalse(result["success"])
        self.assertTrue(result["partial_success"])
        self.assertEqual(result["processed"], 3)
        self.assertEqual(result["succeeded_count"], 2)
        self.assertEqual(result["failed_count"], 1)
        self.assertEqual([item["status"] for item in result["items"]], ["succeeded", "failed", "succeeded"])
        self.assertIsNone(get_current_identity())

    def test_cancellation_keeps_results_completed_before_next_item(self):
        service = Mock()
        service.batch_action.return_value = {
            "succeeded": [{"candidate_id": "candidate_1", "result": {}}], "failed": [],
        }
        with patch.object(main, "_processing_identity", return_value=identity()), \
                patch.object(main, "feishu_knowledge_service", service):
            with self.assertRaises(ProcessingJobCancelled) as raised:
                main._job_feishu_candidate_batch(_Context(cancel_on_item=2), {
                    "candidate_ids": ["candidate_1", "candidate_2"],
                    "action": "exclude", "actor_user_id": "publisher",
                })
        self.assertEqual(raised.exception.result["processed"], 1)
        self.assertEqual(raised.exception.result["total"], 2)
        self.assertEqual(raised.exception.result["items"][0]["candidate_id"], "candidate_1")


if __name__ == "__main__":
    unittest.main()
