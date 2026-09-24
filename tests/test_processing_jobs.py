import asyncio
import tempfile
import unittest
from pathlib import Path

from backend.auth_context import Identity
from backend.processing_jobs import (
    ProcessingJobCancelled,
    ProcessingJobContext,
    ProcessingJobError,
    ProcessingJobRunner,
    ProcessingJobService,
    sanitize_value,
)
from backend.storage.database import Database
from backend.storage.repository import StorageRepository


def identity(user_id="user_system", permissions=frozenset({"job.read", "job.manage"})):
    return Identity(
        user_id=user_id, email=f"{user_id}@example.com", display_name=user_id,
        organization_id="org_default", project_id="project_default", project_name="默认项目",
        role="project_manager", permissions=permissions,
    )


class ProcessingJobRepositoryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repository = StorageRepository(Database(Path(self.temp.name) / "jobs.db"))

    def tearDown(self):
        self.temp.cleanup()

    def create_job(self, **overrides):
        payload = {
            "job_type": "test", "payload": {"value": 1}, "created_by": "user_system",
            "idempotency_key": "test:one", "max_attempts": 3,
        }
        payload.update(overrides)
        return self.repository.create_or_get_processing_job(payload)

    def test_schema_and_open_idempotency(self):
        with self.repository.database.transaction() as connection:
            tables = {row["name"] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )}
        self.assertTrue({"processing_jobs", "processing_job_attempts", "processing_job_events"} <= tables)
        first = self.create_job()
        second = self.create_job()
        self.assertTrue(first["created"])
        self.assertFalse(second["created"])
        self.assertEqual(first["job_id"], second["job_id"])

    def test_claim_failure_retry_and_recovery_are_durable(self):
        job = self.create_job()
        claimed = self.repository.claim_next_processing_job("worker-a")
        self.assertEqual(claimed["job_id"], job["job_id"])
        self.assertEqual(claimed["attempts"], 1)
        state = self.repository.fail_processing_job(
            job["job_id"], error_code="TEMP", error_message="temporary", retry_delay_seconds=0,
        )
        self.assertEqual(state, "retry_wait")
        claimed = self.repository.claim_next_processing_job("worker-b")
        self.assertEqual(claimed["attempts"], 2)
        self.assertEqual(self.repository.recover_processing_jobs(), 1)
        detail = self.repository.get_processing_job(job["job_id"], can_manage=True)
        self.assertEqual(detail["status"], "retry_wait")
        self.assertEqual(detail["attempt_history"][0]["status"], "interrupted")
        self.assertTrue(any(event["event_type"] == "recovered" for event in detail["events"]))

    def test_event_timeline_keeps_insertion_order_with_same_second_timestamps(self):
        job = self.create_job()
        self.repository.claim_next_processing_job("worker")
        self.repository.update_processing_job_progress(
            job["job_id"], stage="work", progress=50, message="processing",
        )
        self.repository.complete_processing_job(job["job_id"], {"ok": True})
        detail = self.repository.get_processing_job(job["job_id"], can_manage=True)
        self.assertEqual(
            [event["event_type"] for event in detail["events"]],
            ["created", "started", "progress", "succeeded"],
        )

    def test_creator_scope_cancel_and_manual_retry(self):
        own = self.create_job(idempotency_key="own", created_by="member-a")
        other = self.create_job(idempotency_key="other", created_by="member-b")
        listing = self.repository.list_processing_jobs(viewer_user_id="member-a", can_manage=False)
        self.assertEqual([item["job_id"] for item in listing["items"]], [own["job_id"]])
        cancelled = self.repository.cancel_processing_job(own["job_id"], actor_user_id="member-a")
        self.assertEqual(cancelled["status"], "cancelled")
        retried = self.repository.retry_processing_job(own["job_id"], actor_user_id="manager")
        self.assertEqual(retried["status"], "queued")
        self.assertIsNone(self.repository.get_processing_job(other["job_id"], viewer_user_id="member-a"))

    def test_list_searches_document_filename_in_payload(self):
        job = self.create_job(
            idempotency_key="document-name",
            payload={"original_filename": "项目技术实现方案.docx"},
        )
        result = self.repository.list_processing_jobs(
            viewer_user_id="user_system", can_manage=True, keyword="技术实现方案",
        )
        self.assertEqual([item["job_id"] for item in result["items"]], [job["job_id"]])

    def test_running_cancel_is_cooperative(self):
        job = self.create_job()
        self.repository.claim_next_processing_job("worker")
        requested = self.repository.cancel_processing_job(job["job_id"], actor_user_id="user_system")
        self.assertEqual(requested["status"], "running")
        self.assertTrue(requested["cancel_requested"])
        self.repository.mark_processing_job_cancelled(job["job_id"])
        self.assertEqual(self.repository.get_processing_job(job["job_id"], can_manage=True)["status"], "cancelled")


class ProcessingJobRunnerTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repository = StorageRepository(Database(Path(self.temp.name) / "runner.db"))
        self.service = ProcessingJobService(lambda: self.repository)
        self.runner = ProcessingJobRunner(self.service, poll_seconds=0.02)

    async def asyncTearDown(self):
        await self.runner.stop()
        self.temp.cleanup()

    async def wait_for(self, job_id, terminal=("succeeded", "failed", "cancelled")):
        for _ in range(200):
            job = self.repository.get_processing_job(job_id, can_manage=True)
            if job["status"] in terminal:
                return job
            await asyncio.sleep(0.02)
        self.fail("processing job did not reach terminal state")

    async def test_runner_executes_registered_handler_and_persists_progress(self):
        def handler(context: ProcessingJobContext, payload):
            context.report("work", 55, "处理中")
            return {"answer": payload["value"] * 2}

        self.service.register("double", handler)
        await self.runner.start()
        job = self.service.enqueue("double", {"value": 4}, created_by="user_system")
        finished = await self.wait_for(job["job_id"])
        self.assertEqual(finished["status"], "succeeded")
        self.assertEqual(finished["result"], {"answer": 8})
        self.assertTrue(any(event["stage"] == "work" for event in finished["events"]))

    async def test_non_retryable_handler_error_is_terminal(self):
        def handler(context, payload):
            raise ProcessingJobError("invalid input", code="INVALID", retryable=False)

        self.service.register("invalid", handler)
        await self.runner.start()
        job = self.service.enqueue("invalid", {}, created_by="user_system")
        finished = await self.wait_for(job["job_id"])
        self.assertEqual(finished["status"], "failed")
        self.assertEqual(finished["error_code"], "INVALID")

    async def test_completed_non_interruptible_handler_commits_after_late_cancel_request(self):
        def handler(context, payload):
            self.repository.cancel_processing_job(context.job_id, actor_user_id="user_system")
            return {"published": True, "_job_commit_on_cancel": True}

        self.service.register("commit_after_cancel", handler)
        await self.runner.start()
        job = self.service.enqueue("commit_after_cancel", {}, created_by="user_system")
        finished = await self.wait_for(job["job_id"])
        self.assertEqual(finished["status"], "succeeded")
        self.assertEqual(finished["result"], {"published": True})

    async def test_cancelled_handler_persists_partial_item_result(self):
        def handler(context, payload):
            raise ProcessingJobCancelled({"processed": 1, "items": [{"id": "one", "status": "succeeded"}]})

        self.service.register("cancel_with_result", handler)
        await self.runner.start()
        job = self.service.enqueue("cancel_with_result", {}, created_by="user_system")
        finished = await self.wait_for(job["job_id"])
        self.assertEqual(finished["status"], "cancelled")
        self.assertEqual(finished["result"]["processed"], 1)
        self.assertEqual(finished["result"]["items"][0]["status"], "succeeded")

    def test_payload_and_error_sanitization(self):
        value = sanitize_value({"api_key": "secret", "nested": {"token": "abc"}, "text": "Bearer xyz"})
        self.assertEqual(value["api_key"], "***")
        self.assertEqual(value["nested"]["token"], "***")
        self.assertEqual(value["text"], "Bearer ***")


class ProcessingJobPermissionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        repository = StorageRepository(Database(Path(self.temp.name) / "permissions.db"))
        self.repository = repository
        self.service = ProcessingJobService(lambda: repository)
        self.service.register("test", lambda context, payload: {})

    def tearDown(self):
        self.temp.cleanup()

    def test_member_reads_only_own_and_can_retry_own_failure(self):
        own = self.service.enqueue(
            "test", {"original_filename": "项目技术实现方案.docx"},
            created_by="member-a", idempotency_key="a",
        )
        other = self.service.enqueue("test", {}, created_by="member-b", idempotency_key="b")
        member = identity("member-a", frozenset({"job.read"}))
        result = self.service.list_jobs(member)
        self.assertEqual([item["job_id"] for item in result["items"]], [own["job_id"]])
        self.assertEqual(result["items"][0]["display_name"], "项目技术实现方案.docx")
        self.assertEqual(result["items"][0]["allowed_actions"], ["cancel"])
        self.repository.cancel_processing_job(own["job_id"], actor_user_id="member-a")
        detail = self.service.get_job(own["job_id"], member)
        self.assertEqual(detail["allowed_actions"], ["retry"])
        retried = self.service.retry(own["job_id"], member)
        self.assertEqual(retried["status"], "queued")
        self.assertEqual(retried["allowed_actions"], ["cancel"])
        with self.assertRaisesRegex(ValueError, "不存在或无权访问"):
            self.service.retry(other["job_id"], member)

    def test_queued_cancel_invokes_registered_cleanup(self):
        cleaned = []
        self.service.register_cleanup("test", lambda job: cleaned.append(job["job_id"]))
        job = self.service.enqueue("test", {}, created_by="member-a", idempotency_key="cleanup")
        result = self.service.cancel(job["job_id"], identity("member-a", frozenset({"job.read"})))
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(cleaned, [job["job_id"]])


if __name__ == "__main__":
    unittest.main()
