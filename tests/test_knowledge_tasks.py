import asyncio
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from backend.auth import ROLE_PERMISSIONS
from backend.auth_context import Identity
from backend.knowledge_assets import KnowledgeAssetService
from backend.knowledge_tasks import KnowledgeTaskService
from backend.processing_jobs import ProcessingJobError, ProcessingJobRunner, ProcessingJobService
from backend.storage.database import Database
from backend.storage.repository import StorageRepository


class KnowledgeTaskWorkflowTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repository = StorageRepository(Database(Path(self.temp.name) / "app.db"))
        self.service = KnowledgeTaskService(lambda: self.repository)
        self.assets = KnowledgeAssetService(lambda: self.repository)
        self.admin = self.repository.provision_project_member(
            "admin@example.com", "项目管理员", "project_admin",
        )
        self.member = self.repository.provision_project_member(
            "member@example.com", "项目成员", "project_member",
        )
        self.other = self.repository.provision_project_member(
            "other@example.com", "其他成员", "project_member",
        )
        self.admin = self.repository.set_demo_login_password(
            self.admin["user_id"], "test-password-hash",
        )
        self.member = self.repository.set_demo_login_password(
            self.member["user_id"], "test-password-hash",
        )
        self.other = self.repository.set_demo_login_password(
            self.other["user_id"], "test-password-hash",
        )
        self.admin_identity = self._identity(self.admin)
        self.member_identity = self._identity(self.member)
        self.other_identity = self._identity(self.other)

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def _identity(member):
        role = str(member["role"])
        return Identity(
            user_id=str(member["user_id"]), email=str(member["email"]),
            display_name=str(member["display_name"]), organization_id="org_default",
            project_id=str(member["project_id"]), project_name=str(member["project_name"]),
            role=role, permissions=ROLE_PERMISSIONS[role], session_id="test",
        )

    def _assigned_task(self):
        return self.service.create({
            "task_type": "knowledge_gap",
            "title": "补齐部署知识",
            "description": "缺少可复用部署说明",
            "dedupe_key": "gap:deploy",
            "assignee_user_id": self.member["user_id"],
        }, self.admin_identity)

    def _due_task(self, suffix: str, due_at: datetime):
        return self.service.create({
            "task_type": "knowledge_gap", "title": f"到期任务-{suffix}",
            "description": "用于验证自动提醒调度",
            "dedupe_key": f"due-task:{suffix}",
            "assignee_user_id": self.member["user_id"],
            "due_at": due_at.astimezone(timezone.utc).isoformat(timespec="seconds"),
            "priority": "high",
        }, self.admin_identity)

    def _set_notification_policy(self, **overrides):
        payload = {
            "enabled": True, "target_chat_id": "oc_task_reminders", "timezone": "UTC",
            "quiet_start": "00:00", "quiet_end": "00:00", "remind_before_hours": 24,
            "reminder_interval_hours": 24, "escalation_after_hours": 24,
            "escalation_interval_hours": 24,
        }
        payload.update(overrides)
        return self.service.update_notification_policy(payload, self.admin_identity)

    def _asset_review_task(self, suffix: str = "default"):
        document = SimpleNamespace(page_content=f"复审内容 {suffix}", metadata={})
        prepared = self.assets.prepare_documents(
            [document], source_type="manual_upload", external_key=f"review-{suffix}.md",
            title=f"复审资料-{suffix}.md", actor=self.admin["user_id"],
            stored_file=f"review-{suffix}.md", owner_user_id=self.member["user_id"],
        )
        self.repository.upsert_document_chunks([document], [f"review-vector-{suffix}"])
        self.assets.record_projection(prepared["version_id"], "vector", succeeded=True)
        self.assets.publish(prepared, actor=self.admin["user_id"])
        asset = self.assets.transition(
            prepared["asset_id"], "mark_review_due", self.admin["user_id"], "到期复审",
        )
        task = self.service.create_asset_review_task(asset, self.admin_identity)
        return prepared, asset, task

    def _asset_publish_task(self, suffix: str = "default"):
        document = SimpleNamespace(page_content=f"待发布内容 {suffix}", metadata={})
        prepared = self.assets.prepare_documents(
            [document], source_type="manual_upload", external_key=f"publish-{suffix}.md",
            title=f"待发布资料-{suffix}.md", actor=self.admin["user_id"],
            stored_file=f"publish-{suffix}.md", owner_user_id=self.member["user_id"],
        )
        self.repository.upsert_document_chunks([document], [f"publish-vector-{suffix}"])
        self.assets.record_projection(prepared["version_id"], "vector", succeeded=True)
        for projection_type in ("graph", "context", "readiness"):
            self.repository.set_asset_projection(prepared["version_id"], projection_type, "ready")
        self.repository.mark_asset_version_ready(prepared["version_id"])
        asset = self.assets.transition(
            prepared["asset_id"], "submit_review", self.admin["user_id"],
        )
        task = self.service.create_asset_publish_task(asset, self.admin_identity)
        return prepared, asset, task

    def _active_topic_asset(
        self, category: str, *, suffix: str, visibility: str = "project",
        owner_user_id: str = "",
    ):
        document = SimpleNamespace(page_content=f"{category} 可验证内容 {suffix}", metadata={})
        prepared = self.assets.prepare_documents(
            [document], source_type="manual_upload", external_key=f"{suffix}.md",
            title=f"{category}-{suffix}.md", actor=self.admin["user_id"],
            stored_file=f"{suffix}.md", categories=[category], visibility=visibility,
            owner_user_id=owner_user_id or self.admin["user_id"],
        )
        self.repository.upsert_document_chunks([document], [f"topic-vector-{suffix}"])
        self.assets.record_projection(prepared["version_id"], "vector", succeeded=True)
        self.assets.publish(prepared, actor=self.admin["user_id"])
        return prepared

    def test_schema_and_dedupe_merge_keep_existing_state(self):
        required = {
            "knowledge_tasks", "knowledge_task_events", "task_notifications",
            "knowledge_task_notification_policies",
        }
        with self.repository.database.transaction() as connection:
            tables = {str(row[0]) for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )}
        self.assertTrue(required.issubset(tables))

        task = self._assigned_task()
        started = self.service.act(task["task_id"], "start", {}, self.member_identity)
        merged = self.service.create({
            "task_type": "knowledge_gap", "title": "重复部署缺口",
            "dedupe_key": "gap:deploy", "assignee_user_id": self.member["user_id"],
        }, self.admin_identity)

        self.assertFalse(merged["created"])
        self.assertEqual(merged["occurrence_count"], 2)
        self.assertEqual(merged["status"], "in_progress")
        self.assertEqual(merged["events"][-1]["event_type"], "merged")
        self.assertEqual(merged["events"][-1]["to_status"], "in_progress")
        self.assertEqual(started["status"], "in_progress")

    def test_allowed_actions_and_block_reason_are_server_authoritative(self):
        task = self._assigned_task()

        member_view = self.service.get(task["task_id"], self.member_identity)
        admin_view = self.service.get(task["task_id"], self.admin_identity)
        self.assertIn("start", member_view["allowed_actions"])
        self.assertIn("start", admin_view["allowed_actions"])
        self.assertEqual(member_view["primary_action_block_reason"], "")

        self.service.act(task["task_id"], "start", {}, self.member_identity)
        submitted = self.service.act(
            task["task_id"], "submit", {"evidence": "已发布部署资料"}, self.member_identity,
        )
        self.assertNotIn("accept", submitted["allowed_actions"])
        self.assertIn("不能验收自己的任务", submitted["primary_action_block_reason"])
        self.assertIn("accept", self.service.get(task["task_id"], self.admin_identity)["allowed_actions"])

    def test_manager_can_transfer_self_assigned_task_then_accept_independently(self):
        task = self.service.create({
            "task_type": "knowledge_gap",
            "title": "补充技术架构资料",
            "assignee_user_id": self.admin["user_id"],
        }, self.admin_identity)
        self.service.act(task["task_id"], "start", {}, self.admin_identity)
        submitted = self.service.act(
            task["task_id"], "submit", {"evidence": "已关联生效架构资料"}, self.admin_identity,
        )

        self.assertNotIn("accept", submitted["allowed_actions"])
        self.assertIn("请先转派给其他成员复核", submitted["primary_action_block_reason"])

        transferred = self.service.act(
            task["task_id"], "assign",
            {"assignee_user_id": self.member["user_id"]}, self.admin_identity,
        )
        self.assertEqual(transferred["status"], "pending_acceptance")
        self.assertEqual(transferred["completion_evidence"], "已关联生效架构资料")
        self.assertIn("accept", transferred["allowed_actions"])
        self.assertEqual(transferred["primary_action_block_reason"], "")

        accepted = self.service.act(task["task_id"], "accept", {}, self.admin_identity)
        self.assertEqual(accepted["status"], "completed")

    def test_effective_visible_asset_automatically_submits_matching_onboarding_gap(self):
        gap = self.service.create_onboarding_gap_tasks(
            "开发工程师",
            [{"category": "proj_architecture", "label": "技术架构", "weight": 2}],
            self.admin_identity, assignee_user_id=self.member["user_id"],
        )["tasks"][0]
        prepared = self._active_topic_asset("proj_architecture", suffix="architecture")

        result = self.service.reconcile_onboarding_gap_tasks(
            actor=self.admin["user_id"], trigger_asset_id=prepared["asset_id"],
        )
        updated = self.service.get(gap["task_id"], self.member_identity)

        self.assertEqual(result["matched_count"], 1)
        self.assertEqual(updated["status"], "pending_acceptance")
        self.assertEqual(updated["linked_asset_id"], prepared["asset_id"])
        self.assertIn("系统已匹配生效知识", updated["completion_evidence"])
        self.assertEqual(updated["events"][-1]["event_type"], "auto_evidence_matched")
        self.assertEqual(
            self.service.reconcile_onboarding_gap_tasks(actor=self.admin["user_id"])["matched_count"],
            0,
        )

    def test_onboarding_gap_does_not_match_asset_hidden_from_assignee(self):
        gap = self.service.create_onboarding_gap_tasks(
            "开发工程师",
            [{"category": "proj_architecture", "label": "技术架构", "weight": 2}],
            self.admin_identity, assignee_user_id=self.member["user_id"],
        )["tasks"][0]
        self._active_topic_asset(
            "proj_architecture", suffix="private-architecture",
            visibility="private", owner_user_id=self.other["user_id"],
        )

        result = self.service.reconcile_onboarding_gap_tasks(actor=self.admin["user_id"])
        updated = self.service.get(gap["task_id"], self.member_identity)

        self.assertEqual(result["matched_count"], 0)
        self.assertEqual(updated["status"], "open")
        self.assertEqual(updated["linked_asset_id"], "")

    def test_cancel_requires_reason_and_reporter_or_manager_permission(self):
        task = self._assigned_task()

        with self.assertRaisesRegex(ValueError, "必须说明原因"):
            self.service.act(task["task_id"], "cancel", {}, self.admin_identity)
        with self.assertRaisesRegex(PermissionError, "报告人或任务管理人员"):
            self.service.act(
                task["task_id"], "cancel", {"note": "误创建"}, self.member_identity,
            )

        cancelled = self.service.act(
            task["task_id"], "cancel", {"note": "误创建，取消并保留审计"}, self.admin_identity,
        )

        self.assertEqual(cancelled["status"], "cancelled")
        self.assertEqual(cancelled["events"][-1]["event_type"], "cancel")
        self.assertEqual(cancelled["events"][-1]["note"], "误创建，取消并保留审计")

    def test_notification_policy_is_guarded_and_validated(self):
        self.assertFalse(self.service.get_notification_policy()["enabled"])
        with self.assertRaisesRegex(PermissionError, "任务管理人员"):
            self.service.update_notification_policy({"enabled": False}, self.member_identity)
        with self.assertRaisesRegex(ValueError, "HH:MM"):
            self._set_notification_policy(quiet_start="25:00")
        with self.assertRaisesRegex(ValueError, "时区无效"):
            self._set_notification_policy(timezone="Mars/Project")
        with self.assertRaisesRegex(ValueError, "目标飞书群"):
            self._set_notification_policy(target_chat_id="")

        saved = self._set_notification_policy(
            quiet_start="21:30", quiet_end="07:45", reminder_interval_hours=12,
        )
        self.assertTrue(saved["enabled"])
        self.assertEqual(saved["quiet_start"], "21:30")
        self.assertEqual(saved["reminder_interval_hours"], 12)

    def test_due_notifications_are_grouped_persistent_and_idempotent(self):
        current = datetime(2026, 8, 28, 9, 0, tzinfo=timezone.utc)
        self._set_notification_policy()
        first = self._due_task("digest-a", current + timedelta(hours=5))
        second = self._due_task("digest-b", current + timedelta(hours=8))

        scheduled = self.service.schedule_due_notifications(now=current)
        repeated = self.service.schedule_due_notifications(now=current + timedelta(minutes=5))

        self.assertEqual(scheduled["scheduled_count"], 2)
        self.assertEqual(scheduled["job_count"], 1)
        self.assertEqual(repeated["scheduled_count"], 0)
        job = scheduled["jobs"][0]
        self.assertEqual(job["job_type"], "knowledge_task_notification")
        self.assertEqual(set(job["payload"]["notification_ids"]), {
            self.service.get(first["task_id"], self.admin_identity)["notifications"][0]["notification_id"],
            self.service.get(second["task_id"], self.admin_identity)["notifications"][0]["notification_id"],
        })
        self.assertTrue(all(
            self.service.get(task["task_id"], self.admin_identity)["notifications"][0]["status"] == "pending"
            for task in (first, second)
        ))

    def test_quiet_hours_defer_delivery_without_duplicate_jobs(self):
        current = datetime(2026, 8, 28, 15, 0, tzinfo=timezone.utc)  # 23:00 in Shanghai.
        self._set_notification_policy(
            timezone="Asia/Shanghai", quiet_start="22:00", quiet_end="08:00",
        )
        self._due_task("quiet", current + timedelta(hours=4))

        scheduled = self.service.schedule_due_notifications(now=current)
        repeated = self.service.schedule_due_notifications(now=current + timedelta(hours=1))

        self.assertTrue(scheduled["deferred_by_quiet_hours"])
        self.assertEqual(scheduled["available_at"], "2026-08-29T00:00:00+00:00")
        self.assertEqual(scheduled["jobs"][0]["available_at"], scheduled["available_at"])
        self.assertEqual(repeated["scheduled_count"], 0)

    def test_overdue_notifications_advance_deterministic_escalation_level(self):
        current = datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)
        self._set_notification_policy(escalation_after_hours=24, escalation_interval_hours=24)
        task = self._due_task("escalation", current - timedelta(hours=50))

        scheduled = self.service.schedule_due_notifications(now=current)
        notification = self.service.get(task["task_id"], self.admin_identity)["notifications"][0]

        self.assertEqual(scheduled["scheduled_count"], 1)
        self.assertEqual(notification["notification_kind"], "escalation")
        self.assertEqual(notification["escalation_level"], 2)
        self.assertIn("level:2", notification["dedupe_key"])

    async def test_scheduled_notification_delivery_records_success_and_failure(self):
        current = datetime.now(timezone.utc)
        self._set_notification_policy(remind_before_hours=48)
        task = self._due_task("delivery", current + timedelta(hours=2))
        scheduled = self.service.schedule_due_notifications(now=current)
        payload = scheduled["jobs"][0]["payload"]

        with patch(
            "backend.feishu_bot.feishu_bot.send_card_message", new=AsyncMock(return_value=False),
        ):
            with self.assertRaises(ProcessingJobError) as failed:
                await self.service.deliver_scheduled_notifications(payload)
        self.assertTrue(failed.exception.retryable)
        notification = self.service.get(task["task_id"], self.admin_identity)["notifications"][0]
        self.assertEqual(notification["status"], "failed")
        self.assertEqual(notification["attempt_count"], 1)

        with patch(
            "backend.feishu_bot.feishu_bot.send_card_message", new=AsyncMock(return_value=True),
        ) as send:
            result = await self.service.deliver_scheduled_notifications(payload)
        notification = self.service.get(task["task_id"], self.admin_identity)["notifications"][0]
        self.assertEqual(result["sent_count"], 1)
        self.assertEqual(notification["status"], "sent")
        self.assertEqual(notification["attempt_count"], 2)
        self.assertTrue(notification["sent_at"])
        self.assertIn("到期任务-delivery", send.await_args.args[2])

    async def test_queued_notification_skips_task_completed_before_delivery(self):
        current = datetime.now(timezone.utc)
        self._set_notification_policy(remind_before_hours=48)
        task = self._due_task("stale", current + timedelta(hours=2))
        scheduled = self.service.schedule_due_notifications(now=current)
        payload = scheduled["jobs"][0]["payload"]
        self.service.act(task["task_id"], "start", {}, self.member_identity)
        self.service.act(
            task["task_id"], "submit", {"evidence": "已提前完成"}, self.member_identity,
        )
        self.service.act(task["task_id"], "accept", {}, self.admin_identity)

        with patch(
            "backend.feishu_bot.feishu_bot.send_card_message", new=AsyncMock(return_value=True),
        ) as send:
            result = await self.service.deliver_scheduled_notifications(payload)
        notification = self.service.get(task["task_id"], self.admin_identity)["notifications"][0]
        self.assertEqual(result["skipped_count"], 1)
        self.assertEqual(notification["status"], "skipped")
        send.assert_not_awaited()

    async def test_notification_processing_job_executes_persisted_digest(self):
        current = datetime.now(timezone.utc)
        self._set_notification_policy(remind_before_hours=48)
        task = self._due_task("runner", current + timedelta(hours=3))
        scheduled = self.service.schedule_due_notifications(now=current)
        job_id = scheduled["jobs"][0]["job_id"]
        processing = ProcessingJobService(lambda: self.repository)

        async def deliver(context, payload):
            context.report("notification_validate", 15, "正在复核提醒任务")
            return await self.service.deliver_scheduled_notifications(payload)

        processing.register("knowledge_task_notification", deliver)
        runner = ProcessingJobRunner(processing, poll_seconds=0.01)
        with patch(
            "backend.feishu_bot.feishu_bot.send_card_message", new=AsyncMock(return_value=True),
        ):
            try:
                await runner.start()
                for _ in range(200):
                    job = self.repository.get_processing_job(job_id, can_manage=True)
                    if job and job["status"] in {"succeeded", "failed"}:
                        break
                    await asyncio.sleep(0.01)
                else:
                    self.fail("notification processing job did not finish")
            finally:
                await runner.stop()

        notification = self.service.get(task["task_id"], self.admin_identity)["notifications"][0]
        self.assertEqual(job["status"], "succeeded")
        self.assertEqual(job["result"]["sent_count"], 1)
        self.assertEqual(notification["status"], "sent")
        self.assertEqual(notification["processing_status"], "succeeded")

    def test_state_machine_requires_evidence_and_independent_acceptance(self):
        task = self._assigned_task()
        with self.assertRaisesRegex(ValueError, "当前状态"):
            self.service.act(task["task_id"], "submit", {"evidence": "x"}, self.member_identity)

        self.service.act(task["task_id"], "start", {}, self.member_identity)
        with self.assertRaisesRegex(ValueError, "完成证据"):
            self.service.act(task["task_id"], "submit", {}, self.member_identity)
        submitted = self.service.act(
            task["task_id"], "submit", {"evidence": "已发布部署手册 v2"}, self.member_identity,
        )
        self.assertEqual(submitted["status"], "pending_acceptance")
        with self.assertRaisesRegex(PermissionError, "不能验收自己的任务"):
            self.service.act(task["task_id"], "accept", {}, self.member_identity)
        accepted = self.service.act(task["task_id"], "accept", {}, self.admin_identity)
        self.assertEqual(accepted["status"], "completed")
        self.assertTrue(accepted["completed_at"])

    def test_completed_asset_review_task_atomically_confirms_exact_asset(self):
        prepared, _, task = self._asset_review_task("atomic")
        self.repository.set_asset_projection(prepared["version_id"], "readiness", "ready")
        self.service.act(task["task_id"], "start", {}, self.member_identity)
        self.service.act(
            task["task_id"], "submit", {"evidence": "已核对来源、有效期与内容准确性"},
            self.member_identity,
        )

        completed = self.service.act(task["task_id"], "accept", {}, self.admin_identity)
        asset = self.repository.get_asset(prepared["asset_id"])
        projection = self.repository.asset_projection_status(prepared["asset_id"])

        self.assertEqual(completed["status"], "completed")
        self.assertTrue(completed["writeback"]["applied"])
        self.assertEqual(completed["writeback"]["type"], "asset_review_confirm")
        self.assertEqual(asset["status"], "active")
        self.assertEqual(asset["review_due_at"], "")
        self.assertEqual(asset["updated_by"], self.admin["user_id"])
        self.assertEqual(projection["projections"]["readiness"]["status"], "repair_required")
        self.assertTrue(completed["writeback"]["processing_job_id"])
        self.assertEqual(completed["writeback"]["processing_job"]["status"], "queued")
        self.assertEqual(completed["writeback"]["processing_job"]["job_type"], "projection_repair")
        persisted = self.service.get(task["task_id"], self.admin_identity)
        self.assertTrue(persisted["writeback"]["applied"])
        with self.repository.database.transaction() as connection:
            event = connection.execute(
                """
                SELECT payload_json FROM domain_outbox
                WHERE aggregate_id=? AND event_type='knowledge.asset_active'
                ORDER BY created_at DESC,rowid DESC LIMIT 1
                """,
                (prepared["asset_id"],),
            ).fetchone()
        self.assertIn(task["task_id"], str(event["payload_json"]))

    async def test_asset_review_reliably_recomputes_readiness_and_reports_job_status(self):
        prepared, _, task = self._asset_review_task("readiness-job")
        self.repository.set_asset_projection(prepared["version_id"], "graph", "ready")
        self.service.act(task["task_id"], "start", {}, self.member_identity)
        self.service.act(
            task["task_id"], "submit", {"evidence": "已完成复审并等待就绪度刷新"},
            self.member_identity,
        )
        completed = self.service.act(task["task_id"], "accept", {}, self.admin_identity)
        job_id = completed["writeback"]["processing_job_id"]
        processing = ProcessingJobService(lambda: self.repository)
        runner = ProcessingJobRunner(processing, poll_seconds=0.01)

        def repair_handler(context, payload):
            context.report("projection_scan", 15, "正在检查知识投影")
            return self.assets.repair_projections(
                str(payload.get("asset_id") or ""), actor=str(payload.get("actor") or "processing_job"),
            )

        processing.register("projection_repair", repair_handler)
        try:
            await runner.start()
            for _ in range(200):
                job = self.repository.get_processing_job(job_id, can_manage=True)
                if job and job["status"] in {"succeeded", "failed"}:
                    break
                await asyncio.sleep(0.01)
            else:
                self.fail("readiness projection job did not finish")
        finally:
            await runner.stop()

        self.assertEqual(job["status"], "succeeded")
        projection = self.repository.asset_projection_status(prepared["asset_id"])
        self.assertEqual(projection["projections"]["readiness"]["status"], "ready")
        persisted = self.service.get(task["task_id"], self.admin_identity)
        self.assertEqual(persisted["writeback"]["processing_job"]["status"], "succeeded")
        self.assertEqual(persisted["writeback"]["processing_job"]["progress"], 100)

    def test_asset_review_completion_does_not_override_manual_lifecycle_change(self):
        prepared, _, task = self._asset_review_task("manual")
        self.service.act(task["task_id"], "start", {}, self.member_identity)
        self.service.act(
            task["task_id"], "submit", {"evidence": "任务复审证据"}, self.member_identity,
        )
        self.assets.transition(prepared["asset_id"], "confirm_valid", self.admin["user_id"])

        completed = self.service.act(task["task_id"], "accept", {}, self.admin_identity)
        asset = self.repository.get_asset(prepared["asset_id"])

        self.assertFalse(completed["writeback"]["applied"])
        self.assertEqual(completed["writeback"]["reason"], "asset_active")
        self.assertEqual(asset["status"], "active")

    def test_repeated_review_due_reopens_task_and_clears_stale_evidence(self):
        prepared, _, task = self._asset_review_task("repeat")
        self.service.act(task["task_id"], "start", {}, self.member_identity)
        self.service.act(
            task["task_id"], "submit", {"evidence": "第一轮复审证据"}, self.member_identity,
        )
        self.assets.transition(prepared["asset_id"], "confirm_valid", self.admin["user_id"])
        asset = self.assets.transition(
            prepared["asset_id"], "mark_review_due", self.admin["user_id"], "再次到期",
        )

        reopened = self.service.create_asset_review_task(asset, self.admin_identity)

        self.assertEqual(reopened["task_id"], task["task_id"])
        self.assertEqual(reopened["status"], "in_progress")
        self.assertEqual(reopened["completion_evidence"], "")
        self.assertEqual(reopened["occurrence_count"], 2)
        self.assertEqual(reopened["events"][-1]["event_type"], "reopened")
        with self.assertRaisesRegex(ValueError, "当前状态"):
            self.service.act(task["task_id"], "accept", {}, self.admin_identity)

    def test_asset_review_task_and_writeback_rollback_together(self):
        prepared, _, task = self._asset_review_task("rollback")
        self.service.act(task["task_id"], "start", {}, self.member_identity)
        self.service.act(
            task["task_id"], "submit", {"evidence": "会触发回滚的复审证据"}, self.member_identity,
        )
        with self.repository.database.transaction(write=True) as connection:
            connection.execute(
                """
                CREATE TRIGGER reject_test_asset_review_writeback
                BEFORE UPDATE OF status ON knowledge_assets
                WHEN OLD.status='review_due' AND NEW.status='active'
                BEGIN SELECT RAISE(ABORT, 'test asset writeback failure'); END
                """
            )

        with self.assertRaisesRegex(sqlite3.IntegrityError, "test asset writeback failure"):
            self.service.act(task["task_id"], "accept", {}, self.admin_identity)

        persisted_task = self.service.get(task["task_id"], self.admin_identity)
        persisted_asset = self.repository.get_asset(prepared["asset_id"])
        self.assertEqual(persisted_task["status"], "pending_acceptance")
        self.assertEqual(persisted_asset["status"], "review_due")
        self.assertFalse(any(event["event_type"] == "accept" for event in persisted_task["events"]))
        with self.repository.database.transaction() as connection:
            queued = connection.execute(
                "SELECT COUNT(*) FROM processing_jobs WHERE linked_asset_id=? AND job_type='projection_repair'",
                (prepared["asset_id"],),
            ).fetchone()[0]
        self.assertEqual(queued, 0)

    def test_asset_publish_task_atomically_publishes_exact_ready_version(self):
        prepared, _, task = self._asset_publish_task("atomic")
        self.assertEqual(task["source_type"], "asset_version")
        self.assertEqual(task["source_key"], prepared["version_id"])
        self.assertEqual(task["metadata"]["linked_version_id"], prepared["version_id"])
        self.service.act(task["task_id"], "start", {}, self.member_identity)
        self.service.act(
            task["task_id"], "submit", {"evidence": "已核对来源、权限和版本内容"},
            self.member_identity,
        )

        completed = self.service.act(task["task_id"], "accept", {}, self.admin_identity)
        asset = self.repository.get_asset(prepared["asset_id"])
        version = self.repository.get_asset_version(prepared["asset_id"], prepared["version_id"])

        self.assertEqual(completed["status"], "completed")
        self.assertTrue(completed["writeback"]["applied"])
        self.assertEqual(completed["writeback"]["type"], "asset_version_publish")
        self.assertEqual(asset["status"], "active")
        self.assertEqual(asset["current_version_id"], prepared["version_id"])
        self.assertEqual(version["status"], "active")
        self.assertEqual(completed["writeback"]["processing_job"]["status"], "queued")
        with self.repository.database.transaction() as connection:
            event = connection.execute(
                """
                SELECT payload_json FROM domain_outbox
                WHERE aggregate_id=? AND event_type='knowledge.asset_activated'
                ORDER BY created_at DESC,rowid DESC LIMIT 1
                """,
                (prepared["asset_id"],),
            ).fetchone()
        self.assertIn(task["task_id"], str(event["payload_json"]))

    def test_asset_publish_accept_requires_publish_permission(self):
        _, _, task = self._asset_publish_task("permission")
        self.service.act(task["task_id"], "start", {}, self.member_identity)
        self.service.act(
            task["task_id"], "submit", {"evidence": "等待具备发布权限的人员验收"},
            self.member_identity,
        )
        task_manager_without_publish = Identity(
            user_id=self.other_identity.user_id, email=self.other_identity.email,
            display_name=self.other_identity.display_name,
            organization_id=self.other_identity.organization_id,
            project_id=self.other_identity.project_id,
            project_name=self.other_identity.project_name,
            role=self.other_identity.role,
            permissions=frozenset({"task.read", "task.manage"}), session_id="test-manager",
        )

        with self.assertRaisesRegex(PermissionError, "知识发布人员"):
            self.service.act(task["task_id"], "accept", {}, task_manager_without_publish)
        self.assertEqual(self.service.get(task["task_id"], self.admin_identity)["status"], "pending_acceptance")

    def test_asset_publish_task_does_not_override_manual_lifecycle_change(self):
        prepared, asset, task = self._asset_publish_task("manual")
        self.service.act(task["task_id"], "start", {}, self.member_identity)
        self.service.act(
            task["task_id"], "submit", {"evidence": "旧发布证据"}, self.member_identity,
        )
        self.assets.transition(
            prepared["asset_id"], "return_to_draft", self.admin["user_id"],
            expected_revision=asset["lifecycle_revision"],
        )

        completed = self.service.act(task["task_id"], "accept", {}, self.admin_identity)
        persisted_asset = self.repository.get_asset(prepared["asset_id"])
        self.assertFalse(completed["writeback"]["applied"])
        self.assertEqual(completed["writeback"]["reason"], "asset_draft")
        self.assertEqual(persisted_asset["status"], "draft")
        self.assertEqual(
            self.repository.get_asset_version(prepared["asset_id"], prepared["version_id"])["status"],
            "ready",
        )

    def test_asset_publish_task_rejects_stale_version_when_newer_ready_exists(self):
        prepared, _, task = self._asset_publish_task("stale")
        newer_document = SimpleNamespace(page_content="更新后的候选版本", metadata={})
        newer = self.assets.prepare_documents(
            [newer_document], source_type="manual_upload", external_key="publish-stale.md",
            title="待发布资料-stale.md", actor=self.admin["user_id"],
            stored_file="publish-stale-v2.md", asset_id=prepared["asset_id"],
            owner_user_id=self.member["user_id"],
        )
        self.repository.upsert_document_chunks([newer_document], ["publish-vector-stale-v2"])
        self.assets.record_projection(newer["version_id"], "vector", succeeded=True)
        for projection_type in ("graph", "context", "readiness"):
            self.repository.set_asset_projection(newer["version_id"], projection_type, "ready")
        self.repository.mark_asset_version_ready(newer["version_id"])
        self.service.act(task["task_id"], "start", {}, self.member_identity)
        self.service.act(
            task["task_id"], "submit", {"evidence": "旧版本审核证据"}, self.member_identity,
        )

        completed = self.service.act(task["task_id"], "accept", {}, self.admin_identity)
        self.assertFalse(completed["writeback"]["applied"])
        self.assertEqual(completed["writeback"]["reason"], "newer_version_exists")
        self.assertEqual(completed["writeback"]["newer_version_id"], newer["version_id"])
        self.assertEqual(self.repository.get_asset(prepared["asset_id"])["status"], "pending_review")

    def test_resubmitted_asset_refreshes_publish_task_revision_and_evidence(self):
        prepared, asset, task = self._asset_publish_task("reopen")
        self.service.act(task["task_id"], "start", {}, self.member_identity)
        self.service.act(
            task["task_id"], "submit", {"evidence": "第一轮发布审核证据"}, self.member_identity,
        )
        draft = self.assets.transition(
            prepared["asset_id"], "return_to_draft", self.admin["user_id"],
            expected_revision=asset["lifecycle_revision"],
        )
        pending = self.assets.transition(
            prepared["asset_id"], "submit_review", self.admin["user_id"],
            expected_revision=draft["lifecycle_revision"],
        )

        reopened = self.service.create_asset_publish_task(pending, self.admin_identity)
        self.assertEqual(reopened["task_id"], task["task_id"])
        self.assertEqual(reopened["status"], "in_progress")
        self.assertEqual(reopened["completion_evidence"], "")
        self.assertEqual(reopened["metadata"]["asset_revision"], pending["lifecycle_revision"])
        self.assertEqual(reopened["events"][-1]["event_type"], "reopened")

    def test_asset_publish_task_and_writeback_rollback_together(self):
        prepared, _, task = self._asset_publish_task("rollback")
        self.service.act(task["task_id"], "start", {}, self.member_identity)
        self.service.act(
            task["task_id"], "submit", {"evidence": "会触发发布事务回滚"}, self.member_identity,
        )
        with self.repository.database.transaction(write=True) as connection:
            connection.execute(
                """
                CREATE TRIGGER reject_test_asset_publish_writeback
                BEFORE UPDATE OF status ON asset_versions
                WHEN OLD.status='ready' AND NEW.status='active'
                BEGIN SELECT RAISE(ABORT, 'test asset publish failure'); END
                """
            )

        with self.assertRaisesRegex(sqlite3.IntegrityError, "test asset publish failure"):
            self.service.act(task["task_id"], "accept", {}, self.admin_identity)

        persisted_task = self.service.get(task["task_id"], self.admin_identity)
        persisted_asset = self.repository.get_asset(prepared["asset_id"])
        persisted_version = self.repository.get_asset_version(prepared["asset_id"], prepared["version_id"])
        self.assertEqual(persisted_task["status"], "pending_acceptance")
        self.assertEqual(persisted_asset["status"], "pending_review")
        self.assertEqual(persisted_version["status"], "ready")
        with self.repository.database.transaction() as connection:
            queued = connection.execute(
                "SELECT COUNT(*) FROM processing_jobs WHERE linked_asset_id=? AND job_type='projection_repair'",
                (prepared["asset_id"],),
            ).fetchone()[0]
        self.assertEqual(queued, 0)

    def test_scope_and_assignment_permissions(self):
        task = self._assigned_task()
        self.assertEqual(len(self.service.list(self.member_identity)["tasks"]), 1)
        self.assertEqual(self.service.list(self.other_identity)["tasks"], [])
        self.assertEqual(len(self.service.list(self.admin_identity)["tasks"]), 1)
        self.assertIsNone(self.service.get(task["task_id"], self.other_identity))
        with self.assertRaisesRegex(PermissionError, "指派给其他成员"):
            self.service.create({
                "task_type": "manual", "title": "越权分派",
                "assignee_user_id": self.other["user_id"],
            }, self.member_identity)

    def test_handover_gap_defaults_to_departing_person(self):
        """补充交接资料属于离职人员义务，非任务管理成员发起时由服务端默认指派。"""
        self.repository.save_handover({
            "id": "ho_gap_default",
            "name": self.member["display_name"],
            "role": "开发工程师",
            "recipient": self.other["display_name"],
            "recipient_user_id": self.other["user_id"],
            "departing_user_id": self.member["user_id"],
            "status": "pending_acceptance",
        })
        task = self.service.create({
            "task_type": "handover_gap", "title": "补充交接资料",
            "handover_id": "ho_gap_default", "dedupe_key": "handover_gap:ho_gap_default",
        }, self.other_identity)
        self.assertEqual(task["assignee_user_id"], self.member["user_id"])
        self.assertEqual(task["reporter_user_id"], self.other["user_id"])

    def test_onboarding_gap_generation_is_idempotent(self):
        gaps = [
            {"category": "proj_risk", "label": "风险日志", "weight": 2},
            {"category": "opa_process", "label": "流程规范", "weight": 1},
        ]
        first = self.service.create_onboarding_gap_tasks(
            "测试工程师", gaps, self.admin_identity, assignee_user_id=self.member["user_id"],
        )
        second = self.service.create_onboarding_gap_tasks(
            "测试工程师", gaps, self.admin_identity, assignee_user_id=self.member["user_id"],
        )
        self.assertEqual((first["created_count"], first["merged_count"]), (2, 0))
        self.assertEqual((second["created_count"], second["merged_count"]), (0, 2))
        self.assertEqual(len(self.service.list(self.admin_identity)["tasks"]), 2)

    async def test_notification_failure_is_recorded_without_rolling_back_task(self):
        task = self._assigned_task()
        with patch(
            "backend.feishu_bot.feishu_bot.send_card_message", new=AsyncMock(return_value=False),
        ):
            result = await self.service.notify_feishu(task["task_id"], "oc_demo", self.admin_identity)
        stored = self.service.get(task["task_id"], self.admin_identity)
        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(stored["status"], "open")
        self.assertEqual(stored["notifications"][0]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
