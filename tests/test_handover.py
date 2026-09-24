import sqlite3
import tempfile
import unittest
from pathlib import Path

from backend.auth import ROLE_PERMISSIONS
from backend.auth_context import Identity
from backend.handover import HandoverService
from backend.knowledge_tasks import KnowledgeTaskService
from backend.onboarding import OnboardingService
from backend.storage.database import Database
from backend.storage.repository import StorageRepository


class HandoverOnboardingWorkflowTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repository = StorageRepository(Database(Path(self.temp.name) / "app.db"))
        self.onboarding = OnboardingService(lambda: self.repository)
        self.tasks = KnowledgeTaskService(lambda: self.repository)
        self.service = HandoverService(lambda: self.repository, self.onboarding, self.tasks)
        self.onboarding.ensure_defaults()
        self.admin = self._active_member("admin@example.com", "项目管理员", "project_admin")
        self.recipient = self._active_member("new@example.com", "接替成员", "new_member")
        self.other = self._active_member("other@example.com", "其他成员", "project_member")
        self.admin_identity = self._identity(self.admin)
        self.recipient_identity = self._identity(self.recipient)
        self.other_identity = self._identity(self.other)

    def tearDown(self):
        self.temp.cleanup()

    def _active_member(self, email, name, role):
        member = self.repository.provision_project_member(email, name, role)
        return self.repository.set_demo_login_password(member["user_id"], "test-password-hash")

    @staticmethod
    def _identity(member):
        role = str(member["role"])
        return Identity(
            user_id=str(member["user_id"]), email=str(member["email"]),
            display_name=str(member["display_name"]), organization_id="org_default",
            project_id=str(member["project_id"]), project_name=str(member["project_name"]),
            role=role, permissions=ROLE_PERMISSIONS[role], session_id="test",
        )

    @staticmethod
    def _asset(topic_key, index):
        return {
            "asset_id": f"asset_{index}_{topic_key}",
            "version_id": f"version_{index}_{topic_key}",
            "title": f"{topic_key} 交接资料",
            "status": "active", "asset_status": "active", "version_status": "active",
            "is_current_version": True, "projection_status": "healthy",
            "topics": [topic_key], "source_id": f"source_{index}_{topic_key}",
            "source_file": f"{topic_key}.md", "authority_score": 0.9,
            "freshness_score": 0.8, "traceability_score": 1.0,
        }

    def _developer_assets(self):
        template = self.onboarding.resolve_template("developer")
        return [
            self._asset(topic["topic_key"], index)
            for index, topic in enumerate(template["topics"], 1)
        ]

    def _record(self, **overrides):
        record = {
            "id": "handover_1", "name": "离职成员", "role": "开发工程师",
            "role_key": "developer", "recipient": self.recipient["display_name"],
            "recipient_user_id": self.recipient["user_id"], "due_date": "2026-09-30",
            "departing_user_id": self.other["user_id"], "created_by_user_id": self.admin["user_id"],
            "status": "pending_acceptance", "files": ["交接说明.md"],
            "asset_ids": ["asset_6_交接文档"], "created_at": "2026-08-26T08:00:00+00:00",
        }
        record.update(overrides)
        if record.get("role_key"):
            template = self.repository.get_onboarding_template(role_key=record["role_key"])
            if template:
                record["role_template_id"] = template["template_id"]
                record["role_template_revision"] = template["revision"]
        return self.repository.save_handover(record, actor=self.admin["user_id"])

    def _owned_asset(self):
        prepared = self.repository.prepare_asset_version(
            source_type="manual_upload", external_key="owned-guide.md", title="离职成员维护手册",
            content_hash="owned-guide-v1", actor=self.other["user_id"],
            owner_user_id=self.other["user_id"], categories=["项目规范"],
        )
        self.repository.set_asset_projection(prepared["version_id"], "vector", "ready")
        for projection in ("graph", "context", "readiness"):
            self.repository.set_asset_projection(prepared["version_id"], projection, "repair_required")
        self.repository.mark_asset_version_ready(prepared["version_id"])
        self.repository.publish_asset_version(
            prepared["asset_id"], prepared["version_id"], self.other["user_id"],
        )
        return prepared

    def _private_owned_asset(self):
        prepared = self.repository.prepare_asset_version(
            source_type="manual_upload", external_key="private-guide.md", title="仅离职人员可见资料",
            content_hash="private-guide-v1", actor=self.other["user_id"], owner_user_id=self.other["user_id"],
            categories=["项目规范"], visibility="private",
        )
        self.repository.set_asset_projection(prepared["version_id"], "vector", "ready")
        for projection in ("graph", "context", "readiness"):
            self.repository.set_asset_projection(prepared["version_id"], projection, "repair_required")
        self.repository.mark_asset_version_ready(prepared["version_id"])
        self.repository.publish_asset_version(
            prepared["asset_id"], prepared["version_id"], self.other["user_id"],
        )
        return prepared

    def _publish_owned_asset_revision(self, content_hash):
        prepared = self.repository.prepare_asset_version(
            source_type="manual_upload", external_key="owned-guide.md", title="离职成员维护手册",
            content_hash=content_hash, actor=self.other["user_id"],
            owner_user_id=self.other["user_id"], categories=["项目规范"],
        )
        self.repository.set_asset_projection(prepared["version_id"], "vector", "ready")
        for projection in ("graph", "context", "readiness"):
            self.repository.set_asset_projection(prepared["version_id"], projection, "repair_required")
        self.repository.mark_asset_version_ready(prepared["version_id"])
        self.repository.publish_asset_version(
            prepared["asset_id"], prepared["version_id"], self.other["user_id"],
        )
        return prepared

    def _submit_and_accept_all_items(self):
        for item in self.repository.get_handover("handover_1")["items"]:
            self.service.item_action(
                "handover_1", item["item_id"], "submit", {"evidence": f"{item['title']} 证据"},
                self.other_identity,
            )
            self.service.item_action(
                "handover_1", item["item_id"], "accept", {}, self.recipient_identity,
            )

    def _rejected_item_with_task(self):
        record = self._record()
        item = record["items"][0]
        self.service.item_action(
            "handover_1", item["item_id"], "submit", {"evidence": "原始交接说明"},
            self.other_identity,
        )
        rejected = self.service.item_action(
            "handover_1", item["item_id"], "reject", {"reason": "缺少回滚路径"},
            self.recipient_identity,
        )
        return item, rejected["task"]

    def test_recipient_confirmation_creates_and_links_plan_idempotently(self):
        assets = self._developer_assets()
        self.repository.readiness_assets = lambda identity=None: assets
        self._record()

        first = self.service.accept("handover_1", self.recipient_identity)
        self.assertTrue(first["success"])
        self.assertFalse(first["partial_success"])
        self.assertTrue(first["created"])
        self.assertEqual(first["record"]["status"], "accepted")
        self.assertEqual(first["record"]["learning_plan_status"], "ready")
        self.assertEqual(first["plan"]["source_type"], "handover")
        self.assertEqual(first["plan"]["source_key"], "handover_1")
        self.assertEqual(first["plan"]["user_id"], self.recipient["user_id"])
        self.assertIn(
            "asset_6_交接文档",
            {item["asset_id"] for item in first["plan"]["items"] if item["item_type"] == "reading"},
        )

        item_count = len(first["plan"]["items"])
        second = self.service.accept("handover_1", self.recipient_identity)
        self.assertFalse(second["created"])
        self.assertEqual(second["plan"]["plan_id"], first["plan"]["plan_id"])
        self.assertEqual(len(second["plan"]["items"]), item_count)
        self.assertEqual(
            len(self.repository.list_onboarding_plans(user_id=self.recipient["user_id"])), 1,
        )

    def test_non_recipient_cannot_confirm_handover(self):
        self._record()
        with self.assertRaisesRegex(PermissionError, "只能确认"):
            self.service.accept("handover_1", self.other_identity)
        self.assertEqual(self.repository.get_handover("handover_1")["status"], "pending_acceptance")

    def test_only_initiator_can_cancel_pending_handover(self):
        self._record(created_by_user_id=self.other["user_id"])
        with self.assertRaisesRegex(PermissionError, "只有发起人"):
            self.service.cancel("handover_1", self.admin_identity)

        result = self.service.cancel("handover_1", self.other_identity)
        self.assertTrue(result["success"])
        self.assertIsNone(self.repository.get_handover("handover_1"))

    def test_accepted_handover_cannot_be_cancelled(self):
        self._record(created_by_user_id=self.other["user_id"], status="accepted")
        with self.assertRaisesRegex(ValueError, "待接替人确认"):
            self.service.cancel("handover_1", self.other_identity)

    def test_missing_role_template_is_persisted_as_retryable_partial_success(self):
        self._record(role="自定义顾问", role_key="")
        result = self.service.accept("handover_1", self.admin_identity)
        self.assertTrue(result["success"])
        self.assertTrue(result["partial_success"])
        self.assertEqual(result["record"]["status"], "accepted")
        self.assertEqual(result["record"]["learning_plan_status"], "needs_attention")
        self.assertIn("岗位模板", result["record"]["learning_plan_error"])
        self.assertIsNone(result["plan"])

    def test_legacy_recipient_name_maps_to_active_project_member(self):
        assets = self._developer_assets()
        self.repository.readiness_assets = lambda identity=None: assets
        self._record(recipient_user_id="", recipient=self.recipient["email"].upper())
        result = self.service.accept("handover_1", self.admin_identity)
        self.assertEqual(result["record"]["recipient_user_id"], self.recipient["user_id"])
        self.assertEqual(result["record"]["recipient"], self.recipient["display_name"])

    def test_departing_admin_cannot_review_even_when_someone_else_submitted(self):
        record = self._record(departing_user_id=self.admin["user_id"])
        item = record["items"][0]
        self.service.item_action("handover_1", item["item_id"], "submit", {"evidence": "已提交资料"}, self.admin_identity)
        with self.repository.database.transaction(write=True) as connection:
            connection.execute("UPDATE handover_items SET submitted_by=? WHERE item_id=?",
                               (self.other["user_id"], item["item_id"]))
        view = self.service.detail("handover_1", self.admin_identity)["items"][0]
        self.assertEqual(view["review_actions"], [])
        self.assertIn("等待接替成员验收", view["review_block_reason"])
        for action in ("accept", "reject"):
            with self.assertRaisesRegex(PermissionError, "不能验收自己"):
                self.service.item_action("handover_1", item["item_id"], action,
                                         {"reason": "补充", "proxy_reason": "管理员代验"}, self.admin_identity)
        self.assertEqual(self.repository.get_handover_item("handover_1", item["item_id"])["status"], "submitted")

    def test_admin_submitter_cannot_review_and_transaction_enforces_same_rule(self):
        item = self._record()["items"][0]
        self.service.item_action("handover_1", item["item_id"], "submit", {"evidence": "管理员提交"}, self.admin_identity)
        for call in (
            lambda: self.service.item_action("handover_1", item["item_id"], "accept", {}, self.admin_identity),
            lambda: self.repository.update_handover_item("handover_1", item["item_id"], "accept", {}, actor=self.admin["user_id"]),
        ):
            with self.assertRaisesRegex(PermissionError, "不能验收自己"):
                call()
        recipient_view = self.service.detail("handover_1", self.recipient_identity)["items"][0]
        self.assertEqual(recipient_view["review_action_mode"], "recipient")
        accepted = self.service.item_action("handover_1", item["item_id"], "accept", {}, self.recipient_identity)
        self.assertEqual(accepted["item"]["review_mode"], "recipient")
        self.assertEqual(accepted["item"]["proxy_reason"], "")

    def test_recipient_who_submitted_item_is_also_excluded(self):
        item = self._record(created_by_user_id=self.recipient["user_id"])["items"][0]
        self.service.item_action("handover_1", item["item_id"], "submit", {"evidence": "接替人代提交"}, self.recipient_identity)
        with self.assertRaisesRegex(PermissionError, "不能验收自己"):
            self.service.item_action("handover_1", item["item_id"], "accept", {}, self.recipient_identity)

    def test_proxy_review_requires_reason_persists_audit_and_resubmit_clears_review(self):
        item = self._record()["items"][0]
        self.service.item_action("handover_1", item["item_id"], "submit", {"evidence": "交接文件"}, self.other_identity)
        view = self.service.detail("handover_1", self.admin_identity)["items"][0]
        self.assertEqual(view["review_action_mode"], "proxy")
        for action in ("accept", "reject"):
            with self.assertRaisesRegex(ValueError, "代验收原因"):
                self.service.item_action("handover_1", item["item_id"], action, {"reason": "补充"}, self.admin_identity)
        with self.assertRaisesRegex(ValueError, "1000 字"):
            self.service.item_action("handover_1", item["item_id"], "reject",
                {"reason": "补充", "proxy_reason": "因" * 1001}, self.admin_identity)
        self.assertEqual(self.repository.list_knowledge_tasks(can_manage=True), [])
        reason = "接替人休假，项目经理代核验资料"
        rejected = self.service.item_action("handover_1", item["item_id"], "reject",
            {"reason": "缺少职责边界", "proxy_reason": reason}, self.admin_identity)
        self.assertEqual(rejected["item"]["proxy_reason"], reason)
        resubmitted = self.service.item_action("handover_1", item["item_id"], "submit",
            {"evidence": "已补充职责边界"}, self.other_identity)
        self.assertEqual(resubmitted["item"]["proxy_reason"], "")
        self.assertEqual(resubmitted["item"]["review_mode"], "")
        accepted = self.service.item_action("handover_1", item["item_id"], "accept",
            {"proxy_reason": reason}, self.admin_identity)
        self.assertEqual(accepted["record"]["items"][0]["review_mode"], "proxy")
        self.assertEqual(accepted["item"]["reviewed_by"], self.admin["user_id"])
        event = self.repository.list_audit_events(action="handover.item_accept")["items"][0]
        self.assertEqual(event["detail"]["proxy_reason"], reason)
        self.assertEqual(event["detail"]["review_mode"], "proxy")

    def test_item_draft_associates_shared_current_asset_and_submission_freezes_version(self):
        item = self._record()["items"][0]
        asset = self._owned_asset()
        draft_result = self.service.save_submission_draft(
            "handover_1", item["item_id"],
            {"evidence": "", "asset_refs": [{
                "asset_id": asset["asset_id"], "version_id": asset["version_id"], "source_kind": "existing",
            }]},
            self.other_identity,
        )
        self.assertEqual(draft_result["draft"]["asset_refs"][0]["title"], "离职成员维护手册")
        self.assertEqual(draft_result["assets"][0]["version_id"], asset["version_id"])

        view = self.service.detail("handover_1", self.other_identity)["items"][0]
        self.assertEqual(view["submission_draft"]["asset_refs"][0]["asset_id"], asset["asset_id"])
        self.assertNotIn("submission_draft", self.service.detail("handover_1", self.recipient_identity)["items"][0])

        submitted = self.service.item_action(
            "handover_1", item["item_id"], "submit",
            {"evidence": "", "asset_refs": draft_result["draft"]["asset_refs"]}, self.other_identity,
        )
        self.assertEqual(submitted["item"]["status"], "submitted")
        self.assertEqual(submitted["item"]["asset_id"], asset["asset_id"])
        self.assertEqual(submitted["item"]["version_id"], asset["version_id"])
        evidence = submitted["record"]["items"][0]["evidence_assets"]
        self.assertEqual(
            [(entry["asset_id"], entry["version_id"]) for entry in evidence],
            [(asset["asset_id"], asset["version_id"])],
        )
        self.assertEqual(
            self.repository.get_handover_item_draft("handover_1", item["item_id"], self.other["user_id"])["asset_refs"],
            [],
        )

    def test_uploaded_draft_reference_and_private_asset_access_are_checked_for_recipient(self):
        item = self._record()["items"][0]
        shared = self._owned_asset()
        appended = self.service.append_uploaded_draft_asset(
            "handover_1", item["item_id"], shared["asset_id"], shared["version_id"], self.other_identity,
        )
        self.assertEqual(appended["draft"]["asset_refs"][0]["source_kind"], "uploaded")

        private = self._private_owned_asset()
        with self.assertRaisesRegex(ValueError, "无权访问"):
            self.service.save_submission_draft(
                "handover_1", item["item_id"],
                {"evidence": "", "asset_refs": [{
                    "asset_id": private["asset_id"], "version_id": private["version_id"],
                }]},
                self.other_identity,
            )

    def test_draft_keeps_selected_version_when_asset_receives_a_new_version(self):
        item = self._record()["items"][0]
        original = self._owned_asset()
        self.service.save_submission_draft(
            "handover_1", item["item_id"],
            {"evidence": "", "asset_refs": [{
                "asset_id": original["asset_id"], "version_id": original["version_id"],
            }]},
            self.other_identity,
        )
        replacement = self._publish_owned_asset_revision("owned-guide-v2")
        self.assertEqual(replacement["asset_id"], original["asset_id"])
        self.assertNotEqual(replacement["version_id"], original["version_id"])

        submitted = self.service.item_action(
            "handover_1", item["item_id"], "submit",
            {"evidence": "", "asset_refs": [{
                "asset_id": original["asset_id"], "version_id": original["version_id"],
            }]},
            self.other_identity,
        )
        self.assertEqual(submitted["item"]["version_id"], original["version_id"])

    def test_default_required_checklist_uses_explainable_completion_weights(self):
        record = self._record()
        view = self.service.detail(record["id"])

        template = self.repository.get_onboarding_template(role_key="developer")
        self.assertEqual(len(view["items"]), 8 + len(template["topics"]))
        self.assertTrue(all(item["required"] for item in view["items"]))
        self.assertEqual(
            {item["title"] for item in view["items"] if item["item_type"] == "role_topic"},
            {f"岗位知识：{topic['label']}" for topic in template["topics"]},
        )
        self.assertEqual(view["completion"], 20)
        self.assertEqual(view["completion_breakdown"]["weights"], {
            "required_items": 60, "risks": 20, "recipient_confirmation": 20,
        })
        self.assertFalse(view["completion_gates"]["required_items_complete"])
        self.assertFalse(view["completion_gates"]["closable"])

    def test_item_rejection_requires_reason_and_creates_deduplicated_gap_task(self):
        record = self._record()
        item = record["items"][0]
        with self.assertRaisesRegex(ValueError, "证据"):
            self.service.item_action("handover_1", item["item_id"], "submit", {}, self.other_identity)

        submitted = self.service.item_action(
            "handover_1", item["item_id"], "submit", {"evidence": "架构说明第 2 节"},
            self.other_identity,
        )
        self.assertEqual(submitted["item"]["status"], "submitted")
        with self.assertRaisesRegex(PermissionError, "不能验收自己"):
            self.service.item_action(
                "handover_1", item["item_id"], "reject", {"reason": "缺少回滚路径"},
                self.other_identity,
            )
        with self.assertRaisesRegex(ValueError, "补充要求"):
            self.service.item_action(
                "handover_1", item["item_id"], "reject", {}, self.recipient_identity,
            )

        rejected = self.service.item_action(
            "handover_1", item["item_id"], "reject", {"reason": "缺少回滚路径"},
            self.recipient_identity,
        )
        self.assertEqual(rejected["item"]["status"], "rejected")
        self.assertEqual(rejected["item"]["validity_status"], "invalid")
        self.assertEqual(rejected["task"]["task_type"], "handover_gap")
        self.assertEqual(rejected["task"]["assignee_user_id"], self.other["user_id"])
        self.assertEqual(rejected["item"]["knowledge_task_id"], rejected["task"]["task_id"])

        self.service.item_action(
            "handover_1", item["item_id"], "submit", {"evidence": "已补充回滚步骤"},
            self.other_identity,
        )
        accepted = self.service.item_action(
            "handover_1", item["item_id"], "accept", {}, self.recipient_identity,
        )
        self.assertEqual(accepted["item"]["status"], "accepted")
        self.assertEqual(accepted["item"]["validity_status"], "valid")

    def test_completed_gap_task_atomically_resubmits_exact_handover_item(self):
        item, task = self._rejected_item_with_task()
        self.tasks.act(task["task_id"], "start", {}, self.other_identity)
        self.tasks.act(
            task["task_id"], "submit", {"evidence": "已补充发布失败回滚步骤 v2"},
            self.other_identity,
        )

        completed = self.tasks.act(task["task_id"], "accept", {}, self.recipient_identity)
        updated_item = self.repository.get_handover_item("handover_1", item["item_id"])

        self.assertEqual(completed["status"], "completed")
        self.assertTrue(completed["writeback"]["applied"])
        self.assertEqual(completed["writeback"]["item_id"], item["item_id"])
        self.assertEqual(updated_item["status"], "submitted")
        self.assertEqual(updated_item["validity_status"], "unknown")
        self.assertEqual(updated_item["evidence"], "已补充发布失败回滚步骤 v2")
        self.assertEqual(updated_item["submitted_by"], self.other["user_id"])
        self.assertEqual(updated_item["reviewed_by"], "")
        self.assertEqual(updated_item["knowledge_task_id"], task["task_id"])
        persisted = self.tasks.get(task["task_id"], self.recipient_identity)
        self.assertTrue(persisted["writeback"]["applied"])

        accepted = self.service.item_action(
            "handover_1", item["item_id"], "accept", {}, self.recipient_identity,
        )
        self.assertEqual(accepted["item"]["status"], "accepted")

    def test_gap_task_completion_does_not_overwrite_manual_resubmission(self):
        item, task = self._rejected_item_with_task()
        self.tasks.act(task["task_id"], "start", {}, self.other_identity)
        self.tasks.act(
            task["task_id"], "submit", {"evidence": "任务中的补充说明"}, self.other_identity,
        )
        self.service.item_action(
            "handover_1", item["item_id"], "submit", {"evidence": "人工重新提交的权威说明"},
            self.other_identity,
        )

        completed = self.tasks.act(task["task_id"], "accept", {}, self.recipient_identity)
        updated_item = self.repository.get_handover_item("handover_1", item["item_id"])

        self.assertFalse(completed["writeback"]["applied"])
        self.assertEqual(completed["writeback"]["reason"], "item_submitted")
        self.assertEqual(updated_item["status"], "submitted")
        self.assertEqual(updated_item["evidence"], "人工重新提交的权威说明")

    def test_repeated_item_rejection_reopens_task_and_clears_stale_evidence(self):
        item, task = self._rejected_item_with_task()
        self.tasks.act(task["task_id"], "start", {}, self.other_identity)
        self.tasks.act(
            task["task_id"], "submit", {"evidence": "第一轮补充证据"}, self.other_identity,
        )
        self.service.item_action(
            "handover_1", item["item_id"], "submit", {"evidence": "第一轮人工重提"},
            self.other_identity,
        )

        second_rejection = self.service.item_action(
            "handover_1", item["item_id"], "reject", {"reason": "仍缺少演练记录"},
            self.recipient_identity,
        )
        reopened = second_rejection["task"]

        self.assertEqual(reopened["task_id"], task["task_id"])
        self.assertEqual(reopened["status"], "in_progress")
        self.assertEqual(reopened["completion_evidence"], "")
        self.assertEqual(reopened["occurrence_count"], 2)
        self.assertEqual(reopened["events"][-1]["event_type"], "reopened")
        with self.assertRaisesRegex(ValueError, "当前状态"):
            self.tasks.act(task["task_id"], "accept", {}, self.recipient_identity)

    def test_gap_task_and_handover_writeback_rollback_together(self):
        item, task = self._rejected_item_with_task()
        self.tasks.act(task["task_id"], "start", {}, self.other_identity)
        self.tasks.act(
            task["task_id"], "submit", {"evidence": "会触发事务回滚的证据"}, self.other_identity,
        )
        with self.repository.database.transaction(write=True) as connection:
            connection.execute(
                """
                CREATE TRIGGER reject_test_handover_writeback
                BEFORE UPDATE OF status ON handover_items
                WHEN OLD.status='rejected' AND NEW.status='submitted'
                BEGIN SELECT RAISE(ABORT, 'test writeback failure'); END
                """
            )

        with self.assertRaisesRegex(sqlite3.IntegrityError, "test writeback failure"):
            self.tasks.act(task["task_id"], "accept", {}, self.recipient_identity)

        persisted_task = self.tasks.get(task["task_id"], self.recipient_identity)
        persisted_item = self.repository.get_handover_item("handover_1", item["item_id"])
        self.assertEqual(persisted_task["status"], "pending_acceptance")
        self.assertEqual(persisted_item["status"], "rejected")
        self.assertFalse(any(event["event_type"] == "accept" for event in persisted_task["events"]))

    def test_blocking_risk_prevents_closure_until_evidence_is_accepted(self):
        assets = self._developer_assets()
        self.repository.readiness_assets = lambda identity=None: assets
        self._record()
        self._submit_and_accept_all_items()
        self.service.accept("handover_1", self.recipient_identity)
        risk = self.service.create_risk("handover_1", {
            "title": "生产密钥轮换未完成", "impact": "接替人无法独立发布",
            "severity": "blocking", "owner_user_id": self.other["user_id"],
            "due_at": "2026-09-10", "mitigation": "由平台管理员完成授权并验证发布",
        }, self.other_identity)["risk"]

        with self.assertRaisesRegex(ValueError, "阻断风险"):
            self.service.complete("handover_1", self.admin_identity)
        with self.assertRaisesRegex(ValueError, "关闭证据"):
            self.service.risk_action(
                "handover_1", risk["risk_id"], "submit_close", {}, self.other_identity,
            )
        pending = self.service.risk_action(
            "handover_1", risk["risk_id"], "submit_close",
            {"close_evidence": "授权工单 IAM-1024 已验证"}, self.other_identity,
        )
        self.assertEqual(pending["risk"]["status"], "pending_close")
        closed = self.service.risk_action(
            "handover_1", risk["risk_id"], "close", {}, self.recipient_identity,
        )
        self.assertEqual(closed["risk"]["status"], "closed")
        self.assertEqual(closed["record"]["completion"], 100)

    def test_closure_is_idempotent_and_snapshot_is_immutable(self):
        assets = self._developer_assets()
        self.repository.readiness_assets = lambda identity=None: assets
        self._record()
        self._submit_and_accept_all_items()
        self.service.accept("handover_1", self.recipient_identity)

        first = self.service.complete("handover_1", self.admin_identity)
        second = self.service.complete("handover_1", self.admin_identity)
        self.assertTrue(first["created"])
        self.assertFalse(second["created"])
        self.assertEqual(first["snapshot"]["snapshot_id"], second["snapshot"]["snapshot_id"])
        self.assertEqual(first["snapshot"]["checksum"], second["snapshot"]["checksum"])
        self.assertEqual(first["record"]["status"], "completed")
        self.assertEqual(first["record"]["completion"], 100)
        self.assertEqual(
            len(first["snapshot"]["payload"]["items"]),
            len(first["record"]["items"]),
        )
        self.assertEqual(first["snapshot"]["payload"]["learning_plan"]["status"], "ready")
        self.assertEqual(first["snapshot"]["payload"]["schema"], "handover_snapshot_v2")

        with self.repository.database.transaction(write=True) as connection:
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute(
                    "UPDATE handover_snapshots SET checksum='changed' WHERE handover_id='handover_1'"
                )
        with self.assertRaisesRegex(ValueError, "已关闭"):
            self.service.item_action(
                "handover_1", first["record"]["items"][0]["item_id"], "submit",
                {"evidence": "试图改写"}, self.admin_identity,
            )

    def test_only_management_role_can_seal_snapshot(self):
        self._record()
        with self.assertRaisesRegex(PermissionError, "管理角色"):
            self.service.complete("handover_1", self.recipient_identity)

    def test_automatic_inventory_requires_review_and_preserves_decisions_on_refresh(self):
        owned = self._owned_asset()
        task = self.repository.create_or_merge_knowledge_task({
            "task_type": "knowledge_gap", "title": "补齐发布回滚说明",
            "assignee_user_id": self.other["user_id"], "reporter_user_id": self.admin["user_id"],
            "priority": "high", "dedupe_key": "inventory-open-task",
        })
        with self.repository.database.transaction(write=True) as connection:
            connection.execute(
                """
                INSERT INTO graph_nodes(project_id,node_id,origin,source,updated_at,data_json)
                VALUES(?,?,?,?,?,?)
                """,
                (
                    self.repository.project_id, "graph_owned_topic", "generated", "owned-guide.md",
                    "2026-08-26T09:00:00+00:00",
                    '{"label":"发布回滚主题","asset_id":"%s","version_id":"%s"}'
                    % (owned["asset_id"], owned["version_id"]),
                ),
            )
        record = self._record()
        self.assertEqual(record["inventory_summary"]["total"], 3)
        self.assertEqual(record["inventory_summary"]["pending"], 3)
        self.assertFalse(self.service.detail("handover_1")["completion_gates"]["inventory_reviewed"])
        self.repository.readiness_assets = lambda identity=None: self._developer_assets()
        self._submit_and_accept_all_items()
        self.service.accept("handover_1", self.recipient_identity)
        with self.assertRaisesRegex(ValueError, "知识盘点项"):
            self.service.complete("handover_1", self.admin_identity)

        by_type = {item["source_type"]: item for item in record["inventory"]}
        with self.assertRaisesRegex(ValueError, "说明原因"):
            self.service.inventory_action(
                "handover_1", by_type["graph_topic"]["inventory_id"], "exclude", {},
                self.recipient_identity,
            )
        self.service.inventory_action(
            "handover_1", by_type["knowledge_asset"]["inventory_id"], "include", {},
            self.recipient_identity,
        )
        self.service.inventory_action(
            "handover_1", by_type["knowledge_task"]["inventory_id"], "include", {},
            self.recipient_identity,
        )
        self.service.inventory_action(
            "handover_1", by_type["graph_topic"]["inventory_id"], "exclude",
            {"reason": "内容已由知识资产主记录覆盖"}, self.recipient_identity,
        )
        refreshed = self.service.refresh_inventory("handover_1", self.recipient_identity)["record"]
        self.assertEqual(refreshed["inventory_summary"]["pending"], 0)
        self.assertEqual(refreshed["inventory_summary"]["included"], 2)
        self.assertTrue(refreshed["completion_gates"]["inventory_reviewed"])

        result = self.service.complete("handover_1", self.admin_identity)
        self.assertEqual(len(result["snapshot"]["payload"]["inventory"]), 3)
        self.assertIn(
            (owned["asset_id"], owned["version_id"]),
            {
                (item["asset_id"], item["version_id"])
                for item in result["snapshot"]["payload"]["asset_versions"]
            },
        )
        self.assertEqual(task["status"], "open")


if __name__ == "__main__":
    unittest.main()
