import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from backend.auth import ROLE_PERMISSIONS
from backend.auth_context import Identity
from backend.knowledge_assets import KnowledgeAssetService
from backend.onboarding import OnboardingService
from backend.storage.database import Database
from backend.storage.repository import StorageRepository


class OnboardingWorkflowTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repository = StorageRepository(Database(Path(self.temp.name) / "app.db"))
        self.service = OnboardingService(lambda: self.repository)
        self.admin = self._active_member("admin@example.com", "项目管理员", "project_admin")
        self.member = self._active_member("member@example.com", "新成员", "new_member")
        self.other_member = self._active_member("other@example.com", "其他成员", "project_member")
        self.admin_identity = self._identity(self.admin)
        self.member_identity = self._identity(self.member)
        self.other_identity = self._identity(self.other_member)

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
    def _asset(topic_key, index=1):
        return {
            "asset_id": f"asset_{index}_{topic_key}",
            "version_id": f"version_{index}_{topic_key}",
            "title": f"{topic_key} 项目资料",
            "status": "active", "asset_status": "active", "version_status": "active",
            "is_current_version": True, "projection_status": "healthy",
            "topics": [topic_key], "source_id": f"source_{index}_{topic_key}",
            "source_file": f"{topic_key}.md", "authority_score": 0.9,
            "freshness_score": 0.8, "traceability_score": 1.0,
        }

    def _all_developer_assets(self):
        template = self.service._template("developer")
        return [self._asset(item["topic_key"], index) for index, item in enumerate(template["topics"], 1)]

    def _publish_asset(self, topic_key, index=1):
        service = KnowledgeAssetService(lambda: self.repository)
        document = SimpleNamespace(
            page_content=f"{topic_key} 可核验项目知识 {index}", metadata={},
        )
        prepared = service.prepare_documents(
            [document], source_type="manual_upload",
            external_key=f"{topic_key}-{index}.md", title=f"{topic_key} 项目资料 {index}",
            actor=self.admin["user_id"], stored_file=f"{topic_key}-{index}.md",
            categories=[topic_key], owner_user_id=self.admin["user_id"],
        )
        self.repository.upsert_document_chunks([document], [f"vector-{topic_key}-{index}"])
        service.record_projection(prepared["version_id"], "vector", succeeded=True)
        return service.publish(
            prepared, actor=self.admin["user_id"], graph_ready=True,
            context_ready=True, readiness_ready=True,
        )

    def test_schema_defaults_and_template_management_permissions(self):
        required = {
            "onboarding_role_templates", "onboarding_template_topics",
            "onboarding_learning_plans", "onboarding_learning_items",
        }
        with self.repository.database.transaction() as connection:
            tables = {str(row[0]) for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )}
        self.assertTrue(required.issubset(tables))

        templates = self.service.list_templates(self.member_identity)
        self.assertEqual(len(templates), 5)
        developer = next(item for item in templates if item["role_key"] == "developer")
        self.assertFalse(developer["can_manage"])
        with self.assertRaisesRegex(PermissionError, "无权维护"):
            self.service.save_template(developer, self.member_identity)

        updated = self.service.save_template(
            {**developer, "focus": "先理解调用链、权限边界和交付标准"},
            self.admin_identity,
        )
        self.assertEqual(updated["revision"], developer["revision"] + 1)
        self.assertIn("权限边界", updated["focus"])

    def test_guide_and_personal_progress_are_separate_and_explainable(self):
        assets = self._all_developer_assets()
        requested_by = []
        self.repository.readiness_assets = lambda identity=None: (
            requested_by.append(identity.user_id) or assets
        )
        guide = self.service.guide(
            "developer", self.admin_identity, user_id=self.member["user_id"],
        )
        self.assertEqual(guide["target_user"]["user_id"], self.member["user_id"])
        self.assertEqual(guide["readiness"], 94.0)
        self.assertEqual(set(guide["dimensions"]), {
            "topic_coverage", "authority", "freshness", "traceability",
        })
        self.assertIsNone(guide["plan"])
        self.assertEqual(requested_by[-1], self.member["user_id"])

        plan = self.service.create_plan({
            "user_id": self.member["user_id"], "template_id": guide["template"]["template_id"],
            "target_date": "2026-09-30",
        }, self.admin_identity)
        self.assertTrue(plan["created"])
        self.assertEqual(plan["progress"], 0.0)
        self.assertEqual(plan["formula"], "必读完成 60% + 实践任务 25% + 负责人确认 15%")

        for item in plan["items"]:
            if item["item_type"] == "manager_confirmation":
                continue
            plan = self.service.update_item(
                plan["plan_id"], item["item_id"],
                {"status": "completed", "evidence": "已完成项目实践并提交结果"},
                self.member_identity,
            )
        self.assertEqual(plan["progress"], 85.0)
        confirmation = next(
            item for item in plan["items"] if item["item_type"] == "manager_confirmation"
        )
        with self.assertRaisesRegex(PermissionError, "负责人确认"):
            self.service.update_item(
                plan["plan_id"], confirmation["item_id"], {"status": "completed"},
                self.member_identity,
            )
        plan = self.service.update_item(
            plan["plan_id"], confirmation["item_id"],
            {"status": "completed", "evidence": "已核验阅读与实践证据"},
            self.admin_identity,
        )
        self.assertEqual(plan["progress"], 100.0)
        self.assertEqual(plan["status"], "completed")

        reading = next(item for item in plan["items"] if item["item_type"] == "reading")
        reopened = self.service.update_item(
            plan["plan_id"], reading["item_id"], {"status": "pending"}, self.member_identity,
        )
        reset_confirmation = next(
            item for item in reopened["items"] if item["item_type"] == "manager_confirmation"
        )
        self.assertEqual(reopened["status"], "active")
        self.assertEqual(reset_confirmation["status"], "pending")
        self.assertLess(reopened["progress"], 85.0)

    def test_supervisor_worklist_is_scoped_and_surfaces_confirmation_readiness(self):
        assets = self._all_developer_assets()
        self.repository.readiness_assets = lambda identity=None: assets
        second_member_identity = self._identity(self.other_member)
        other_manager = self._active_member("other-manager@example.com", "其他主管", "project_manager")
        other_manager_identity = self._identity(other_manager)

        ready_plan = self.service.create_plan({
            "user_id": self.member["user_id"], "role": "developer",
        }, self.admin_identity)
        waiting_plan = self.service.create_plan({
            "user_id": self.other_member["user_id"], "role": "developer",
        }, self.admin_identity)
        other_manager_plan = self.service.create_plan({
            "user_id": other_manager["user_id"], "role": "developer",
        }, other_manager_identity)
        for item in ready_plan["items"]:
            if item["item_type"] != "manager_confirmation":
                ready_plan = self.service.update_item(
                    ready_plan["plan_id"], item["item_id"],
                    {"status": "completed", "evidence": "已完成可核验的阅读或实践任务"},
                    self.member_identity,
                )

        plans = self.service.list_supervised_plans(self.admin_identity)

        self.assertEqual([plan["plan_id"] for plan in plans], [
            ready_plan["plan_id"], waiting_plan["plan_id"],
        ])
        self.assertNotIn(other_manager_plan["plan_id"], {plan["plan_id"] for plan in plans})
        self.assertTrue(plans[0]["manager_confirmation"]["can_confirm"])
        self.assertEqual(plans[0]["manager_confirmation"]["pending_count"], 0)
        self.assertFalse(plans[1]["manager_confirmation"]["can_confirm"])
        self.assertGreater(plans[1]["manager_confirmation"]["pending_count"], 0)
        self.assertEqual(plans[0]["role_key"], "developer")
        with self.assertRaisesRegex(PermissionError, "主管学习计划"):
            self.service.list_supervised_plans(second_member_identity)

    def test_missing_authorized_material_blocks_completion(self):
        self.repository.readiness_assets = lambda identity=None: [self._asset("proj_architecture")]
        plan = self.service.create_plan({
            "user_id": self.member["user_id"], "role": "developer",
        }, self.admin_identity)
        blocked = next(
            item for item in plan["items"]
            if item["item_type"] == "reading" and item["status"] == "blocked"
        )
        self.assertGreater(plan["blocked_count"], 0)
        with self.assertRaisesRegex(ValueError, "缺少可用资料"):
            self.service.update_item(
                plan["plan_id"], blocked["item_id"], {"status": "completed"},
                self.member_identity,
            )

    def test_refresh_recovers_blocked_readings_and_is_idempotent(self):
        published = [
            self._publish_asset(topic["topic_key"], index)
            for index, topic in enumerate(self.service._template("developer")["topics"], 1)
        ]
        actual_readiness = self.repository.readiness_assets
        visible = {"enabled": False}
        self.repository.readiness_assets = lambda identity=None: (
            actual_readiness(identity=identity) if visible["enabled"] else []
        )
        plan = self.service.create_plan({
            "user_id": self.member["user_id"], "role": "developer",
        }, self.admin_identity)
        self.assertEqual(plan["blocked_count"], 6)
        practice = next(item for item in plan["items"] if item["item_type"] == "practice")
        plan = self.service.update_item(
            plan["plan_id"], practice["item_id"],
            {"status": "completed", "evidence": "已提交可核验项目实践记录"},
            self.member_identity,
        )

        visible["enabled"] = True
        refreshed = self.service.refresh_plan_materials(plan["plan_id"], self.member_identity)
        self.assertEqual(refreshed["blocked_count"], 0)
        self.assertEqual(refreshed["refresh"]["unblocked_count"], 6)
        self.assertEqual(
            {item["asset_id"] for item in refreshed["items"] if item["item_type"] == "reading"},
            {item["asset_id"] for item in published},
        )
        completed_practice = next(
            item for item in refreshed["items"] if item["item_id"] == practice["item_id"]
        )
        self.assertEqual(completed_practice["status"], "completed")
        self.assertEqual(completed_practice["evidence"], "已提交可核验项目实践记录")

        repeated = self.service.refresh_plan_materials(plan["plan_id"], self.member_identity)
        self.assertFalse(repeated["refresh"]["changed"])
        self.assertEqual(len(repeated["items"]), len(refreshed["items"]))

    def test_refresh_keeps_missing_topics_blocked_and_enforces_plan_owner(self):
        self._publish_asset("proj_architecture")
        plan = self.service.create_plan({
            "user_id": self.member["user_id"], "role": "developer",
        }, self.admin_identity)
        with self.assertRaisesRegex(PermissionError, "只能刷新自己的学习计划"):
            self.service.refresh_plan_materials(plan["plan_id"], self.other_identity)

        refreshed = self.service.refresh_plan_materials(plan["plan_id"], self.admin_identity)
        self.assertGreater(refreshed["blocked_count"], 0)
        architecture = next(
            item for item in refreshed["items"]
            if item["item_type"] == "reading" and item["topic_key"] == "proj_architecture"
        )
        self.assertEqual(architecture["status"], "pending")

    def test_refresh_rebinds_pending_reading_to_current_asset_version(self):
        original = self._publish_asset("proj_architecture")
        plan = self.service.create_plan({
            "user_id": self.member["user_id"], "role": "developer",
        }, self.admin_identity)
        original_item = next(
            item for item in plan["items"]
            if item["item_type"] == "reading" and item["topic_key"] == "proj_architecture"
        )

        service = KnowledgeAssetService(lambda: self.repository)
        document = SimpleNamespace(
            page_content="proj_architecture 已更新的架构说明", metadata={},
        )
        prepared = service.prepare_documents(
            [document], source_type="manual_upload", external_key="proj_architecture-1.md",
            title="proj_architecture 项目资料 1", actor=self.admin["user_id"],
            stored_file="proj_architecture-1-v2.md", categories=["proj_architecture"],
            owner_user_id=self.admin["user_id"],
        )
        self.repository.upsert_document_chunks([document], ["vector-proj-architecture-v2"])
        service.record_projection(prepared["version_id"], "vector", succeeded=True)
        current = service.publish(
            prepared, actor=self.admin["user_id"], graph_ready=True,
            context_ready=True, readiness_ready=True,
        )

        refreshed = self.service.refresh_plan_materials(plan["plan_id"], self.member_identity)
        current_item = next(
            item for item in refreshed["items"]
            if item["item_type"] == "reading" and item["topic_key"] == "proj_architecture"
        )
        self.assertEqual(current_item["asset_id"], original["asset_id"])
        self.assertNotEqual(current_item["version_id"], original_item["version_id"])
        self.assertEqual(current_item["version_id"], current["current_version_id"])
        self.assertEqual(current_item["status"], "pending")


if __name__ == "__main__":
    unittest.main()
