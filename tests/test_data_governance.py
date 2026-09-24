import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from backend.data_governance import DataGovernanceService
from backend.onboarding import OnboardingService
from backend.storage.database import Database, utc_now
from backend.storage.repository import StorageRepository


class ReportContext:
    def __init__(self):
        self.events = []

    def report(self, stage, progress, message=""):
        self.events.append((stage, progress, message))


def seed_targeted_asset(repository, *, owner="member", stored_file="private/source.md"):
    now = utc_now()
    with repository.database.transaction(write=True) as connection:
        connection.execute(
            "INSERT OR IGNORE INTO users(user_id,display_name,user_type,organization_id,status,created_at,updated_at) "
            "VALUES(?,?,'local','org_default','active',?,?)",
            (owner, "待治理成员", now, now),
        )
        connection.execute(
            """
            INSERT OR IGNORE INTO project_memberships(
                project_id,user_id,role,created_at,status,granted_by,updated_at
            ) VALUES(?,?,'project_member',?,'active','admin',?)
            """,
            (repository.project_id, owner, now, now),
        )
        connection.execute(
            """
            INSERT INTO knowledge_sources(
                source_id,organization_id,project_id,source_type,external_key,display_name,
                owner_user_id,status,metadata_json,created_at,updated_at
            ) VALUES('source-target','org_default',?,'feishu_conversation','candidate-target',
                '敏感来源',?,'active','{}',?,?)
            """,
            (repository.project_id, owner, now, now),
        )
        connection.execute(
            """
            INSERT INTO knowledge_assets(
                asset_id,organization_id,project_id,primary_source_id,asset_key,title,summary,
                owner_user_id,status,current_version_id,lifecycle_revision,created_by,updated_by,
                metadata_json,created_at,updated_at
            ) VALUES('asset-target','org_default',?,'source-target','target:key','敏感资产','敏感摘要',
                ?,'active','version-target',1,?,?,'{}',?,?)
            """,
            (repository.project_id, owner, owner, owner, now, now),
        )
        connection.execute(
            """
            INSERT INTO asset_versions(
                version_id,asset_id,project_id,version_no,status,content_hash,summary,created_by,
                metadata_json,created_at,updated_at
            ) VALUES('version-target','asset-target',?,1,'active','content-hash','版本摘要',?,'{}',?,?)
            """,
            (repository.project_id, owner, now, now),
        )
        connection.execute(
            """
            INSERT INTO documents(
                document_id,project_id,source_file,stored_file,uploader,status,content_hash,
                registered_at,metadata_json,created_at,updated_at,organization_id,source_id,
                asset_id,version_id,owner_user_id
            ) VALUES('document-target',?,'source.md',?,?, 'active','document-hash',?,'{}',?,?,
                'org_default','source-target','asset-target','version-target',?)
            """,
            (repository.project_id, stored_file, owner, now, now, now, owner),
        )
        connection.execute(
            "INSERT INTO document_chunks(chunk_id,document_id,chunk_index,vector_id,text_content,metadata_json,created_at) "
            "VALUES('chunk-target','document-target',0,'vector-target','需要删除的正文','{}',?)",
            (now,),
        )
        connection.execute(
            "INSERT INTO asset_version_documents(version_id,document_id,created_at) "
            "VALUES('version-target','document-target',?)",
            (now,),
        )
        for projection in ("vector", "graph", "context", "readiness"):
            connection.execute(
                "INSERT INTO asset_projection_states(version_id,projection_type,status,updated_at) "
                "VALUES('version-target',?,'ready',?)",
                (projection, now),
            )
        connection.execute(
            "INSERT INTO graph_nodes(project_id,node_id,origin,source,updated_at,data_json) "
            "VALUES(?,'node-target','auto','source.md',?,?)",
            (repository.project_id, now, json.dumps({"asset_id": "asset-target"})),
        )
        connection.execute(
            "INSERT INTO feishu_candidates(candidate_id,project_id,chat_id,status,updated_at,data_json) "
            "VALUES('candidate-target',?,'chat-target','published',?,'{}')",
            (repository.project_id, now),
        )
        connection.execute(
            """
            INSERT INTO feishu_messages(
                message_id,project_id,chat_id,message_type,content_status,content_hash,
                create_time,updated_at,data_json
            ) VALUES('message-target',?,'chat-target','text','published','message-hash',?,?,?)
            """,
            (repository.project_id, now, now, json.dumps({"sender": owner, "content": "敏感消息"})),
        )
        connection.execute(
            "INSERT INTO feishu_candidate_messages(candidate_id,message_id) "
            "VALUES('candidate-target','message-target')"
        )
        connection.execute(
            "INSERT INTO feishu_assets(asset_id,project_id,candidate_id,status,stored_file,updated_at,data_json) "
            "VALUES('legacy-asset-target',?,'candidate-target','published',?,?,?)",
            (repository.project_id, stored_file, now, json.dumps({
                "candidate_id": "candidate-target", "stored_file": stored_file,
            })),
        )
        connection.execute(
            "INSERT INTO knowledge_asset_aliases(project_id,alias_type,alias_value,asset_id,created_at) "
            "VALUES(?,'feishu_asset','legacy-asset-target','asset-target',?)",
            (repository.project_id, now),
        )
    return {
        "source_id": "source-target", "asset_id": "asset-target", "user_id": owner,
        "stored_file": stored_file,
    }


class DataGovernanceRepositoryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repository = StorageRepository(Database(self.root / "data" / "app.db"))

    def tearDown(self):
        self.temp.cleanup()

    def test_policy_defaults_update_validation_and_audit(self):
        policy = self.repository.get_data_governance_policy()
        self.assertEqual(policy["raw_message_retention_days"], 90)
        self.assertTrue(policy["daily_backup_enabled"])
        self.assertFalse(policy["daily_retention_enabled"])
        self.assertEqual(policy["schedule_time"], "02:00")
        updated = self.repository.update_data_governance_policy(
            {"raw_message_retention_days": 30, "default_sensitivity": "confidential"},
            actor_user_id="admin",
        )
        self.assertEqual(updated["raw_message_retention_days"], 30)
        self.assertEqual(updated["default_sensitivity"], "confidential")
        with self.assertRaises(ValueError):
            self.repository.update_data_governance_policy(
                {"backup_retention_count": 0}, actor_user_id="admin",
            )
        audit = self.repository.list_audit_events(action="governance.policy_updated")
        self.assertEqual(audit["total"], 1)

    def test_daily_schedule_policy_validates_timezone_time_and_booleans(self):
        updated = self.repository.update_data_governance_policy(
            {
                "daily_backup_enabled": False, "daily_retention_enabled": True,
                "schedule_time": "23:15", "timezone": "America/New_York",
            },
            actor_user_id="admin",
        )
        self.assertFalse(updated["daily_backup_enabled"])
        self.assertTrue(updated["daily_retention_enabled"])
        self.assertEqual(updated["schedule_time"], "23:15")
        self.assertEqual(updated["timezone"], "America/New_York")
        for changes in ({"schedule_time": "25:00"}, {"timezone": "Mars/Base"}, {"daily_backup_enabled": "yes"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.repository.update_data_governance_policy(changes, actor_user_id="admin")

    def test_retention_preview_and_apply_preserve_linked_and_key_audit(self):
        old = "2020-01-01T00:00:00+00:00"
        with self.repository.database.transaction(write=True) as connection:
            for message_id in ("old-free", "old-linked"):
                connection.execute(
                    """
                    INSERT INTO feishu_messages(
                        message_id,project_id,chat_id,message_type,content_status,content_hash,
                        create_time,updated_at,data_json
                    ) VALUES(?,?,'chat','text','archived','hash',?,?,?)
                    """,
                    (message_id, self.repository.project_id, old, old, json.dumps({"text": "sensitive"})),
                )
            connection.execute(
                """
                INSERT INTO feishu_candidates(candidate_id,project_id,chat_id,status,updated_at,data_json)
                VALUES('candidate',?,'chat','published',?,'{}')
                """,
                (self.repository.project_id, utc_now()),
            )
            connection.execute(
                "INSERT INTO feishu_candidate_messages(candidate_id,message_id) VALUES('candidate','old-linked')"
            )
        self.repository.write_security_audit("member", "access.knowledge_graph", "api", "graph", {})
        self.repository.write_security_audit("admin", "knowledge.asset_revoked", "asset", "asset-1", {})
        job = self.repository.create_or_get_processing_job({
            "job_type": "test", "created_by": "admin", "payload": {},
        })
        self.repository.claim_next_processing_job("worker")
        self.repository.complete_processing_job(job["job_id"], {})
        with self.repository.database.transaction(write=True) as connection:
            connection.execute(
                "UPDATE audit_events SET created_at=? WHERE action='access.knowledge_graph'", (old,)
            )
            connection.execute(
                "UPDATE processing_jobs SET completed_at=?,updated_at=? WHERE job_id=?",
                (old, old, job["job_id"]),
            )
        preview = self.repository.preview_data_retention()
        self.assertEqual(preview["raw_message_redaction_count"], 1)
        self.assertEqual(preview["access_audit_expiry_count"], 1)
        self.assertEqual(preview["processing_job_expiry_count"], 1)
        result = self.repository.apply_data_retention(actor_user_id="admin")
        self.assertEqual(result["raw_messages_redacted"], 1)
        with self.repository.database.transaction() as connection:
            free = json.loads(connection.execute(
                "SELECT data_json FROM feishu_messages WHERE message_id='old-free'"
            ).fetchone()[0])
            linked = json.loads(connection.execute(
                "SELECT data_json FROM feishu_messages WHERE message_id='old-linked'"
            ).fetchone()[0])
            self.assertTrue(free["retention_redacted"])
            self.assertEqual(linked["text"], "sensitive")
            self.assertEqual(connection.execute(
                "SELECT COUNT(*) FROM audit_events WHERE action='access.knowledge_graph'"
            ).fetchone()[0], 0)
            self.assertEqual(connection.execute(
                "SELECT COUNT(*) FROM audit_events WHERE action='knowledge.asset_revoked'"
            ).fetchone()[0], 1)
            self.assertEqual(connection.execute(
                "SELECT COUNT(*) FROM processing_jobs WHERE job_id=?", (job["job_id"],)
            ).fetchone()[0], 0)

    def test_audit_output_redacts_sensitive_detail(self):
        self.repository.write_security_audit(
            "admin", "test", "config", "one", {"api_key": "secret", "nested": {"token": "abc"}},
        )
        event = self.repository.list_audit_events(action="test")["items"][0]
        self.assertEqual(event["detail"]["api_key"], "***")
        self.assertEqual(event["detail"]["nested"]["token"], "***")

    def test_targeted_preview_export_and_source_delete_are_audited_and_idempotent(self):
        target = seed_targeted_asset(self.repository)
        preview = self.repository.preview_targeted_governance("source", target["source_id"])
        self.assertEqual(preview["asset_count"], 1)
        self.assertEqual(preview["document_count"], 1)
        self.assertEqual(preview["chunk_count"], 1)
        self.assertEqual(preview["message_count"], 1)

        exported = self.repository.export_targeted_governance(
            "source", target["source_id"], actor_user_id="admin",
        )
        self.assertEqual(exported["records"]["chunks"][0]["text_content"], "需要删除的正文")
        self.assertEqual(
            self.repository.list_audit_events(action="governance.target_exported")["total"], 1,
        )

        result = self.repository.apply_targeted_governance_deletion(
            "source", target["source_id"], actor_user_id="admin",
        )
        self.assertEqual(result["assets_deleted"], 1)
        self.assertEqual(result["chunks_deleted"], 1)
        self.assertEqual(result["messages_redacted"], 1)
        with self.repository.database.transaction() as connection:
            self.assertEqual(connection.execute(
                "SELECT status FROM knowledge_assets WHERE asset_id='asset-target'"
            ).fetchone()[0], "deleted")
            self.assertEqual(connection.execute(
                "SELECT COUNT(*) FROM document_chunks WHERE document_id='document-target'"
            ).fetchone()[0], 0)
            self.assertEqual(connection.execute(
                "SELECT COUNT(*) FROM graph_nodes WHERE node_id='node-target'"
            ).fetchone()[0], 0)
            message = json.loads(connection.execute(
                "SELECT data_json FROM feishu_messages WHERE message_id='message-target'"
            ).fetchone()[0])
            self.assertTrue(message["targeted_deleted"])
            legacy_asset = connection.execute(
                "SELECT status,stored_file,data_json FROM feishu_assets WHERE asset_id='legacy-asset-target'"
            ).fetchone()
            self.assertEqual(legacy_asset["status"], "reverted")
            self.assertEqual(legacy_asset["stored_file"], "")
            self.assertTrue(json.loads(legacy_asset["data_json"])["targeted_deleted"])
            legacy_candidate = connection.execute(
                "SELECT status,data_json FROM feishu_candidates WHERE candidate_id='candidate-target'"
            ).fetchone()
            self.assertEqual(legacy_candidate["status"], "reverted")
            self.assertTrue(json.loads(legacy_candidate["data_json"])["targeted_deleted"])
            self.assertEqual(connection.execute(
                "SELECT COUNT(*) FROM targeted_deletion_runs"
            ).fetchone()[0], 1)

        self.repository.complete_targeted_file_cleanup(
            result["run_id"], removed=result["files"], actor_user_id="admin",
        )
        repeated = self.repository.apply_targeted_governance_deletion(
            "source", target["source_id"], actor_user_id="admin",
        )
        self.assertEqual(repeated["assets_deleted"], 0)
        self.assertEqual(repeated["documents_deleted"], 0)
        self.assertEqual(repeated["messages_redacted"], 0)

    def test_targeted_user_delete_disables_identity_and_rejects_self_delete(self):
        target = seed_targeted_asset(self.repository)
        with self.assertRaises(ValueError):
            self.repository.apply_targeted_governance_deletion(
                "user", target["user_id"], actor_user_id=target["user_id"],
            )
        result = self.repository.apply_targeted_governance_deletion(
            "user", target["user_id"], actor_user_id="admin",
        )
        self.assertEqual(result["assets_deleted"], 1)
        with self.repository.database.transaction() as connection:
            user = connection.execute(
                "SELECT display_name,status,email FROM users WHERE user_id=?", (target["user_id"],)
            ).fetchone()
            membership = connection.execute(
                "SELECT status FROM project_memberships WHERE project_id=? AND user_id=?",
                (self.repository.project_id, target["user_id"]),
            ).fetchone()
            self.assertEqual(dict(user), {"display_name": "[已删除用户]", "status": "disabled", "email": ""})
            self.assertEqual(membership["status"], "disabled")


class DataGovernanceServiceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repository = StorageRepository(Database(self.root / "data" / "app.db"))
        self.service = DataGovernanceService(
            lambda: self.repository, lambda: self.root / "backups",
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_backup_is_verified_listed_and_pruned(self):
        self.repository.update_data_governance_policy(
            {"backup_retention_count": 1}, actor_user_id="admin",
        )
        context = ReportContext()
        first = self.service.create_backup(context, {"actor_user_id": "admin"})
        second = self.service.create_backup(context, {"actor_user_id": "admin"})
        backups = self.service.list_backups()
        self.assertEqual(len(backups), 1)
        self.assertTrue(backups[0]["valid"])
        self.assertEqual(backups[0]["name"], second["name"])
        self.assertIn(first["name"], second["pruned"])
        self.assertTrue(any(stage == "backup_verify" for stage, _, _ in context.events))

    def test_retention_requires_exact_confirmation(self):
        with self.assertRaises(ValueError):
            self.service.apply_retention(ReportContext(), {"confirmation": "yes"})

    def test_targeted_service_removes_original_and_refreshes_projections(self):
        target = seed_targeted_asset(self.repository)
        original = self.root / "uploads" / target["stored_file"]
        original.parent.mkdir(parents=True)
        original.write_text("敏感原件", encoding="utf-8")
        refreshed = []
        service = DataGovernanceService(
            lambda: self.repository,
            lambda: self.root / "backups",
            lambda: self.root / "uploads",
            lambda: refreshed.append(True) or {"vector": {"changed": True}, "graph": None, "context": None},
        )
        context = ReportContext()
        result = service.apply_targeted_deletion(context, {
            "target_type": "asset", "target_id": target["asset_id"],
            "confirmation": "永久删除定向数据", "actor_user_id": "admin",
        })
        self.assertFalse(original.exists())
        self.assertEqual(result["file_cleanup"]["removed"], [target["stored_file"]])
        self.assertEqual(refreshed, [True])
        self.assertTrue(any(stage == "target_delete_projections" for stage, _, _ in context.events))

    def test_reset_knowledge_library_removes_content_without_backup_or_member_deletion(self):
        target = seed_targeted_asset(self.repository)
        original = self.root / "uploads" / target["stored_file"]
        original.parent.mkdir(parents=True)
        original.write_text("待清空的演示资料", encoding="utf-8")
        self.repository.create_or_merge_knowledge_task({
            "task_type": "knowledge_gap", "title": "待清空知识任务",
            "reporter_user_id": "admin", "source_type": "manual",
        })
        self.repository.create_or_get_processing_job({
            "job_type": "demo_history", "payload": {}, "created_by": "admin",
        })
        self.repository.save_handover({
            "id": "handover-reset", "name": "待清空交接", "files": [],
        })
        onboarding = OnboardingService(lambda: self.repository)
        onboarding.ensure_defaults()
        template = self.repository.get_onboarding_template(role_key="developer")
        self.repository.create_onboarding_plan(
            {
                "plan_id": "plan-reset", "user_id": target["user_id"],
                "template_id": template["template_id"], "source_type": "handover",
                "source_key": "handover-reset",
            },
            [{"item_id": "item-reset", "title": "待清空学习任务", "item_type": "reading"}],
            actor="admin",
        )
        now = utc_now()
        with self.repository.database.transaction(write=True) as connection:
            connection.execute(
                """
                INSERT INTO handover_snapshots(
                    snapshot_id,handover_id,project_id,snapshot_version,snapshot_json,checksum,created_by,created_at
                ) VALUES(?,?,?,?,?,?,?,?)
                """,
                ("snapshot-reset", "handover-reset", self.repository.project_id, 1, "{}", "checksum", "admin", now),
            )
        refreshed = []
        service = DataGovernanceService(
            lambda: self.repository,
            lambda: self.root / "backups",
            lambda: self.root / "uploads",
            lambda: refreshed.append(True) or {"vector": {"changed": True}},
        )

        result = service.reset_knowledge_library(actor_user_id="admin")

        self.assertEqual(result["sources_deleted"], 1)
        self.assertEqual(result["documents_deleted"], 1)
        self.assertEqual(result["history"], {
            "processing_jobs": 1, "knowledge_tasks": 1,
            "onboarding_learning_plans": 1, "handover_cases": 1,
        })
        self.assertFalse(original.exists())
        self.assertEqual(refreshed, [True])
        self.assertFalse((self.root / "backups").exists())
        with self.repository.database.transaction() as connection:
            self.assertEqual(connection.execute(
                "SELECT COUNT(*) FROM knowledge_sources WHERE status='active'"
            ).fetchone()[0], 0)
            self.assertEqual(connection.execute(
                "SELECT COUNT(*) FROM documents WHERE status='active'"
            ).fetchone()[0], 0)
            self.assertEqual(connection.execute(
                "SELECT status FROM users WHERE user_id=?", (target["user_id"],)
            ).fetchone()[0], "active")
            self.assertEqual(connection.execute(
                "SELECT COUNT(*) FROM processing_jobs WHERE project_id=?", (self.repository.project_id,)
            ).fetchone()[0], 0)
            self.assertEqual(connection.execute(
                "SELECT COUNT(*) FROM knowledge_tasks WHERE project_id=?", (self.repository.project_id,)
            ).fetchone()[0], 0)
            self.assertEqual(connection.execute(
                "SELECT COUNT(*) FROM handover_cases WHERE project_id=?", (self.repository.project_id,)
            ).fetchone()[0], 0)
            self.assertEqual(connection.execute(
                "SELECT COUNT(*) FROM handover_snapshots WHERE project_id=?", (self.repository.project_id,)
            ).fetchone()[0], 0)
            self.assertEqual(connection.execute(
                "SELECT COUNT(*) FROM onboarding_learning_plans WHERE project_id=?", (self.repository.project_id,)
            ).fetchone()[0], 0)
            self.assertEqual(connection.execute(
                "SELECT COUNT(*) FROM onboarding_learning_items"
            ).fetchone()[0], 0)
        self.assertEqual(self.repository.list_audit_events(action="knowledge.library_reset")["total"], 1)

    def test_reset_knowledge_library_also_clears_expired_and_revoked_sources(self):
        target = seed_targeted_asset(self.repository)
        original = self.root / "uploads" / target["stored_file"]
        original.parent.mkdir(parents=True)
        original.write_text("已失效资料也必须被清空", encoding="utf-8")
        self.repository.transition_asset(target["asset_id"], "expire", "admin")
        self.assertEqual(self.repository.active_knowledge_source_ids(), [])
        self.assertEqual(self.repository.resettable_knowledge_source_ids(), [target["source_id"]])

        service = DataGovernanceService(
            lambda: self.repository,
            lambda: self.root / "backups",
            lambda: self.root / "uploads",
        )
        result = service.reset_knowledge_library(actor_user_id="admin")

        self.assertEqual(result["sources_processed"], 1)
        self.assertEqual(result["assets_deleted"], 1)
        self.assertEqual(result["documents_deleted"], 1)
        self.assertFalse(original.exists())
        self.assertEqual(self.repository.resettable_knowledge_source_ids(), [])
        with self.repository.database.transaction() as connection:
            self.assertEqual(connection.execute(
                "SELECT status FROM knowledge_sources WHERE source_id=?", (target["source_id"],)
            ).fetchone()[0], "deleted")
            self.assertEqual(connection.execute(
                "SELECT status FROM knowledge_assets WHERE asset_id=?", (target["asset_id"],)
            ).fetchone()[0], "deleted")
            self.assertEqual(connection.execute(
                "SELECT status FROM documents WHERE document_id='document-target'"
            ).fetchone()[0], "deleted")

    def test_reset_knowledge_library_clears_feishu_history_and_resources_but_keeps_sources(self):
        resource_dir = self.root / "uploads" / "feishu_resources"
        resource_dir.mkdir(parents=True)
        (resource_dir / "message-image.png").write_bytes(b"image evidence")
        # An untracked file could belong to another project sharing the same
        # deployment upload root, so it must not be deleted by this project reset.
        (resource_dir / "other-project-image.png").write_bytes(b"other project image")
        generated_document = self.root / "uploads" / "feishu-generated.md"
        generated_document.write_text("飞书沉淀知识正文", encoding="utf-8")
        self.repository.save_feishu_workspace({
            "groups": [{
                "chat_id": "oc_reset", "name": "知识采集群", "collection_mode": "review",
                "visibility": "project",
            }],
            "messages": [{
                "message_id": "fm_reset", "chat_id": "oc_reset", "message_type": "image",
                "content_status": "review_required", "content": "上线前完成回滚演练",
                "resource_files": [{"stored_file": "message-image.png", "content_type": "image/png"}],
                "create_time": utc_now(),
            }],
            "candidates": [{
                "candidate_id": "fc_reset", "chat_id": "oc_reset", "status": "review_required",
                "source_message_ids": ["fm_reset"],
            }],
            "assets": [{
                "asset_id": "fa_reset", "candidate_id": "fc_reset", "status": "published",
                "stored_file": "feishu-generated.md", "source_message_ids": ["fm_reset"],
            }],
            "diagnostics": {"last_event": "同步完成"},
        })
        prior_revision = self.repository.feishu_revision()
        service = DataGovernanceService(
            lambda: self.repository,
            lambda: self.root / "backups",
            lambda: self.root / "uploads",
        )

        result = service.reset_knowledge_library(actor_user_id="admin")

        feishu_history = result["feishu_history"]
        self.assertEqual(feishu_history["messages_deleted"], 1)
        self.assertEqual(feishu_history["candidates_deleted"], 1)
        self.assertEqual(feishu_history["assets_deleted"], 1)
        self.assertEqual(feishu_history["diagnostics_deleted"], 1)
        self.assertEqual(feishu_history["groups_retained"], 1)
        self.assertEqual(feishu_history["revision"], prior_revision + 1)
        self.assertEqual(
            set(feishu_history["resource_file_cleanup"]["removed"]),
            {"message-image.png"},
        )
        self.assertFalse(generated_document.exists())
        self.assertFalse((resource_dir / "message-image.png").exists())
        self.assertTrue((resource_dir / "other-project-image.png").exists())
        workspace = self.repository.load_feishu_workspace()
        self.assertEqual([group["chat_id"] for group in workspace["groups"]], ["oc_reset"])
        self.assertEqual(workspace["messages"], [])
        self.assertEqual(workspace["candidates"], [])
        self.assertEqual(workspace["assets"], [])
        self.assertEqual(workspace["diagnostics"], {})
        with self.repository.database.transaction() as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM feishu_candidate_messages").fetchone()[0], 0)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM feishu_asset_messages").fetchone()[0], 0)

    def test_daily_schedule_is_timezone_aware_durable_and_idempotent(self):
        current = datetime(2026, 8, 27, 19, 30, tzinfo=timezone.utc)  # 上海 03:30
        first = self.service.schedule_daily_operations(now=current)
        repeated = self.service.schedule_daily_operations(now=current)
        self.assertTrue(first["enabled"])
        self.assertEqual(first["local_date"], "2026-08-28")
        self.assertEqual(first["job_count"], 1)
        self.assertEqual(first["jobs"][0]["job_type"], "database_backup")
        self.assertEqual(first["jobs"][0]["payload"]["trigger"], "daily_governance_schedule")
        self.assertEqual(repeated["job_count"], 0)
        self.assertEqual(repeated["existing_operations"], ["database_backup"])
        status = self.service.schedule_status(now=current)
        self.assertEqual(len(status["latest_intents"]), 1)
        self.assertEqual(status["latest_intents"][0]["processing_status"], "queued")

    def test_daily_schedule_catches_up_previous_day_and_optionally_runs_retention(self):
        self.repository.update_data_governance_policy(
            {"daily_retention_enabled": True}, actor_user_id="admin",
        )
        before_window = datetime(2026, 8, 27, 16, 30, tzinfo=timezone.utc)  # 上海 00:30
        result = self.service.schedule_daily_operations(now=before_window)
        self.assertEqual(result["local_date"], "2026-08-27")
        self.assertEqual(result["job_count"], 2)
        jobs = {item["job_type"]: item for item in result["jobs"]}
        self.assertEqual(set(jobs), {"database_backup", "data_retention"})
        self.assertEqual(jobs["data_retention"]["payload"]["confirmation"], "执行数据清理")


if __name__ == "__main__":
    unittest.main()
