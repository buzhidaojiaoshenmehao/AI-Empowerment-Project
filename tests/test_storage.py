import asyncio
import hashlib
import io
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException, UploadFile
from langchain_core.documents import Document

from backend import main
from backend.config import settings
from backend.feishu_workspace import FeishuWorkspace
from backend.knowledge_base.vector_store import LocalEmbeddings, VectorStore
from backend.storage import get_repository, reset_repository_for_tests
from backend.storage.database import DEFAULT_PROJECT_ID, SYSTEM_USER_ID, Database, SCHEMA_MIGRATIONS, utc_now
from backend.storage.legacy import LegacyMigrationError, LegacyMigrator
from backend.storage.repository import StorageRepository


class SQLiteStorageTest(unittest.TestCase):
    def _repository(self, root: Path) -> StorageRepository:
        return StorageRepository(Database(root / "data" / "app.db"))

    def _write_legacy(self, root: Path) -> Path:
        chroma = root / "chroma_db"
        chroma.mkdir(parents=True, exist_ok=True)
        (root / "categories.json").write_bytes(
            json.dumps(["交接文档", "项目规范"], ensure_ascii=False).encode("gbk")
        )
        metadata = {
            "source_file": "规范.md", "stored_file": "stored_规范.md",
            "category": "项目规范", "categories": ["项目规范"], "uploader": "测试用户",
        }
        (chroma / "vector_store.json").write_text(json.dumps({
            "vectors": [[1.0, 0.0]], "texts": ["项目规范正文"],
            "metadatas": [metadata], "ids": ["vec_1"],
        }, ensure_ascii=False), encoding="utf-8")
        (chroma / "project_context.json").write_text(json.dumps({
            "project_name": "迁移测试项目", "metadata": {
                "stored_规范.md": metadata,
                "stored_仅元数据.md": {
                    "source_file": "仅元数据.md", "stored_file": "stored_仅元数据.md",
                    "category": "项目规范", "categories": ["项目规范"],
                },
            },
        }, ensure_ascii=False), encoding="utf-8")
        (chroma / "feishu_workspace.json").write_text(json.dumps({
            "version": 4,
            "groups": [{"chat_id": "oc_legacy", "name": "迁移群", "collection_mode": "auto"}],
            "messages": [{
                "message_id": "om_legacy", "chat_id": "oc_legacy", "message_type": "text",
                "content": "迁移消息", "content_status": "archived", "create_time": "1",
            }],
            "candidates": [{
                "candidate_id": "candidate_legacy", "chat_id": "oc_legacy",
                "aggregation_key": "legacy-topic", "source_message_ids": ["om_legacy"],
            }],
            "assets": [{
                "asset_id": "asset_legacy", "candidate_id": "candidate_legacy",
                "stored_file": "legacy.md", "source_message_ids": ["om_legacy"],
            }],
            "audit": [{
                "audit_id": "audit_legacy", "action": "legacy_import", "created_at": "1",
            }],
            "diagnostics": {},
        }, ensure_ascii=False), encoding="utf-8")
        (root / "knowledge_graph_state.json").write_text(json.dumps({
            "nodes": [
                {"id": "node_a", "label": "节点 A", "origin": "manual"},
                {"id": "node_b", "label": "节点 B", "origin": "manual"},
            ],
            "edges": [{"id": "edge_ab", "source": "node_a", "target": "node_b", "method": "manual"}],
        }, ensure_ascii=False), encoding="utf-8")
        uploads = root / "uploads"
        uploads.mkdir(exist_ok=True)
        (uploads / "legacy.md").write_text("legacy upload", encoding="utf-8")
        return chroma

    def test_schema_pragmas_placeholders_and_quick_check(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = self._repository(Path(directory))
            database = repository.database
            with database.transaction() as connection:
                self.assertEqual(connection.execute("PRAGMA foreign_keys").fetchone()[0], 1)
                self.assertEqual(connection.execute("PRAGMA journal_mode").fetchone()[0].lower(), "wal")
                self.assertGreaterEqual(connection.execute("PRAGMA busy_timeout").fetchone()[0], 1000)
                self.assertIsNotNone(connection.execute(
                    "SELECT 1 FROM projects WHERE project_id=?", (DEFAULT_PROJECT_ID,)
                ).fetchone())
                self.assertIsNotNone(connection.execute(
                    "SELECT 1 FROM users WHERE user_id=? AND user_type='placeholder'", (SYSTEM_USER_ID,)
                ).fetchone())
            self.assertTrue(database.quick_check()["ok"])
            self.assertEqual(database.migration_status()["pending_versions"], [])
            with database.transaction() as connection:
                document_columns = {row[1] for row in connection.execute("PRAGMA table_info(documents)")}
                self.assertTrue({
                    "organization_id", "source_id", "asset_id", "version_id", "visibility",
                    "sensitivity_level", "access_policy_id", "acl_revision",
                }.issubset(document_columns))

    def test_legacy_migration_is_idempotent_and_accepts_gbk_categories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chroma = self._write_legacy(root)
            repository = self._repository(root)
            migrator = LegacyMigrator(repository, root, chroma)

            first = migrator.migrate_once()
            second = migrator.migrate_once()

            self.assertEqual(first["status"], "success")
            self.assertEqual(second["status"], "already_migrated")
            self.assertEqual(repository.list_categories(), ["交接文档", "项目规范"])
            self.assertEqual(len(repository.list_documents()), 2)
            self.assertEqual(len(repository.document_chunks()), 1)
            self.assertEqual(len(repository.load_feishu_workspace()["messages"]), 1)
            self.assertTrue(first["reconciled"])
            self.assertEqual(first["counts"], {
                "categories": 2,
                "documents": 2,
                "chunks": 1,
                "handovers": 0,
                "feishu_groups": 1,
                "feishu_messages": 1,
                "feishu_candidates": 1,
                "feishu_assets": 1,
                "audit_events": 1,
                "graph_nodes": 2,
                "graph_edges": 1,
                "upload_files": 1,
            })
            self.assertTrue(all(item["matched"] for item in first["reconciliation"].values()))
            self.assertTrue(Path(first["report_path"]).is_file())
            backup_manifest = Path(first["backup_dir"]) / "manifest.json"
            self.assertTrue(backup_manifest.is_file())

    def test_empty_migration_reopens_when_legacy_files_appear_later(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chroma = root / "chroma_db"
            chroma.mkdir(parents=True)
            repository = self._repository(root)
            migrator = LegacyMigrator(repository, root, chroma)

            self.assertEqual(migrator.migrate_once()["status"], "success")
            self._write_legacy(root)
            reopened = migrator.migrate_once()

            self.assertEqual(reopened["status"], "success")
            self.assertEqual(len(repository.list_documents()), 2)
            self.assertEqual(len(repository.document_chunks()), 1)
            self.assertEqual(len(repository.load_feishu_workspace()["messages"]), 1)

    def test_corrupt_json_fails_closed_and_keeps_backup_and_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chroma = root / "chroma_db"
            chroma.mkdir(parents=True)
            (chroma / "vector_store.json").write_text('{"texts": [', encoding="utf-8")
            repository = self._repository(root)
            migrator = LegacyMigrator(repository, root, chroma)

            with self.assertRaises(LegacyMigrationError):
                migrator.migrate_once()

            self.assertEqual(repository.list_documents(), [])
            run = repository.migration_run("legacy_json_v1")
            self.assertEqual(run["status"], "failed")
            self.assertTrue(Path(run["report_path"]).is_file())
            reports = json.loads(Path(run["report_path"]).read_text(encoding="utf-8"))
            self.assertTrue((Path(reports["backup_dir"]) / "chroma_db" / "vector_store.json").is_file())

    def test_concurrent_feishu_writes_preserve_all_messages(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "data" / "app.db"
            initial = FeishuWorkspace(StorageRepository(Database(database_path)))
            initial.upsert_group({"chat_id": "oc_parallel", "name": "并发群", "collection_mode": "archive_only"})

            def write_message(index: int) -> None:
                workspace = FeishuWorkspace(StorageRepository(Database(database_path)))
                workspace.upsert_messages("oc_parallel", [{
                    "message_id": f"om_{index}", "msg_type": "text",
                    "body": {"content": json.dumps({"text": f"并发消息 {index}"}, ensure_ascii=False)},
                    "create_time": str(int(datetime.now().timestamp() * 1000) + index),
                }])

            with ThreadPoolExecutor(max_workers=20) as executor:
                list(executor.map(write_message, range(40)))

            restarted = FeishuWorkspace(StorageRepository(Database(database_path)))
            self.assertEqual({item["message_id"] for item in restarted.list_items(limit=100)}, {
                f"om_{index}" for index in range(40)
            })
            self.assertTrue(restarted.repository.database.quick_check()["ok"])

    def test_feishu_event_id_is_idempotent_across_different_message_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = self._repository(Path(directory))
            workspace = FeishuWorkspace(repository)
            workspace.upsert_group({"chat_id": "oc_event", "collection_mode": "archive_only"})
            first = {
                "message_id": "om_first", "event_id": "evt_same", "msg_type": "text",
                "body": {"content": json.dumps({"text": "同一事件"}, ensure_ascii=False)},
            }
            second = {**first, "message_id": "om_second"}

            self.assertEqual(workspace.upsert_messages("oc_event", [first]), 1)
            self.assertEqual(workspace.upsert_messages("oc_event", [second]), 0)
            self.assertEqual(len(workspace.list_items(limit=100)), 1)

    def test_process_restart_reads_same_sqlite_database(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = {
                **os.environ,
                "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
                "AI_EMPOWERMENT_BASE_DIR": str(root),
            }
            write_script = (
                "from types import SimpleNamespace; "
                "from backend.storage import get_repository; "
                "get_repository().upsert_document_chunks([SimpleNamespace("
                "page_content='跨进程正文', metadata={'source_file':'跨进程.md','stored_file':'跨进程.md'})])"
            )
            read_script = (
                "from backend.storage import get_repository; "
                "docs=get_repository().list_documents(); "
                "assert len(docs)==1 and docs[0]['source_file']=='跨进程.md'"
            )
            subprocess.run([sys.executable, "-c", write_script], cwd=root, env=environment, check=True)
            subprocess.run([sys.executable, "-c", read_script], cwd=root, env=environment, check=True)

    def test_deployment_paths_cannot_be_changed_by_editable_user_config(self):
        with tempfile.TemporaryDirectory() as directory:
            config_file = Path(directory) / "config.json"
            config_file.write_text(json.dumps({
                "DATABASE_PATH": "/tmp/should-not-be-used.db",
                "DATABASE_BACKUP_DIR": "/tmp/should-not-be-used",
                "LLM_MODEL": "saved-model",
            }), encoding="utf-8")
            original_database = settings.DATABASE_PATH
            original_backup = settings.DATABASE_BACKUP_DIR
            original_model = settings.LLM_MODEL
            try:
                with patch.object(main, "CONFIG_FILE", config_file):
                    main._load_saved_config()
                self.assertEqual(settings.DATABASE_PATH, original_database)
                self.assertEqual(settings.DATABASE_BACKUP_DIR, original_backup)
                self.assertEqual(settings.LLM_MODEL, "saved-model")
            finally:
                settings.DATABASE_PATH = original_database
                settings.DATABASE_BACKUP_DIR = original_backup
                settings.LLM_MODEL = original_model

    def test_existing_v1_database_upgrades_to_current_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "data" / "app.db"
            database_path.parent.mkdir(parents=True)
            version, name, sql = SCHEMA_MIGRATIONS[0]
            checksum = hashlib.sha256(sql.encode("utf-8")).hexdigest()
            connection = sqlite3.connect(str(database_path))
            try:
                connection.execute(
                    "CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY,name TEXT NOT NULL,checksum TEXT NOT NULL,applied_at TEXT NOT NULL)"
                )
                connection.executescript(sql)
                connection.execute(
                    "INSERT INTO schema_migrations(version,name,checksum,applied_at) VALUES(?,?,?,?)",
                    (version, name, checksum, utc_now()),
                )
                connection.commit()
            finally:
                connection.close()

            database = Database(database_path)
            database.initialize()

            self.assertEqual(database.migration_status()["current_version"], len(SCHEMA_MIGRATIONS))
            with database.transaction() as upgraded:
                document_columns = {row[1] for row in upgraded.execute("PRAGMA table_info(documents)")}
                message_columns = {row[1] for row in upgraded.execute("PRAGMA table_info(feishu_messages)")}
            self.assertIn("access_policy_id", document_columns)
            self.assertIn("event_id", message_columns)

    def test_legacy_import_rolls_back_the_whole_transaction(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = self._repository(Path(directory))
            snapshot = {
                "categories": ["项目规范"],
                "documents": [
                    {"document": {"source_file": "一.md", "stored_file": "一.md"}, "chunks": []},
                    {"document": {"source_file": "二.md", "stored_file": "二.md"}, "chunks": []},
                ],
            }
            original = repository._upsert_document_connection
            calls = 0

            def fail_on_second(connection, document, chunks):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("injected migration failure")
                return original(connection, document, chunks)

            with patch.object(repository, "_upsert_document_connection", side_effect=fail_on_second):
                with self.assertRaises(RuntimeError):
                    repository.import_legacy_snapshot(snapshot)

            self.assertEqual(repository.list_documents(), [])
            self.assertEqual(repository.list_categories(), [])

    def test_document_source_identity_is_stable_across_versions(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = self._repository(Path(directory))
            repository.upsert_document_chunks([SimpleNamespace(
                page_content="第一版",
                metadata={"source_file": "规范.md", "stored_file": "规范-v1.md"},
            )])
            first = repository.list_documents()[0]
            repository.upsert_document_chunks([SimpleNamespace(
                page_content="第二版",
                metadata={"source_file": "规范.md", "stored_file": "规范-v2.md"},
            )])
            versions = repository.list_documents()

            self.assertEqual({item["source_id"] for item in versions}, {first["source_id"]})
            self.assertEqual(len({item["version_id"] for item in versions}), 2)

    def test_restart_reads_documents_handovers_and_graph_edits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self._repository(root)
            docs = [SimpleNamespace(page_content="第一块正文", metadata={
                "source_file": "设计.md", "stored_file": "stored_设计.md",
                "category": "技术架构", "categories": ["技术架构"],
            })]
            first.upsert_document_chunks(docs, ["vec_1"])
            first.save_handovers([{
                "id": "handover_1", "name": "张三", "role": "开发工程师",
                "recipient": "李四", "status": "pending_acceptance", "files": ["设计.md"],
            }])
            first.save_graph_edits({
                "nodes": [{"id": "manual_1", "label": "人工节点", "origin": "manual"}],
                "edges": [],
            })

            restarted = self._repository(root)
            self.assertEqual(restarted.list_documents()[0]["chunks"], 1)
            self.assertEqual(restarted.list_handovers()[0]["id"], "handover_1")
            self.assertEqual(restarted.load_graph_edits()["nodes"][0]["id"], "manual_1")

    def test_derived_graph_edges_are_not_authoritative_and_filename_delete_is_literal(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = self._repository(Path(directory))
            for source_file in ("a_b.md", "aXb.md"):
                repository.upsert_document_chunks([SimpleNamespace(
                    page_content=source_file,
                    metadata={
                        "source_file": source_file,
                        "stored_file": f"stored_{source_file}",
                        "category": "项目规范",
                        "categories": ["项目规范"],
                    },
                )])
            repository.save_graph_edits({
                "nodes": [
                    {"id": "generated_1", "label": "自动节点", "origin": "generated", "x": 10},
                    {"id": "manual_1", "label": "人工节点", "origin": "manual"},
                ],
                "edges": [{
                    "id": "auto_1", "source": "generated_1", "target": "manual_1",
                    "method": "hybrid_auto", "auto_generated": True,
                }],
            })

            self.assertEqual(repository.delete_document("a_b.md"), 1)
            self.assertEqual([item["source_file"] for item in repository.list_documents()], ["aXb.md"])
            persisted_graph = repository.load_graph_edits()
            self.assertEqual(persisted_graph["edges"], [])
            self.assertEqual({node["id"] for node in persisted_graph["nodes"]}, {"generated_1", "manual_1"})

    def test_backup_restore_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = self._repository(root)
            repository.replace_categories(["备份前"])
            backup = repository.database.backup(root / "backups")
            repository.replace_categories(["备份后"])
            self.assertIn("备份后", repository.list_categories())

            restored = repository.database.restore(backup["path"], root / "backups")

            self.assertTrue(restored["success"])
            self.assertEqual(repository.list_categories(), ["备份前"])
            self.assertTrue(repository.database.quick_check()["ok"])
            self.assertTrue(Path(restored["safety_backup"]["path"]).is_file())

    def test_restore_refuses_while_another_process_holds_runtime_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = Database(root / "data" / "app.db")
            database.initialize()
            backup = database.backup(root / "backups")
            holder_script = """
import sys
from backend.storage.database import Database
lock = Database(sys.argv[1]).process_lock('test-holder').acquire()
print('LOCKED', flush=True)
sys.stdin.readline()
lock.release()
"""
            holder = subprocess.Popen(
                [sys.executable, "-c", holder_script, str(database.path)],
                cwd=Path(__file__).resolve().parents[1],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                self.assertEqual(holder.stdout.readline().strip(), "LOCKED")
                with self.assertRaisesRegex(RuntimeError, "请先停止服务"):
                    database.restore(backup["path"], root / "backups")
            finally:
                if holder.stdin:
                    holder.stdin.write("stop\n")
                    holder.stdin.flush()
                holder.communicate(timeout=5)

            restored = database.restore(backup["path"], root / "backups")
            self.assertTrue(restored["success"])

    def test_derived_vector_index_is_rebuilt_from_sqlite_after_restart(self):
        original_database = settings.DATABASE_PATH
        original_chroma = settings.CHROMA_PERSIST_DIR
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings.DATABASE_PATH = str(root / "data" / "app.db")
            settings.CHROMA_PERSIST_DIR = str(root / "chroma_db")
            reset_repository_for_tests()
            try:
                first = VectorStore()
                first._embeddings = LocalEmbeddings()
                first.add_documents([Document(
                    page_content="可从 SQLite 恢复的知识正文",
                    metadata={
                        "source_file": "恢复.md", "stored_file": "stored_恢复.md",
                        "category": "项目规范", "categories": ["项目规范"],
                    },
                )])
                vector_file = Path(settings.CHROMA_PERSIST_DIR) / "vector_store.json"
                vector_file.write_text(json.dumps({
                    "vectors": [], "texts": [], "metadatas": [], "ids": [],
                }), encoding="utf-8")

                restarted = VectorStore()
                restarted._embeddings = LocalEmbeddings()
                result = restarted.reconcile_with_storage()

                self.assertTrue(result["changed"])
                self.assertEqual(result["recreated"], 1)
                self.assertEqual(restarted.count(), 1)
            finally:
                settings.DATABASE_PATH = original_database
                settings.CHROMA_PERSIST_DIR = original_chroma
                reset_repository_for_tests()

    def test_retrieval_reconciles_revoked_or_deleted_sqlite_documents(self):
        original_database = settings.DATABASE_PATH
        original_chroma = settings.CHROMA_PERSIST_DIR
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings.DATABASE_PATH = str(root / "data" / "app.db")
            settings.CHROMA_PERSIST_DIR = str(root / "chroma_db")
            reset_repository_for_tests()
            try:
                store = VectorStore()
                store._embeddings = LocalEmbeddings()
                store.add_documents([Document(
                    page_content="删除后绝不能继续召回的内容",
                    metadata={"source_file": "撤销.md", "stored_file": "stored_撤销.md"},
                )])
                get_repository().delete_document("撤销.md")

                self.assertEqual(store.similarity_search("删除后绝不能继续召回", k=3), [])
                self.assertEqual(store.count(), 0)
            finally:
                settings.DATABASE_PATH = original_database
                settings.CHROMA_PERSIST_DIR = original_chroma
                reset_repository_for_tests()

    def test_duplicate_legacy_vector_ids_keep_distinct_versions_by_chunk_id(self):
        original_database = settings.DATABASE_PATH
        original_chroma = settings.CHROMA_PERSIST_DIR
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings.DATABASE_PATH = str(root / "data" / "app.db")
            settings.CHROMA_PERSIST_DIR = str(root / "chroma_db")
            reset_repository_for_tests()
            try:
                repository = get_repository()
                first = Document(
                    page_content="第一版独立知识",
                    metadata={"source_file": "规范.md", "stored_file": "v1_规范.md"},
                )
                second = Document(
                    page_content="第二版独立知识",
                    metadata={"source_file": "规范.md", "stored_file": "v2_规范.md"},
                )
                repository.upsert_document_chunks([first], ["legacy_collision"])
                repository.upsert_document_chunks([second], ["legacy_collision"])

                store = VectorStore()
                store._embeddings = LocalEmbeddings()
                rebuilt = store.reconcile_with_storage()

                self.assertEqual(rebuilt["chunks"], 2)
                self.assertEqual(rebuilt["identity"], "chunk_id")
                self.assertEqual(len(set(store.get()["ids"])), 2)
                self.assertEqual(set(store.get()["ids"]), {
                    item["chunk_id"] for item in repository.document_chunks()
                })

                with self.assertRaisesRegex(ValueError, "多个版本"):
                    repository.delete_document("规范.md")

                self.assertEqual(store.delete_document("v1_规范.md"), 1)
                self.assertEqual(store.count(), 1)
                self.assertEqual(store.get()["metadatas"][0]["stored_file"], "v2_规范.md")

                self.assertEqual(repository.delete_document("v2_规范.md"), 1)
                self.assertEqual(repository.list_documents(), [])
            finally:
                settings.DATABASE_PATH = original_database
                settings.CHROMA_PERSIST_DIR = original_chroma
                reset_repository_for_tests()

    def test_restore_can_replace_a_corrupt_current_database(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = self._repository(root)
            repository.replace_categories(["可恢复分类"])
            backup = repository.database.backup(root / "backups")
            repository.database.path.write_bytes(b"not-a-sqlite-database")

            restored_database = Database(repository.database.path)
            result = restored_database.restore(backup["path"], root / "backups")
            restored_repository = StorageRepository(restored_database)

            self.assertEqual(restored_repository.list_categories(), ["可恢复分类"])
            self.assertFalse(result["safety_backup"]["valid"])
            self.assertTrue(Path(result["safety_backup"]["path"]).is_file())

    def test_context_failure_keeps_published_upload_and_marks_projection_for_repair(self):
        with tempfile.TemporaryDirectory() as directory:
            upload_dir = Path(directory)
            uploaded = UploadFile(filename="失败.md", file=io.BytesIO(b"rollback body"))
            documents = [SimpleNamespace(page_content="rollback body", metadata={})]
            prepared = {
                "asset_id": "asset_context_soft_failure",
                "source_id": "source_context_soft_failure",
                "version_id": "version_context_soft_failure",
                "version_no": 1,
                "duplicate": False,
            }
            with (
                patch.object(main, "UPLOAD_DIR", upload_dir),
                patch.object(main, "load_document", return_value=documents),
                patch.object(main.vector_store, "add_documents", return_value=1),
                patch.object(main.vector_store, "delete_documents_by_source", return_value=1) as delete_index,
                patch.object(main.project_context, "register_document", side_effect=RuntimeError("metadata failed")),
                patch.object(main.knowledge_graph, "build_graph"),
                patch.object(main.knowledge_asset_service, "prepare_documents", return_value=prepared),
                patch.object(main.knowledge_asset_service, "record_projection"),
                patch.object(
                    main.knowledge_asset_service,
                    "publish",
                    return_value={"asset_id": prepared["asset_id"], "status": "active"},
                ) as publish,
                patch.object(main.knowledge_asset_service, "mark_post_publish_projection"),
            ):
                result = asyncio.run(main.upload_document(uploaded, category="项目规范", uploader="测试用户"))

            self.assertTrue(result["success"])
            self.assertTrue(result["partial_success"])
            self.assertEqual(result["projection_status"], "repair_required")
            self.assertIn("项目语境", result["repair_required"])
            self.assertFalse(result["context_synced"])
            self.assertIn("待后台修复", result["message"])
            delete_index.assert_not_called()
            self.assertEqual(len(list(upload_dir.iterdir())), 1)
            self.assertFalse(publish.call_args.kwargs["context_ready"])
            self.assertIn("metadata failed", publish.call_args.kwargs["context_error"])

    def test_handover_projection_failure_is_reported_without_rolling_back_asset(self):
        with tempfile.TemporaryDirectory() as directory:
            upload_dir = Path(directory)
            uploaded = UploadFile(filename="交接说明.md", file=io.BytesIO(b"handover body"))
            documents = [SimpleNamespace(page_content="handover body", metadata={})]
            prepared = {
                "asset_id": "asset_handover_partial",
                "source_id": "source_handover_partial",
                "version_id": "version_handover_partial",
                "version_no": 1,
                "duplicate": False,
            }
            with (
                patch.object(main, "UPLOAD_DIR", upload_dir),
                patch.object(main, "load_document", return_value=documents),
                patch.object(main.vector_store, "add_documents", return_value=1),
                patch.object(main.project_context, "extract_project_metadata", return_value={}),
                patch.object(main.project_context, "register_document"),
                patch.object(main.knowledge_graph, "build_graph", side_effect=RuntimeError("graph failed")),
                patch.object(main.knowledge_asset_service, "prepare_documents", return_value=prepared),
                patch.object(main.knowledge_asset_service, "record_projection"),
                patch.object(
                    main.knowledge_asset_service,
                    "publish",
                    return_value={"asset_id": prepared["asset_id"], "status": "active"},
                ),
                patch.object(main.knowledge_asset_service, "mark_post_publish_projection"),
                patch.object(main, "_load_handover_records", return_value=[]),
                patch.object(main, "_save_handover_records"),
                patch.object(
                    main, "get_current_identity",
                    return_value=SimpleNamespace(
                        user_id="张三", display_name="张三", email="zhangsan@example.com",
                        project_id="project_default",
                    ),
                ),
            ):
                result = asyncio.run(main.resignation_submit(
                    "冒名成员", "开发", "李四", "2026-09-01", [uploaded],
                ))

            self.assertTrue(result["success"])
            self.assertTrue(result["partial_success"])
            self.assertEqual(result["projection_status"], "repair_required")
            self.assertEqual(result["repair_required"][0]["projections"], ["知识图谱"])
            self.assertEqual(result["files"][0]["asset_id"], prepared["asset_id"])
            self.assertEqual(result["person"]["name"], "张三")
            self.assertEqual(result["handover"]["departing_user_id"], "张三")
            self.assertTrue(next(upload_dir.iterdir()).is_file())

    def test_handover_record_failure_rolls_back_uploaded_knowledge(self):
        with tempfile.TemporaryDirectory() as directory:
            upload_dir = Path(directory)
            uploaded = UploadFile(filename="交接.md", file=io.BytesIO(b"handover body"))
            documents = [SimpleNamespace(page_content="handover body", metadata={})]
            prepared = {
                "asset_id": "asset_handover_rollback",
                "source_id": "source_handover_rollback",
                "version_id": "version_handover_rollback",
                "version_no": 1,
                "duplicate": False,
            }
            with (
                patch.object(main, "UPLOAD_DIR", upload_dir),
                patch.object(main, "load_document", return_value=documents),
                patch.object(main.vector_store, "add_documents", return_value=1),
                patch.object(main.vector_store, "delete_documents_by_source", return_value=1) as delete_index,
                patch.object(main.project_context, "register_document"),
                patch.object(main.project_context, "unregister_document", return_value=True),
                patch.object(main.knowledge_graph, "build_graph"),
                patch.object(main.knowledge_asset_service, "prepare_documents", return_value=prepared),
                patch.object(main.knowledge_asset_service, "record_projection"),
                patch.object(
                    main.knowledge_asset_service,
                    "publish",
                    return_value={"asset_id": prepared["asset_id"], "status": "active"},
                ),
                patch.object(main.knowledge_asset_service, "mark_post_publish_projection"),
                patch.object(main.knowledge_asset_service, "transition") as transition,
                patch.object(
                    main,
                    "get_repository",
                    return_value=SimpleNamespace(
                        project_id="project_default",
                        list_project_members=lambda: [],
                        readiness_assets=lambda identity=None: [],
                    ),
                ),
                patch.object(main, "_load_handover_records", return_value=[]),
                patch.object(main, "_save_handover_records", side_effect=RuntimeError("record failed")),
                patch.object(
                    main, "get_current_identity",
                    return_value=SimpleNamespace(
                        user_id="张三", display_name="张三", email="zhangsan@example.com",
                        project_id="project_default",
                    ),
                ),
            ):
                with self.assertRaises(HTTPException):
                    asyncio.run(main.resignation_submit("张三", "开发", "李四", "", [uploaded]))

            self.assertEqual(delete_index.call_count, 1)
            transition.assert_called_once_with(
                prepared["asset_id"], "revoke", "张三", reason="交接记录保存失败",
            )
            self.assertEqual(list(upload_dir.iterdir()), [])

    def test_graph_build_api_accepts_background_job_without_claiming_completion(self):
        with patch.object(main.processing_job_service, "enqueue", return_value={"job_id": "job_graph", "status": "queued"}):
            result = asyncio.run(main.build_graph())
        self.assertTrue(result["accepted"])
        self.assertEqual(result["job"]["status"], "queued")
        self.assertNotIn("完成", result["message"])

    def test_failed_derived_delete_restores_authoritative_document(self):
        original_database = settings.DATABASE_PATH
        original_chroma = settings.CHROMA_PERSIST_DIR
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings.DATABASE_PATH = str(root / "data" / "app.db")
            settings.CHROMA_PERSIST_DIR = str(root / "chroma_db")
            reset_repository_for_tests()
            try:
                store = VectorStore()
                store._embeddings = LocalEmbeddings()
                store.add_documents([Document(
                    page_content="撤销失败后必须继续可查",
                    metadata={
                        "source_file": "回滚.md",
                        "stored_file": "stored_回滚.md",
                        "source_type": "feishu",
                        "status": "active",
                        "content_hash": "rollback-content-hash",
                    },
                )])
                repository = get_repository()
                before = repository.get_document("stored_回滚.md")

                with patch.object(store, "_save", side_effect=RuntimeError("disk full")):
                    with self.assertRaisesRegex(RuntimeError, "disk full"):
                        store.delete_document("stored_回滚.md")

                restored = repository.get_document("stored_回滚.md")
                self.assertIsNotNone(restored)
                for field in ("document_id", "source_id", "asset_id", "version_id", "source_type", "status", "content_hash", "created_at"):
                    self.assertEqual(restored[field], before[field])
                self.assertEqual(len(repository.document_chunks()), 1)
                self.assertEqual(store.count(), 1)
                self.assertIn("撤销失败后必须继续可查", store.similarity_search("继续可查", k=1)[0].page_content)
            finally:
                settings.DATABASE_PATH = original_database
                settings.CHROMA_PERSIST_DIR = original_chroma
                reset_repository_for_tests()

    def test_feishu_revert_transaction_rolls_back_document_and_state_together(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = self._repository(Path(directory))
            workspace = FeishuWorkspace(repository)
            data = workspace.load()
            data["messages"] = [{
                "message_id": "om_atomic", "chat_id": "oc_atomic", "content_status": "published",
                "review_status": "approved", "asset_id": "fa_atomic",
            }]
            data["candidates"] = [{
                "candidate_id": "fc_atomic", "chat_id": "oc_atomic", "status": "published",
                "asset_id": "fa_atomic", "source_message_ids": ["om_atomic"],
            }]
            data["assets"] = [{
                "asset_id": "fa_atomic", "candidate_id": "fc_atomic", "status": "published",
                "source_file": "原子撤销.md", "stored_file": "stored_原子撤销.md",
                "source_message_ids": ["om_atomic"],
            }]
            workspace.save(data)
            repository.upsert_document_chunks([Document(
                page_content="事务失败时仍应可查",
                metadata={"source_file": "原子撤销.md", "stored_file": "stored_原子撤销.md"},
            )])

            with patch.object(repository, "_replace_feishu_connection", side_effect=RuntimeError("commit failed")):
                with self.assertRaisesRegex(RuntimeError, "commit failed"):
                    workspace.revert_asset("fa_atomic")

            self.assertEqual(workspace.get_asset("fa_atomic")["status"], "published")
            self.assertEqual(workspace.get_candidate("fc_atomic")["status"], "published")
            self.assertEqual(workspace.get_message("om_atomic")["content_status"], "published")
            self.assertIsNotNone(repository.get_document("stored_原子撤销.md"))
            self.assertEqual(len(repository.document_chunks()), 1)

    def test_feishu_revert_survives_process_exit_before_projection_rebuild(self):
        original_database = settings.DATABASE_PATH
        original_chroma = settings.CHROMA_PERSIST_DIR
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database_path = root / "data" / "app.db"
            chroma_path = root / "chroma_db"
            script = """
import os
import sys
from langchain_core.documents import Document
from backend.config import settings

settings.DATABASE_PATH = sys.argv[1]
settings.CHROMA_PERSIST_DIR = sys.argv[2]

from backend.storage import get_repository, reset_repository_for_tests
from backend.feishu_workspace import FeishuWorkspace
from backend.knowledge_base.vector_store import LocalEmbeddings, VectorStore

reset_repository_for_tests()
repository = get_repository()
workspace = FeishuWorkspace(repository)
store = VectorStore()
store._embeddings = LocalEmbeddings()
store.add_documents([Document(
    page_content="进程退出后不得继续检索",
    metadata={"source_file": "崩溃撤销.md", "stored_file": "stored_崩溃撤销.md", "source_type": "feishu_conversation"},
)])
data = workspace.load()
data["messages"] = [{
    "message_id": "om_crash", "chat_id": "oc_crash", "content_status": "published",
    "review_status": "approved", "asset_id": "fa_crash",
}]
data["candidates"] = [{
    "candidate_id": "fc_crash", "chat_id": "oc_crash", "status": "published",
    "asset_id": "fa_crash", "source_message_ids": ["om_crash"],
}]
data["assets"] = [{
    "asset_id": "fa_crash", "candidate_id": "fc_crash", "status": "published",
    "source_file": "崩溃撤销.md", "stored_file": "stored_崩溃撤销.md",
    "source_message_ids": ["om_crash"],
}]
workspace.save(data)
workspace.revert_asset("fa_crash")
os._exit(23)
"""
            completed = subprocess.run(
                [sys.executable, "-c", script, str(database_path), str(chroma_path)],
                cwd=Path(__file__).resolve().parents[1],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(completed.returncode, 23, completed.stderr)
            stale_projection = json.loads((chroma_path / "vector_store.json").read_text(encoding="utf-8"))
            self.assertEqual(len(stale_projection["ids"]), 1)

            settings.DATABASE_PATH = str(database_path)
            settings.CHROMA_PERSIST_DIR = str(chroma_path)
            reset_repository_for_tests()
            try:
                repository = get_repository()
                workspace = FeishuWorkspace(repository)
                self.assertEqual(workspace.get_asset("fa_crash")["status"], "reverted")
                self.assertEqual(workspace.get_asset("fa_crash")["projection_status"], "pending_refresh")
                self.assertIsNone(repository.get_document("stored_崩溃撤销.md"))
                self.assertEqual(repository.document_chunks(), [])

                restarted_store = VectorStore()
                restarted_store._embeddings = LocalEmbeddings()
                self.assertEqual(restarted_store.similarity_search("进程退出后不得继续检索", k=1), [])
                self.assertEqual(restarted_store.count(), 0)
            finally:
                settings.DATABASE_PATH = original_database
                settings.CHROMA_PERSIST_DIR = original_chroma
                reset_repository_for_tests()

    def test_retrieval_filters_document_revoked_during_ranking(self):
        original_database = settings.DATABASE_PATH
        original_chroma = settings.CHROMA_PERSIST_DIR
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings.DATABASE_PATH = str(root / "data" / "app.db")
            settings.CHROMA_PERSIST_DIR = str(root / "chroma_db")
            reset_repository_for_tests()
            try:
                store = VectorStore()
                store._embeddings = LocalEmbeddings()
                store.add_documents([Document(
                    page_content="并发撤销后不得进入模型上下文",
                    metadata={"source_file": "并发撤销.md", "stored_file": "stored_并发撤销.md"},
                )])
                original_embed_query = store.embeddings.embed_query

                def revoke_during_ranking(query):
                    get_repository().delete_document("stored_并发撤销.md")
                    return original_embed_query(query)

                with patch.object(store.embeddings, "embed_query", side_effect=revoke_during_ranking):
                    self.assertEqual(store.similarity_search("并发撤销", k=1), [])
            finally:
                settings.DATABASE_PATH = original_database
                settings.CHROMA_PERSIST_DIR = original_chroma
                reset_repository_for_tests()


if __name__ == "__main__":
    unittest.main()
