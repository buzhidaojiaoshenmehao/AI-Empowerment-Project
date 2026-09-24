"""SQLite connection, schema migration, integrity and backup primitives."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional


DEFAULT_PROJECT_ID = "project_default"
SYSTEM_USER_ID = "user_system"
DEFAULT_ORGANIZATION_ID = "org_default"


class StorageProcessLock:
    """Cross-process exclusive lock used by the app runtime and offline restore."""

    def __init__(self, path: Path, purpose: str) -> None:
        self.path = path
        self.purpose = purpose
        self._handle = None

    def acquire(self) -> "StorageProcessLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            handle.close()
            raise RuntimeError(
                "数据库正被运行中的服务占用；请先停止服务，再执行恢复或启动其他实例"
            ) from exc
        self._handle = handle
        return self

    def release(self) -> None:
        handle, self._handle = self._handle, None
        if handle is None:
            return
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

    def __enter__(self) -> "StorageProcessLock":
        return self.acquire()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


SCHEMA_MIGRATIONS = (
    (
        1,
        "initial_storage",
        """
        CREATE TABLE IF NOT EXISTS projects (
            project_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            slug TEXT NOT NULL UNIQUE,
            is_default INTEGER NOT NULL DEFAULT 0 CHECK (is_default IN (0, 1)),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            user_type TEXT NOT NULL DEFAULT 'placeholder',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS project_memberships (
            project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
            role TEXT NOT NULL DEFAULT 'system_placeholder',
            created_at TEXT NOT NULL,
            PRIMARY KEY (project_id, user_id)
        );
        CREATE TABLE IF NOT EXISTS categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            sort_order INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE (project_id, name)
        );
        CREATE TABLE IF NOT EXISTS documents (
            document_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            source_file TEXT NOT NULL,
            stored_file TEXT NOT NULL,
            primary_category TEXT NOT NULL DEFAULT '未分类',
            uploader TEXT NOT NULL DEFAULT '',
            role TEXT NOT NULL DEFAULT '',
            handover_id TEXT NOT NULL DEFAULT '',
            source_type TEXT NOT NULL DEFAULT 'document',
            status TEXT NOT NULL DEFAULT 'active',
            content_hash TEXT NOT NULL DEFAULT '',
            upload_date TEXT NOT NULL DEFAULT '',
            registered_at TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE (project_id, stored_file)
        );
        CREATE INDEX IF NOT EXISTS idx_documents_project_source
            ON documents(project_id, source_file);
        CREATE INDEX IF NOT EXISTS idx_documents_handover
            ON documents(handover_id);
        CREATE TABLE IF NOT EXISTS document_categories (
            document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
            category_id INTEGER NOT NULL REFERENCES categories(category_id) ON DELETE RESTRICT,
            is_primary INTEGER NOT NULL DEFAULT 0 CHECK (is_primary IN (0, 1)),
            PRIMARY KEY (document_id, category_id)
        );
        CREATE TABLE IF NOT EXISTS document_chunks (
            chunk_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            vector_id TEXT NOT NULL DEFAULT '',
            text_content TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            UNIQUE (document_id, chunk_index)
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_document ON document_chunks(document_id, chunk_index);
        CREATE TABLE IF NOT EXISTS handover_cases (
            handover_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT '',
            recipient TEXT NOT NULL DEFAULT '',
            due_date TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'pending_acceptance',
            accepted_by TEXT NOT NULL DEFAULT '',
            accepted_at TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            data_json TEXT NOT NULL DEFAULT '{}'
        );
        CREATE TABLE IF NOT EXISTS handover_files (
            handover_id TEXT NOT NULL REFERENCES handover_cases(handover_id) ON DELETE CASCADE,
            document_id TEXT REFERENCES documents(document_id) ON DELETE SET NULL,
            source_file TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (handover_id, source_file)
        );
        CREATE TABLE IF NOT EXISTS feishu_groups (
            chat_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            name TEXT NOT NULL DEFAULT '',
            collection_mode TEXT NOT NULL DEFAULT 'review',
            updated_at TEXT NOT NULL,
            data_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS feishu_messages (
            message_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            chat_id TEXT NOT NULL,
            message_type TEXT NOT NULL DEFAULT 'text',
            content_status TEXT NOT NULL DEFAULT 'archived',
            candidate_id TEXT NOT NULL DEFAULT '',
            asset_id TEXT NOT NULL DEFAULT '',
            content_hash TEXT NOT NULL DEFAULT '',
            create_time TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL,
            data_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_feishu_messages_chat_time
            ON feishu_messages(chat_id, create_time DESC);
        CREATE INDEX IF NOT EXISTS idx_feishu_messages_status
            ON feishu_messages(content_status);
        CREATE TABLE IF NOT EXISTS feishu_candidates (
            candidate_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            chat_id TEXT NOT NULL DEFAULT '',
            aggregation_key TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'review_required',
            category TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL,
            data_json TEXT NOT NULL
        );
        CREATE UNIQUE INDEX IF NOT EXISTS uq_feishu_candidate_aggregation
            ON feishu_candidates(project_id, aggregation_key) WHERE aggregation_key <> '';
        CREATE TABLE IF NOT EXISTS feishu_candidate_messages (
            candidate_id TEXT NOT NULL REFERENCES feishu_candidates(candidate_id) ON DELETE CASCADE,
            message_id TEXT NOT NULL REFERENCES feishu_messages(message_id) ON DELETE CASCADE,
            PRIMARY KEY (candidate_id, message_id)
        );
        CREATE TABLE IF NOT EXISTS feishu_assets (
            asset_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            candidate_id TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'published',
            stored_file TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL,
            data_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS feishu_asset_messages (
            asset_id TEXT NOT NULL REFERENCES feishu_assets(asset_id) ON DELETE CASCADE,
            message_id TEXT NOT NULL REFERENCES feishu_messages(message_id) ON DELETE CASCADE,
            PRIMARY KEY (asset_id, message_id)
        );
        CREATE TABLE IF NOT EXISTS audit_events (
            audit_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            actor TEXT NOT NULL DEFAULT 'system',
            action TEXT NOT NULL,
            object_type TEXT NOT NULL DEFAULT '',
            object_id TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            data_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_events(created_at DESC);
        CREATE TABLE IF NOT EXISTS feishu_diagnostics (
            project_id TEXT PRIMARY KEY REFERENCES projects(project_id) ON DELETE CASCADE,
            data_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS graph_nodes (
            project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            node_id TEXT NOT NULL,
            origin TEXT NOT NULL DEFAULT 'manual',
            source TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL,
            data_json TEXT NOT NULL,
            PRIMARY KEY (project_id, node_id)
        );
        CREATE TABLE IF NOT EXISTS graph_edges (
            project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            edge_id TEXT NOT NULL,
            source_node_id TEXT NOT NULL,
            target_node_id TEXT NOT NULL,
            origin TEXT NOT NULL DEFAULT 'manual',
            updated_at TEXT NOT NULL,
            data_json TEXT NOT NULL,
            PRIMARY KEY (project_id, edge_id)
        );
        CREATE TABLE IF NOT EXISTS system_state (
            state_key TEXT PRIMARY KEY,
            value_text TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS migration_runs (
            migration_key TEXT PRIMARY KEY,
            source_hash TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL,
            report_path TEXT NOT NULL DEFAULT '',
            detail TEXT NOT NULL DEFAULT '',
            started_at TEXT NOT NULL,
            completed_at TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS storage_operations (
            operation_id TEXT PRIMARY KEY,
            operation_type TEXT NOT NULL,
            target_path TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            attempts INTEGER NOT NULL DEFAULT 0,
            last_error TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """,
    ),
    (
        2,
        "knowledge_identity_and_access_foundation",
        """
        ALTER TABLE documents ADD COLUMN organization_id TEXT NOT NULL DEFAULT 'org_default';
        ALTER TABLE documents ADD COLUMN source_id TEXT NOT NULL DEFAULT '';
        ALTER TABLE documents ADD COLUMN asset_id TEXT NOT NULL DEFAULT '';
        ALTER TABLE documents ADD COLUMN version_id TEXT NOT NULL DEFAULT '';
        ALTER TABLE documents ADD COLUMN supersedes_version_id TEXT NOT NULL DEFAULT '';
        ALTER TABLE documents ADD COLUMN visibility TEXT NOT NULL DEFAULT 'project';
        ALTER TABLE documents ADD COLUMN sensitivity_level TEXT NOT NULL DEFAULT 'internal';
        ALTER TABLE documents ADD COLUMN access_policy_id TEXT NOT NULL DEFAULT '';
        ALTER TABLE documents ADD COLUMN acl_revision INTEGER NOT NULL DEFAULT 0;
        ALTER TABLE documents ADD COLUMN valid_from TEXT NOT NULL DEFAULT '';
        ALTER TABLE documents ADD COLUMN valid_until TEXT NOT NULL DEFAULT '';
        CREATE INDEX IF NOT EXISTS idx_documents_source_version
            ON documents(project_id, source_id, version_id);
        CREATE INDEX IF NOT EXISTS idx_documents_access_scope
            ON documents(organization_id, project_id, visibility, status);

        ALTER TABLE feishu_messages ADD COLUMN provider_account_id TEXT NOT NULL DEFAULT 'feishu_default';
        ALTER TABLE feishu_messages ADD COLUMN event_id TEXT NOT NULL DEFAULT '';
        CREATE UNIQUE INDEX IF NOT EXISTS uq_feishu_provider_event
            ON feishu_messages(project_id, provider_account_id, event_id)
            WHERE event_id <> '';
        """,
    ),
    (
        3,
        "local_iam_and_source_access",
        """
        CREATE TABLE IF NOT EXISTS organizations (
            organization_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            slug TEXT NOT NULL UNIQUE,
            status TEXT NOT NULL DEFAULT 'active',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        ALTER TABLE projects ADD COLUMN organization_id TEXT NOT NULL DEFAULT 'org_default';
        ALTER TABLE projects ADD COLUMN status TEXT NOT NULL DEFAULT 'active';

        ALTER TABLE users ADD COLUMN organization_id TEXT NOT NULL DEFAULT 'org_default';
        ALTER TABLE users ADD COLUMN email TEXT NOT NULL DEFAULT '';
        ALTER TABLE users ADD COLUMN email_normalized TEXT NOT NULL DEFAULT '';
        ALTER TABLE users ADD COLUMN status TEXT NOT NULL DEFAULT 'invited';
        ALTER TABLE users ADD COLUMN password_hash TEXT NOT NULL DEFAULT '';
        ALTER TABLE users ADD COLUMN password_changed_at TEXT NOT NULL DEFAULT '';
        ALTER TABLE users ADD COLUMN failed_login_count INTEGER NOT NULL DEFAULT 0;
        ALTER TABLE users ADD COLUMN locked_until TEXT NOT NULL DEFAULT '';
        ALTER TABLE users ADD COLUMN last_login_at TEXT NOT NULL DEFAULT '';
        CREATE UNIQUE INDEX IF NOT EXISTS uq_users_email_normalized
            ON users(email_normalized) WHERE email_normalized <> '';
        CREATE INDEX IF NOT EXISTS idx_users_organization_status
            ON users(organization_id, status);

        ALTER TABLE project_memberships ADD COLUMN status TEXT NOT NULL DEFAULT 'active';
        ALTER TABLE project_memberships ADD COLUMN granted_by TEXT NOT NULL DEFAULT 'user_system';
        ALTER TABLE project_memberships ADD COLUMN updated_at TEXT NOT NULL DEFAULT '';
        CREATE INDEX IF NOT EXISTS idx_memberships_user_status
            ON project_memberships(user_id, status);

        CREATE TABLE IF NOT EXISTS user_invitations (
            invitation_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
            project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            token_hash TEXT NOT NULL UNIQUE,
            expires_at TEXT NOT NULL,
            consumed_at TEXT NOT NULL DEFAULT '',
            created_by TEXT NOT NULL DEFAULT 'user_system',
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_invitations_user_expiry
            ON user_invitations(user_id, expires_at);

        CREATE TABLE IF NOT EXISTS auth_sessions (
            session_id TEXT PRIMARY KEY,
            token_hash TEXT NOT NULL UNIQUE,
            csrf_hash TEXT NOT NULL,
            user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
            current_project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            expires_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            created_at TEXT NOT NULL,
            user_agent_hash TEXT NOT NULL DEFAULT '',
            remote_address_hash TEXT NOT NULL DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_auth_sessions_user_expiry
            ON auth_sessions(user_id, expires_at);

        CREATE TABLE IF NOT EXISTS access_policies (
            access_policy_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            policy_type TEXT NOT NULL DEFAULT 'restricted',
            status TEXT NOT NULL DEFAULT 'active',
            created_by TEXT NOT NULL DEFAULT 'user_system',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(project_id, name)
        );
        CREATE TABLE IF NOT EXISTS access_policy_users (
            access_policy_id TEXT NOT NULL REFERENCES access_policies(access_policy_id) ON DELETE CASCADE,
            user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
            created_at TEXT NOT NULL,
            PRIMARY KEY(access_policy_id, user_id)
        );
        CREATE TABLE IF NOT EXISTS access_policy_roles (
            access_policy_id TEXT NOT NULL REFERENCES access_policies(access_policy_id) ON DELETE CASCADE,
            role TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY(access_policy_id, role)
        );

        ALTER TABLE documents ADD COLUMN owner_user_id TEXT NOT NULL DEFAULT '';
        ALTER TABLE feishu_groups ADD COLUMN visibility TEXT NOT NULL DEFAULT 'project';
        ALTER TABLE feishu_groups ADD COLUMN access_policy_id TEXT NOT NULL DEFAULT '';
        CREATE INDEX IF NOT EXISTS idx_documents_owner_access
            ON documents(project_id, owner_user_id, visibility, access_policy_id, status);
        """,
    ),
    (
        4,
        "unified_knowledge_assets",
        """
        CREATE TABLE IF NOT EXISTS knowledge_sources (
            source_id TEXT PRIMARY KEY,
            organization_id TEXT NOT NULL REFERENCES organizations(organization_id) ON DELETE RESTRICT,
            project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            source_type TEXT NOT NULL,
            external_key TEXT NOT NULL,
            display_name TEXT NOT NULL DEFAULT '',
            owner_user_id TEXT NOT NULL DEFAULT '',
            visibility TEXT NOT NULL DEFAULT 'project'
                CHECK (visibility IN ('project','organization','restricted','private')),
            sensitivity_level TEXT NOT NULL DEFAULT 'internal',
            access_policy_id TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'active'
                CHECK (status IN ('active','revoked','deleted')),
            retention_policy TEXT NOT NULL DEFAULT 'project_default',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(project_id, source_type, external_key)
        );
        CREATE INDEX IF NOT EXISTS idx_knowledge_sources_access
            ON knowledge_sources(project_id,status,visibility,owner_user_id,access_policy_id);

        CREATE TABLE IF NOT EXISTS knowledge_assets (
            asset_id TEXT PRIMARY KEY,
            organization_id TEXT NOT NULL REFERENCES organizations(organization_id) ON DELETE RESTRICT,
            project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            primary_source_id TEXT NOT NULL REFERENCES knowledge_sources(source_id) ON DELETE RESTRICT,
            asset_key TEXT NOT NULL,
            title TEXT NOT NULL,
            topic TEXT NOT NULL DEFAULT '',
            summary TEXT NOT NULL DEFAULT '',
            owner_user_id TEXT NOT NULL DEFAULT '',
            authority_level INTEGER NOT NULL DEFAULT 50 CHECK (authority_level BETWEEN 0 AND 100),
            sensitivity_level TEXT NOT NULL DEFAULT 'internal',
            visibility TEXT NOT NULL DEFAULT 'project'
                CHECK (visibility IN ('project','organization','restricted','private')),
            access_policy_id TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'draft'
                CHECK (status IN ('draft','pending_review','active','review_due','expired','revoked','deleted')),
            current_version_id TEXT NOT NULL DEFAULT '',
            review_due_at TEXT NOT NULL DEFAULT '',
            published_at TEXT NOT NULL DEFAULT '',
            revoked_at TEXT NOT NULL DEFAULT '',
            lifecycle_revision INTEGER NOT NULL DEFAULT 0,
            created_by TEXT NOT NULL DEFAULT 'user_system',
            updated_by TEXT NOT NULL DEFAULT 'user_system',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            UNIQUE(project_id, asset_key)
        );
        CREATE INDEX IF NOT EXISTS idx_knowledge_assets_current
            ON knowledge_assets(project_id,status,current_version_id,review_due_at);
        CREATE INDEX IF NOT EXISTS idx_knowledge_assets_access
            ON knowledge_assets(project_id,visibility,owner_user_id,access_policy_id,status);

        CREATE TABLE IF NOT EXISTS asset_versions (
            version_id TEXT PRIMARY KEY,
            asset_id TEXT NOT NULL REFERENCES knowledge_assets(asset_id) ON DELETE CASCADE,
            project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            version_no INTEGER NOT NULL CHECK (version_no > 0),
            status TEXT NOT NULL DEFAULT 'preparing'
                CHECK (status IN ('preparing','ready','active','superseded','failed','revoked')),
            content_hash TEXT NOT NULL DEFAULT '',
            summary TEXT NOT NULL DEFAULT '',
            supersedes_version_id TEXT NOT NULL DEFAULT '',
            valid_from TEXT NOT NULL DEFAULT '',
            valid_until TEXT NOT NULL DEFAULT '',
            review_due_at TEXT NOT NULL DEFAULT '',
            created_by TEXT NOT NULL DEFAULT 'user_system',
            published_by TEXT NOT NULL DEFAULT '',
            published_at TEXT NOT NULL DEFAULT '',
            failure_reason TEXT NOT NULL DEFAULT '',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(asset_id, version_no)
        );
        CREATE UNIQUE INDEX IF NOT EXISTS uq_asset_one_active_version
            ON asset_versions(asset_id) WHERE status='active';
        CREATE INDEX IF NOT EXISTS idx_asset_versions_project_status
            ON asset_versions(project_id,status,asset_id);
        CREATE INDEX IF NOT EXISTS idx_asset_versions_content
            ON asset_versions(asset_id,content_hash);

        CREATE TABLE IF NOT EXISTS asset_version_documents (
            version_id TEXT NOT NULL REFERENCES asset_versions(version_id) ON DELETE CASCADE,
            document_id TEXT NOT NULL UNIQUE REFERENCES documents(document_id) ON DELETE CASCADE,
            document_role TEXT NOT NULL DEFAULT 'authoritative',
            created_at TEXT NOT NULL,
            PRIMARY KEY(version_id, document_id)
        );
        CREATE INDEX IF NOT EXISTS idx_asset_version_documents_document
            ON asset_version_documents(document_id,version_id);

        CREATE TABLE IF NOT EXISTS asset_categories (
            asset_id TEXT NOT NULL REFERENCES knowledge_assets(asset_id) ON DELETE CASCADE,
            category_id INTEGER NOT NULL REFERENCES categories(category_id) ON DELETE RESTRICT,
            is_primary INTEGER NOT NULL DEFAULT 0 CHECK (is_primary IN (0,1)),
            created_at TEXT NOT NULL,
            PRIMARY KEY(asset_id, category_id)
        );

        CREATE TABLE IF NOT EXISTS asset_applicable_roles (
            asset_id TEXT NOT NULL REFERENCES knowledge_assets(asset_id) ON DELETE CASCADE,
            role_key TEXT NOT NULL,
            requirement_level TEXT NOT NULL DEFAULT 'recommended'
                CHECK (requirement_level IN ('required','recommended','optional')),
            weight INTEGER NOT NULL DEFAULT 1 CHECK (weight BETWEEN 0 AND 100),
            created_at TEXT NOT NULL,
            PRIMARY KEY(asset_id, role_key)
        );

        CREATE TABLE IF NOT EXISTS asset_projection_states (
            version_id TEXT NOT NULL REFERENCES asset_versions(version_id) ON DELETE CASCADE,
            projection_type TEXT NOT NULL
                CHECK (projection_type IN ('vector','graph','context','readiness')),
            status TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending','building','ready','repair_required','obsolete')),
            desired_revision INTEGER NOT NULL DEFAULT 1,
            applied_revision INTEGER NOT NULL DEFAULT 0,
            attempts INTEGER NOT NULL DEFAULT 0,
            last_error TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL,
            PRIMARY KEY(version_id, projection_type)
        );
        CREATE INDEX IF NOT EXISTS idx_asset_projection_repair
            ON asset_projection_states(status,projection_type,updated_at);

        CREATE TABLE IF NOT EXISTS domain_outbox (
            event_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            aggregate_type TEXT NOT NULL,
            aggregate_id TEXT NOT NULL,
            aggregate_revision INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            payload_json TEXT NOT NULL DEFAULT '{}',
            status TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending','processing','done','failed','dead')),
            attempts INTEGER NOT NULL DEFAULT 0,
            available_at TEXT NOT NULL,
            locked_at TEXT NOT NULL DEFAULT '',
            processed_at TEXT NOT NULL DEFAULT '',
            last_error TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(aggregate_type,aggregate_id,aggregate_revision,event_type)
        );
        CREATE INDEX IF NOT EXISTS idx_domain_outbox_pending
            ON domain_outbox(status,available_at,created_at);

        CREATE TABLE IF NOT EXISTS knowledge_asset_aliases (
            project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            alias_type TEXT NOT NULL,
            alias_value TEXT NOT NULL,
            asset_id TEXT NOT NULL REFERENCES knowledge_assets(asset_id) ON DELETE CASCADE,
            created_at TEXT NOT NULL,
            PRIMARY KEY(project_id,alias_type,alias_value)
        );
        CREATE INDEX IF NOT EXISTS idx_knowledge_asset_alias_asset
            ON knowledge_asset_aliases(asset_id);
        """,
    ),
    (
        5,
        "external_identity_mapping",
        """
        CREATE TABLE IF NOT EXISTS user_external_identities (
            provider TEXT NOT NULL,
            external_user_id TEXT NOT NULL,
            user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
            organization_id TEXT NOT NULL REFERENCES organizations(organization_id) ON DELETE CASCADE,
            verified_email TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY(provider, external_user_id),
            UNIQUE(provider, user_id)
        );
        CREATE INDEX IF NOT EXISTS idx_external_identity_user
            ON user_external_identities(user_id, provider);
        """,
    ),
    (
        6,
        "knowledge_task_workflow",
        """
        CREATE TABLE IF NOT EXISTS knowledge_tasks (
            task_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            task_type TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            source_type TEXT NOT NULL DEFAULT 'manual',
            source_key TEXT NOT NULL DEFAULT '',
            dedupe_key TEXT NOT NULL DEFAULT '',
            linked_asset_id TEXT NOT NULL DEFAULT '',
            role_key TEXT NOT NULL DEFAULT '',
            handover_id TEXT NOT NULL DEFAULT '',
            priority TEXT NOT NULL DEFAULT 'medium'
                CHECK (priority IN ('low','medium','high','critical')),
            assignee_user_id TEXT NOT NULL DEFAULT '',
            reporter_user_id TEXT NOT NULL DEFAULT '',
            due_at TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'unassigned'
                CHECK (status IN ('unassigned','open','in_progress','pending_acceptance','completed','cancelled')),
            completion_evidence TEXT NOT NULL DEFAULT '',
            occurrence_count INTEGER NOT NULL DEFAULT 1 CHECK (occurrence_count > 0),
            last_occurrence_at TEXT NOT NULL,
            completed_at TEXT NOT NULL DEFAULT '',
            accepted_at TEXT NOT NULL DEFAULT '',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE UNIQUE INDEX IF NOT EXISTS uq_open_knowledge_task_dedupe
            ON knowledge_tasks(project_id,dedupe_key)
            WHERE dedupe_key <> '' AND status IN ('unassigned','open','in_progress','pending_acceptance');
        CREATE INDEX IF NOT EXISTS idx_knowledge_tasks_project_status
            ON knowledge_tasks(project_id,status,priority,due_at,updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_knowledge_tasks_assignee
            ON knowledge_tasks(project_id,assignee_user_id,status,updated_at DESC);

        CREATE TABLE IF NOT EXISTS knowledge_task_events (
            event_id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL REFERENCES knowledge_tasks(task_id) ON DELETE CASCADE,
            project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            actor_user_id TEXT NOT NULL DEFAULT 'user_system',
            event_type TEXT NOT NULL,
            from_status TEXT NOT NULL DEFAULT '',
            to_status TEXT NOT NULL DEFAULT '',
            note TEXT NOT NULL DEFAULT '',
            data_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_knowledge_task_events_task
            ON knowledge_task_events(task_id,created_at,event_id);

        CREATE TABLE IF NOT EXISTS task_notifications (
            notification_id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL REFERENCES knowledge_tasks(task_id) ON DELETE CASCADE,
            project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            channel TEXT NOT NULL DEFAULT 'feishu',
            target_id TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending','sent','failed','skipped')),
            error TEXT NOT NULL DEFAULT '',
            attempted_at TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_task_notifications_task
            ON task_notifications(task_id,attempted_at DESC);
        """,
    ),
    (
        7,
        "persistent_processing_jobs",
        """
        CREATE TABLE IF NOT EXISTS processing_jobs (
            job_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            job_type TEXT NOT NULL,
            idempotency_key TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'queued'
                CHECK (status IN ('queued','running','retry_wait','succeeded','failed','cancelled')),
            stage TEXT NOT NULL DEFAULT 'queued',
            progress INTEGER NOT NULL DEFAULT 0 CHECK (progress BETWEEN 0 AND 100),
            priority INTEGER NOT NULL DEFAULT 50 CHECK (priority BETWEEN 0 AND 100),
            payload_json TEXT NOT NULL DEFAULT '{}',
            result_json TEXT NOT NULL DEFAULT '{}',
            attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
            max_attempts INTEGER NOT NULL DEFAULT 3 CHECK (max_attempts BETWEEN 1 AND 20),
            available_at TEXT NOT NULL,
            locked_at TEXT NOT NULL DEFAULT '',
            locked_by TEXT NOT NULL DEFAULT '',
            heartbeat_at TEXT NOT NULL DEFAULT '',
            cancel_requested INTEGER NOT NULL DEFAULT 0 CHECK (cancel_requested IN (0,1)),
            created_by TEXT NOT NULL DEFAULT 'user_system',
            linked_asset_id TEXT NOT NULL DEFAULT '',
            source_id TEXT NOT NULL DEFAULT '',
            error_code TEXT NOT NULL DEFAULT '',
            last_error TEXT NOT NULL DEFAULT '',
            started_at TEXT NOT NULL DEFAULT '',
            completed_at TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE UNIQUE INDEX IF NOT EXISTS uq_open_processing_job_idempotency
            ON processing_jobs(project_id,idempotency_key)
            WHERE idempotency_key <> '' AND status IN ('queued','running','retry_wait');
        CREATE INDEX IF NOT EXISTS idx_processing_jobs_claim
            ON processing_jobs(project_id,status,available_at,priority DESC,created_at);
        CREATE INDEX IF NOT EXISTS idx_processing_jobs_creator
            ON processing_jobs(project_id,created_by,updated_at DESC);

        CREATE TABLE IF NOT EXISTS processing_job_attempts (
            attempt_id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL REFERENCES processing_jobs(job_id) ON DELETE CASCADE,
            project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            attempt_number INTEGER NOT NULL,
            worker_id TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'running'
                CHECK (status IN ('running','succeeded','retry_wait','failed','cancelled','interrupted')),
            error_code TEXT NOT NULL DEFAULT '',
            error_message TEXT NOT NULL DEFAULT '',
            started_at TEXT NOT NULL,
            completed_at TEXT NOT NULL DEFAULT '',
            UNIQUE(job_id,attempt_number)
        );
        CREATE INDEX IF NOT EXISTS idx_processing_job_attempts_job
            ON processing_job_attempts(job_id,attempt_number DESC);

        CREATE TABLE IF NOT EXISTS processing_job_events (
            event_id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL REFERENCES processing_jobs(job_id) ON DELETE CASCADE,
            project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            event_type TEXT NOT NULL,
            from_status TEXT NOT NULL DEFAULT '',
            to_status TEXT NOT NULL DEFAULT '',
            stage TEXT NOT NULL DEFAULT '',
            progress INTEGER NOT NULL DEFAULT 0,
            message TEXT NOT NULL DEFAULT '',
            data_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_processing_job_events_job
            ON processing_job_events(job_id,created_at,event_id);
        """,
    ),
    (
        8,
        "project_data_governance",
        """
        CREATE TABLE IF NOT EXISTS data_governance_policies (
            project_id TEXT PRIMARY KEY REFERENCES projects(project_id) ON DELETE CASCADE,
            raw_message_retention_days INTEGER NOT NULL DEFAULT 180
                CHECK (raw_message_retention_days BETWEEN 7 AND 3650),
            access_audit_retention_days INTEGER NOT NULL DEFAULT 365
                CHECK (access_audit_retention_days BETWEEN 30 AND 3650),
            processing_history_retention_days INTEGER NOT NULL DEFAULT 90
                CHECK (processing_history_retention_days BETWEEN 7 AND 3650),
            backup_retention_count INTEGER NOT NULL DEFAULT 10
                CHECK (backup_retention_count BETWEEN 1 AND 100),
            default_sensitivity TEXT NOT NULL DEFAULT 'internal'
                CHECK (default_sensitivity IN ('public','internal','confidential','restricted')),
            updated_by TEXT NOT NULL DEFAULT 'user_system',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS data_retention_runs (
            run_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            status TEXT NOT NULL CHECK (status IN ('succeeded','failed')),
            actor_user_id TEXT NOT NULL DEFAULT 'user_system',
            policy_json TEXT NOT NULL DEFAULT '{}',
            result_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_retention_runs_project
            ON data_retention_runs(project_id,created_at DESC);
        """,
    ),
    (
        9,
        "governance_default_retention_alignment",
        """
        UPDATE data_governance_policies
        SET raw_message_retention_days=90,updated_at=strftime('%Y-%m-%dT%H:%M:%SZ','now')
        WHERE raw_message_retention_days=180 AND updated_by='user_system';
        """,
    ),
    (
        10,
        "runtime_observability",
        """
        CREATE TABLE IF NOT EXISTS runtime_metric_buckets (
            project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            metric_key TEXT NOT NULL,
            route_key TEXT NOT NULL DEFAULT '',
            bucket_start TEXT NOT NULL,
            sample_count INTEGER NOT NULL DEFAULT 0,
            success_count INTEGER NOT NULL DEFAULT 0,
            error_count INTEGER NOT NULL DEFAULT 0,
            total_duration_ms REAL NOT NULL DEFAULT 0,
            max_duration_ms REAL NOT NULL DEFAULT 0,
            histogram_json TEXT NOT NULL DEFAULT '{}',
            updated_at TEXT NOT NULL,
            PRIMARY KEY (project_id, metric_key, route_key, bucket_start)
        );
        CREATE INDEX IF NOT EXISTS idx_runtime_metrics_window
            ON runtime_metric_buckets(project_id,metric_key,bucket_start DESC);

        CREATE TABLE IF NOT EXISTS runtime_alerts (
            alert_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            rule_key TEXT NOT NULL,
            severity TEXT NOT NULL CHECK (severity IN ('info','warning','critical')),
            status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open','resolved')),
            title TEXT NOT NULL,
            summary TEXT NOT NULL DEFAULT '',
            evidence_json TEXT NOT NULL DEFAULT '{}',
            occurrence_count INTEGER NOT NULL DEFAULT 1,
            first_seen_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            resolved_at TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL
        );
        CREATE UNIQUE INDEX IF NOT EXISTS uq_runtime_open_alert
            ON runtime_alerts(project_id,rule_key) WHERE status='open';
        CREATE INDEX IF NOT EXISTS idx_runtime_alerts_project
            ON runtime_alerts(project_id,status,severity,last_seen_at DESC);

        CREATE TABLE IF NOT EXISTS runtime_capacity_runs (
            run_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            status TEXT NOT NULL CHECK (status IN ('passed','failed')),
            message_count INTEGER NOT NULL DEFAULT 0,
            p50_ms REAL NOT NULL DEFAULT 0,
            p95_ms REAL NOT NULL DEFAULT 0,
            max_ms REAL NOT NULL DEFAULT 0,
            report_json TEXT NOT NULL DEFAULT '{}',
            actor_user_id TEXT NOT NULL DEFAULT 'user_system',
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_runtime_capacity_project
            ON runtime_capacity_runs(project_id,created_at DESC);
        """,
    ),
    (
        11,
        "onboarding_role_templates_and_learning_plans",
        """
        CREATE TABLE IF NOT EXISTS onboarding_role_templates (
            template_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            role_key TEXT NOT NULL,
            name TEXT NOT NULL,
            focus TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'active'
                CHECK (status IN ('active','inactive')),
            revision INTEGER NOT NULL DEFAULT 1,
            created_by TEXT NOT NULL DEFAULT 'user_system',
            updated_by TEXT NOT NULL DEFAULT 'user_system',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE (project_id, role_key)
        );
        CREATE INDEX IF NOT EXISTS idx_onboarding_templates_project
            ON onboarding_role_templates(project_id,status,name);

        CREATE TABLE IF NOT EXISTS onboarding_template_topics (
            topic_id TEXT PRIMARY KEY,
            template_id TEXT NOT NULL REFERENCES onboarding_role_templates(template_id) ON DELETE CASCADE,
            topic_key TEXT NOT NULL,
            label TEXT NOT NULL,
            weight REAL NOT NULL DEFAULT 1 CHECK (weight > 0 AND weight <= 100),
            required_asset_ids_json TEXT NOT NULL DEFAULT '[]',
            practice_task TEXT NOT NULL DEFAULT '',
            owner_user_id TEXT NOT NULL DEFAULT '',
            completion_standard TEXT NOT NULL DEFAULT '',
            position INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE (template_id, topic_key)
        );
        CREATE INDEX IF NOT EXISTS idx_onboarding_topics_template
            ON onboarding_template_topics(template_id,position,topic_key);

        CREATE TABLE IF NOT EXISTS onboarding_learning_plans (
            plan_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
            template_id TEXT NOT NULL REFERENCES onboarding_role_templates(template_id),
            template_revision INTEGER NOT NULL DEFAULT 1,
            role_name TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active'
                CHECK (status IN ('active','completed','cancelled')),
            target_date TEXT NOT NULL DEFAULT '',
            manager_user_id TEXT NOT NULL DEFAULT '',
            source_type TEXT NOT NULL DEFAULT 'manual',
            source_key TEXT NOT NULL DEFAULT '',
            created_by TEXT NOT NULL DEFAULT 'user_system',
            started_at TEXT NOT NULL,
            completed_at TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE UNIQUE INDEX IF NOT EXISTS uq_onboarding_active_plan
            ON onboarding_learning_plans(project_id,user_id,template_id)
            WHERE status='active';
        CREATE INDEX IF NOT EXISTS idx_onboarding_plans_user
            ON onboarding_learning_plans(project_id,user_id,status,updated_at DESC);

        CREATE TABLE IF NOT EXISTS onboarding_learning_items (
            item_id TEXT PRIMARY KEY,
            plan_id TEXT NOT NULL REFERENCES onboarding_learning_plans(plan_id) ON DELETE CASCADE,
            topic_key TEXT NOT NULL DEFAULT '',
            item_type TEXT NOT NULL
                CHECK (item_type IN ('reading','practice','manager_confirmation')),
            title TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            asset_id TEXT NOT NULL DEFAULT '',
            version_id TEXT NOT NULL DEFAULT '',
            weight REAL NOT NULL DEFAULT 1 CHECK (weight > 0),
            status TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending','completed','blocked')),
            evidence TEXT NOT NULL DEFAULT '',
            completed_by TEXT NOT NULL DEFAULT '',
            completed_at TEXT NOT NULL DEFAULT '',
            position INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_onboarding_items_plan
            ON onboarding_learning_items(plan_id,item_type,position,item_id);
        """,
    ),
    (
        12,
        "handover_items_risks_and_immutable_snapshots",
        """
        CREATE TABLE IF NOT EXISTS handover_items (
            item_id TEXT PRIMARY KEY,
            handover_id TEXT NOT NULL REFERENCES handover_cases(handover_id) ON DELETE CASCADE,
            project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            item_key TEXT NOT NULL,
            item_type TEXT NOT NULL DEFAULT 'knowledge',
            title TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            required INTEGER NOT NULL DEFAULT 1 CHECK (required IN (0,1)),
            owner_user_id TEXT NOT NULL DEFAULT '',
            recipient_user_id TEXT NOT NULL DEFAULT '',
            due_at TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending','submitted','accepted','rejected')),
            validity_status TEXT NOT NULL DEFAULT 'unknown'
                CHECK (validity_status IN ('unknown','valid','invalid','expired')),
            evidence TEXT NOT NULL DEFAULT '',
            asset_id TEXT NOT NULL DEFAULT '',
            version_id TEXT NOT NULL DEFAULT '',
            rejection_reason TEXT NOT NULL DEFAULT '',
            knowledge_task_id TEXT NOT NULL DEFAULT '',
            submitted_by TEXT NOT NULL DEFAULT '',
            submitted_at TEXT NOT NULL DEFAULT '',
            reviewed_by TEXT NOT NULL DEFAULT '',
            reviewed_at TEXT NOT NULL DEFAULT '',
            revision INTEGER NOT NULL DEFAULT 1 CHECK (revision > 0),
            position INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE (handover_id,item_key)
        );
        CREATE INDEX IF NOT EXISTS idx_handover_items_case
            ON handover_items(handover_id,required,status,position,item_id);

        CREATE TABLE IF NOT EXISTS handover_risks (
            risk_id TEXT PRIMARY KEY,
            handover_id TEXT NOT NULL REFERENCES handover_cases(handover_id) ON DELETE CASCADE,
            project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            title TEXT NOT NULL,
            impact TEXT NOT NULL DEFAULT '',
            severity TEXT NOT NULL DEFAULT 'medium'
                CHECK (severity IN ('low','medium','high','blocking')),
            owner_user_id TEXT NOT NULL DEFAULT '',
            due_at TEXT NOT NULL DEFAULT '',
            mitigation TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'open'
                CHECK (status IN ('open','mitigating','pending_close','closed')),
            close_evidence TEXT NOT NULL DEFAULT '',
            submitted_by TEXT NOT NULL DEFAULT '',
            submitted_at TEXT NOT NULL DEFAULT '',
            closed_by TEXT NOT NULL DEFAULT '',
            closed_at TEXT NOT NULL DEFAULT '',
            revision INTEGER NOT NULL DEFAULT 1 CHECK (revision > 0),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_handover_risks_case
            ON handover_risks(handover_id,status,severity,due_at,risk_id);

        CREATE TABLE IF NOT EXISTS handover_snapshots (
            snapshot_id TEXT PRIMARY KEY,
            handover_id TEXT NOT NULL UNIQUE REFERENCES handover_cases(handover_id) ON DELETE RESTRICT,
            project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            snapshot_version INTEGER NOT NULL DEFAULT 1,
            snapshot_json TEXT NOT NULL,
            checksum TEXT NOT NULL,
            created_by TEXT NOT NULL DEFAULT 'user_system',
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_handover_snapshots_project
            ON handover_snapshots(project_id,created_at DESC);
        CREATE TRIGGER IF NOT EXISTS trg_handover_snapshots_no_update
        BEFORE UPDATE ON handover_snapshots
        BEGIN
            SELECT RAISE(ABORT, 'handover snapshot is immutable');
        END;
        CREATE TRIGGER IF NOT EXISTS trg_handover_snapshots_no_delete
        BEFORE DELETE ON handover_snapshots
        BEGIN
            SELECT RAISE(ABORT, 'handover snapshot is immutable');
        END;
        """,
    ),
    (
        13,
        "handover_automatic_knowledge_inventory",
        """
        CREATE TABLE IF NOT EXISTS handover_inventory_entries (
            inventory_id TEXT PRIMARY KEY,
            handover_id TEXT NOT NULL REFERENCES handover_cases(handover_id) ON DELETE CASCADE,
            project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            source_type TEXT NOT NULL
                CHECK (source_type IN ('knowledge_asset','knowledge_task','graph_topic')),
            reference_id TEXT NOT NULL,
            version_id TEXT NOT NULL DEFAULT '',
            title TEXT NOT NULL,
            relation_type TEXT NOT NULL,
            relation_detail TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending','included','excluded')),
            exclusion_reason TEXT NOT NULL DEFAULT '',
            reviewed_by TEXT NOT NULL DEFAULT '',
            reviewed_at TEXT NOT NULL DEFAULT '',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE (handover_id,source_type,reference_id,version_id)
        );
        CREATE INDEX IF NOT EXISTS idx_handover_inventory_case
            ON handover_inventory_entries(handover_id,status,source_type,created_at,inventory_id);
        """,
    ),
    (
        14,
        "rag_quality_evaluation_and_feedback",
        """
        CREATE TABLE IF NOT EXISTS rag_evaluation_sets (
            evaluation_set_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            version INTEGER NOT NULL DEFAULT 1 CHECK (version > 0),
            description TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'draft'
                CHECK (status IN ('draft','active','archived')),
            min_case_target INTEGER NOT NULL DEFAULT 30 CHECK (min_case_target BETWEEN 1 AND 500),
            created_by TEXT NOT NULL DEFAULT 'user_system',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE (project_id,name,version)
        );
        CREATE INDEX IF NOT EXISTS idx_rag_evaluation_sets_project
            ON rag_evaluation_sets(project_id,status,updated_at DESC,evaluation_set_id);

        CREATE TABLE IF NOT EXISTS rag_evaluation_cases (
            evaluation_case_id TEXT PRIMARY KEY,
            evaluation_set_id TEXT NOT NULL REFERENCES rag_evaluation_sets(evaluation_set_id) ON DELETE CASCADE,
            project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            question TEXT NOT NULL,
            scenario TEXT NOT NULL
                CHECK (scenario IN ('fact','process','decision','risk','handover','no_answer')),
            expected_points_json TEXT NOT NULL DEFAULT '[]',
            allowed_asset_ids_json TEXT NOT NULL DEFAULT '[]',
            allowed_source_files_json TEXT NOT NULL DEFAULT '[]',
            expect_no_answer INTEGER NOT NULL DEFAULT 0 CHECK (expect_no_answer IN (0,1)),
            position INTEGER NOT NULL DEFAULT 0,
            created_by TEXT NOT NULL DEFAULT 'user_system',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_rag_evaluation_cases_set
            ON rag_evaluation_cases(evaluation_set_id,position,evaluation_case_id);

        CREATE TABLE IF NOT EXISTS rag_evaluation_runs (
            evaluation_run_id TEXT PRIMARY KEY,
            evaluation_set_id TEXT NOT NULL REFERENCES rag_evaluation_sets(evaluation_set_id) ON DELETE CASCADE,
            project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            status TEXT NOT NULL DEFAULT 'completed'
                CHECK (status IN ('running','completed','failed')),
            strategy_json TEXT NOT NULL DEFAULT '{}',
            metrics_json TEXT NOT NULL DEFAULT '{}',
            created_by TEXT NOT NULL DEFAULT 'user_system',
            started_at TEXT NOT NULL,
            completed_at TEXT NOT NULL DEFAULT '',
            error TEXT NOT NULL DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_rag_evaluation_runs_set
            ON rag_evaluation_runs(evaluation_set_id,started_at DESC,evaluation_run_id);

        CREATE TABLE IF NOT EXISTS rag_evaluation_results (
            evaluation_result_id TEXT PRIMARY KEY,
            evaluation_run_id TEXT NOT NULL REFERENCES rag_evaluation_runs(evaluation_run_id) ON DELETE CASCADE,
            evaluation_case_id TEXT NOT NULL REFERENCES rag_evaluation_cases(evaluation_case_id) ON DELETE CASCADE,
            project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            no_answer INTEGER NOT NULL DEFAULT 0 CHECK (no_answer IN (0,1)),
            citation_validity_rate REAL NOT NULL DEFAULT 0,
            allowed_source_precision REAL NOT NULL DEFAULT 0,
            expected_evidence_coverage REAL NOT NULL DEFAULT 0,
            passed INTEGER NOT NULL DEFAULT 0 CHECK (passed IN (0,1)),
            citations_json TEXT NOT NULL DEFAULT '[]',
            matched_points_json TEXT NOT NULL DEFAULT '[]',
            detail TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            UNIQUE (evaluation_run_id,evaluation_case_id)
        );

        CREATE TABLE IF NOT EXISTS rag_answer_feedback (
            feedback_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            answer_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            question TEXT NOT NULL DEFAULT '',
            helpful INTEGER NOT NULL CHECK (helpful IN (0,1)),
            reason TEXT NOT NULL DEFAULT ''
                CHECK (reason IN ('','citation_error','knowledge_outdated','incomplete','permission_blocked','other')),
            note TEXT NOT NULL DEFAULT '',
            citations_json TEXT NOT NULL DEFAULT '[]',
            knowledge_task_id TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE (project_id,answer_id,user_id)
        );
        CREATE INDEX IF NOT EXISTS idx_rag_answer_feedback_project
            ON rag_answer_feedback(project_id,helpful,reason,updated_at DESC,feedback_id);
        """,
    ),
    (
        15,
        "production_slo_windows_and_recovery_drills",
        """
        CREATE TABLE IF NOT EXISTS runtime_maintenance_windows (
            window_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            title TEXT NOT NULL,
            reason TEXT NOT NULL DEFAULT '',
            starts_at TEXT NOT NULL,
            ends_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'scheduled'
                CHECK (status IN ('scheduled','cancelled')),
            created_by TEXT NOT NULL DEFAULT 'user_system',
            cancelled_by TEXT NOT NULL DEFAULT '',
            cancelled_at TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_runtime_maintenance_project
            ON runtime_maintenance_windows(project_id,status,starts_at,ends_at);

        CREATE TABLE IF NOT EXISTS runtime_slo_events (
            event_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            metric_key TEXT NOT NULL,
            subject_id TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'running'
                CHECK (status IN ('running','succeeded','failed')),
            started_at TEXT NOT NULL,
            completed_at TEXT NOT NULL DEFAULT '',
            duration_ms REAL NOT NULL DEFAULT 0,
            detail_json TEXT NOT NULL DEFAULT '{}',
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_runtime_slo_events_window
            ON runtime_slo_events(project_id,metric_key,started_at,status);

        CREATE TABLE IF NOT EXISTS runtime_recovery_drills (
            drill_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            status TEXT NOT NULL CHECK (status IN ('passed','failed')),
            backup_name TEXT NOT NULL DEFAULT '',
            backup_created_at TEXT NOT NULL DEFAULT '',
            rpo_hours REAL NOT NULL DEFAULT 0,
            rto_seconds REAL NOT NULL DEFAULT 0,
            report_json TEXT NOT NULL DEFAULT '{}',
            actor_user_id TEXT NOT NULL DEFAULT 'user_system',
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_runtime_recovery_project
            ON runtime_recovery_drills(project_id,created_at DESC);
        """,
    ),
    (
        16,
        "knowledge_task_notification_policy_and_delivery",
        """
        CREATE TABLE IF NOT EXISTS knowledge_task_notification_policies (
            project_id TEXT PRIMARY KEY REFERENCES projects(project_id) ON DELETE CASCADE,
            enabled INTEGER NOT NULL DEFAULT 0 CHECK (enabled IN (0,1)),
            target_chat_id TEXT NOT NULL DEFAULT '',
            timezone TEXT NOT NULL DEFAULT 'Asia/Shanghai',
            quiet_start TEXT NOT NULL DEFAULT '22:00',
            quiet_end TEXT NOT NULL DEFAULT '08:00',
            remind_before_hours INTEGER NOT NULL DEFAULT 24
                CHECK (remind_before_hours BETWEEN 1 AND 168),
            reminder_interval_hours INTEGER NOT NULL DEFAULT 24
                CHECK (reminder_interval_hours BETWEEN 1 AND 168),
            escalation_after_hours INTEGER NOT NULL DEFAULT 24
                CHECK (escalation_after_hours BETWEEN 1 AND 720),
            escalation_interval_hours INTEGER NOT NULL DEFAULT 24
                CHECK (escalation_interval_hours BETWEEN 1 AND 720),
            updated_by TEXT NOT NULL DEFAULT 'user_system',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        ALTER TABLE task_notifications ADD COLUMN notification_kind TEXT NOT NULL DEFAULT 'manual';
        ALTER TABLE task_notifications ADD COLUMN dedupe_key TEXT NOT NULL DEFAULT '';
        ALTER TABLE task_notifications ADD COLUMN processing_job_id TEXT NOT NULL DEFAULT '';
        ALTER TABLE task_notifications ADD COLUMN scheduled_at TEXT NOT NULL DEFAULT '';
        ALTER TABLE task_notifications ADD COLUMN sent_at TEXT NOT NULL DEFAULT '';
        ALTER TABLE task_notifications ADD COLUMN attempt_count INTEGER NOT NULL DEFAULT 0;
        ALTER TABLE task_notifications ADD COLUMN escalation_level INTEGER NOT NULL DEFAULT 0;
        CREATE UNIQUE INDEX IF NOT EXISTS uq_task_notification_dedupe
            ON task_notifications(project_id,dedupe_key) WHERE dedupe_key<>'';
        CREATE INDEX IF NOT EXISTS idx_task_notifications_job
            ON task_notifications(project_id,processing_job_id,status);
        CREATE INDEX IF NOT EXISTS idx_task_notifications_schedule
            ON task_notifications(project_id,scheduled_at,status);
        """,
    ),
    (
        17,
        "targeted_data_governance",
        """
        CREATE TABLE IF NOT EXISTS targeted_deletion_runs (
            run_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            target_type TEXT NOT NULL CHECK (target_type IN ('source','user','asset')),
            target_id TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('succeeded','failed')),
            actor_user_id TEXT NOT NULL DEFAULT 'user_system',
            preview_json TEXT NOT NULL DEFAULT '{}',
            result_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_targeted_deletion_runs_project
            ON targeted_deletion_runs(project_id,created_at DESC,run_id);
        """,
    ),
    (
        18,
        "daily_data_governance_schedule",
        """
        ALTER TABLE data_governance_policies ADD COLUMN daily_backup_enabled INTEGER NOT NULL DEFAULT 1;
        ALTER TABLE data_governance_policies ADD COLUMN daily_retention_enabled INTEGER NOT NULL DEFAULT 0;
        ALTER TABLE data_governance_policies ADD COLUMN schedule_time TEXT NOT NULL DEFAULT '02:00';
        ALTER TABLE data_governance_policies ADD COLUMN timezone TEXT NOT NULL DEFAULT 'Asia/Shanghai';

        CREATE TABLE IF NOT EXISTS governance_schedule_intents (
            intent_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            operation TEXT NOT NULL CHECK (operation IN ('database_backup','data_retention')),
            local_date TEXT NOT NULL,
            scheduled_for TEXT NOT NULL,
            processing_job_id TEXT NOT NULL REFERENCES processing_jobs(job_id) ON DELETE RESTRICT,
            created_at TEXT NOT NULL,
            UNIQUE(project_id,operation,local_date)
        );
        CREATE INDEX IF NOT EXISTS idx_governance_schedule_intents_project
            ON governance_schedule_intents(project_id,local_date DESC,operation);
        """,
    ),
    (
        19,
        "handover_independent_item_review",
        """
        ALTER TABLE handover_items ADD COLUMN review_mode TEXT NOT NULL DEFAULT '';
        ALTER TABLE handover_items ADD COLUMN proxy_reason TEXT NOT NULL DEFAULT '';
        """,
    ),
    (
        20,
        "handover_item_evidence_assets_and_drafts",
        """
        CREATE TABLE IF NOT EXISTS handover_item_evidence_assets (
            evidence_id TEXT PRIMARY KEY,
            handover_id TEXT NOT NULL REFERENCES handover_cases(handover_id) ON DELETE CASCADE,
            item_id TEXT NOT NULL REFERENCES handover_items(item_id) ON DELETE CASCADE,
            project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            asset_id TEXT NOT NULL,
            version_id TEXT NOT NULL,
            source_kind TEXT NOT NULL DEFAULT 'existing'
                CHECK (source_kind IN ('existing','uploaded','legacy')),
            attached_by TEXT NOT NULL DEFAULT '',
            attached_at TEXT NOT NULL,
            UNIQUE (item_id,asset_id,version_id)
        );
        CREATE INDEX IF NOT EXISTS idx_handover_item_evidence_item
            ON handover_item_evidence_assets(item_id,attached_at,evidence_id);

        CREATE TABLE IF NOT EXISTS handover_item_drafts (
            handover_id TEXT NOT NULL REFERENCES handover_cases(handover_id) ON DELETE CASCADE,
            item_id TEXT NOT NULL REFERENCES handover_items(item_id) ON DELETE CASCADE,
            project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
            evidence TEXT NOT NULL DEFAULT '',
            asset_refs_json TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (handover_id,item_id,user_id)
        );
        CREATE INDEX IF NOT EXISTS idx_handover_item_drafts_item
            ON handover_item_drafts(handover_id,item_id,updated_at DESC);

        INSERT OR IGNORE INTO handover_item_evidence_assets(
            evidence_id,handover_id,item_id,project_id,asset_id,version_id,source_kind,attached_by,attached_at
        )
        SELECT
            'legacy:' || item_id || ':' || asset_id || ':' || version_id,
            handover_id,item_id,project_id,asset_id,version_id,'legacy',submitted_by,
            CASE WHEN submitted_at<>'' THEN submitted_at ELSE updated_at END
        FROM handover_items
        WHERE asset_id<>'' AND version_id<>'';
        """,
    ),
    (
        21,
        "user_duty_profile",
        """
        ALTER TABLE users ADD COLUMN duty TEXT NOT NULL DEFAULT '';
        """,
    ),
)


class Database:
    """Small, explicit sqlite3 database wrapper used by all repositories."""

    def __init__(self, path: Path | str, timeout_seconds: float = 30.0) -> None:
        self.path = Path(path).expanduser().resolve()
        self.timeout_seconds = timeout_seconds
        self._lock = threading.RLock()
        self._initialized = False

    def _connect(self, path: Optional[Path] = None) -> sqlite3.Connection:
        target = path or self.path
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(target.parent, 0o700)
        except OSError:
            pass
        connection = sqlite3.connect(
            str(target),
            timeout=self.timeout_seconds,
            isolation_level=None,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {int(self.timeout_seconds * 1000)}")
        if path is None or target == self.path:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")
        return connection

    def process_lock(self, purpose: str = "application") -> StorageProcessLock:
        lock_path = self.path.with_suffix(self.path.suffix + ".runtime.lock")
        return StorageProcessLock(lock_path, purpose)

    def initialize(self) -> None:
        with self._lock:
            if self._initialized:
                return
            connection = self._connect()
            try:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        version INTEGER PRIMARY KEY,
                        name TEXT NOT NULL,
                        checksum TEXT NOT NULL,
                        applied_at TEXT NOT NULL
                    )
                    """
                )
                applied = {
                    int(row["version"]): row["checksum"]
                    for row in connection.execute("SELECT version, checksum FROM schema_migrations")
                }
                for version, name, sql in SCHEMA_MIGRATIONS:
                    checksum = hashlib.sha256(sql.encode("utf-8")).hexdigest()
                    if version in applied:
                        if applied[version] != checksum:
                            raise RuntimeError(f"数据库迁移 {version} 校验和不一致")
                        continue
                    script = (
                        "BEGIN IMMEDIATE;\n"
                        + sql
                        + "\nINSERT INTO schema_migrations(version, name, checksum, applied_at) VALUES ("
                        + f"{version}, {json.dumps(name)}, {json.dumps(checksum)}, {json.dumps(utc_now())});\n"
                        + "COMMIT;"
                    )
                    try:
                        connection.executescript(script)
                    except Exception:
                        connection.execute("ROLLBACK") if connection.in_transaction else None
                        raise
                self._seed_defaults(connection)
                self._initialized = True
                try:
                    os.chmod(self.path, 0o600)
                except OSError:
                    pass
            finally:
                connection.close()

    def _seed_defaults(self, connection: sqlite3.Connection) -> None:
        now = utc_now()
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute(
                "INSERT OR IGNORE INTO organizations(organization_id,name,slug,status,created_at,updated_at) VALUES(?,?,?,?,?,?)",
                (DEFAULT_ORGANIZATION_ID, "默认组织", "default", "active", now, now),
            )
            connection.execute(
                "INSERT OR IGNORE INTO projects(project_id,name,slug,is_default,created_at,updated_at) VALUES(?,?,?,?,?,?)",
                (DEFAULT_PROJECT_ID, "默认项目", "default", 1, now, now),
            )
            connection.execute(
                "UPDATE projects SET organization_id=?,status='active' WHERE project_id=?",
                (DEFAULT_ORGANIZATION_ID, DEFAULT_PROJECT_ID),
            )
            connection.execute(
                "INSERT OR IGNORE INTO users(user_id,display_name,user_type,created_at,updated_at) VALUES(?,?,?,?,?)",
                (SYSTEM_USER_ID, "系统用户（占位）", "placeholder", now, now),
            )
            connection.execute(
                "UPDATE users SET organization_id=?,status='active' WHERE user_id=?",
                (DEFAULT_ORGANIZATION_ID, SYSTEM_USER_ID),
            )
            connection.execute(
                "INSERT OR IGNORE INTO project_memberships(project_id,user_id,role,created_at) VALUES(?,?,?,?)",
                (DEFAULT_PROJECT_ID, SYSTEM_USER_ID, "system_placeholder", now),
            )
            connection.execute(
                "UPDATE project_memberships SET status='active',updated_at=? WHERE project_id=? AND user_id=?",
                (now, DEFAULT_PROJECT_ID, SYSTEM_USER_ID),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    @contextmanager
    def transaction(self, write: bool = False) -> Iterator[sqlite3.Connection]:
        self.initialize()
        with self._lock:
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
                yield connection
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.close()

    def quick_check(self) -> dict:
        self.initialize()
        with self.transaction() as connection:
            quick_rows = [row[0] for row in connection.execute("PRAGMA quick_check")]
            foreign_rows = [tuple(row) for row in connection.execute("PRAGMA foreign_key_check")]
        return {
            "ok": quick_rows == ["ok"] and not foreign_rows,
            "quick_check": quick_rows,
            "foreign_key_errors": foreign_rows,
            "database_path": str(self.path),
        }

    def migration_status(self) -> dict:
        self.initialize()
        expected = {version: name for version, name, _ in SCHEMA_MIGRATIONS}
        with self.transaction() as connection:
            applied = [dict(row) for row in connection.execute(
                "SELECT version,name,checksum,applied_at FROM schema_migrations ORDER BY version"
            )]
            legacy = [dict(row) for row in connection.execute(
                "SELECT migration_key,source_hash,status,report_path,detail,started_at,completed_at "
                "FROM migration_runs ORDER BY started_at"
            )]
        applied_versions = {int(item["version"]) for item in applied}
        return {
            "current_version": max(applied_versions, default=0),
            "expected_version": max(expected, default=0),
            "pending_versions": [version for version in expected if version not in applied_versions],
            "schema_migrations": applied,
            "legacy_migrations": legacy,
        }

    def backup(self, backup_dir: Path | str, name: str = "") -> dict:
        self.initialize()
        backup_root = Path(backup_dir).expanduser().resolve()
        backup_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        safe_name = Path(name).name if name else f"app-{stamp}.db"
        if not safe_name.endswith(".db"):
            safe_name += ".db"
        target = backup_root / safe_name
        with self._lock:
            source_connection = self._connect()
            target_connection = self._connect(target)
            try:
                source_connection.backup(target_connection)
            finally:
                target_connection.close()
                source_connection.close()
        check = self._check_external_database(target)
        if not check["ok"]:
            target.unlink(missing_ok=True)
            raise RuntimeError(f"数据库备份完整性检查失败: {check['quick_check']}")
        digest = self.file_hash(target)
        try:
            os.chmod(target, 0o600)
        except OSError:
            pass
        manifest = {
            "database": target.name,
            "sha256": digest,
            "size": target.stat().st_size,
            "created_at": utc_now(),
            "schema": self.migration_status(),
        }
        manifest_path = target.with_suffix(".manifest.json")
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            os.chmod(manifest_path, 0o600)
        except OSError:
            pass
        return {**manifest, "path": str(target), "manifest_path": str(manifest_path)}

    def restore(self, backup_path: Path | str, backup_dir: Optional[Path | str] = None) -> dict:
        with self.process_lock("restore"):
            return self._restore_offline(backup_path, backup_dir)

    def _restore_offline(self, backup_path: Path | str, backup_dir: Optional[Path | str] = None) -> dict:
        source_path = Path(backup_path).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError("数据库备份不存在")
        manifest_path = source_path.with_suffix(".manifest.json")
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError("数据库备份清单损坏") from exc
            expected_hash = str(manifest.get("sha256") or "")
            if expected_hash and self.file_hash(source_path) != expected_hash:
                raise RuntimeError("数据库备份哈希校验失败")
        check = self._check_external_database(source_path)
        if not check["ok"]:
            raise RuntimeError("拒绝恢复损坏的数据库备份")
        safety_name = f"pre-restore-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
        backup_root = Path(backup_dir or self.path.parent / "backups").expanduser().resolve()
        safety_backup = None
        if self.path.is_file():
            try:
                current_check = self._check_external_database(self.path)
            except sqlite3.DatabaseError:
                current_check = {"ok": False}
            if current_check.get("ok"):
                self._initialized = False
                safety_backup = self.backup(backup_root, name=safety_name)
            else:
                backup_root.mkdir(parents=True, exist_ok=True)
                corrupt_copy = backup_root / f"{safety_name}-corrupt.db"
                shutil.copy2(self.path, corrupt_copy)
                safety_backup = {
                    "path": str(corrupt_copy), "sha256": self.file_hash(corrupt_copy),
                    "size": corrupt_copy.stat().st_size, "valid": False,
                }
        restore_temp = self.path.with_suffix(".restore.tmp")
        restore_temp.unlink(missing_ok=True)
        with self._lock:
            source_connection = self._connect(source_path)
            target_connection = self._connect(restore_temp)
            try:
                source_connection.backup(target_connection)
            finally:
                target_connection.close()
                source_connection.close()
            restored_temp_check = self._check_external_database(restore_temp)
            if not restored_temp_check["ok"]:
                restore_temp.unlink(missing_ok=True)
                raise RuntimeError("恢复临时数据库完整性检查失败")
            for sidecar in (Path(f"{self.path}-wal"), Path(f"{self.path}-shm")):
                sidecar.unlink(missing_ok=True)
            os.replace(restore_temp, self.path)
            self._initialized = False
            self.initialize()
        restored_check = self.quick_check()
        if not restored_check["ok"]:
            raise RuntimeError("恢复后数据库完整性检查失败")
        return {"success": True, "restored_from": str(source_path), "safety_backup": safety_backup}

    def _check_external_database(self, path: Path) -> dict:
        if not path.is_file():
            return {"ok": False, "quick_check": ["database file missing"]}
        # A SQLite backup retains the source journal mode. A normal connection is
        # needed here so WAL sidecars may be created while integrity is checked.
        connection = sqlite3.connect(str(path), timeout=self.timeout_seconds)
        try:
            quick_rows = [row[0] for row in connection.execute("PRAGMA quick_check")]
            foreign_rows = [tuple(row) for row in connection.execute("PRAGMA foreign_key_check")]
            return {"ok": quick_rows == ["ok"] and not foreign_rows, "quick_check": quick_rows}
        finally:
            connection.close()

    @staticmethod
    def file_hash(path: Path | str) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
