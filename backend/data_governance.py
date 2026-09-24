"""Deterministic project data-governance operations."""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from zoneinfo import ZoneInfo

from backend.config import settings
from backend.storage import backup_dir, get_repository


logger = logging.getLogger(__name__)


class DataGovernanceService:
    def __init__(
        self,
        repository_provider=get_repository,
        backup_dir_provider=backup_dir,
        upload_dir_provider: Callable[[], Path] = lambda: Path("uploads"),
        projection_refresher: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> None:
        self.repository_provider = repository_provider
        self.backup_dir_provider = backup_dir_provider
        self.upload_dir_provider = upload_dir_provider
        self.projection_refresher = projection_refresher

    @property
    def repository(self):
        return self.repository_provider()

    def policy(self) -> Dict[str, Any]:
        return self.repository.get_data_governance_policy()

    def update_policy(self, payload: Dict[str, Any], actor_user_id: str) -> Dict[str, Any]:
        return self.repository.update_data_governance_policy(payload, actor_user_id=actor_user_id)

    @staticmethod
    def _schedule_clock(policy: Dict[str, Any], now: Optional[datetime] = None) -> Dict[str, Any]:
        zone = ZoneInfo(str(policy.get("timezone") or "Asia/Shanghai"))
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        local_now = current.astimezone(zone)
        hour, minute = (int(item) for item in str(policy.get("schedule_time") or "02:00").split(":"))
        scheduled_time = time(hour=hour, minute=minute, tzinfo=zone)
        scheduled_today = datetime.combine(local_now.date(), scheduled_time)
        due_local = scheduled_today if local_now >= scheduled_today else scheduled_today - timedelta(days=1)
        next_local = scheduled_today + (timedelta(days=1) if local_now >= scheduled_today else timedelta())
        return {
            "local_now": local_now, "due_local": due_local, "next_local": next_local,
            "local_date": due_local.date().isoformat(),
            "scheduled_for": due_local.astimezone(timezone.utc).isoformat(timespec="seconds"),
            "next_scheduled_at": next_local.astimezone(timezone.utc).isoformat(timespec="seconds"),
        }

    def schedule_daily_operations(self, *, now: Optional[datetime] = None) -> Dict[str, Any]:
        policy = self.policy()
        clock = self._schedule_clock(policy, now)
        operations = []
        if policy.get("daily_backup_enabled"):
            operations.append("database_backup")
        if policy.get("daily_retention_enabled"):
            operations.append("data_retention")
        if not operations:
            return {
                "enabled": False, "job_count": 0, "jobs": [],
                "local_date": clock["local_date"], "next_scheduled_at": clock["next_scheduled_at"],
            }
        result = self.repository.schedule_daily_governance_jobs(
            local_date=clock["local_date"], scheduled_for=clock["scheduled_for"], operations=operations,
        )
        return {"enabled": True, "next_scheduled_at": clock["next_scheduled_at"], **result}

    def schedule_status(self, *, now: Optional[datetime] = None) -> Dict[str, Any]:
        policy = self.policy()
        clock = self._schedule_clock(policy, now)
        return {
            "policy": policy,
            "next_scheduled_at": clock["next_scheduled_at"],
            "latest_intents": self.repository.list_governance_schedule_intents(limit=20),
        }

    def retention_preview(self) -> Dict[str, Any]:
        return self.repository.preview_data_retention()

    def apply_retention(self, context: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
        if str(payload.get("confirmation") or "") != "执行数据清理":
            raise ValueError("数据清理确认短语不正确")
        context.report("retention_scan", 20, "正在重新核对保留策略与引用关系")
        result = self.repository.apply_data_retention(
            actor_user_id=str(payload.get("actor_user_id") or "user_system")
        )
        context.report("retention_audit", 90, "正在保存清理结果与审计")
        return result

    def targeted_preview(self, target_type: str, target_id: str) -> Dict[str, Any]:
        return self.repository.preview_targeted_governance(target_type, target_id)

    def targeted_export(
        self, target_type: str, target_id: str, *, actor_user_id: str,
    ) -> Dict[str, Any]:
        return self.repository.export_targeted_governance(
            target_type, target_id, actor_user_id=actor_user_id,
        )

    @staticmethod
    def _default_projection_refresher(stored_files: Optional[List[str]] = None) -> Dict[str, Any]:
        from backend.knowledge_base.knowledge_graph import knowledge_graph
        from backend.knowledge_base.project_context import project_context
        from backend.knowledge_base.vector_store import vector_store

        vector = vector_store.reconcile_with_storage()
        removed_sources = []
        for stored_file in sorted(set(stored_files or [])):
            source_name = Path(stored_file).name
            if source_name:
                knowledge_graph.remove_source(source_name)
                removed_sources.append(source_name)
        graph = knowledge_graph.build_graph()
        context = project_context.reload() if hasattr(project_context, "reload") else None
        return {"vector": vector, "graph": graph, "context": context,
                "removed_sources": removed_sources}

    def _delete_original_files(self, stored_files: List[str]) -> Dict[str, Any]:
        root = Path(self.upload_dir_provider()).expanduser().resolve()
        removed: List[str] = []
        already_absent: List[str] = []
        skipped: List[str] = []
        for stored_file in sorted(set(str(item or "") for item in stored_files if str(item or ""))):
            candidate = (root / stored_file).resolve()
            try:
                candidate.relative_to(root)
            except ValueError:
                skipped.append(stored_file)
                continue
            if candidate.is_file():
                candidate.unlink()
                removed.append(stored_file)
            else:
                already_absent.append(stored_file)
        return {"removed": removed, "already_absent": already_absent, "skipped": skipped}

    def _delete_feishu_resource_files(self, saved_files: List[str]) -> Dict[str, Any]:
        """Remove archived Feishu resources without allowing path traversal.

        The resource directory can be shared by multiple projects in a deployment,
        so only files attributed to the project at reset start are eligible.
        """
        root = (Path(self.upload_dir_provider()).expanduser().resolve() / "feishu_resources").resolve()
        removed: List[str] = []
        already_absent: List[str] = []
        skipped: List[str] = []
        expected = {str(item or "").strip() for item in saved_files if str(item or "").strip()}
        candidates = {}
        for stored_file in expected:
            if Path(stored_file).name != stored_file:
                skipped.append(stored_file)
                continue
            candidates[stored_file] = root / stored_file
        for stored_file, candidate in sorted(candidates.items()):
            try:
                candidate.resolve().relative_to(root)
            except ValueError:
                skipped.append(stored_file)
                continue
            if candidate.is_file() or candidate.is_symlink():
                candidate.unlink()
                removed.append(stored_file)
            else:
                already_absent.append(stored_file)
        return {"removed": removed, "already_absent": already_absent, "skipped": skipped}

    def apply_targeted_deletion(self, context: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
        if str(payload.get("confirmation") or "") != "永久删除定向数据":
            raise ValueError("定向删除确认短语不正确")
        target_type = str(payload.get("target_type") or "")
        target_id = str(payload.get("target_id") or "")
        actor_user_id = str(payload.get("actor_user_id") or "user_system")
        context.report("target_delete_validate", 10, "正在重新核对定向删除范围")
        result = self.repository.apply_targeted_governance_deletion(
            target_type, target_id, actor_user_id=actor_user_id,
        )
        context.report("target_delete_originals", 55, "正在清除原件与正文")
        stored_files = list(result.pop("files", []))
        files = self._delete_original_files(stored_files)
        if files["skipped"]:
            raise RuntimeError("存在不安全的原件路径，已拒绝结束定向删除")
        context.report("target_delete_projections", 70, "正在清除索引、图谱与缓存投影")
        projections = (
            self.projection_refresher()
            if self.projection_refresher
            else self._default_projection_refresher(stored_files)
        )
        cleanup = self.repository.complete_targeted_file_cleanup(
            str(result["run_id"]), removed=files["removed"] + files["already_absent"],
            actor_user_id=actor_user_id,
        )
        context.report("target_delete_complete", 95, "定向删除已完成并留下治理审计")
        return {**result, "file_cleanup_complete": cleanup["file_cleanup_complete"],
                "file_cleanup": files, "projections": projections}

    def reset_knowledge_library(self, *, actor_user_id: str) -> Dict[str, Any]:
        """Clear every non-tombstoned knowledge lifecycle state for a project.

        The reset is a deliberate administrative exception to normal retention:
        active, review-due, expired and revoked knowledge are all converted to
        content-free deletion tombstones. Persisted Feishu history and resource
        files are cleared as well; accounts, collection-source configuration and
        security audit history remain intact.
        """
        source_ids = self.repository.resettable_knowledge_source_ids()
        # Capture Feishu file ownership before targeted deletion redacts linked
        # messages and legacy rows. This keeps file cleanup project-scoped.
        feishu_file_references = self.repository.project_feishu_file_references()
        totals = {
            "sources_deleted": 0,
            "assets_deleted": 0,
            "versions_revoked": 0,
            "documents_deleted": 0,
            "chunks_deleted": 0,
            "messages_redacted": 0,
        }
        stored_files: List[str] = []
        files_by_run: Dict[str, List[str]] = {}
        for source_id in source_ids:
            item = self.repository.apply_targeted_governance_deletion(
                "source", source_id, actor_user_id=actor_user_id,
            )
            for key in totals:
                totals[key] += int(item.get(key) or 0)
            run_id = str(item["run_id"])
            run_files = [str(value) for value in item.get("files", []) if str(value)]
            files_by_run[run_id] = run_files
            stored_files.extend(run_files)

        feishu_history = self.repository.clear_project_feishu_history()
        stored_files.extend(feishu_file_references.get("asset_files", []))
        # File references are used only for controlled cleanup; return aggregate
        # counts instead of exposing archived file names in the reset response.
        feishu_history.pop("asset_files", None)
        feishu_history.pop("resource_files", None)
        files = self._delete_original_files(stored_files)
        feishu_resources = self._delete_feishu_resource_files(
            feishu_file_references.get("resource_files", [])
        )
        removed_files = set(files["removed"] + files["already_absent"])
        for run_id, run_files in files_by_run.items():
            self.repository.complete_targeted_file_cleanup(
                run_id,
                removed=[value for value in run_files if value in removed_files],
                actor_user_id=actor_user_id,
            )
        projections = (
            self.projection_refresher()
            if self.projection_refresher
            else self._default_projection_refresher(stored_files)
        )
        history = self.repository.clear_project_demo_history()
        result = {
            **totals,
            "sources_processed": len(source_ids),
            "file_cleanup": files,
            "feishu_history": {**feishu_history, "resource_file_cleanup": feishu_resources},
            "projections": projections,
            "history": history,
        }
        self.repository.write_security_audit(
            actor_user_id, "knowledge.library_reset", "knowledge_library", "project",
            {
                **totals,
                "sources_processed": len(source_ids),
                "files_removed": len(files["removed"]),
                "files_skipped": len(files["skipped"]),
                "feishu_messages_deleted": feishu_history["messages_deleted"],
                "feishu_candidates_deleted": feishu_history["candidates_deleted"],
                "feishu_assets_deleted": feishu_history["assets_deleted"],
                "feishu_resources_removed": len(feishu_resources["removed"]),
                **history,
            },
        )
        return result

    def list_backups(self) -> List[Dict[str, Any]]:
        root = Path(self.backup_dir_provider()).expanduser().resolve()
        if not root.is_dir():
            return []
        items = []
        for path in root.glob("*.db"):
            manifest_path = path.with_suffix(".manifest.json")
            manifest: Dict[str, Any] = {}
            if manifest_path.is_file():
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    manifest = {"valid": False, "error": "备份清单损坏"}
            valid = bool(manifest) and manifest.get("sha256") == self.repository.database.file_hash(path)
            items.append({
                "name": path.name,
                "size": path.stat().st_size,
                "created_at": str(manifest.get("created_at") or ""),
                "sha256": str(manifest.get("sha256") or ""),
                "schema_version": int((manifest.get("schema") or {}).get("current_version") or 0),
                "valid": valid,
                "kind": "safety" if path.name.startswith("pre-restore-") else "regular",
            })
        return sorted(items, key=lambda item: (item["created_at"], item["name"]), reverse=True)

    def create_backup(self, context: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
        context.report("backup_database", 25, "正在创建 SQLite 一致性备份")
        backup = self.repository.database.backup(self.backup_dir_provider())
        context.report("backup_verify", 75, "正在校验备份完整性")
        retention_count = int(self.policy()["backup_retention_count"])
        regular = [item for item in self.list_backups() if item["kind"] == "regular"]
        removed = []
        root = Path(self.backup_dir_provider()).expanduser().resolve()
        for item in regular[retention_count:]:
            path = root / item["name"]
            manifest = path.with_suffix(".manifest.json")
            path.unlink(missing_ok=True)
            manifest.unlink(missing_ok=True)
            removed.append(item["name"])
        self.repository.write_security_audit(
            str(payload.get("actor_user_id") or "user_system"),
            "governance.backup_created", "database_backup", Path(backup["path"]).name,
            {"size": backup["size"], "sha256": backup["sha256"], "pruned": removed},
        )
        context.report("backup_complete", 95, "备份已创建并通过完整性校验")
        return {
            "name": Path(backup["path"]).name,
            "size": backup["size"], "sha256": backup["sha256"], "pruned": removed,
        }


data_governance_service = DataGovernanceService()


class DataGovernanceScheduler:
    """In-process clock backed by durable daily intents and processing jobs."""

    def __init__(
        self, service: DataGovernanceService, *, interval_seconds: float = 300,
        wake_callback: Optional[Callable[[], None]] = None,
    ) -> None:
        self.service = service
        self.interval_seconds = max(1.0, float(interval_seconds))
        self.wake_callback = wake_callback
        self._wake = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._stopping = False

    @property
    def running(self) -> bool:
        return bool(self._task and not self._task.done())

    def wake(self) -> None:
        self._wake.set()

    async def start(self) -> None:
        if self.running:
            return
        self._stopping = False
        self._task = asyncio.create_task(self._run(), name="data-governance-scheduler")
        self.wake()

    async def stop(self) -> None:
        self._stopping = True
        self.wake()
        task, self._task = self._task, None
        if task:
            try:
                await asyncio.wait_for(task, timeout=5)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                if not task.done():
                    task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    async def _run(self) -> None:
        while not self._stopping:
            try:
                result = self.service.schedule_daily_operations()
                if result.get("job_count") and self.wake_callback:
                    self.wake_callback()
            except Exception:
                logger.exception("每日数据治理调度扫描失败")
            self._wake.clear()
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=self.interval_seconds)
            except asyncio.TimeoutError:
                pass


data_governance_scheduler = DataGovernanceScheduler(
    data_governance_service, interval_seconds=settings.GOVERNANCE_SCAN_SECONDS,
)


__all__ = [
    "DataGovernanceScheduler", "DataGovernanceService",
    "data_governance_scheduler", "data_governance_service",
]
