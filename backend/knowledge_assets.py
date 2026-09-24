"""Unified knowledge-asset lifecycle service.

The service coordinates authoritative lifecycle state. Vector, graph and
context implementations remain replaceable projections and never decide which
version is retrievable.
"""
from __future__ import annotations

import hashlib
import logging
import threading
from typing import Any, Dict, Iterable, Optional, Sequence

from backend.storage import get_repository
from backend.storage.database import StorageProcessLock

logger = logging.getLogger(__name__)


class ProjectionRepairBusyError(RuntimeError):
    """Raised when another projection repair is already running in this process."""


class KnowledgeAssetService:
    _projection_repair_lock = threading.Lock()

    def __init__(self, repository_provider=get_repository) -> None:
        self.repository_provider = repository_provider

    @property
    def repository(self):
        return self.repository_provider()

    @staticmethod
    def content_hash(documents: Sequence[Any]) -> str:
        digest = hashlib.sha256()
        for document in documents:
            digest.update(str(getattr(document, "page_content", "") or "").encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()

    def prepare_documents(
        self,
        documents: Sequence[Any],
        *,
        source_type: str,
        external_key: str,
        title: str,
        actor: str,
        stored_file: str,
        categories: Sequence[str] = (),
        applicable_roles: Sequence[str] = (),
        asset_id: str = "",
        source_id: str = "",
        visibility: str = "project",
        sensitivity_level: str = "",
        access_policy_id: str = "",
        owner_user_id: str = "",
        summary: str = "",
        valid_from: str = "",
        valid_until: str = "",
        review_due_at: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        content_hash_override: str = "",
    ) -> Dict[str, Any]:
        if not documents:
            raise ValueError("文档没有可发布的知识内容")
        if not sensitivity_level:
            sensitivity_level = str(
                self.repository.get_data_governance_policy().get("default_sensitivity") or "internal"
            )
        content_hash = str(content_hash_override or self.content_hash(documents))
        prepared = self.repository.prepare_asset_version(
            source_type=source_type,
            external_key=external_key,
            title=title,
            content_hash=content_hash,
            actor=actor,
            asset_id=asset_id,
            source_id=source_id,
            categories=categories,
            applicable_roles=applicable_roles,
            visibility=visibility,
            sensitivity_level=sensitivity_level,
            access_policy_id=access_policy_id,
            owner_user_id=owner_user_id,
            summary=summary,
            valid_from=valid_from,
            valid_until=valid_until,
            review_due_at=review_due_at,
            metadata=metadata,
        )
        if prepared.get("duplicate"):
            return prepared
        primary = next((str(item).strip() for item in categories if str(item).strip()), "未分类")
        for document in documents:
            document.metadata.update({
                "source_file": title,
                "stored_file": stored_file,
                "source_type": source_type,
                "source_id": prepared["source_id"],
                "asset_id": prepared["asset_id"],
                "version_id": prepared["version_id"],
                "version_no": prepared["version_no"],
                "content_hash": content_hash,
                "status": "preparing",
                "categories": list(categories) or [primary],
                "category": primary,
                "visibility": visibility,
                "sensitivity_level": sensitivity_level,
                "access_policy_id": access_policy_id,
                "owner_user_id": owner_user_id,
                "lifecycle_managed": True,
            })
        return prepared

    def record_projection(self, version_id: str, projection_type: str, *, succeeded: bool, error: str = "") -> Dict[str, Any]:
        status = "ready" if succeeded else "repair_required"
        if projection_type == "vector" and not succeeded:
            self.repository.fail_asset_version(version_id, error or "向量索引失败")
            raise RuntimeError(error or "向量索引失败")
        return self.repository.set_asset_projection(version_id, projection_type, status, error)

    def publish(
        self,
        prepared: Dict[str, Any],
        *,
        actor: str,
        graph_ready: bool = False,
        context_ready: bool = True,
        readiness_ready: bool = False,
        graph_error: str = "等待发布后重建图谱",
        context_error: str = "",
    ) -> Dict[str, Any]:
        if prepared.get("duplicate") and prepared.get("status") == "active":
            return self.repository.get_asset(str(prepared["asset_id"])) or prepared
        version_id = str(prepared["version_id"])
        self.repository.set_asset_projection(version_id, "graph", "ready" if graph_ready else "repair_required", graph_error)
        self.repository.set_asset_projection(version_id, "context", "ready" if context_ready else "repair_required", context_error)
        self.repository.set_asset_projection(
            version_id, "readiness", "ready" if readiness_ready else "repair_required",
            "等待新人赋能投影刷新" if not readiness_ready else "",
        )
        self.repository.mark_asset_version_ready(version_id)
        return self.repository.publish_asset_version(str(prepared["asset_id"]), version_id, actor)

    def mark_post_publish_projection(self, version_id: str, projection_type: str, succeeded: bool, error: str = "") -> None:
        self.repository.set_asset_projection(
            version_id, projection_type, "ready" if succeeded else "repair_required", error,
        )

    def fail(self, prepared: Dict[str, Any], error: Exception | str, actor: str = "system") -> None:
        version_id = str(prepared.get("version_id") or "")
        if version_id and not prepared.get("duplicate"):
            self.repository.fail_asset_version(version_id, str(error), actor=actor)

    def list_assets(
        self, identity: Any = None, include_inactive: bool = False,
        include_deleted: bool = False,
    ):
        return self.repository.list_assets(
            identity=identity,
            include_inactive=include_inactive,
            include_deleted=include_deleted,
        )

    def list_versions(self, asset_id: str, identity: Any = None):
        return self.repository.list_asset_versions(asset_id, identity=identity)

    def get_asset(self, asset_id: str, identity: Any = None):
        return self.repository.get_asset_detail(asset_id, identity=identity)

    def get_version(self, asset_id: str, version_id: str, identity: Any = None):
        return self.repository.get_asset_version(asset_id, version_id, identity=identity)

    def projection_status(self, asset_id: str, identity: Any = None):
        return self.repository.asset_projection_status(asset_id, identity=identity)

    def repair_projections(
        self, asset_id: str = "", *, actor: str = "system", identity: Any = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        if not self._projection_repair_lock.acquire(blocking=False):
            raise ProjectionRepairBusyError("知识投影修复任务正在运行")
        process_lock = None
        try:
            lock_path = self.repository.database.path.with_suffix(
                self.repository.database.path.suffix + ".projection-repair.lock"
            )
            try:
                process_lock = StorageProcessLock(lock_path, "knowledge projection repair").acquire()
            except RuntimeError as exc:
                raise ProjectionRepairBusyError("知识投影修复任务正在运行") from exc
            if asset_id and not self.repository.get_asset(asset_id, identity=identity):
                raise ValueError("知识资产不存在或无权访问")
            targets = self.repository.current_projection_targets(asset_id, include_ready=force)
            grouped: Dict[str, list[Dict[str, Any]]] = {}
            for target in targets:
                grouped.setdefault(str(target["projection_type"]), []).append(target)
            results: Dict[str, Any] = {}
            # Graph/context/readiness consume the current retrieval projection.
            # Rebuilding them before the vector index can persist an empty but
            # apparently successful graph when the derived index is missing.
            projection_order = ("vector", "graph", "context", "readiness")
            ordered_types = [item for item in projection_order if item in grouped]
            ordered_types.extend(item for item in grouped if item not in projection_order)
            for projection_type in ordered_types:
                items = grouped[projection_type]
                for item in items:
                    self.repository.set_asset_projection(item["version_id"], projection_type, "building")
                try:
                    if projection_type == "vector":
                        from backend.knowledge_base.vector_store import vector_store
                        detail = vector_store.reconcile_with_storage()
                    elif projection_type == "graph":
                        from backend.knowledge_base.knowledge_graph import knowledge_graph
                        detail = knowledge_graph.build_graph()
                    elif projection_type == "context":
                        from backend.knowledge_base.project_context import project_context
                        detail = project_context.reload() if hasattr(project_context, "reload") else None
                    elif projection_type == "readiness":
                        detail = {"assets": len(self.repository.readiness_assets(identity=identity))}
                    else:
                        raise ValueError("不支持的知识投影类型")
                    for item in items:
                        self.repository.set_asset_projection(item["version_id"], projection_type, "ready")
                    results[projection_type] = {"success": True, "detail": detail, "count": len(items)}
                except Exception as exc:
                    logger.exception("repair knowledge projection failed: %s", projection_type)
                    for item in items:
                        self.repository.set_asset_projection(
                            item["version_id"], projection_type, "repair_required", str(exc),
                        )
                    results[projection_type] = {"success": False, "error": str(exc)[:500], "count": len(items)}
            status = self.repository.asset_projection_status(asset_id, identity=identity) if asset_id else None
            self.repository.write_security_audit(
                actor, "knowledge.projections_repaired", "knowledge_asset", asset_id or "all",
                {"force": force, "results": results}, project_id=self.repository.project_id,
            )
            return {"success": all(item["success"] for item in results.values()), "results": results, "status": status}
        finally:
            if process_lock is not None:
                process_lock.release()
            self._projection_repair_lock.release()

    def patch_metadata(self, asset_id: str, changes: Dict[str, Any], actor: str):
        return self.repository.patch_asset_metadata(asset_id, changes, actor)

    def transition(
        self, asset_id: str, action: str, actor: str, reason: str = "",
        expected_revision: Optional[int] = None, version_id: str = "",
    ):
        if str(action or "").strip().lower() == "publish":
            if not version_id:
                raise ValueError("发布操作必须指定 version_id")
            self.repository.mark_asset_version_ready(version_id)
            return self.repository.publish_asset_version(asset_id, version_id, actor, expected_revision)
        return self.repository.transition_asset(asset_id, action, actor, reason, expected_revision)


knowledge_asset_service = KnowledgeAssetService()


__all__ = ["KnowledgeAssetService", "ProjectionRepairBusyError", "knowledge_asset_service"]
