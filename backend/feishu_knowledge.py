"""飞书知识候选发布与撤销服务。

该模块承接向量库、项目上下文和知识图谱副作用，让机器人事件处理与
FastAPI 路由都只依赖一个稳定服务，避免互相循环导入。
"""
from __future__ import annotations

import json
import hashlib
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from backend.feishu_workspace import FeishuWorkspace, feishu_workspace
from backend.knowledge_assets import KnowledgeAssetService


logger = logging.getLogger(__name__)


def _safe_filename(value: str, fallback: str = "项目群") -> str:
    cleaned = re.sub(r"[\\/:*?\"<>|\s]+", "_", str(value or "").strip()).strip("_")
    return cleaned[:48] or fallback


class FeishuKnowledgeService:
    """把聚合候选发布为一份可追溯知识资产，并支持完整撤销。"""

    def __init__(
        self,
        workspace: Optional[FeishuWorkspace] = None,
        upload_dir: Optional[Path] = None,
        document_loader=None,
        vector_store=None,
        project_context=None,
        knowledge_graph=None,
        asset_service=None,
    ) -> None:
        self.workspace = workspace or feishu_workspace
        self.upload_dir = Path(upload_dir or "uploads")
        self._document_loader = document_loader
        self._vector_store = vector_store
        self._project_context = project_context
        self._knowledge_graph = knowledge_graph
        self._asset_service = asset_service or KnowledgeAssetService(lambda: self.workspace.repository)

    def _dependencies(self):
        if self._document_loader is None:
            from backend.knowledge_base.document_loader import load_document
            self._document_loader = load_document
        if self._vector_store is None:
            from backend.knowledge_base.vector_store import vector_store
            self._vector_store = vector_store
        if self._project_context is None:
            from backend.knowledge_base.project_context import project_context
            self._project_context = project_context
        if self._knowledge_graph is None:
            from backend.knowledge_base.knowledge_graph import knowledge_graph
            self._knowledge_graph = knowledge_graph
        return self._document_loader, self._vector_store, self._project_context, self._knowledge_graph

    def _register_workspace_projection(
        self,
        candidate_id: str,
        asset: Dict[str, Any],
        *,
        actor: str,
        previous_source_ids: Iterable[str],
    ) -> tuple[Optional[Dict[str, Any]], str]:
        """Write the legacy workspace projection without changing authoritative truth."""
        try:
            published = self.workspace.publish_candidate(candidate_id, asset)
            if not published:
                raise RuntimeError("兼容工作台未返回发布结果")
            return published, ""
        except Exception as exc:
            diagnostic = f"兼容工作台登记失败: {exc}"[:500]
            logger.exception("Feishu workspace projection failed for %s", candidate_id)
            try:
                self.workspace.update_candidate(
                    candidate_id,
                    "failed",
                    error=diagnostic,
                    preserve_published_ids=list(previous_source_ids),
                )
                self.workspace.record_audit(
                    "workspace_projection_failed",
                    "candidate",
                    candidate_id,
                    diagnostic,
                    actor=actor,
                )
            except Exception:
                logger.exception("Failed to record Feishu workspace projection diagnostic")
            return None, diagnostic

    @staticmethod
    def _render_candidate(candidate: Dict[str, Any], group: Dict[str, Any]) -> str:
        lines = [
            f"# {candidate.get('title') or '飞书项目沟通知识'}",
            "",
            f"- 来源群聊：{group.get('name') or '项目群'}",
            f"- 知识分类：{candidate.get('category') or 'proj_communication'}",
            f"- 聚合消息：{len(candidate.get('messages') or [])} 条",
            f"- 候选置信度：{candidate.get('confidence') or 0}",
            f"- 形成原因：{'；'.join(candidate.get('reasons') or [])}",
            "",
            "## 沟通原文",
            "",
        ]
        for message in candidate.get("messages") or []:
            resource_count = int(message.get("resource_count") or 0)
            lines.extend([
                f"### {message.get('sender') or '未知成员'}  {message.get('create_time') or ''}",
                "",
                str(message.get("content") or ""),
                "",
                f"内容类型：{message.get('message_type') or 'text'}",
                f"提取方式：{message.get('extraction_method') or '原始文本'}",
                f"图片原件：已保留 {resource_count} 张" if resource_count else "图片原件：无",
                f"消息 ID：{message.get('message_id') or ''}",
                "",
            ])
        return "\n".join(lines).strip() + "\n"

    def ingest_candidate(self, candidate_id: str, category: str = "", actor: str = "system") -> Dict[str, Any]:
        candidate = self.workspace.get_candidate(candidate_id)
        if not candidate:
            raise ValueError("知识候选不存在")
        if category:
            self.workspace.update_candidate(candidate_id, "category", category=category)
            candidate = self.workspace.get_candidate(candidate_id) or candidate
        failed_images = [
            message for message in candidate.get("messages") or []
            if message.get("message_type") in {"image", "post"}
            and message.get("extraction_status") == "failed"
        ]
        if failed_images:
            raise ValueError("候选中仍有未识别图片，请先在内容中心重新识别")

        active_asset = self.workspace.get_asset(str(candidate.get("asset_id") or "")) if candidate.get("asset_id") else None
        current_source_ids = list(candidate.get("source_message_ids") or [])
        if (
            candidate.get("status") == "published"
            and active_asset
            and active_asset.get("status") == "published"
            and list(active_asset.get("source_message_ids") or []) == current_source_ids
        ):
            return {"success": True, "already_ingested": True, "asset": active_asset, "candidate": candidate}

        group = self.workspace.get_group(str(candidate.get("chat_id") or "")) or {"name": "项目群"}
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        content = self._render_candidate(candidate, group)
        revision = hashlib.sha256(
            (content + "\n" + json.dumps(current_source_ids, ensure_ascii=False)).encode("utf-8")
        ).hexdigest()[:10]
        stored_file = f"feishu_candidate_{candidate_id}_{revision}.md"
        source_file = f"飞书知识_{_safe_filename(group.get('name') or '')}_{candidate_id[-8:]}.md"
        file_path = self.upload_dir / stored_file
        loader, vector_store, project_context, knowledge_graph = self._dependencies()

        previous_asset = active_asset if active_asset and active_asset.get("status") == "published" else None
        previous_source_ids = list(previous_asset.get("source_message_ids") or []) if previous_asset else []
        prepared_asset: Dict[str, Any] = {}
        authoritative_published = False
        try:
            file_path.write_text(content, encoding="utf-8")
            docs = loader(str(file_path))
            category_value = str(candidate.get("category") or group.get("default_category") or "proj_communication")
            categories = [category_value]
            source_ids_json = json.dumps(current_source_ids, ensure_ascii=False)
            for doc in docs:
                doc.metadata.update({
                    "source_file": source_file,
                    "stored_file": stored_file,
                    "categories": categories,
                    "category": category_value,
                    "uploader": "飞书知识自动化",
                    "upload_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "source_type": "feishu_conversation",
                    "feishu_chat_id": str(candidate.get("chat_id") or ""),
                    "feishu_candidate_id": candidate_id,
                    "feishu_message_ids": source_ids_json,
                    "visibility": str(group.get("visibility") or "project"),
                    "access_policy_id": str(group.get("access_policy_id") or ""),
                })
            compatibility_asset_id = str(candidate.get("asset_id") or f"fa_{candidate_id.removeprefix('fc_')}")
            prepared_asset = self._asset_service.prepare_documents(
                docs,
                source_type="feishu_conversation",
                external_key=candidate_id,
                title=source_file,
                actor=actor,
                stored_file=stored_file,
                categories=categories,
                asset_id=compatibility_asset_id,
                visibility=str(group.get("visibility") or "project"),
                access_policy_id=str(group.get("access_policy_id") or ""),
                summary=str(candidate.get("summary") or candidate.get("title") or ""),
                metadata={
                    "feishu_chat_id": str(candidate.get("chat_id") or ""),
                    "feishu_candidate_id": candidate_id,
                    "feishu_message_ids": current_source_ids,
                },
                content_hash_override=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            )
            self._asset_service.repository.add_asset_alias(
                str(prepared_asset["asset_id"]), "feishu_asset", compatibility_asset_id,
            )
            if prepared_asset.get("duplicate") and prepared_asset.get("status") == "active":
                canonical_asset = self._asset_service.get_asset(str(prepared_asset["asset_id"])) or prepared_asset
                asset = {
                    "asset_id": compatibility_asset_id,
                    "canonical_asset_id": str(prepared_asset["asset_id"]),
                    "source_file": source_file,
                    "stored_file": stored_file,
                    "chunks": int((active_asset or {}).get("chunks") or 0),
                    "category": category_value,
                    "published_by": actor,
                }
                published, compatibility_error = self._register_workspace_projection(
                    candidate_id,
                    asset,
                    actor=actor,
                    previous_source_ids=previous_source_ids,
                )
                if compatibility_error:
                    return {
                        "success": True,
                        "partial_success": True,
                        "already_ingested": True,
                        "authoritative_asset": canonical_asset,
                        "asset": canonical_asset,
                        "candidate": self.workspace.get_candidate(candidate_id),
                        "compatibility_registered": False,
                        "repair_required": True,
                        "diagnostic": compatibility_error,
                    }
                return {
                    "success": True, "already_ingested": True,
                    "asset": self.workspace.get_asset(compatibility_asset_id), "candidate": published,
                    "authoritative_asset": canonical_asset,
                    "compatibility_registered": True,
                }
            if prepared_asset.get("duplicate"):
                raise RuntimeError("相同飞书知识版本正在处理中，请稍后重试")
            chunk_count = vector_store.add_documents(docs)
            self._asset_service.record_projection(
                str(prepared_asset["version_id"]), "vector", succeeded=True,
            )
            metadata = project_context.extract_project_metadata(content)
            metadata.update({
                "source_file": source_file,
                "stored_file": stored_file,
                "categories": categories,
                "category": category_value,
                "uploader": "飞书知识自动化",
                "upload_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "chunks": chunk_count,
                "source_type": "feishu_conversation",
                "feishu_chat_id": str(candidate.get("chat_id") or ""),
                "feishu_candidate_id": candidate_id,
                "feishu_message_ids": source_ids_json,
                "visibility": str(group.get("visibility") or "project"),
                "access_policy_id": str(group.get("access_policy_id") or ""),
                "asset_id": prepared_asset["asset_id"],
                "source_id": prepared_asset["source_id"],
                "version_id": prepared_asset["version_id"],
                "status": "preparing",
            })
            context_ready = True
            context_error = ""
            try:
                project_context.register_document(stored_file, metadata)
            except Exception as exc:
                context_ready = False
                context_error = str(exc)
                logger.warning("Feishu context projection failed: %s", exc)
            asset = {
                "asset_id": compatibility_asset_id,
                "canonical_asset_id": str(prepared_asset["asset_id"]),
                "source_file": source_file,
                "stored_file": stored_file,
                "chunks": chunk_count,
                "category": category_value,
                "published_by": actor,
            }
            canonical_asset = self._asset_service.publish(
                prepared_asset,
                actor=actor,
                graph_ready=False,
                context_ready=context_ready,
                context_error=context_error,
            )
            authoritative_published = True
            try:
                from backend.onboarding import OnboardingService
                OnboardingService(lambda: self._asset_service.repository).refresh_active_plans(
                    actor=actor,
                    trigger_asset_id=str(canonical_asset.get("asset_id") or prepared_asset["asset_id"]),
                    trigger="feishu_asset_published",
                )
            except Exception:
                logger.exception("Feishu knowledge onboarding reconciliation failed: %s", candidate_id)
            published, compatibility_error = self._register_workspace_projection(
                candidate_id,
                asset,
                actor=actor,
                previous_source_ids=previous_source_ids,
            )
            if compatibility_error:
                return {
                    "success": True,
                    "partial_success": True,
                    "already_ingested": False,
                    "authoritative_asset": canonical_asset,
                    "asset": canonical_asset,
                    "candidate": self.workspace.get_candidate(candidate_id),
                    "compatibility_registered": False,
                    "repair_required": True,
                    "diagnostic": compatibility_error,
                }
        except Exception as exc:
            if authoritative_published:
                raise RuntimeError("统一资产已发布，兼容工作台待修复") from exc
            if prepared_asset:
                self._asset_service.fail(prepared_asset, exc, actor=actor)
            # 新版本始终使用独立 stored_file，因此补偿删除不会触碰仍在服务的旧版本。
            try:
                vector_store.delete_documents_by_source(stored_file)
            except Exception:
                pass
            try:
                project_context.unregister_document(stored_file)
            except Exception:
                pass
            file_path.unlink(missing_ok=True)
            self.workspace.update_candidate(
                candidate_id,
                "failed",
                error=str(exc),
                preserve_published_ids=previous_source_ids,
            )
            raise

        cleanup_errors = []
        try:
            knowledge_graph.build_graph()
            self._asset_service.mark_post_publish_projection(
                str(prepared_asset["version_id"]), "graph", True,
            )
        except Exception as exc:
            cleanup_errors.append(f"图谱刷新失败: {exc}")
            self._asset_service.mark_post_publish_projection(
                str(prepared_asset["version_id"]), "graph", False, str(exc),
            )
        if cleanup_errors:
            self.workspace.record_audit(
                "asset_cleanup_warning",
                "asset",
                asset["asset_id"],
                "；".join(cleanup_errors),
                actor=actor,
            )
        return {
            "success": True,
            "already_ingested": False,
            "asset": self.workspace.get_asset(asset["asset_id"]),
            "authoritative_asset": canonical_asset,
            "candidate": published,
            "compatibility_registered": True,
            "cleanup_warnings": cleanup_errors,
        }

    def publish_ready_candidates(self, candidate_ids: Optional[Iterable[str]] = None) -> Dict[str, Any]:
        selected_ids = set(candidate_ids or [])
        ready = self.workspace.list_candidates(status="ready_for_auto", limit=500)
        if selected_ids:
            ready = [item for item in ready if item.get("candidate_id") in selected_ids]
        results = {"published": [], "partial": [], "failed": []}
        for candidate in ready:
            candidate_id = str(candidate.get("candidate_id") or "")
            try:
                result = self.ingest_candidate(candidate_id)
                target = "partial" if result.get("partial_success") else "published"
                results[target].append({
                    "candidate_id": candidate_id,
                    "asset": result.get("asset"),
                    "diagnostic": result.get("diagnostic", ""),
                })
            except Exception as exc:
                results["failed"].append({"candidate_id": candidate_id, "error": str(exc)[:300]})
        return results

    def batch_action(
        self,
        candidate_ids: Iterable[str],
        action: str,
        category: str = "",
        actor: str = "管理员",
    ) -> Dict[str, Any]:
        """批量处理候选并返回逐项结果，供 API 与测试复用。"""
        selected_ids = list(dict.fromkeys(str(item).strip() for item in candidate_ids if str(item).strip()))
        if not selected_ids:
            raise ValueError("请至少选择一个知识候选")
        if action not in {"approve", "exclude", "category", "retry"}:
            raise ValueError("不支持的批量操作")
        if action == "category" and not category:
            raise ValueError("请选择要应用的知识分类")

        succeeded = []
        failed = []
        for candidate_id in selected_ids:
            try:
                if not self.workspace.get_candidate(candidate_id):
                    raise ValueError("知识候选不存在")
                if action == "exclude":
                    result = self.workspace.update_candidate(candidate_id, "exclude")
                elif action == "category":
                    result = self.workspace.update_candidate(candidate_id, "category", category=category)
                else:
                    if action == "retry":
                        self.workspace.update_candidate(candidate_id, "retry")
                    result = self.ingest_candidate(candidate_id, category=category, actor=actor)
                succeeded.append({"candidate_id": candidate_id, "result": result})
            except Exception as exc:
                failed.append({"candidate_id": candidate_id, "error": str(exc)[:300]})
        return {
            "success": not failed,
            "processed": len(selected_ids),
            "succeeded": succeeded,
            "failed": failed,
            "message": f"已处理 {len(succeeded)} 项" + (f"，{len(failed)} 项失败" if failed else ""),
        }

    def resolve_workspace_asset_id(self, asset_id: str) -> str:
        """Resolve either a legacy Feishu id or a unified id to the workspace record."""
        value = str(asset_id or "").strip()
        if not value:
            return ""
        if self.workspace.get_asset(value):
            return value
        resolved = self._asset_service.repository.resolve_asset_id(value)
        if not resolved:
            return ""
        for item in self.workspace.list_assets():
            candidate = str(item.get("asset_id") or "")
            if candidate and self._asset_service.repository.resolve_asset_id(candidate) == resolved:
                return candidate
        return ""

    def revert_asset(self, asset_id: str, reason: str = "", actor: str = "system") -> Dict[str, Any]:
        reason = str(reason or "").strip()
        if not reason:
            raise ValueError("撤销原因不能为空")
        workspace_asset_id = self.resolve_workspace_asset_id(asset_id)
        asset = self.workspace.get_asset(workspace_asset_id)
        if not asset:
            raise ValueError("知识资产不存在")
        _, vector_store, project_context, knowledge_graph = self._dependencies()
        unified_asset_id = self._asset_service.repository.resolve_asset_id(workspace_asset_id)
        authoritative_asset = None
        if unified_asset_id:
            # Fail closed before any best-effort derived cleanup or legacy UI update.
            authoritative_asset = self._asset_service.transition(
                unified_asset_id, "revoke", actor, reason=reason,
            )
        authoritative = self.workspace.revert_asset_authoritatively(
            workspace_asset_id, reason=reason, actor=actor,
        )
        if not authoritative:
            raise ValueError("知识资产不存在")
        record = dict(authoritative.get("asset") or asset)
        source_file = str(authoritative.get("source_file") or record.get("source_file") or "")
        stored_file = str(authoritative.get("stored_file") or record.get("stored_file") or "")
        deleted_chunks = int(authoritative.get("deleted_chunks") or 0)

        sync_warnings = []
        try:
            vector_store.reconcile_with_storage()
        except Exception as exc:
            sync_warnings.append("知识检索索引同步待重试")
            logger.warning("vector index refresh after asset revert failed: %s", exc)
        try:
            if hasattr(project_context, "reload"):
                project_context.reload()
            else:
                project_context.unregister_document(stored_file or source_file)
        except Exception as exc:
            sync_warnings.append("项目语境缓存同步待重试")
            logger.warning("project context refresh after asset revert failed: %s", exc)
        try:
            knowledge_graph.build_graph()
        except Exception as exc:
            sync_warnings.append("知识图谱同步待重试")
        if sync_warnings:
            try:
                self.workspace.record_audit(
                    "asset_revert_cleanup_warning", "asset", workspace_asset_id,
                    "；".join(sync_warnings), actor=actor,
                )
            except Exception:
                logger.exception("record asset revert cleanup warning failed: %s", asset_id)
        projection_status = "repair_required" if sync_warnings else "healthy"
        diagnostic_id = (
            hashlib.sha256(f"{asset_id}:{datetime.now().isoformat()}".encode("utf-8")).hexdigest()[:10]
            if sync_warnings else ""
        )
        try:
            record = self.workspace.update_asset_projection(workspace_asset_id, projection_status, sync_warnings)
        except Exception:
            logger.exception("persist asset projection status failed: %s", asset_id)
            sync_warnings.append("派生状态记录待修复")
            projection_status = "repair_required"
        try:
            from backend.onboarding import OnboardingService
            OnboardingService(lambda: self._asset_service.repository).refresh_active_plans(
                actor=actor, trigger_asset_id=str(unified_asset_id or asset_id),
                trigger="feishu_asset_revoked",
            )
        except Exception:
            logger.exception("Feishu knowledge onboarding reconciliation failed after revert: %s", asset_id)
        return {
            "success": True,
            "partial_success": bool(sync_warnings),
            "already_reverted": bool(authoritative.get("already_reverted")),
            "deleted_chunks": deleted_chunks,
            "deleted_documents": int(authoritative.get("deleted_documents") or 0),
            "archived_documents": int(authoritative.get("archived_documents") or 0),
            "vector_synced": "知识检索索引同步待重试" not in sync_warnings,
            "graph_synced": "知识图谱同步待重试" not in sync_warnings,
            "projection_status": projection_status,
            "diagnostic_id": diagnostic_id,
            "repair_action": "再次执行撤销以重试派生同步" if sync_warnings else "",
            "sync_warnings": sync_warnings,
            "asset": record,
            "knowledge_asset": authoritative_asset,
        }

    def batch_revert(
        self, asset_ids: Iterable[str], reason: str = "", actor: str = "system",
    ) -> Dict[str, Any]:
        reason = str(reason or "").strip()
        if not reason:
            raise ValueError("撤销原因不能为空")
        selected_ids = list(dict.fromkeys(str(item).strip() for item in asset_ids if str(item).strip()))
        if not selected_ids:
            raise ValueError("请至少选择一个知识资产")
        succeeded = []
        failed = []
        for asset_id in selected_ids:
            try:
                succeeded.append({
                    "asset_id": asset_id,
                    "result": self.revert_asset(asset_id, reason=reason, actor=actor),
                })
            except Exception as exc:
                failed.append({"asset_id": asset_id, "error": str(exc)[:300]})
        pending_sync = sum(
            1 for item in succeeded if item.get("result", {}).get("partial_success")
        )
        return {
            "success": not failed,
            "processed": len(selected_ids),
            "succeeded": succeeded,
            "failed": failed,
            "message": (
                f"已撤销 {len(succeeded)} 项"
                + (f"，{pending_sync} 项派生数据待修复" if pending_sync else "")
                + (f"，{len(failed)} 项失败" if failed else "")
            ),
        }


feishu_knowledge_service = FeishuKnowledgeService()
