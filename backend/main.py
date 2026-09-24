"""
FastAPI 主入口 —— AI 赋能项目知识沉淀系统

包含：
- 文档管理（上传/删除/列表/预览/按分类）
- 分类管理（CRUD）
- 向量检索 + RAG 对话
- 项目知识语境管理
- 知识关系图谱
- 飞书集成 Webhook
- 离职交接（多文件上传）
"""
import os
import asyncio
import uuid
import json
import logging
import csv
import io
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Body
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.knowledge_base.document_loader import load_document
from backend.knowledge_base.vector_store import vector_store
from backend.knowledge_base.llm_service import llm_service
from backend.knowledge_base.project_context import KnowledgeCategory, project_context
from backend.knowledge_base.knowledge_graph import knowledge_graph
from backend.feishu_workspace import feishu_workspace
from backend.feishu_knowledge import feishu_knowledge_service
from backend.feishu_bot import feishu_bot
from backend.document_ocr import (
    DocumentOCRFailure,
    IMAGE_DOCUMENT_EXTENSIONS,
    prepare_document_for_ingestion,
)
from backend.knowledge_assets import ProjectionRepairBusyError, knowledge_asset_service
from backend.knowledge_tasks import knowledge_task_reminder_scheduler, knowledge_task_service
from backend.rag_quality import NO_ANSWER_TEXT, rag_quality_service
from backend.data_governance import data_governance_scheduler, data_governance_service
from backend.observability import (
    RuntimeObservabilityMiddleware,
    observability_service,
    runtime_metrics,
)
from backend.onboarding import OnboardingService, onboarding_service
from backend.handover import handover_service, summarize_handover
from backend.processing_jobs import (
    ProcessingJobCancelled,
    ProcessingJobError,
    processing_job_runner,
    processing_job_service,
)
from backend.feishu_connection import feishu_long_connection
from backend.storage import get_repository, initialize_storage, release_runtime_lock
from backend.storage.legacy import LegacyMigrationError
from backend.auth import (
    AuthenticationError,
    CSRF_COOKIE,
    ROLE_LABELS,
    ROLE_PERMISSIONS,
    SESSION_COOKIE,
    auth_service,
)
from backend.auth_middleware import authentication_middleware
from backend.auth_context import Identity, get_current_identity, reset_current_identity, set_current_identity

logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI 赋能项目知识沉淀系统",
    version="2.1.0",
    description="文档知识库 + 智能对话 + 知识图谱 + 飞书集成 + 离职交接",
)

# CORS —— 允许前端跨域访问
cors_origins = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]
if settings.FRONTEND_URL:
    cors_origins.append(settings.FRONTEND_URL.rstrip("/"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.middleware("http")(authentication_middleware)
app.add_middleware(RuntimeObservabilityMiddleware)


def _job_graph_rebuild(context, payload):
    context.report("graph_build", 20, "正在读取有效知识资产")
    result = knowledge_graph.build_graph()
    context.report("graph_save", 85, "正在保存知识关系投影")
    return result if isinstance(result, dict) else {"message": "知识图谱已重建"}


def _job_projection_repair(context, payload):
    asset_id = str(payload.get("asset_id") or "")
    context.report("projection_scan", 15, "正在检查知识投影")
    try:
        result = knowledge_asset_service.repair_projections(
            asset_id or None, actor=str(payload.get("actor") or "processing_job"),
            force=bool(payload.get("force", False)),
        )
    except ProjectionRepairBusyError as exc:
        raise ProcessingJobError(str(exc), code="PROJECTION_BUSY", retryable=True) from exc
    if not result.get("success"):
        raise ProcessingJobError("部分知识投影修复失败", code="PROJECTION_PARTIAL_FAILURE", retryable=True)
    result["onboarding_sync"] = _refresh_onboarding_after_knowledge_change(
        asset_id=asset_id,
        actor=str(payload.get("actor") or "processing_job"),
        trigger="projection_repaired",
    )
    context.report("projection_save", 90, "正在确认投影状态")
    return result


async def _job_feishu_group_sync(context, payload):
    context.report("feishu_download", 10, "正在读取飞书群历史消息")
    return await _sync_feishu_group_history(
        str(payload.get("chat_id") or ""), int(payload.get("page_size") or 50), context=context,
    )


async def _job_knowledge_task_notification(context, payload):
    context.report("notification_validate", 15, "正在复核提醒策略与任务状态")
    result = await knowledge_task_service.deliver_scheduled_notifications(payload)
    context.report("notification_deliver", 90, str(result.get("message") or "提醒投递已完成"))
    return result


def _job_capacity_validation(context, payload):
    return observability_service.run_capacity_validation(context, payload)


def _job_recovery_drill(context, payload):
    return observability_service.run_recovery_drill(context, payload)


def _processing_identity(
    user_id: str, required_permission: str = "knowledge.upload", code_prefix: str = "UPLOAD",
) -> Identity:
    member = get_repository().get_user(str(user_id), project_id=get_repository().project_id)
    if not member or member.get("status") != "active" or member.get("membership_status") != "active":
        raise ProcessingJobError(
            "发起人已不是当前项目的有效成员", code=f"{code_prefix}_ACTOR_INACTIVE", retryable=False,
        )
    role = str(member.get("role") or "")
    permissions = ROLE_PERMISSIONS.get(role, frozenset())
    if "*" not in permissions and required_permission not in permissions:
        raise ProcessingJobError(
            "发起人当前无权执行该处理任务", code=f"{code_prefix}_PERMISSION_REVOKED", retryable=False,
        )
    return Identity(
        user_id=str(member["user_id"]), email=str(member.get("email") or ""),
        display_name=str(member.get("display_name") or ""), organization_id="",
        project_id=str(member.get("project_id") or get_repository().project_id),
        project_name=str(member.get("project_name") or ""), role=role, permissions=permissions,
    )


async def _job_document_ingestion(context, payload):
    staging_name = str(payload.get("staging_name") or "")
    if not staging_name or Path(staging_name).name != staging_name:
        raise ProcessingJobError("上传暂存标识无效", code="UPLOAD_STAGING_INVALID", retryable=False)
    staging_path = (UPLOAD_STAGING_DIR / staging_name).resolve()
    if staging_path.parent != UPLOAD_STAGING_DIR.resolve() or not staging_path.is_file():
        raise ProcessingJobError("上传暂存文件不存在", code="UPLOAD_STAGING_MISSING", retryable=False)
    identity = _processing_identity(str(payload.get("actor_user_id") or ""))
    context.report("document_validate", 10, "正在复核上传者和目标资产权限")
    token = set_current_identity(identity)
    try:
        original_filename = str(payload.get("original_filename") or "document")
        try:
            prepared_documents, extraction = await prepare_document_for_ingestion(
                str(staging_path), original_filename, report=context.report,
                max_ocr_pages=settings.MAX_DOCUMENT_OCR_PAGES,
            )
        except DocumentOCRFailure as exc:
            raise ProcessingJobError(str(exc), code=exc.code, retryable=exc.retryable) from exc
        with staging_path.open("rb") as handle:
            uploaded = UploadFile(file=handle, filename=original_filename)
            context.report("document_ingest", 55, "正在切分、索引并发布知识资产")
            result = await _upload_document_impl(
                uploaded, category=str(payload.get("category") or "") or None,
                uploader=identity.display_name,
                asset_id=str(payload.get("asset_id") or "") or None,
                source_id=str(payload.get("source_id") or "") or None,
                prepared_documents=prepared_documents, extraction_summary=extraction,
            )
        staging_path.unlink(missing_ok=True)
        result["_job_commit_on_cancel"] = True
        return result
    finally:
        reset_current_identity(token)


def _cleanup_document_ingestion(job):
    staging_name = str((job.get("payload") or {}).get("staging_name") or "")
    if not staging_name or Path(staging_name).name != staging_name:
        return
    staging_path = (UPLOAD_STAGING_DIR / staging_name).resolve()
    if staging_path.parent == UPLOAD_STAGING_DIR.resolve():
        staging_path.unlink(missing_ok=True)


def _feishu_batch_result(action, selected_ids, succeeded, failed):
    succeeded_by_id = {str(item.get("candidate_id") or ""): item for item in succeeded}
    failed_by_id = {str(item.get("candidate_id") or ""): item for item in failed}
    items = []
    for candidate_id in selected_ids:
        if candidate_id in succeeded_by_id:
            items.append({"candidate_id": candidate_id, "status": "succeeded"})
        elif candidate_id in failed_by_id:
            items.append({
                "candidate_id": candidate_id, "status": "failed",
                "error": failed_by_id[candidate_id].get("error", ""),
            })
    return {
        "success": not failed,
        "partial_success": bool(succeeded and failed),
        "action": action,
        "processed": len(succeeded) + len(failed),
        "total": len(selected_ids),
        "succeeded_count": len(succeeded),
        "failed_count": len(failed),
        "succeeded": succeeded,
        "failed": failed,
        "items": items,
        "message": f"已处理 {len(succeeded)} 项" + (f"，{len(failed)} 项失败" if failed else ""),
    }


def _job_feishu_candidate_batch(context, payload):
    selected_ids = list(dict.fromkeys(
        str(item).strip() for item in (payload.get("candidate_ids") or []) if str(item).strip()
    ))
    action = str(payload.get("action") or "").strip()
    category = str(payload.get("category") or "").strip()
    if not selected_ids or len(selected_ids) > 100:
        raise ProcessingJobError("批量候选数量无效", code="FEISHU_BATCH_INVALID", retryable=False)
    if action not in {"approve", "exclude", "category", "retry"}:
        raise ProcessingJobError("不支持的批量操作", code="FEISHU_BATCH_INVALID", retryable=False)
    if action == "category" and not category:
        raise ProcessingJobError("请选择要应用的知识分类", code="FEISHU_BATCH_INVALID", retryable=False)

    identity = _processing_identity(
        str(payload.get("actor_user_id") or ""), "knowledge.publish", "FEISHU_BATCH",
    )
    succeeded, failed = [], []
    token = set_current_identity(identity)
    try:
        context.report("feishu_batch_validate", 5, "正在复核批量操作权限和候选范围")
        for index, candidate_id in enumerate(selected_ids):
            try:
                context.report(
                    "feishu_batch_item", 10 + int(index * 80 / max(1, len(selected_ids))),
                    f"正在处理第 {index + 1}/{len(selected_ids)} 项",
                )
                result = feishu_knowledge_service.batch_action(
                    [candidate_id], action=action, category=category, actor=identity.user_id,
                )
                if result.get("failed"):
                    failed.extend(result["failed"])
                else:
                    succeeded.extend(result.get("succeeded") or [])
            except ProcessingJobCancelled:
                raise
            except Exception as exc:
                failed.append({"candidate_id": candidate_id, "error": str(exc)[:300]})
        context.report("feishu_batch_finalize", 95, "正在汇总逐项处理结果")
        return _feishu_batch_result(action, selected_ids, succeeded, failed)
    except ProcessingJobCancelled as exc:
        raise ProcessingJobCancelled(
            _feishu_batch_result(action, selected_ids, succeeded, failed)
        ) from exc
    finally:
        reset_current_identity(token)


def _feishu_enrichment_result(record, published=0, publish_failures=0):
    return {
        "success": True,
        "message_id": str((record or {}).get("message_id") or ""),
        "chat_id": str((record or {}).get("chat_id") or ""),
        "message_type": str((record or {}).get("message_type") or ""),
        "extraction_status": str((record or {}).get("extraction_status") or ""),
        "extraction_method": str((record or {}).get("extraction_method") or ""),
        "resource_count": int((record or {}).get("resource_count") or 0),
        "content_status": str((record or {}).get("content_status") or ""),
        "candidate_id": str((record or {}).get("candidate_id") or ""),
        "published": int(published or 0),
        "publish_failures": int(publish_failures or 0),
    }


async def _job_feishu_message_enrichment(context, payload):
    message_id = str(payload.get("message_id") or "").strip()
    trigger = str(payload.get("trigger") or "manual").strip()
    if not message_id or trigger not in {"manual", "webhook"}:
        raise ProcessingJobError("飞书识别任务输入无效", code="FEISHU_ENRICHMENT_INVALID", retryable=False)

    token = None
    if trigger == "manual":
        identity = _processing_identity(
            str(payload.get("actor_user_id") or ""), "knowledge.publish", "FEISHU_ENRICHMENT",
        )
        token = set_current_identity(identity)
    try:
        context.report("feishu_enrich_validate", 10, "正在复核消息、采集源和识别权限")
        message = feishu_workspace.get_message(message_id)
        if not message:
            raise ProcessingJobError("消息记录不存在", code="FEISHU_MESSAGE_NOT_FOUND", retryable=False)
        if message.get("message_type") not in {"image", "post"}:
            raise ProcessingJobError("该消息不需要图片识别", code="FEISHU_ENRICHMENT_NOT_REQUIRED", retryable=False)
        chat_id = str(message.get("chat_id") or "")
        group = feishu_workspace.get_group(chat_id) or {}
        if not group or group.get("collection_mode") == "off":
            raise ProcessingJobError("采集源不存在或已关闭", code="FEISHU_COLLECTION_OFF", retryable=False)

        context.report("feishu_enrich_extract", 25, "正在下载资源并执行图片识别回退链")
        raw_message = await feishu_bot.retry_enrich_message(message)
        context.report("feishu_enrich_archive", 70, "正在保存识别结果并重新评估知识候选")
        updated = feishu_workspace.replace_enriched_message(chat_id, raw_message)
        if not updated:
            raise ProcessingJobError("消息识别结果保存失败", code="FEISHU_ENRICHMENT_SAVE_FAILED")
        if updated.get("extraction_status") == "failed":
            raise ProcessingJobError(
                str(updated.get("extraction_error") or "图片识别仍未成功"),
                code="FEISHU_ENRICHMENT_FAILED", retryable=True,
            )

        context.report("feishu_enrich_publish", 85, "正在按当前采集规则同步知识候选")
        published = publish_failures = 0
        if group.get("collection_mode") == "auto":
            ready_ids = [
                item["candidate_id"]
                for item in feishu_workspace.list_candidates(status="ready_for_auto", chat_id=chat_id)
            ]
            publish_result = feishu_knowledge_service.publish_ready_candidates(ready_ids)
            published = len(publish_result.get("published") or [])
            publish_failures = len(publish_result.get("failed") or [])
        result = _feishu_enrichment_result(updated, published, publish_failures)
        result["message"] = "图片与富文本内容已完成后台识别" + (f"，自动入库 {published} 项" if published else "")
        result["_job_commit_on_cancel"] = True
        return result
    except ProcessingJobCancelled as exc:
        current = feishu_workspace.get_message(message_id)
        raise ProcessingJobCancelled(
            _feishu_enrichment_result(current) if current else exc.result
        ) from exc
    finally:
        if token is not None:
            reset_current_identity(token)


processing_job_service.register("graph_rebuild", _job_graph_rebuild)
processing_job_service.register("projection_repair", _job_projection_repair)
processing_job_service.register("feishu_group_sync", _job_feishu_group_sync)
processing_job_service.register("database_backup", data_governance_service.create_backup)
processing_job_service.register("data_retention", data_governance_service.apply_retention)
processing_job_service.register("targeted_data_deletion", data_governance_service.apply_targeted_deletion)
processing_job_service.register("capacity_validation", _job_capacity_validation)
processing_job_service.register("recovery_drill", _job_recovery_drill)
processing_job_service.register("document_ingestion", _job_document_ingestion)
processing_job_service.register("feishu_candidate_batch", _job_feishu_candidate_batch)
processing_job_service.register("feishu_message_enrichment", _job_feishu_message_enrichment)
processing_job_service.register("knowledge_task_notification", _job_knowledge_task_notification)
processing_job_service.register_cleanup("document_ingestion", _cleanup_document_ingestion)
knowledge_task_reminder_scheduler.wake_callback = processing_job_runner.wake
data_governance_scheduler.wake_callback = processing_job_runner.wake


@app.on_event("startup")
async def start_feishu_event_connection():
    """Initialize durable data before accepting events, then start Feishu."""
    _load_saved_config()
    try:
        storage_result = initialize_storage(run_legacy_migration=settings.STORAGE_AUTO_MIGRATE)
        print(f"[存储] SQLite 已就绪: {storage_result['database']}")
        demo_accounts = auth_service.ensure_demo_accounts()
        if settings.DEMO_SIMPLE_AUTH:
            print(f"[认证] 演示模式已启用，本次初始化 {demo_accounts} 个成员账号")
        onboarding_sync = _refresh_onboarding_after_knowledge_change(
            actor="user_system", trigger="startup_reconciliation",
        )
        if onboarding_sync.get("changed_plan_count"):
            print(
                "[新人赋能] 已恢复学习资料: "
                f"plans={onboarding_sync['changed_plan_count']}, "
                f"items={onboarding_sync['unblocked_count']}"
            )
        recovered = await processing_job_runner.start()
        await knowledge_task_reminder_scheduler.start()
        await data_governance_scheduler.start()
        runtime_metrics.start()
        if observability_service.should_schedule_recovery_drill():
            current_month = datetime.now().strftime("%Y-%m")
            processing_job_service.enqueue(
                "recovery_drill", {"actor_user_id": "user_system", "trigger": "monthly_startup"},
                created_by="user_system", idempotency_key=f"recovery-drill:{current_month}", priority=30,
            )
        pending_projections = get_repository().count_current_projection_targets()
        vector_consistency = vector_store.consistency_status()
        graph_snapshot = knowledge_graph.get_graph_data()
        graph_missing = bool(
            vector_consistency.get("required_chunks", 0)
            and not graph_snapshot.get("nodes")
        )
        force_projection_repair = not vector_consistency.get("consistent", False) or graph_missing
        if pending_projections or force_projection_repair:
            processing_job_service.enqueue(
                "projection_repair", {
                    "actor": "startup", "force": force_projection_repair,
                    "reason": "derived_index_inconsistent" if force_projection_repair else "pending_projection",
                },
                created_by="user_system", priority=70,
            )
            print(
                "[投影] 已安排启动修复: "
                f"pending={pending_projections}, vector_consistent={vector_consistency.get('consistent')}, "
                f"graph_missing={graph_missing}"
            )
        print(f"[任务] 处理执行器已启动，恢复 {recovered} 个中断任务")
        if settings.FEISHU_APP_ID and settings.FEISHU_APP_SECRET:
            feishu_long_connection.start(asyncio.get_running_loop())
    except LegacyMigrationError as exc:
        # Fail closed: running against a partially imported database is unsafe.
        release_runtime_lock()
        raise RuntimeError(str(exc)) from exc
    except Exception:
        release_runtime_lock()
        raise


@app.on_event("shutdown")
async def stop_feishu_event_connection():
    feishu_long_connection.stop()
    await runtime_metrics.stop()
    await data_governance_scheduler.stop()
    await knowledge_task_reminder_scheduler.stop()
    await processing_job_runner.stop()
    release_runtime_lock()

# 确保上传目录存在
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
UPLOAD_STAGING_DIR = UPLOAD_DIR / ".staging"
UPLOAD_STAGING_DIR.mkdir(parents=True, exist_ok=True)
try:
    os.chmod(UPLOAD_STAGING_DIR, 0o700)
except OSError:
    pass

DEFAULT_CATEGORIES = ["交接文档", "新人培训"]

CATEGORY_LABELS = {
    "eef_culture": "组织文化与结构",
    "eef_standard": "行业标准与法规",
    "eef_market": "市场环境",
    "eef_infrastructure": "基础设施",
    "opa_process": "流程与规范",
    "opa_template": "模板与工具",
    "opa_lessons": "经验教训库",
    "opa_training": "培训材料",
    "opa_historical": "历史数据",
    "proj_plan": "项目管理计划",
    "proj_tailoring": "裁剪指南",
    "proj_decision": "决策记录",
    "proj_risk": "风险日志",
    "proj_report": "报告与状态",
    "proj_requirement": "需求文档",
    "proj_architecture": "技术架构",
    "proj_communication": "沟通记录",
    "proj_acceptance": "验收标准",
    "交接文档": "交接文档",
    "新人培训": "新人培训",
}

# ── 配置持久化 ──
CONFIG_FILE = Path("config.json")
USER_CONFIG_KEYS = (
    "OPENAI_API_KEY", "OPENAI_BASE_URL", "LLM_MODEL",
    "HTTP_PROXY_ENABLED", "HTTP_PROXY_URL", "HTTP_VERIFY_SSL",
    "FEISHU_APP_ID", "FEISHU_APP_SECRET", "FEISHU_WEBHOOK_SECRET", "FEISHU_VERIFICATION_TOKEN",
    "FEISHU_DEFAULT_CHAT_ID",
)

def _load_saved_config():
    """从 config.json 恢复用户保存的配置"""
    if not CONFIG_FILE.exists():
        return
    try:
        data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        # Database and filesystem paths are deployment settings. Keeping them out
        # of the editable UI config prevents a model-setting save from switching data stores.
        for key, value in data.items():
            if key not in USER_CONFIG_KEYS:
                continue
            if hasattr(settings, key) and value is not None:
                setattr(settings, key, value)
        print(f"[配置] 已从 {CONFIG_FILE} 恢复配置")
    except Exception as e:
        print(f"[配置] 读取 config.json 失败: {e}")

def _save_config():
    """将当前可持久化的配置写入 config.json"""
    data = {key: getattr(settings, key, None) for key in USER_CONFIG_KEYS}
    try:
        CONFIG_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        CONFIG_FILE.chmod(0o600)
    except Exception as e:
        print(f"[配置] 写入 config.json 失败: {e}")


def _safe_llm_error(error: Exception) -> str:
    """返回可展示的模型连接错误，避免把敏感信息带到前端。"""
    text = str(error)
    if settings.OPENAI_API_KEY:
        text = text.replace(settings.OPENAI_API_KEY, "***")
    return text[:500]


def _internal_http_error(action: str, user_message: str) -> HTTPException:
    """Log internal details and return a user-safe error with a support identifier."""
    diagnostic_id = uuid.uuid4().hex[:10]
    logger.exception("%s [%s]", action, diagnostic_id)
    return HTTPException(
        status_code=500,
        detail=f"{user_message}（诊断编号：{diagnostic_id}）",
    )


def _sync_project_context(strict: bool = False):
    """Refresh the in-process context cache from authoritative SQLite data."""
    try:
        project_context.reload()
        return len(project_context.list_documents())
    except Exception as exc:
        print(f"[项目语境] 恢复失败: {exc}")
        if strict:
            raise
        return 0


def _load_handover_records() -> list:
    return get_repository().list_handovers()


def _save_handover_records(records: list):
    get_repository().save_handovers(records)


def _handover_risks(record: dict) -> list:
    return list(summarize_handover(record).get("risks") or [])


def _handover_view(record: dict) -> dict:
    return handover_service.view(record, get_current_identity())


def _onboarding_guide(role: str, user_id: str = "") -> dict:
    identity = get_current_identity()
    if identity is None:
        # Internal workflows and legacy contract tests do not run inside an HTTP
        # request. Give those deterministic system calls an explicit identity;
        # user-facing routes are still protected by authentication middleware.
        repository = get_repository()
        identity = Identity(
            user_id="user_system", email="", display_name="系统",
            organization_id="org_default", project_id=repository.project_id,
            project_name=repository.project_name(), role="project_admin",
            permissions=frozenset({"*"}), is_system=True,
        )
        service = OnboardingService(lambda: repository)
    else:
        service = onboarding_service
    try:
        return service.guide(role, identity, user_id=user_id)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _refresh_onboarding_after_knowledge_change(
    *, asset_id: str = "", actor: str = "user_system", trigger: str = "asset_published",
) -> Dict[str, Any]:
    """Keep plans and their knowledge-gap tasks aligned after knowledge changes."""
    try:
        result = onboarding_service.refresh_active_plans(
            actor=actor, trigger_asset_id=asset_id, trigger=trigger,
        )
        result["knowledge_task_sync"] = knowledge_task_service.reconcile_onboarding_gap_tasks(
            actor=actor, trigger_asset_id=asset_id, trigger=trigger,
        )
        result["success"] = bool(result.get("success")) and bool(
            result["knowledge_task_sync"].get("success")
        )
        return result
    except Exception as exc:
        logger.exception("新人资料与知识缺口任务自动匹配失败: %s", asset_id or trigger)
        return {
            "success": False, "plan_count": 0, "changed_plan_count": 0,
            "unblocked_count": 0, "trigger_asset_id": asset_id,
            "results": [], "errors": [{"plan_id": "", "error": str(exc)[:300]}],
            "knowledge_task_sync": {
                "success": False, "checked_count": 0, "matched_count": 0,
                "trigger_asset_id": asset_id, "matched": [],
                "errors": [{"task_id": "", "error": str(exc)[:300]}],
            },
        }

# ── 工具函数 ──

def _load_categories():
    """从 SQLite 加载分类列表，空库使用最小默认分类。"""
    repository = get_repository()
    categories = repository.list_categories()
    return categories or repository.replace_categories(DEFAULT_CATEGORIES)

def _sanitize_upload_filename(filename: str) -> str:
    """Return a basename-only filename so uploaded names cannot create paths."""
    cleaned = Path((filename or "document").replace("\\", "/")).name.strip()
    return cleaned or "document"

def _matches_uploaded_file(stored_name: str, requested_name: str) -> bool:
    requested_name = _sanitize_upload_filename(requested_name)
    return stored_name == requested_name or stored_name.endswith(f"_{requested_name}")


_PLAIN_TEXT_PREVIEW_EXTENSIONS = {".txt", ".md", ".json", ".csv"}
_PARSED_PREVIEW_EXTENSIONS = {".pdf", ".docx"}


def _read_document_preview_text(file_path: Path) -> str:
    """Return readable preview text; never surface raw binary bytes as text."""
    ext = file_path.suffix.lower()
    if ext in _PLAIN_TEXT_PREVIEW_EXTENSIONS:
        return file_path.read_text("utf-8", errors="ignore")
    if ext in _PARSED_PREVIEW_EXTENSIONS:
        documents = load_document(str(file_path))
        return "\n\n".join(str(document.page_content or "") for document in documents).strip()
    raise ValueError(f"暂不支持 .{ext.lstrip('.') or '未知'} 格式的文本预览")

# ═══════════════════════════════════════════════
#  0. 身份认证与项目成员
# ═══════════════════════════════════════════════


@app.get("/api/health/live")
async def liveness():
    """Unauthenticated process liveness without configuration or business data."""
    return {"status": "ok"}


@app.get("/api/auth/config")
async def auth_config():
    """Expose only the login UX mode; never return credentials or secrets."""
    return {
        "success": True,
        "demo_simple_auth": settings.DEMO_SIMPLE_AUTH,
        "password_rule": "email_prefix" if settings.DEMO_SIMPLE_AUTH else "invitation_strong_password",
    }


@app.post("/api/auth/activate")
async def activate_user(payload: Dict[str, Any] = Body(...)):
    try:
        user = auth_service.activate(str(payload.get("token") or ""), str(payload.get("password") or ""))
        return {"success": True, "message": "账号已激活，请登录", "user": user}
    except AuthenticationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/auth/login")
async def login_user(request: Request, payload: Dict[str, Any] = Body(...)):
    try:
        result = auth_service.login(
            str(payload.get("email") or ""),
            str(payload.get("password") or ""),
            user_agent=request.headers.get("User-Agent", ""),
            remote_address=request.client.host if request.client else "",
        )
    except AuthenticationError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    response = JSONResponse({
        "success": True,
        "message": "登录成功",
        "user": auth_service.identity_payload(result["identity"]),
    })
    secure = request.url.scheme == "https"
    response.set_cookie(
        SESSION_COOKIE, result["session_token"], max_age=12 * 60 * 60,
        httponly=True, secure=secure, samesite="strict", path="/",
    )
    response.set_cookie(
        CSRF_COOKIE, result["csrf_token"], max_age=12 * 60 * 60,
        httponly=False, secure=secure, samesite="strict", path="/",
    )
    response.headers["Cache-Control"] = "no-store"
    return response


@app.post("/api/auth/logout")
async def logout_user(request: Request):
    auth_service.logout(request.cookies.get(SESSION_COOKIE, ""))
    response = JSONResponse({"success": True, "message": "已安全退出"})
    response.delete_cookie(SESSION_COOKIE, path="/")
    response.delete_cookie(CSRF_COOKIE, path="/")
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/api/auth/me")
async def current_user(request: Request):
    return {"success": True, "user": auth_service.identity_payload(request.state.identity)}


@app.get("/api/auth/projects")
async def current_user_projects(request: Request):
    identity = request.state.identity
    return {
        "success": True,
        "projects": [{
            "project_id": identity.project_id,
            "name": identity.project_name,
            "role": identity.role,
            "role_label": ROLE_LABELS.get(identity.role, identity.role),
        }],
    }


@app.get("/api/projects/current/members")
async def list_current_project_members(request: Request):
    identity = request.state.identity
    members = get_repository().list_project_members(identity.project_id)
    return {
        "success": True,
        "members": [
            {
                **member,
                "role_label": ROLE_LABELS.get(str(member.get("role") or ""), str(member.get("role") or "")),
            }
            for member in members
        ],
    }


@app.post("/api/projects/current/members")
async def provision_current_project_member(request: Request, payload: Dict[str, Any] = Body(...)):
    role = str(payload.get("role") or "project_member")
    if role not in ROLE_PERMISSIONS:
        raise HTTPException(status_code=400, detail="不支持的项目角色")
    try:
        member = get_repository().provision_project_member(
            str(payload.get("email") or ""),
            str(payload.get("display_name") or ""),
            role,
            granted_by=request.state.identity.user_id,
            duty=str(payload.get("duty") or ""),
        )
        if settings.DEMO_SIMPLE_AUTH:
            member = auth_service.enable_demo_login(
                str(member["user_id"]), actor=request.state.identity.user_id,
            )
            invitation = None
            message = "演示成员已创建，可使用企业邮箱和邮箱前缀密码直接登录"
        else:
            invitation = auth_service.create_invitation(str(member["user_id"]), request.state.identity.user_id)
            message = "成员已创建，请通过安全渠道发送一次性激活链接"
        get_repository().write_security_audit(
            request.state.identity.user_id,
            "iam.member_provisioned",
            "user",
            str(member["user_id"]),
            {"role": role, "email": member.get("email", "")},
            project_id=request.state.identity.project_id,
        )
        return {
            "success": True,
            "member": {**member, "password_hash": ""},
            "invitation": invitation,
            "message": message,
        }
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.patch("/api/projects/current/members/{user_id}")
async def update_current_project_member(user_id: str, request: Request, payload: Dict[str, Any] = Body(...)):
    role = str(payload.get("role") or "").strip()
    status = str(payload.get("status") or "").strip()
    duty = payload.get("duty")
    duty = str(duty).strip() if duty is not None else None
    if role and role not in ROLE_PERMISSIONS:
        raise HTTPException(status_code=400, detail="不支持的项目角色")
    if status and status not in {"active", "suspended", "revoked"}:
        raise HTTPException(status_code=400, detail="不支持的成员状态")
    identity = get_current_identity()
    if identity and identity.user_id == user_id and status and status != "active":
        raise HTTPException(status_code=409, detail="不能停用当前登录账号")
    try:
        previous = get_repository().get_user(user_id, project_id=request.state.identity.project_id) or {}
        member = get_repository().update_project_membership(user_id, role=role, status=status, duty=duty)
        get_repository().write_security_audit(
            request.state.identity.user_id,
            "iam.membership_updated",
            "user",
            user_id,
            {
                "previous_role": previous.get("role", ""),
                "role": member.get("role", ""),
                "previous_status": previous.get("membership_status", ""),
                "status": member.get("membership_status", ""),
                "previous_duty": previous.get("duty", ""),
                "duty": member.get("duty", ""),
            },
            project_id=request.state.identity.project_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"success": True, "member": member}


@app.post("/api/access-policies")
async def create_access_policy(request: Request, payload: Dict[str, Any] = Body(...)):
    name = str(payload.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="策略名称不能为空")
    identity = get_current_identity()
    policy = get_repository().create_access_policy(
        name,
        user_ids=list(payload.get("user_ids") or []),
        roles=list(payload.get("roles") or []),
        created_by=identity.user_id if identity else "user_system",
    )
    get_repository().write_security_audit(
        request.state.identity.user_id,
        "knowledge.access_policy_saved",
        "access_policy",
        str(policy["access_policy_id"]),
        {"name": name, "user_count": len(policy["user_ids"]), "roles": policy["roles"]},
        project_id=request.state.identity.project_id,
    )
    return {"success": True, "policy": policy}


# ═══════════════════════════════════════════════
#  1. 分类管理
# ═══════════════════════════════════════════════

@app.get("/api/categories")
async def list_categories():
    """获取所有分类"""
    return {"success": True, "categories": _load_categories()}

@app.post("/api/categories")
async def add_category(name: str = Form(...)):
    """添加分类"""
    cats = _load_categories()
    name = name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="分类名称不能为空")
    if name in cats:
        raise HTTPException(status_code=400, detail="分类已存在")
    return {"success": True, "categories": get_repository().add_category(name)}

@app.put("/api/categories")
async def update_category(old_name: str = Form(...), new_name: str = Form(...)):
    """编辑分类"""
    cats = _load_categories()
    old_name = old_name.strip()
    new_name = new_name.strip()
    if not new_name:
        raise HTTPException(status_code=400, detail="分类名称不能为空")
    if old_name not in cats:
        raise HTTPException(status_code=404, detail="原分类不存在")
    if new_name != old_name and new_name in cats:
        raise HTTPException(status_code=400, detail="新分类名称已存在")
    return {"success": True, "categories": get_repository().rename_category(old_name, new_name)}

@app.delete("/api/categories/{name}")
async def delete_category(name: str):
    """删除分类"""
    cats = _load_categories()
    if name not in cats:
        raise HTTPException(status_code=404, detail="分类不存在")
    try:
        categories = get_repository().delete_category(name)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"success": True, "categories": categories}

# ═══════════════════════════════════════════════
#  2. 文档管理
# ═══════════════════════════════════════════════

@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    category: Optional[str] = Form(None),
    uploader: Optional[str] = Form(None),
    asset_id: Optional[str] = Form(None),
    source_id: Optional[str] = Form(None),
):
    return await _upload_document_impl(
        file, category=category, uploader=uploader, asset_id=asset_id, source_id=source_id,
    )


async def _upload_document_impl(
    file: UploadFile,
    category: Optional[str] = None,
    uploader: Optional[str] = None,
    asset_id: Optional[str] = None,
    source_id: Optional[str] = None,
    *,
    prepared_documents: Optional[List[Any]] = None,
    extraction_summary: Optional[Dict[str, Any]] = None,
):
    """上传文档并自动向量化"""
    identity = get_current_identity()
    effective_uploader = identity.display_name if identity else str(uploader or "").strip()
    stable_asset_id = asset_id.strip() if isinstance(asset_id, str) else ""
    stable_source_id = source_id.strip() if isinstance(source_id, str) else ""
    stable_asset = get_repository().get_asset(stable_asset_id, identity=identity) if stable_asset_id else None
    if stable_asset_id and not stable_asset:
        raise HTTPException(status_code=404, detail="指定的知识资产不存在或无权更新")
    stable_source = get_repository().get_source(stable_source_id, identity=identity) if stable_source_id else None
    if stable_source_id and not stable_source:
        raise HTTPException(status_code=404, detail="指定的知识来源不存在或无权更新")
    if stable_asset and stable_source_id and stable_asset.get("primary_source_id") != stable_source_id:
        raise HTTPException(status_code=409, detail="指定资产与知识来源不匹配")
    stable_acl = stable_asset or stable_source or {}
    original_filename = _sanitize_upload_filename(file.filename)
    ext = original_filename.rsplit(".", 1)[-1].lower() if "." in original_filename else ""
    allowed_extensions = set(settings.ALLOWED_EXTENSIONS)
    if prepared_documents is not None:
        allowed_extensions.update(IMAGE_DOCUMENT_EXTENSIONS)
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式: .{ext}，支持: {', '.join(sorted(allowed_extensions))}",
        )

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.MAX_UPLOAD_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"文件大小 {size_mb:.1f}MB 超过限制 {settings.MAX_UPLOAD_SIZE_MB}MB",
        )

    safe_name = f"{uuid.uuid4().hex}_{original_filename}"
    file_path = UPLOAD_DIR / safe_name
    indexed = False
    prepared_asset: Dict[str, Any] = {}
    slo_event_id = get_repository().start_runtime_slo_event(
        "document_processing", safe_name.split("_", 1)[0],
        {"extension": ext, "size_bytes": len(content)},
    )
    with open(file_path, "wb") as f:
        f.write(content)

    try:
        docs = prepared_documents if prepared_documents is not None else load_document(str(file_path))
        raw_text = content.decode("utf-8", errors="ignore")
        content_text = (
            "\n\n".join(str(document.page_content or "") for document in docs)
            if prepared_documents is not None else raw_text
        )
        graph_import = {}
        if ext == "json" and hasattr(knowledge_graph, "graph_from_json_content"):
            graph_import = knowledge_graph.graph_from_json_content(raw_text, original_filename)

        # 分类
        if category:
            categories_list = [category.strip()]
        elif graph_import:
            categories_list = ["知识图谱"]
        else:
            categories_list = project_context.classify_document(original_filename, content_text)

        # 写入 metadata
        for doc in docs:
            doc.metadata["source_file"] = original_filename
            doc.metadata["stored_file"] = safe_name
            doc.metadata["categories"] = categories_list
            doc.metadata["category"] = categories_list[0] if categories_list else "未分类"
            if effective_uploader:
                doc.metadata["uploader"] = effective_uploader
            doc.metadata["upload_date"] = datetime.now().strftime("%Y-%m-%d %H:%M")

        actor = identity.user_id if identity else (effective_uploader or "system")
        prepared_asset = knowledge_asset_service.prepare_documents(
            docs,
            source_type="manual_upload",
            external_key=original_filename,
            title=original_filename,
            actor=actor,
            stored_file=safe_name,
            categories=categories_list,
            asset_id=stable_asset_id,
            source_id=stable_source_id,
            visibility=str(stable_acl.get("visibility") or "project"),
            sensitivity_level=str(stable_acl.get("sensitivity_level") or ""),
            access_policy_id=str(stable_acl.get("access_policy_id") or ""),
            owner_user_id=str(stable_acl.get("owner_user_id") or (identity.user_id if identity else "")),
            metadata={
                "upload_filename": original_filename,
                "extraction": dict(extraction_summary or {}),
            },
        )
        if prepared_asset.get("duplicate") and prepared_asset.get("status") == "active":
            file_path.unlink(missing_ok=True)
            existing = get_repository().get_asset(str(prepared_asset["asset_id"])) or prepared_asset
            get_repository().complete_runtime_slo_event(
                slo_event_id, succeeded=True, detail={"outcome": "already_ingested"},
            )
            return {
                "success": True, "already_ingested": True, "filename": original_filename,
                "chunks": 0, "categories": categories_list, "asset": existing,
                **dict(extraction_summary or {}),
                "message": f"{original_filename} 内容未变化，继续使用当前知识版本",
            }
        if prepared_asset.get("duplicate"):
            raise RuntimeError("相同内容版本正在处理中，请稍后重试")

        chunk_count = vector_store.add_documents(docs)
        indexed = True
        knowledge_asset_service.record_projection(
            str(prepared_asset["version_id"]), "vector", succeeded=True,
        )

        # 注册到项目语境
        meta = project_context.extract_project_metadata(content_text)
        meta["categories"] = categories_list
        meta["source_file"] = original_filename
        meta["stored_file"] = safe_name
        meta["chunks"] = chunk_count
        if effective_uploader:
            meta["uploader"] = effective_uploader
        meta.update({
            "asset_id": prepared_asset["asset_id"], "source_id": prepared_asset["source_id"],
            "version_id": prepared_asset["version_id"], "status": "preparing",
        })
        context_ready = True
        context_error = ""
        try:
            project_context.register_document(safe_name, meta)
        except Exception as exc:
            context_ready = False
            context_error = str(exc)[:1000]
            logger.warning("文档已入库，但项目语境投影待修复: %s", safe_name)

        asset = knowledge_asset_service.publish(
            prepared_asset,
            actor=actor,
            graph_ready=False,
            context_ready=context_ready,
            context_error=context_error,
        )

        # 增量更新知识图谱；图谱 JSON 优先作为可视化快照导入。
        graph_message = ""
        graph_synced = True
        graph_nodes = 0
        graph_edges = 0
        try:
            if graph_import:
                knowledge_graph.import_graph_data(graph_import)
                graph_nodes = len(graph_import.get("nodes", []))
                graph_edges = len(graph_import.get("edges", []))
                graph_message = f"，并导入图谱 {graph_nodes} 个节点 / {graph_edges} 条关系"
            else:
                knowledge_graph.build_graph()
            knowledge_asset_service.mark_post_publish_projection(
                str(prepared_asset["version_id"]), "graph", True,
            )
        except Exception as e:
            graph_synced = False
            graph_message = "，文档已入库，但图谱尚未同步，请在知识图谱页面重新构建"
            logger.exception("文档入库后的图谱更新失败: %s", safe_name)
            knowledge_asset_service.mark_post_publish_projection(
                str(prepared_asset["version_id"]), "graph", False, str(e),
            )

        readiness_synced = True
        try:
            # Readiness is calculated from authoritative current assets on demand;
            # validating the projection input is sufficient to mark this revision ready.
            get_repository().readiness_assets(identity=identity)
            knowledge_asset_service.mark_post_publish_projection(
                str(prepared_asset["version_id"]), "readiness", True,
            )
        except Exception as exc:
            readiness_synced = False
            knowledge_asset_service.mark_post_publish_projection(
                str(prepared_asset["version_id"]), "readiness", False, str(exc),
            )
        sync_warnings = []
        if not graph_synced:
            sync_warnings.append("知识图谱")
        if not context_ready:
            sync_warnings.append("项目语境")
        if not readiness_synced:
            sync_warnings.append("新人就绪度")

        onboarding_sync = _refresh_onboarding_after_knowledge_change(
            asset_id=str(asset.get("asset_id") or prepared_asset["asset_id"]),
            actor=actor, trigger="document_published",
        )

        get_repository().complete_runtime_slo_event(
            slo_event_id, succeeded=True,
            detail={
                "outcome": "published",
                "partial_success": bool(sync_warnings)
                    or str((extraction_summary or {}).get("extraction_status") or "") == "partial",
            },
        )
        extraction = dict(extraction_summary or {})
        return {
            "success": True,
            "partial_success": bool(sync_warnings) or extraction.get("extraction_status") == "partial",
            "repair_required": sync_warnings,
            "projection_status": "repair_required" if sync_warnings else "healthy",
            "filename": original_filename,
            "chunks": chunk_count,
            "categories": categories_list,
            "graph_imported": bool(graph_import),
            "graph_synced": graph_synced,
            "graph_nodes": graph_nodes,
            "graph_edges": graph_edges,
            "context_synced": context_ready,
            "readiness_synced": readiness_synced,
            "onboarding_sync": onboarding_sync,
            "sync_warnings": sync_warnings,
            **extraction,
            "asset": asset,
            "message": (
                f"成功导入 {original_filename}，拆分为 {chunk_count} 个知识块{graph_message}"
                + ("；项目语境待后台修复" if not context_ready else "")
                + (f"；{extraction.get('extraction_warning')}" if extraction.get("extraction_warning") else "")
            ),
        }
    except Exception as e:
        if prepared_asset:
            knowledge_asset_service.fail(prepared_asset, e, actor=(identity.user_id if identity else "system"))
        if indexed:
            try:
                vector_store.delete_documents_by_source(safe_name)
                project_context.unregister_document(safe_name)
            except Exception:
                logger.exception("回滚失败的文档导入: %s", safe_name)
        if file_path.exists():
            file_path.unlink()
        diagnostic_id = uuid.uuid4().hex[:10]
        get_repository().complete_runtime_slo_event(
            slo_event_id, succeeded=False,
            detail={"outcome": "failed", "diagnostic_id": diagnostic_id, "error_type": type(e).__name__},
        )
        logger.exception("文档处理失败 [%s]: %s", diagnostic_id, original_filename)
        raise HTTPException(status_code=500, detail=f"文档处理失败，请稍后重试（诊断编号：{diagnostic_id}）") from e


@app.post("/api/documents/uploads", status_code=202)
async def enqueue_document_upload(
    request: Request,
    file: UploadFile = File(...),
    category: Optional[str] = Form(None),
    uploader: Optional[str] = Form(None),
    asset_id: Optional[str] = Form(None),
    source_id: Optional[str] = Form(None),
):
    """Persist an accepted upload and return a restart-safe processing job immediately."""
    identity = request.state.identity
    original_filename = _sanitize_upload_filename(file.filename)
    ext = original_filename.rsplit(".", 1)[-1].lower() if "." in original_filename else ""
    async_extensions = set(settings.ALLOWED_EXTENSIONS) | IMAGE_DOCUMENT_EXTENSIONS
    if ext not in async_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式: .{ext}，支持: {', '.join(sorted(async_extensions))}",
        )
    limit_bytes = int(settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024)
    content = await file.read(limit_bytes + 1)
    if len(content) > limit_bytes:
        raise HTTPException(status_code=400, detail=f"文件超过限制 {settings.MAX_UPLOAD_SIZE_MB}MB")

    stable_asset_id = asset_id.strip() if isinstance(asset_id, str) else ""
    stable_source_id = source_id.strip() if isinstance(source_id, str) else ""
    stable_asset = get_repository().get_asset(stable_asset_id, identity=identity) if stable_asset_id else None
    if stable_asset_id and not stable_asset:
        raise HTTPException(status_code=404, detail="指定的知识资产不存在或无权更新")
    stable_source = get_repository().get_source(stable_source_id, identity=identity) if stable_source_id else None
    if stable_source_id and not stable_source:
        raise HTTPException(status_code=404, detail="指定的知识来源不存在或无权更新")
    if stable_asset and stable_source_id and stable_asset.get("primary_source_id") != stable_source_id:
        raise HTTPException(status_code=409, detail="指定资产与知识来源不匹配")

    staging_name = f"upload_{uuid.uuid4().hex}.pending"
    staging_path = UPLOAD_STAGING_DIR / staging_name
    try:
        with staging_path.open("xb") as handle:
            handle.write(content)
        try:
            os.chmod(staging_path, 0o600)
        except OSError:
            pass
        digest = hashlib.sha256(content).hexdigest()
        job = processing_job_service.enqueue(
            "document_ingestion",
            {
                "staging_name": staging_name, "original_filename": original_filename,
                "category": category.strip() if isinstance(category, str) else "", "asset_id": stable_asset_id,
                "source_id": stable_source_id, "actor_user_id": identity.user_id,
            },
            created_by=identity.user_id,
            idempotency_key=f"document-upload:{identity.project_id}:{stable_asset_id or original_filename}:{digest}",
            priority=65, max_attempts=3, linked_asset_id=stable_asset_id, source_id=stable_source_id,
        )
    except Exception:
        staging_path.unlink(missing_ok=True)
        raise
    if not job.get("created"):
        staging_path.unlink(missing_ok=True)
    get_repository().write_security_audit(
        identity.user_id, "knowledge.document_upload_accepted", "processing_job", str(job["job_id"]),
        {"extension": ext, "size_bytes": len(content), "asset_id": stable_asset_id, "source_id": stable_source_id},
        project_id=identity.project_id,
    )
    return {
        "success": True, "message": "上传已受理，可在处理任务中心查看进度", "job": job,
        "filename": original_filename,
    }


@app.post("/api/documents/upload-batches", status_code=202)
async def enqueue_document_upload_batch(
    request: Request,
    files: List[UploadFile] = File(...),
    category: Optional[str] = Form(None),
    uploader: Optional[str] = Form(None),
):
    """Accept multiple new knowledge files as independently recoverable ingestion jobs."""
    identity = request.state.identity
    max_files = max(1, int(settings.MAX_BATCH_UPLOAD_FILES))
    if not files:
        raise HTTPException(status_code=400, detail="请至少选择一个文件")
    if len(files) > max_files:
        raise HTTPException(status_code=400, detail=f"单批最多上传 {max_files} 个文件")

    prepared: List[Dict[str, Any]] = []
    seen_names = set()
    per_file_limit = int(settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024)
    batch_limit = int(settings.MAX_BATCH_UPLOAD_SIZE_MB * 1024 * 1024)
    total_size = 0
    for file in files:
        original_filename = _sanitize_upload_filename(file.filename)
        normalized_name = original_filename.casefold()
        if normalized_name in seen_names:
            raise HTTPException(status_code=400, detail=f"同一批次不能包含重名文件：{original_filename}")
        seen_names.add(normalized_name)
        ext = original_filename.rsplit(".", 1)[-1].lower() if "." in original_filename else ""
        async_extensions = set(settings.ALLOWED_EXTENSIONS) | IMAGE_DOCUMENT_EXTENSIONS
        if ext not in async_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"{original_filename} 格式不受支持，支持: {', '.join(sorted(async_extensions))}",
            )
        content = await file.read(per_file_limit + 1)
        if len(content) > per_file_limit:
            raise HTTPException(
                status_code=400, detail=f"{original_filename} 超过单文件限制 {settings.MAX_UPLOAD_SIZE_MB}MB",
            )
        total_size += len(content)
        if total_size > batch_limit:
            raise HTTPException(
                status_code=400, detail=f"本批文件总大小超过限制 {settings.MAX_BATCH_UPLOAD_SIZE_MB}MB",
            )
        prepared.append({
            "filename": original_filename, "extension": ext, "content": content,
            "digest": hashlib.sha256(content).hexdigest(),
        })

    normalized_category = category.strip() if isinstance(category, str) else ""
    batch_id = f"batch_{uuid.uuid4().hex}"
    accepted = []
    failed = []
    for item in prepared:
        staging_name = f"upload_{uuid.uuid4().hex}.pending"
        staging_path = UPLOAD_STAGING_DIR / staging_name
        try:
            with staging_path.open("xb") as handle:
                handle.write(item["content"])
            try:
                os.chmod(staging_path, 0o600)
            except OSError:
                pass
            job = processing_job_service.enqueue(
                "document_ingestion",
                {
                    "staging_name": staging_name, "original_filename": item["filename"],
                    "category": normalized_category, "asset_id": "", "source_id": "",
                    "actor_user_id": identity.user_id, "batch_id": batch_id,
                },
                created_by=identity.user_id,
                idempotency_key=(
                    f"document-upload:{identity.project_id}:{item['filename']}:{item['digest']}"
                ),
                priority=65, max_attempts=3, source_id=item["filename"],
            )
            if not job.get("created"):
                staging_path.unlink(missing_ok=True)
            accepted.append({
                "filename": item["filename"], "status": "accepted", "job": job,
                "already_queued": not bool(job.get("created")),
            })
        except Exception:
            staging_path.unlink(missing_ok=True)
            logger.exception("批量上传受理失败: %s", item["filename"])
            failed.append({
                "filename": item["filename"], "status": "failed", "error": "任务受理失败，请重新提交该文件",
            })

    get_repository().write_security_audit(
        identity.user_id, "knowledge.document_batch_upload_accepted", "processing_job_batch",
        batch_id,
        {
            "submitted_count": len(prepared), "accepted_count": len(accepted),
            "failed_count": len(failed), "total_size_bytes": total_size,
            "extensions": sorted({item["extension"] for item in prepared}),
        },
        project_id=identity.project_id,
    )
    if not accepted:
        raise HTTPException(status_code=500, detail="本批文件均未能进入处理队列，请稍后重试")
    return {
        "success": True,
        "message": (
            f"已受理 {len(accepted)} 个文件，可在处理任务中心逐项查看"
            + (f"；{len(failed)} 个文件受理失败" if failed else "")
        ),
        "submitted_count": len(prepared), "accepted_count": len(accepted),
        "failed_count": len(failed), "items": accepted + failed,
        "jobs": [item["job"] for item in accepted], "batch_id": batch_id,
    }


@app.patch("/api/documents/{filename:path}/access")
async def update_document_access(filename: str, request: Request, payload: Dict[str, Any] = Body(...)):
    try:
        document = get_repository().set_document_access(
            filename,
            str(payload.get("visibility") or "project"),
            access_policy_id=str(payload.get("access_policy_id") or ""),
            owner_user_id=str(payload.get("owner_user_id") or ""),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    get_repository().write_security_audit(
        request.state.identity.user_id,
        "knowledge.document_access_updated",
        "document",
        str(document.get("document_id") or filename),
        {
            "stored_file": filename,
            "visibility": document.get("visibility", ""),
            "access_policy_id": document.get("access_policy_id", ""),
            "owner_user_id": document.get("owner_user_id", ""),
        },
        project_id=request.state.identity.project_id,
    )
    return {"success": True, "document": document}


@app.get("/api/documents")
async def list_documents():
    """获取文档列表"""
    docs = _get_documents()
    return {"success": True, "documents": docs}


def _require_knowledge_permission(permission: str):
    identity = get_current_identity()
    if identity is not None and not identity.can(permission):
        raise HTTPException(status_code=403, detail="当前账号没有执行该知识资产操作的权限")
    return identity


@app.get("/api/knowledge/assets")
async def list_knowledge_assets(
    include_inactive: bool = False, include_deleted: bool = False,
):
    identity = _require_knowledge_permission("knowledge.read")
    if (include_inactive or include_deleted) and identity is not None and not identity.can("knowledge.manage"):
        raise HTTPException(status_code=403, detail="只有知识运营人员可以查看非生效资产")
    assets = knowledge_asset_service.list_assets(
        identity,
        include_inactive=include_inactive,
        include_deleted=include_deleted,
    )
    status_counts: Dict[str, int] = {}
    for asset in assets:
        status = str(asset.get("status") or "draft")
        status_counts[status] = status_counts.get(status, 0) + 1
    processing_summary = get_repository().summarize_asset_processing(
        [str(asset.get("asset_id") or "") for asset in assets]
    )
    return {
        "success": True,
        "assets": assets,
        "summary": {
            **status_counts,
            "active_count": status_counts.get("active", 0),
            "pending_count": status_counts.get("pending_review", 0),
            "review_due_count": status_counts.get("review_due", 0),
            **processing_summary,
            # Backward-compatible aggregate for older clients.
            "failed_count": (
                processing_summary["failed_version_count"]
                + processing_summary["repair_required_count"]
            ),
        },
    }


@app.get("/api/knowledge/assets/{asset_id}")
async def get_knowledge_asset(asset_id: str):
    identity = _require_knowledge_permission("knowledge.read")
    asset = knowledge_asset_service.get_asset(asset_id, identity=identity)
    if not asset:
        raise HTTPException(status_code=404, detail="知识资产不存在或无权访问")
    return {"success": True, "asset": asset}


@app.get("/api/knowledge/assets/{asset_id}/versions")
async def list_knowledge_asset_versions(asset_id: str):
    identity = _require_knowledge_permission("knowledge.read")
    versions = knowledge_asset_service.list_versions(asset_id, identity=identity)
    if not versions and not get_repository().get_asset(asset_id, identity=identity):
        raise HTTPException(status_code=404, detail="知识资产不存在或无权访问")
    return {"success": True, "versions": versions}


@app.get("/api/knowledge/assets/{asset_id}/versions/{version_id}/preview")
async def preview_knowledge_asset_version(asset_id: str, version_id: str):
    identity = _require_knowledge_permission("knowledge.read")
    version = knowledge_asset_service.get_version(asset_id, version_id, identity=identity)
    if not version:
        raise HTTPException(status_code=404, detail="知识资产版本不存在或无权访问")
    documents = list(version.get("documents") or [])
    if not documents:
        raise HTTPException(status_code=404, detail="该版本没有可访问的原始文档")
    document = documents[0]
    stored_file = Path(str(document.get("stored_file") or "")).name
    if not stored_file:
        raise HTTPException(status_code=404, detail="该版本的原始文档不存在")
    file_path = UPLOAD_DIR / stored_file
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="该版本的原始文档不存在")
    try:
        loaded = await asyncio.to_thread(load_document, str(file_path))
        content = "\n\n".join(str(item.page_content or "") for item in loaded)
    except Exception as exc:
        raise HTTPException(status_code=422, detail="该版本原件暂时无法解析") from exc
    limit = 100_000
    return {
        "success": True,
        "asset_id": asset_id,
        "version_id": version_id,
        "filename": str(document.get("source_file") or stored_file),
        "stored_file": stored_file,
        "content": content[:limit],
        "truncated": len(content) > limit,
        "document": document,
    }


@app.get("/api/knowledge/assets/{asset_id}/projection-status")
async def get_knowledge_asset_projection_status(asset_id: str):
    identity = _require_knowledge_permission("knowledge.read")
    status = knowledge_asset_service.projection_status(asset_id, identity=identity)
    if not status:
        raise HTTPException(status_code=404, detail="知识资产不存在或无权访问")
    return {"success": True, **status}


@app.post("/api/knowledge/assets/{asset_id}/repair-projections")
async def repair_knowledge_asset_projections(asset_id: str, payload: Dict[str, Any] = Body(default={})):
    identity = _require_knowledge_permission("knowledge.manage")
    if not knowledge_asset_service.projection_status(asset_id, identity=identity):
        raise HTTPException(status_code=404, detail="知识资产不存在或无权访问")
    job = processing_job_service.enqueue(
        "projection_repair",
        {"asset_id": asset_id, "actor": identity.user_id if identity else "user_system", "force": bool(payload.get("force", False))},
        created_by=identity.user_id if identity else "user_system",
        idempotency_key=f"projection-repair:{asset_id}",
        linked_asset_id=asset_id,
        priority=75,
    )
    return {"success": True, "accepted": True, "job": job, "message": "投影修复已进入处理队列"}


@app.patch("/api/knowledge/assets/{asset_id}")
async def patch_knowledge_asset(asset_id: str, payload: Dict[str, Any] = Body(...)):
    identity = _require_knowledge_permission("knowledge.manage")
    try:
        asset = knowledge_asset_service.patch_metadata(
            asset_id, payload, identity.user_id if identity else "system",
        )
        return {"success": True, "asset": asset}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/knowledge/assets/{asset_id}/transition")
async def transition_knowledge_asset(asset_id: str, payload: Dict[str, Any] = Body(...)):
    identity = _require_knowledge_permission("knowledge.publish")
    actor = identity.user_id if identity else "system"
    action = str(payload.get("action") or "")
    try:
        if action.strip().lower() == "revoke":
            workspace_asset_id = feishu_knowledge_service.resolve_workspace_asset_id(asset_id)
            if workspace_asset_id:
                result = feishu_knowledge_service.revert_asset(
                    workspace_asset_id,
                    reason=str(payload.get("reason") or ""),
                    actor=actor,
                )
                return {
                    "success": True,
                    "asset": result.get("knowledge_asset"),
                    "feishu": result,
                }
        asset = knowledge_asset_service.transition(
            asset_id, action, actor, str(payload.get("reason") or ""),
            payload.get("expected_revision"), str(payload.get("version_id") or ""),
        )
        task_sync = None
        normalized_action = action.strip().lower()
        if normalized_action == "mark_review_due":
            try:
                task_sync = knowledge_task_service.create_asset_review_task(asset, identity)
            except Exception as exc:
                logger.exception("知识资产复审任务创建失败: %s", asset_id)
                task_sync = {"success": False, "error": str(exc)[:500]}
        elif normalized_action in {"submit_review", "request_review"}:
            try:
                task_sync = knowledge_task_service.create_asset_publish_task(asset, identity)
            except Exception as exc:
                logger.exception("知识资产发布任务创建失败: %s", asset_id)
                task_sync = {"success": False, "error": str(exc)[:500]}
        onboarding_sync = None
        if normalized_action in {"publish", "revoke", "expire", "mark_review_due"}:
            onboarding_sync = _refresh_onboarding_after_knowledge_change(
                asset_id=str(asset.get("asset_id") or asset_id), actor=actor,
                trigger=f"asset_{normalized_action}",
            )
        return {
            "success": True, "asset": asset, "task_sync": task_sync,
            "onboarding_sync": onboarding_sync,
        }
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/documents/by-category")
async def list_documents_by_category():
    """按主分类返回文档列表，避免多标签文档在多个分类下重复显示。"""
    docs = _get_documents()
    grouped = {}
    for doc in docs:
        categories = doc.get("categories") or []
        primary_category = doc.get("category") or (categories[0] if categories else "未分类")
        grouped.setdefault(primary_category, []).append(doc)
    return {"success": True, "grouped": grouped, "categories": _load_categories()}


@app.get("/api/documents/preview/{filename:path}")
async def preview_document(filename: str):
    """预览文档内容"""
    requested = _sanitize_upload_filename(filename)
    candidate_names = []
    for document in _get_documents():
        identifiers = {
            _sanitize_upload_filename(str(document.get(key) or ""))
            for key in ("name", "source_file", "stored_file")
            if document.get(key)
        }
        if requested in identifiers:
            candidate_names.extend(str(document.get(key) or "") for key in ("stored_file", "name", "source_file"))
    # Internal/background callers retain the legacy projection fallback.
    # Authenticated requests must resolve through permission-filtered documents.
    if get_current_identity() is None:
        for asset in feishu_workspace.list_assets(status="published"):
            identifiers = {
                _sanitize_upload_filename(str(asset.get(key) or ""))
                for key in ("source_file", "stored_file")
                if asset.get(key)
            }
            if requested in identifiers:
                candidate_names.extend(str(asset.get(key) or "") for key in ("stored_file", "source_file"))

    safe_names = list(dict.fromkeys(
        _sanitize_upload_filename(name) for name in candidate_names if str(name or "").strip()
    ))
    if not safe_names:
        raise HTTPException(status_code=404, detail="文件不存在或已失效")
    for f in UPLOAD_DIR.iterdir():
        if any(_matches_uploaded_file(f.name, candidate) for candidate in safe_names):
            try:
                text = _read_document_preview_text(f)
            except ValueError as exc:
                raise HTTPException(status_code=415, detail=f"{exc}，请下载原件查看") from exc
            except Exception as e:
                raise _internal_http_error("读取文档预览失败", "读取文件失败，请稍后重试") from e
            if not text.strip():
                text = "该文件没有可提取的文本内容，请下载原件查看。"
            return {
                "success": True,
                "filename": requested,
                "stored_file": f.name,
                "content": text[:30000],
                "truncated": len(text) > 30000,
            }
    raise HTTPException(status_code=404, detail="文件不存在")


@app.delete("/api/documents/{filename}")
async def delete_document(filename: str):
    """删除文档"""
    identity = get_current_identity()
    repository = get_repository()
    asset_id = repository.resolve_asset_id(filename)
    try:
        if asset_id:
            knowledge_asset_service.transition(
                asset_id, "revoke", identity.user_id if identity else "system", reason="删除文档",
            )
            repair = vector_store.reconcile_with_storage()
            deleted = int(bool(repair.get("changed")))
        else:
            deleted = vector_store.delete_document(filename)
    except Exception as exc:
        diagnostic_id = uuid.uuid4().hex[:10]
        logger.exception("删除文档失败 [%s]: %s", diagnostic_id, filename)
        raise HTTPException(status_code=500, detail=f"删除失败，请稍后重试（诊断编号：{diagnostic_id}）") from exc
    context_deleted = bool(deleted)
    sync_warnings = []
    try:
        project_context.reload()
    except Exception:
        logger.exception("刷新文档元数据缓存失败: %s", filename)
        sync_warnings.append("文档列表缓存同步待重试")
    graph_cleanup = {"nodes": 0, "edges": 0, "imported_nodes": 0}
    if hasattr(knowledge_graph, "remove_source"):
        try:
            graph_cleanup = knowledge_graph.remove_source(filename)
        except Exception:
            logger.exception("清理文档图谱来源失败: %s", filename)
            sync_warnings.append("知识图谱来源清理待重试")
    # Managed assets retain source files as immutable version evidence.
    if not asset_id:
        for f in UPLOAD_DIR.iterdir():
            if _matches_uploaded_file(f.name, filename):
                try:
                    f.unlink()
                except OSError:
                    logger.exception("删除文档原件失败: %s", f.name)
                    sync_warnings.append("文档原件清理待重试")
                break
    graph_rebuilt = False
    try:
        knowledge_graph.build_graph()
        graph_rebuilt = True
    except Exception:
        logger.exception("删除文档后重建知识图谱失败: %s", filename)
        sync_warnings.append("知识图谱重建待重试")
    return {
        "success": True,
        "partial_success": bool(sync_warnings),
        "deleted": deleted,
        "metadata_deleted": context_deleted,
        "graph_cleanup": graph_cleanup,
        "graph_rebuilt": graph_rebuilt,
        "sync_warnings": sync_warnings,
        "asset_id": asset_id,
    }


def _get_documents():
    """Return documents from authoritative SQLite through the context adapter."""
    _sync_project_context(strict=True)
    return project_context.list_documents()


def _list_docs_fallback():
    """备用：从 uploads 目录和 project_context 获取文档列表"""
    return vector_store.list_documents() if hasattr(vector_store, 'list_documents') else []


def _ingest_feishu_message(message: Dict[str, Any], group: Dict[str, Any], category: str = "") -> Dict[str, Any]:
    """兼容旧单条审核入口，实际按候选聚合后发布。"""
    candidate = feishu_workspace.ensure_candidate_for_messages(
        [str(message.get("message_id") or "")],
        category=category or str(group.get("default_category") or ""),
    )
    if not candidate:
        raise ValueError("无法为该消息创建知识候选")
    result = feishu_knowledge_service.ingest_candidate(str(candidate["candidate_id"]), category=category, actor="管理员")
    asset = result.get("asset") or {}
    return {
        "filename": asset.get("source_file") or "飞书项目沟通知识.md",
        "chunks": int(asset.get("chunks") or 0),
        "already_ingested": bool(result.get("already_ingested")),
        "asset_id": asset.get("asset_id") or "",
        "candidate_id": candidate.get("candidate_id") or "",
    }

# ═══════════════════════════════════════════════
#  3. 检索 & 对话
# ═══════════════════════════════════════════════

@app.post("/api/search")
async def search_documents(query: str = Form(...), k: int = Form(4)):
    from backend.knowledge_base.retriever import retriever
    docs = retriever.retrieve(query, k=k)
    results = [{"content": d.page_content, "source": d.metadata.get("source_file", "未知")} for d in docs]
    return {"success": True, "query": query, "results": results}


@app.post("/api/search/contextual")
async def search_with_context(
    query: str = Form(...),
    k: int = Form(4),
    category: Optional[str] = Form(None),
):
    """带项目语境的增强检索（支持按知识类别筛选）"""
    from backend.knowledge_base.retriever import retriever
    docs = retriever.retrieve(query, k=k)
    if category:
        matched = [
            doc for doc in docs
            if category == doc.metadata.get("category") or category in (doc.metadata.get("categories") or [])
        ]
        docs = matched or docs
    _sync_project_context()
    context_prompt = project_context.get_context_prompt(query, [category] if category else None)
    results = [{"content": d.page_content, "source": d.metadata.get("source_file", "未知")} for d in docs]
    return {"success": True, "query": query, "results": results, "context": context_prompt}


def _trace_line(trace_id: str, title: str, detail: str, status: str = "active", evidence: Optional[List[str]] = None) -> str:
    """向前端发送可解释的分析事件，不传递模型私有推理内容。"""
    payload = {
        "id": trace_id,
        "title": title,
        "detail": detail,
        "status": status,
        "evidence": evidence or [],
    }
    return "__TRACE__:" + json.dumps(payload, ensure_ascii=False) + "\n"


def _answer_meta_line(answer_id: str, *, no_answer: bool, citation_validity_rate: float = 0.0) -> str:
    return "__ANSWER_META__:" + json.dumps({
        "answer_id": answer_id,
        "no_answer": bool(no_answer),
        "citation_validity_rate": round(float(citation_validity_rate or 0), 4),
    }, ensure_ascii=False) + "\n"


def _question_excerpt(message: str, limit: int = 48) -> str:
    compact = " ".join((message or "").split())
    return compact if len(compact) <= limit else compact[:limit] + "..."


def _source_names(docs: List[Any]) -> List[str]:
    names: List[str] = []
    for doc in docs:
        name = str(doc.metadata.get("source_file", "未知资料")).strip() or "未知资料"
        if name not in names:
            names.append(name)
    return names


def _rag_prompt(message: str, context: str) -> str:
    context_prompt = project_context.get_context_prompt(message)
    return f"""{context_prompt}

基于以上项目知识语境，请回答以下问题。

相关知识库内容：
{context}

用户问题：{message}

请基于上述知识库内容回答。如果知识库中没有相关信息，请如实告知。"""


@app.get("/api/rag/evaluation-sets")
async def list_rag_evaluation_sets():
    return {"success": True, **rag_quality_service.list_evaluation_sets()}


@app.post("/api/rag/evaluation-sets")
async def create_rag_evaluation_set(request: Request, payload: Dict[str, Any] = Body(...)):
    try:
        evaluation_set = rag_quality_service.create_evaluation_set(payload, request.state.identity)
        return {"success": True, "evaluation_set": evaluation_set, "message": "评测集已创建"}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/rag/evaluation-sets/{evaluation_set_id}")
async def get_rag_evaluation_set(evaluation_set_id: str):
    evaluation_set = rag_quality_service.get_evaluation_set(evaluation_set_id)
    if not evaluation_set:
        raise HTTPException(status_code=404, detail="评测集不存在")
    return {"success": True, "evaluation_set": evaluation_set}


@app.post("/api/rag/evaluation-sets/{evaluation_set_id}/cases")
async def add_rag_evaluation_case(
    evaluation_set_id: str, request: Request, payload: Dict[str, Any] = Body(...),
):
    try:
        case = rag_quality_service.add_evaluation_case(
            evaluation_set_id, payload, request.state.identity,
        )
        return {"success": True, "case": case, "message": "评测问题已加入"}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/rag/evaluation-sets/{evaluation_set_id}/runs")
async def run_rag_evaluation(evaluation_set_id: str, request: Request):
    try:
        run = rag_quality_service.run_evaluation(evaluation_set_id, request.state.identity)
        return {"success": True, "run": run, "message": "离线检索评测已完成"}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/rag/feedback")
async def record_rag_feedback(request: Request, payload: Dict[str, Any] = Body(...)):
    try:
        result = rag_quality_service.record_feedback(
            payload, request.state.identity, knowledge_task_service,
        )
        created_task = result.get("knowledge_task")
        return {
            "success": True,
            **result,
            "message": "感谢反馈" if not created_task else "反馈已保存，并已创建知识改进任务",
        }
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/chat")
async def chat(
    message: str = Form(...),
    use_rag: bool = Form(True),
    history: Optional[str] = Form("[]"),
):
    """流式对话，先发送可解释的分析轨迹，再发送回答正文。"""
    import ast

    hist = ast.literal_eval(history) if isinstance(history, str) else (history or [])
    _sync_project_context()

    async def generate():
        answer_id = rag_quality_service.answer_id()
        question = _question_excerpt(message)
        yield _trace_line("analysis", "理解问题", f"已识别问题焦点：“{question}”。正在确定回答范围和需要核实的项目资料。", "done")

        if use_rag:
            yield _trace_line("retrieval", "检索项目资料", "正在查询知识库、交接资料和知识图谱索引。", "active")
            evidence = rag_quality_service.retrieve_evidence(message, get_current_identity(), k=4)
            docs = evidence["documents"]
            sources = evidence["citations"]
            source_names = _source_names(docs)
            context = "\n\n".join(d.page_content for d in docs)
            yield _trace_line(
                "retrieval",
                "检索项目资料",
                f"检索到 {len(docs)} 个相关知识片段，来自 {len(source_names)} 份项目资料。",
                "done",
                source_names,
            )
            yield _trace_line(
                "evidence",
                "核对证据",
                "已优先选择与问题直接相关的片段，用于支撑结论并避免超出资料范围。",
                "done",
                source_names[:3],
            )
            yield "__SOURCES__:" + json.dumps(sources, ensure_ascii=False) + "\n"
            if evidence["no_answer"]:
                yield _trace_line(
                    "answer", "返回知识边界",
                    "没有找到足够且全部有效的可访问证据，已停止模型生成，避免补写项目事实。", "done",
                )
                yield _answer_meta_line(
                    answer_id, no_answer=True,
                    citation_validity_rate=evidence["citation_validity_rate"],
                )
                yield "__ANSWER__\n"
                yield NO_ANSWER_TEXT
                return
            full_prompt = _rag_prompt(message, context)
        else:
            full_prompt = message

        yield _trace_line("answer", "组织回答", "正在按“结论、依据、补充说明”的结构生成回复。", "active")
        yield _answer_meta_line(
            answer_id, no_answer=False,
            citation_validity_rate=evidence["citation_validity_rate"] if use_rag else 0.0,
        )
        yield "__ANSWER__\n"
        async for chunk in llm_service.chat(query=full_prompt, history=hist, use_rag=False):
            if chunk:
                yield chunk

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")


@app.post("/api/chat-with-sources")
async def chat_with_sources(
    message: str = Form(...),
    history: Optional[str] = Form("[]"),
):
    """RAG 流式对话，返回可解释过程、来源和回答正文。"""
    import ast

    hist = ast.literal_eval(history) if isinstance(history, str) else (history or [])
    _sync_project_context()

    async def generate():
        answer_id = rag_quality_service.answer_id()
        question = _question_excerpt(message)
        yield _trace_line("analysis", "理解问题", f"已识别问题焦点：“{question}”。正在确定需要查证的项目知识。", "done")
        yield _trace_line("retrieval", "检索项目资料", "正在从知识库、交接资料和图谱索引中查找相关内容。", "active")

        evidence = rag_quality_service.retrieve_evidence(message, get_current_identity(), k=4)
        docs = evidence["documents"]
        source_names = _source_names(docs)
        context = "\n\n".join(d.page_content for d in docs)
        sources = evidence["citations"]

        yield _trace_line(
            "retrieval",
            "检索项目资料",
            f"检索到 {len(docs)} 个相关知识片段，来自 {len(source_names)} 份项目资料。",
            "done",
            source_names,
        )
        yield _trace_line(
            "evidence",
            "核对证据",
            "已聚焦与问题直接相关的片段，交叉确认结论的适用范围。",
            "done",
            source_names[:3],
        )
        yield "__SOURCES__:" + json.dumps(sources, ensure_ascii=False) + "\n"
        yield _trace_line(
            "citation",
            "关联引用",
            f"已核验 {len(sources)} 条引用的来源、版本、知识块与当前访问权限。",
            "done",
            source_names,
        )
        if evidence["no_answer"]:
            yield _trace_line(
                "answer", "返回知识边界",
                "没有找到足够且全部有效的可访问证据，已停止模型生成，避免补写项目事实。", "done",
            )
            yield _answer_meta_line(
                answer_id, no_answer=True,
                citation_validity_rate=evidence["citation_validity_rate"],
            )
            yield "__ANSWER__\n"
            yield NO_ANSWER_TEXT
            return
        yield _trace_line("answer", "组织回答", "正在按“结论、依据、补充说明”的结构生成回复。", "active")
        yield _answer_meta_line(
            answer_id, no_answer=False,
            citation_validity_rate=evidence["citation_validity_rate"],
        )
        yield "__ANSWER__\n"

        full_prompt = _rag_prompt(message, context)
        async for chunk in llm_service.chat(query=full_prompt, history=hist, use_rag=False):
            if chunk:
                yield chunk

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")


@app.post("/api/smart-chat")
async def smart_chat(
    message: str = Form(...),
    history: Optional[str] = Form("[]"),
):
    """Enhance Smart Mode with read-only tool routing and an existing-RAG fallback."""
    try:
        hist = json.loads(history) if isinstance(history, str) else (history or [])
        if not isinstance(hist, list):
            hist = []
    except (TypeError, json.JSONDecodeError):
        hist = []
    _sync_project_context()

    async def generate():
        answer_id = rag_quality_service.answer_id()
        question = _question_excerpt(message)
        yield _trace_line(
            "agent-routing",
            "分析问题并选择检索路径",
            f"已识别问题焦点：“{question}”。正在选择适合的项目知识能力进行核验。",
            "active",
        )

        try:
            from backend.knowledge_base.project_agent import ProjectAgentService

            service = ProjectAgentService(
                onboarding_provider=_onboarding_guide,
                handover_provider=lambda: [_handover_view(record) for record in _load_handover_records()],
            )
            sources: List[Dict[str, str]] = []
            answer = ""
            agent_no_answer = False
            async for event in service.stream(message, hist):
                event_type = event.get("type")
                if event_type == "trace":
                    payload = event.get("payload") or {}
                    yield _trace_line(
                        str(payload.get("id") or "agent-tool"),
                        str(payload.get("title") or "查询项目数据"),
                        str(payload.get("detail") or "正在核验项目数据。"),
                        str(payload.get("status") or "active"),
                        payload.get("evidence") or [],
                    )
                elif event_type == "sources":
                    sources = event.get("sources") or sources
                    yield "__SOURCES__:" + json.dumps(sources, ensure_ascii=False) + "\n"
                elif event_type == "answer":
                    answer = str(event.get("content") or "")
                    sources = event.get("sources") or sources
                    agent_no_answer = bool(event.get("no_answer"))

            yield _trace_line(
                "agent-routing",
                "分析问题并选择检索路径",
                "已根据问题完成检索路径选择与项目证据核验。",
                "done",
            )
            stable_sources = [
                item for item in sources if item.get("document_id") or item.get("chunk_id")
            ]
            if stable_sources:
                validated_sources = rag_quality_service.validate_citations(
                    stable_sources, get_current_identity(),
                )
                validated_by_source = {
                    str(item.get("source") or ""): item for item in validated_sources
                }
                sources = [
                    validated_by_source.get(str(item.get("source") or ""), item)
                    for item in sources
                ]
                citations_valid = all(item.get("valid") for item in validated_sources)
                citation_validity_rate = sum(
                    bool(item.get("valid")) for item in validated_sources
                ) / len(validated_sources)
            else:
                citations_valid = True
                citation_validity_rate = 0.0
            if sources:
                yield "__SOURCES__:" + json.dumps(sources, ensure_ascii=False) + "\n"
            no_answer = agent_no_answer or not bool(answer.strip()) or not citations_valid
            yield _trace_line(
                "answer", "返回知识边界" if no_answer else "组织回答",
                "引用已失效或不可访问，已停止返回未经支持的项目事实。"
                if no_answer else "正在基于工具返回的可核验结果生成回复。",
                "done" if no_answer else "active",
            )
            yield _answer_meta_line(
                answer_id, no_answer=no_answer,
                citation_validity_rate=citation_validity_rate,
            )
            yield "__ANSWER__\n"
            yield NO_ANSWER_TEXT if no_answer else answer
            return
        except Exception as agent_error:
            # Keep the established RAG path as a transparent compatibility fallback.
            yield _trace_line(
                "agent-fallback",
                "切换兼容检索",
                "当前智能检索未完成，已自动切换到原有 RAG 流程，不影响本次问答。",
                "done",
            )

        evidence = rag_quality_service.retrieve_evidence(message, get_current_identity(), k=4)
        docs = evidence["documents"]
        source_names = _source_names(docs)
        sources = evidence["citations"]
        context = "\n\n".join(doc.page_content for doc in docs)
        yield _trace_line(
            "retrieval",
            "检索项目资料",
            f"兼容流程检索到 {len(docs)} 个相关知识片段，来自 {len(source_names)} 份项目资料。",
            "done",
            source_names,
        )
        yield "__SOURCES__:" + json.dumps(sources, ensure_ascii=False) + "\n"
        if evidence["no_answer"]:
            yield _trace_line(
                "answer", "返回知识边界",
                "兼容检索也没有找到足够且有效的可访问证据，已停止模型生成。", "done",
            )
            yield _answer_meta_line(
                answer_id, no_answer=True,
                citation_validity_rate=evidence["citation_validity_rate"],
            )
            yield "__ANSWER__\n"
            yield NO_ANSWER_TEXT
            return
        yield _trace_line("answer", "组织回答", "正在沿用原有 RAG 生成稳定回复。", "active")
        yield _answer_meta_line(
            answer_id, no_answer=False,
            citation_validity_rate=evidence["citation_validity_rate"],
        )
        yield "__ANSWER__\n"
        full_prompt = _rag_prompt(message, context)
        async for chunk in llm_service.chat(query=full_prompt, history=hist, use_rag=False):
            if chunk:
                yield chunk

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")


@app.get("/api/project/context")
async def get_project_context_api():
    _sync_project_context()
    context = project_context.get_full_context() if hasattr(project_context, 'get_full_context') else {}
    return {"success": True, "context": context}


@app.post("/api/project/classify")
async def classify_document_text(text: str = Form(...)):
    cats = project_context.classify_document("manual.txt", text)
    return {"success": True, "categories": cats}


@app.get("/api/onboarding/templates")
async def onboarding_templates(request: Request):
    """Return project role templates with explainable topic requirements."""
    return {
        "success": True,
        "templates": onboarding_service.list_templates(request.state.identity),
    }


@app.put("/api/onboarding/templates/{template_id}")
async def save_onboarding_template(
    template_id: str, request: Request, payload: Dict[str, Any] = Body(...),
):
    try:
        template = onboarding_service.save_template(
            {**payload, "template_id": template_id}, request.state.identity,
        )
        return {"success": True, "template": template, "message": "岗位模板已保存"}
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/onboarding/guide")
async def onboarding_guide(role: str = "developer", user_id: str = ""):
    """按岗位和目标成员权限生成项目就绪度、缺口与个人计划。"""
    return _onboarding_guide(role, user_id)


@app.get("/api/onboarding/knowledge-gaps")
async def onboarding_knowledge_gaps(role: str = "开发工程师"):
    """返回指定岗位的项目知识覆盖度和待补充项。"""
    guide = _onboarding_guide(role)
    return {
        "success": True,
        "role": guide["role"],
        "readiness": guide["readiness"],
        "gaps": guide["gaps"],
        "coverage": guide["coverage"],
    }


@app.get("/api/onboarding/plans")
async def list_onboarding_plans(request: Request, user_id: str = ""):
    try:
        return {
            "success": True,
            "plans": onboarding_service.list_plans(request.state.identity, user_id=user_id),
        }
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/onboarding/supervised-plans")
async def list_supervised_onboarding_plans(request: Request):
    """Return the current supervisor's compact active learning-plan worklist."""
    try:
        return {
            "success": True,
            "plans": onboarding_service.list_supervised_plans(request.state.identity),
        }
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@app.post("/api/onboarding/plans", status_code=201)
async def create_onboarding_plan(request: Request, payload: Dict[str, Any] = Body(...)):
    try:
        plan = onboarding_service.create_plan(payload, request.state.identity)
        return {
            "success": True, "plan": plan,
            "created": bool(plan.get("created")),
            "message": "学习计划已创建" if plan.get("created") else "该成员已有进行中的学习计划",
        }
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/onboarding/plans/{plan_id}/refresh-materials")
async def refresh_onboarding_plan_materials(plan_id: str, request: Request):
    try:
        plan = onboarding_service.refresh_plan_materials(plan_id, request.state.identity)
        summary = dict(plan.get("refresh") or {})
        matched = int(summary.get("unblocked_count") or 0) + int(summary.get("added_count") or 0)
        remaining = int(summary.get("remaining_blocked_count") or 0)
        if matched:
            message = f"已匹配 {matched} 项必读资料"
        elif summary.get("changed"):
            message = "学习资料已更新"
        else:
            message = "当前学习资料没有变化"
        message += f"，仍有 {remaining} 项待补充" if remaining else "，全部必读主题已有可用资料"
        return {"success": True, "plan": plan, "refresh": summary, "message": message}
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.patch("/api/onboarding/plans/{plan_id}/items/{item_id}")
async def update_onboarding_item(
    plan_id: str, item_id: str, request: Request, payload: Dict[str, Any] = Body(...),
):
    try:
        plan = onboarding_service.update_item(
            plan_id, item_id, payload, request.state.identity,
        )
        return {"success": True, "plan": plan, "message": "学习进度已更新"}
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


# ═══════════════════════════════════════════════
#  3.0 系统处理任务
# ═══════════════════════════════════════════════

@app.get("/api/processing/jobs")
async def list_processing_jobs(
    request: Request, status: str = "", job_type: str = "", keyword: str = "",
    page: int = 1, page_size: int = 50,
):
    result = processing_job_service.list_jobs(
        request.state.identity, status=status.strip(), job_type=job_type.strip(),
        keyword=keyword.strip(), page=page, page_size=page_size,
    )
    return {"success": True, **result}


@app.get("/api/processing/jobs/{job_id}")
async def get_processing_job(job_id: str, request: Request):
    job = processing_job_service.get_job(job_id, request.state.identity)
    if not job:
        raise HTTPException(status_code=404, detail="处理任务不存在或无权访问")
    return {"success": True, "job": job}


@app.post("/api/processing/jobs/{job_id}/cancel")
async def cancel_processing_job(job_id: str, request: Request):
    try:
        job = processing_job_service.cancel(job_id, request.state.identity)
        return {"success": True, "job": job, "message": "取消请求已保存"}
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/processing/jobs/{job_id}/retry")
async def retry_processing_job(job_id: str, request: Request):
    try:
        existing = processing_job_service.get_job(job_id, request.state.identity)
        if existing and existing.get("job_type") == "document_ingestion" and existing.get("status") == "cancelled":
            raise ValueError("已取消的上传原文已清理，请重新上传文件")
        job = processing_job_service.retry(job_id, request.state.identity)
        return {"success": True, "job": job, "message": "任务已重新进入队列"}
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


# ═══════════════════════════════════════════════
#  3.1 知识治理任务
# ═══════════════════════════════════════════════

@app.get("/api/knowledge/tasks")
async def list_knowledge_tasks(
    request: Request,
    status: str = "",
    task_type: str = "",
    assignee_user_id: str = "",
    keyword: str = "",
    limit: int = 200,
):
    result = knowledge_task_service.list(
        request.state.identity,
        status=status.strip(), task_type=task_type.strip(),
        assignee_user_id=assignee_user_id.strip(), keyword=keyword.strip(), limit=limit,
    )
    return {"success": True, **result}


@app.post("/api/knowledge/tasks")
async def create_knowledge_task(request: Request, payload: Dict[str, Any] = Body(...)):
    try:
        task = knowledge_task_service.create(payload, request.state.identity)
        return {
            "success": True,
            "task": task,
            "created": bool(task.get("created")),
            "message": "知识任务已创建" if task.get("created") else "已合并到现有知识任务",
        }
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/knowledge/tasks/onboarding-gaps")
async def create_onboarding_gap_tasks(request: Request, payload: Dict[str, Any] = Body(...)):
    role = str(payload.get("role") or "开发工程师")
    guide = _onboarding_guide(role)
    try:
        result = knowledge_task_service.create_onboarding_gap_tasks(
            str(guide["role"]), guide["gaps"], request.state.identity,
            assignee_user_id=str(payload.get("assignee_user_id") or ""),
            due_days=int(payload.get("due_days") or 7),
        )
        return {
            "success": True,
            **result,
            "message": f"已新建 {result['created_count']} 项，合并 {result['merged_count']} 项重复缺口",
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/knowledge/tasks/notification-policy")
async def get_knowledge_task_notification_policy(request: Request):
    return {"success": True, "policy": knowledge_task_service.get_notification_policy()}


@app.put("/api/knowledge/tasks/notification-policy")
async def update_knowledge_task_notification_policy(
    request: Request, payload: Dict[str, Any] = Body(...),
):
    try:
        policy = knowledge_task_service.update_notification_policy(payload, request.state.identity)
        knowledge_task_reminder_scheduler.wake()
        return {"success": True, "policy": policy, "message": "自动提醒策略已保存"}
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/knowledge/tasks/notification-policy/scan")
async def scan_knowledge_task_notifications(request: Request):
    if not request.state.identity.can("task.manage"):
        raise HTTPException(status_code=403, detail="只有任务管理人员可以立即扫描提醒")
    result = knowledge_task_service.schedule_due_notifications()
    if result.get("job_count"):
        processing_job_runner.wake()
    return {
        "success": True, "result": result,
        "message": f"已安排 {result.get('scheduled_count', 0)} 项提醒",
    }


@app.get("/api/knowledge/tasks/{task_id}")
async def get_knowledge_task(task_id: str, request: Request):
    task = knowledge_task_service.get(task_id, request.state.identity)
    if not task:
        raise HTTPException(status_code=404, detail="知识任务不存在或无权访问")
    return {"success": True, "task": task}


@app.post("/api/knowledge/tasks/{task_id}/actions")
async def act_on_knowledge_task(task_id: str, request: Request, payload: Dict[str, Any] = Body(...)):
    try:
        task = knowledge_task_service.act(
            task_id, str(payload.get("action") or ""), payload, request.state.identity,
        )
        writeback = task.get("writeback") if isinstance(task.get("writeback"), dict) else {}
        if writeback.get("processing_job_id"):
            processing_job_runner.wake()
        return {
            "success": True, "task": task,
            "message": str(writeback.get("message") or "任务状态已更新"),
        }
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/knowledge/tasks/{task_id}/notify")
async def notify_knowledge_task(task_id: str, request: Request, payload: Dict[str, Any] = Body(...)):
    try:
        result = await knowledge_task_service.notify_feishu(
            task_id, str(payload.get("chat_id") or ""), request.state.identity,
        )
        return {
            "success": bool(result.get("success")),
            "notification": result,
            "message": "飞书提醒已发送" if result.get("success") else "提醒发送失败，任务数据未受影响",
        }
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


# ═══════════════════════════════════════════════
#  4. 离职交接
# ═══════════════════════════════════════════════

@app.post("/api/resignation/submit")
async def resignation_submit(
    name: str = Form(""),
    role: str = Form(""),
    recipient: str = Form(""),
    due_date: str = Form(""),
    files: List[UploadFile] = File(None),
    recipient_user_id: str = Form(""),
):
    """
    离职交接提交
    - 支持多文件上传
    - 文档自动归入「交接文档」分类
    - 按职能（role）标记，按人员（name）归档
    """
    identity = get_current_identity()
    if identity is None:
        raise HTTPException(status_code=401, detail="登录已失效，请重新登录")

    # A departure handover is always initiated for the authenticated member.
    # Keep the form field for backwards-compatible clients, but never let it
    # select another project member.
    name = str(identity.display_name or identity.email or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="当前账号缺少可用姓名")
    # 职能同样取本人账号维护的职能，表单字段仅在本人职能为空时兜底。
    actor = identity.user_id
    repository = get_repository()
    departing_user_id = identity.user_id
    resolved_recipient_user_id = (
        recipient_user_id.strip() if isinstance(recipient_user_id, str) else ""
    )
    resolved_recipient = recipient.strip()
    if resolved_recipient_user_id:
        member = repository.get_user(
            resolved_recipient_user_id,
            project_id=identity.project_id,
        )
        if (
            not member
            or member.get("status") != "active"
            or member.get("membership_status") != "active"
        ):
            raise HTTPException(status_code=400, detail="接替人不是当前项目的有效成员")
        resolved_recipient = str(member.get("display_name") or member.get("email") or "")

    role_name = str(getattr(identity, "duty", "") or "").strip() or role.strip()
    role_key = ""
    role_template_id = ""
    role_template_revision = 0
    if role_name:
        try:
            template = onboarding_service.resolve_template(role_name, fallback=False)
            role_name = str(template.get("name") or role_name)
            role_key = str(template.get("role_key") or "")
            role_template_id = str(template.get("template_id") or "")
            role_template_revision = int(template.get("revision") or 0)
        except ValueError:
            # Historical/custom roles remain recordable. Acceptance will surface
            # the missing template instead of silently assigning another role.
            pass

    handover_id = uuid.uuid4().hex[:12]
    # 记录离职人员到本地 JSON
    resign_record = {
        "id": handover_id,
        "name": name.strip(),
        "role": role_name,
        "role_key": role_key,
        "role_template_id": role_template_id,
        "role_template_revision": role_template_revision,
        "recipient": resolved_recipient,
        "recipient_user_id": resolved_recipient_user_id,
        "departing_user_id": departing_user_id,
        "created_by_user_id": actor,
        "due_date": due_date.strip(),
        "status": "pending_acceptance",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "files": [],
    }

    # 处理多个上传文件
    file_results = []
    failed_files = []
    projection_warnings = []
    processed_uploads = []
    if files:
        for file in files:
            if not file or not file.filename:
                continue
            original_filename = _sanitize_upload_filename(file.filename)
            ext = original_filename.rsplit(".", 1)[-1].lower() if "." in original_filename else ""
            if ext not in settings.ALLOWED_EXTENSIONS:
                failed_files.append({"filename": original_filename, "reason": "文件格式不支持"})
                continue

            content = await file.read()
            if len(content) > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
                failed_files.append({"filename": original_filename, "reason": "文件超过大小限制"})
                continue
            safe_name = f"{uuid.uuid4().hex}_{original_filename}"
            file_path = UPLOAD_DIR / safe_name
            indexed = False
            prepared_asset: Dict[str, Any] = {}
            file_sync_warnings = []
            with open(file_path, "wb") as f:
                f.write(content)

            try:
                docs = load_document(str(file_path))
                content_text = content.decode("utf-8", errors="ignore")

                # 强制归入「交接文档」分类
                categories_list = ["交接文档"]
                for doc in docs:
                    doc.metadata["source_file"] = original_filename
                    doc.metadata["stored_file"] = safe_name
                    doc.metadata["categories"] = categories_list
                    doc.metadata["category"] = "交接文档"
                    doc.metadata["uploader"] = name.strip()
                    doc.metadata["role"] = role_name
                    doc.metadata["handover_id"] = handover_id
                    doc.metadata["handover_recipient"] = resolved_recipient
                    doc.metadata["handover_due_date"] = due_date.strip()
                    doc.metadata["upload_date"] = datetime.now().strftime("%Y-%m-%d %H:%M")

                prepared_asset = knowledge_asset_service.prepare_documents(
                    docs,
                    source_type="handover_document",
                    external_key=f"{handover_id}:{original_filename}",
                    title=original_filename,
                    actor=actor,
                    stored_file=safe_name,
                    categories=categories_list,
                    applicable_roles=[role_key or role_name] if (role_key or role_name) else [],
                    owner_user_id=departing_user_id,
                    metadata={
                        "handover_id": handover_id,
                        "handover_recipient": resolved_recipient,
                        "handover_recipient_user_id": resolved_recipient_user_id,
                    },
                )

                chunk_count = vector_store.add_documents(docs)
                indexed = True
                knowledge_asset_service.record_projection(
                    str(prepared_asset["version_id"]), "vector", succeeded=True,
                )

                meta = project_context.extract_project_metadata(content_text)
                meta["categories"] = categories_list
                meta["source_file"] = original_filename
                meta["stored_file"] = safe_name
                meta["chunks"] = chunk_count
                meta["uploader"] = name.strip()
                meta["role"] = role_name
                meta["handover_id"] = handover_id
                meta["handover_recipient"] = resolved_recipient
                meta["handover_due_date"] = due_date.strip()
                meta.update({
                    "asset_id": prepared_asset["asset_id"], "source_id": prepared_asset["source_id"],
                    "version_id": prepared_asset["version_id"], "status": "preparing",
                })
                context_ready = True
                context_error = ""
                try:
                    project_context.register_document(safe_name, meta)
                except Exception as exc:
                    context_ready = False
                    context_error = str(exc)
                    file_sync_warnings.append("项目语境")
                    logger.exception("交接附件项目语境更新失败: %s", safe_name)

                asset = knowledge_asset_service.publish(
                    prepared_asset, actor=actor, graph_ready=False,
                    context_ready=context_ready, context_error=context_error,
                )

                try:
                    knowledge_graph.build_graph()
                    knowledge_asset_service.mark_post_publish_projection(
                        str(prepared_asset["version_id"]), "graph", True,
                    )
                except Exception as exc:
                    file_sync_warnings.append("知识图谱")
                    knowledge_asset_service.mark_post_publish_projection(
                        str(prepared_asset["version_id"]), "graph", False, str(exc),
                    )

                try:
                    get_repository().readiness_assets(identity=identity)
                    knowledge_asset_service.mark_post_publish_projection(
                        str(prepared_asset["version_id"]), "readiness", True,
                    )
                except Exception as exc:
                    file_sync_warnings.append("新人就绪度")
                    knowledge_asset_service.mark_post_publish_projection(
                        str(prepared_asset["version_id"]), "readiness", False, str(exc),
                    )

                file_results.append({
                    "filename": original_filename,
                    "chunks": chunk_count,
                    "asset_id": asset.get("asset_id"),
                    "projection_status": "repair_required" if file_sync_warnings else "healthy",
                    "sync_warnings": file_sync_warnings,
                })
                if file_sync_warnings:
                    projection_warnings.append({
                        "filename": original_filename,
                        "asset_id": asset.get("asset_id"),
                        "projections": file_sync_warnings,
                    })
                processed_uploads.append({
                    "stored_file": safe_name, "path": file_path,
                    "asset_id": prepared_asset.get("asset_id"), "version_id": prepared_asset.get("version_id"),
                })
            except Exception as e:
                if prepared_asset:
                    knowledge_asset_service.fail(prepared_asset, e, actor=actor)
                if indexed:
                    try:
                        vector_store.delete_documents_by_source(safe_name)
                        project_context.unregister_document(safe_name)
                    except Exception:
                        logger.exception("回滚失败的交接附件: %s", safe_name)
                if file_path.exists():
                    file_path.unlink()
                diagnostic_id = uuid.uuid4().hex[:10]
                logger.exception("交接附件处理失败 [%s]: %s", diagnostic_id, original_filename)
                failed_files.append({
                    "filename": original_filename,
                    "reason": f"处理失败（诊断编号：{diagnostic_id}）",
                })

    onboarding_sync = _refresh_onboarding_after_knowledge_change(
        asset_id=str(file_results[-1].get("asset_id") or "") if file_results else "",
        actor=actor, trigger="handover_assets_published",
    ) if file_results else None

    resign_record["files"] = [item["filename"] for item in file_results]
    resign_record["asset_ids"] = [
        str(item["asset_id"]) for item in file_results if item.get("asset_id")
    ]
    records = _load_handover_records()
    records.insert(0, resign_record)
    try:
        _save_handover_records(records)
    except Exception as e:
        for uploaded in processed_uploads:
            try:
                if uploaded.get("asset_id"):
                    knowledge_asset_service.transition(
                        str(uploaded["asset_id"]), "revoke", actor, reason="交接记录保存失败",
                    )
                vector_store.delete_documents_by_source(uploaded["stored_file"])
                project_context.unregister_document(uploaded["stored_file"])
                uploaded["path"].unlink(missing_ok=True)
            except Exception:
                logger.exception("交接记录失败后的附件回滚失败: %s", uploaded["stored_file"])
        diagnostic_id = uuid.uuid4().hex[:10]
        logger.exception("保存交接记录失败 [%s]", diagnostic_id)
        raise HTTPException(
            status_code=500,
            detail=f"交接记录保存失败，附件未入库（诊断编号：{diagnostic_id}）",
        ) from e

    partial_success = bool(failed_files or projection_warnings)
    return {
        "success": True,
        "partial_success": partial_success,
        "repair_required": projection_warnings,
        "projection_status": "repair_required" if projection_warnings else "healthy",
        "message": (
            f"离职交接已提交，{len(file_results)} 个交接文档已归档至知识库"
            + (f"，{len(failed_files)} 个文件未处理" if failed_files else "")
            + (f"，{len(projection_warnings)} 个文档派生同步待修复" if projection_warnings else "")
            + f"（分类：交接文档，作者：{name.strip()}）"
        ),
        "person": {"name": name.strip(), "role": role_name, "status": "pending_acceptance"},
        "files": file_results,
        "onboarding_sync": onboarding_sync,
        "failed_files": failed_files,
        "sync_warnings": projection_warnings,
        "handover": _handover_view(resign_record),
    }


@app.get("/api/resignation/records")
async def list_resignation_records():
    """返回可追踪的交接记录、接替确认状态和风险。"""
    return {"success": True, "records": [_handover_view(record) for record in _load_handover_records()]}


@app.get("/api/resignation/{handover_id}")
async def get_resignation_handover(handover_id: str):
    try:
        return {"success": True, "record": handover_service.detail(handover_id, get_current_identity())}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/resignation/{handover_id}/items/{item_id}/actions")
async def act_on_handover_item(
    handover_id: str, item_id: str, payload: Dict[str, Any] = Body(...),
):
    identity = get_current_identity()
    if identity is None:
        raise HTTPException(status_code=401, detail="登录已失效，请重新登录")
    try:
        return handover_service.item_action(
            handover_id, item_id, str(payload.get("action") or ""), payload, identity,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        status = 404 if str(exc) in {"交接记录不存在", "交接项不存在"} else 409
        raise HTTPException(status_code=status, detail=str(exc)) from exc


@app.get("/api/resignation/{handover_id}/items/{item_id}/submission-context")
async def get_handover_item_submission_context(handover_id: str, item_id: str):
    identity = get_current_identity()
    if identity is None:
        raise HTTPException(status_code=401, detail="登录已失效，请重新登录")
    try:
        return handover_service.submission_context(handover_id, item_id, identity)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        status = 404 if str(exc) in {"交接记录不存在", "交接项不存在"} else 409
        raise HTTPException(status_code=status, detail=str(exc)) from exc


@app.put("/api/resignation/{handover_id}/items/{item_id}/draft")
async def save_handover_item_submission_draft(
    handover_id: str, item_id: str, payload: Dict[str, Any] = Body(...),
):
    identity = get_current_identity()
    if identity is None:
        raise HTTPException(status_code=401, detail="登录已失效，请重新登录")
    try:
        return handover_service.save_submission_draft(handover_id, item_id, payload, identity)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        status = 404 if str(exc) in {"交接记录不存在", "交接项不存在"} else 409
        raise HTTPException(status_code=status, detail=str(exc)) from exc


@app.post("/api/resignation/{handover_id}/items/{item_id}/attachments")
async def upload_handover_item_attachment(
    handover_id: str, item_id: str, file: UploadFile = File(...),
):
    """Ingest a supplemental file and keep it in the submitter's item draft."""
    identity = get_current_identity()
    if identity is None:
        raise HTTPException(status_code=401, detail="登录已失效，请重新登录")
    try:
        # Check submission and recipient access prerequisites before accepting a file.
        context = handover_service.submission_context(handover_id, item_id, identity)
        if len(context["draft"].get("asset_refs") or []) >= 10:
            raise ValueError("每个必交项最多关联 10 份资料，请先移除草稿中的资料后再上传")
        upload = await _upload_document_impl(
            file, category="交接文档", uploader=identity.display_name,
        )
        asset = dict(upload.get("asset") or {})
        asset_id = str(asset.get("asset_id") or "")
        version_id = str(asset.get("current_version_id") or "")
        if not asset_id or not version_id:
            raise RuntimeError("补充文件已入库，但未取得可关联的知识资产版本")
        draft_result = handover_service.append_uploaded_draft_asset(
            handover_id, item_id, asset_id, version_id, identity,
        )
        return {
            "success": True,
            "filename": str(upload.get("filename") or file.filename or ""),
            "asset_id": asset_id,
            "version_id": version_id,
            "draft": draft_result["draft"],
            "message": draft_result["message"],
        }
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        status = 404 if str(exc) in {"交接记录不存在", "交接项不存在"} else 409
        raise HTTPException(status_code=status, detail=str(exc)) from exc


@app.post("/api/resignation/{handover_id}/inventory/refresh")
async def refresh_handover_inventory(handover_id: str):
    identity = get_current_identity()
    if identity is None:
        raise HTTPException(status_code=401, detail="登录已失效，请重新登录")
    try:
        return handover_service.refresh_inventory(handover_id, identity)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        status = 404 if str(exc) == "交接记录不存在" else 409
        raise HTTPException(status_code=status, detail=str(exc)) from exc


@app.post("/api/resignation/{handover_id}/inventory/{inventory_id}/actions")
async def act_on_handover_inventory(
    handover_id: str, inventory_id: str, payload: Dict[str, Any] = Body(...),
):
    identity = get_current_identity()
    if identity is None:
        raise HTTPException(status_code=401, detail="登录已失效，请重新登录")
    try:
        return handover_service.inventory_action(
            handover_id, inventory_id, str(payload.get("action") or ""), payload, identity,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        status = 404 if str(exc) in {"交接记录不存在", "知识盘点项不存在"} else 409
        raise HTTPException(status_code=status, detail=str(exc)) from exc


@app.post("/api/resignation/{handover_id}/risks")
async def create_handover_risk(handover_id: str, payload: Dict[str, Any] = Body(...)):
    identity = get_current_identity()
    if identity is None:
        raise HTTPException(status_code=401, detail="登录已失效，请重新登录")
    try:
        return handover_service.create_risk(handover_id, payload, identity)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        status = 404 if str(exc) == "交接记录不存在" else 409
        raise HTTPException(status_code=status, detail=str(exc)) from exc


@app.post("/api/resignation/{handover_id}/risks/{risk_id}/actions")
async def act_on_handover_risk(
    handover_id: str, risk_id: str, payload: Dict[str, Any] = Body(...),
):
    identity = get_current_identity()
    if identity is None:
        raise HTTPException(status_code=401, detail="登录已失效，请重新登录")
    try:
        return handover_service.risk_action(
            handover_id, risk_id, str(payload.get("action") or ""), payload, identity,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        status = 404 if str(exc) in {"交接记录不存在", "交接风险不存在"} else 409
        raise HTTPException(status_code=status, detail=str(exc)) from exc


@app.post("/api/resignation/{handover_id}/complete")
async def complete_resignation_handover(handover_id: str):
    identity = get_current_identity()
    if identity is None:
        raise HTTPException(status_code=401, detail="登录已失效，请重新登录")
    try:
        return handover_service.complete(handover_id, identity)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        status = 404 if str(exc) == "交接记录不存在" else 409
        raise HTTPException(status_code=status, detail=str(exc)) from exc


@app.post("/api/resignation/{handover_id}/cancel")
async def cancel_resignation_handover(handover_id: str):
    """Cancel a still-unaccepted handover at its initiator's request."""
    identity = get_current_identity()
    if identity is None:
        raise HTTPException(status_code=401, detail="登录已失效，请重新登录")
    try:
        return handover_service.cancel(handover_id, identity)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        status = 404 if str(exc) == "交接记录不存在" else 409
        raise HTTPException(status_code=status, detail=str(exc)) from exc


@app.get("/api/resignation/{handover_id}/snapshot")
async def get_resignation_snapshot(handover_id: str):
    snapshot = get_repository().get_handover_snapshot(handover_id)
    if not snapshot:
        raise HTTPException(status_code=404, detail="交接快照不存在")
    return {"success": True, "snapshot": snapshot}


@app.post("/api/resignation/{handover_id}/accept")
async def accept_resignation_handover(handover_id: str, accepted_by: str = Form("")):
    identity = get_current_identity()
    if identity is None:
        raise HTTPException(status_code=401, detail="登录已失效，请重新登录")
    try:
        result = handover_service.accept(handover_id, identity)
        return {
            **result,
            "record": result["record"],
            "learning_plan": result.get("plan"),
        }
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        status_code = 404 if str(exc) == "交接记录不存在" else 409
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc


@app.get("/api/onboarding/handover")
async def onboarding_handover_report(handover_id: str):
    """为接替人提供交接资产盘点和待办摘要。"""
    for record in _load_handover_records():
        if record.get("id") == handover_id:
            view = _handover_view(record)
            return {
                "success": True,
                "handover": view,
                "summary": f"{view['name']} 的 {view['role'] or '项目'} 交接包含 {view['file_count']} 份资料。",
                "next_actions": [
                    "阅读交接资料并标记关键文档",
                    "核对未关闭风险和责任人",
                    (
                        "在新人赋能中执行已生成的接替人学习计划"
                        if view.get("learning_plan_status") == "ready"
                        else "确认接收并生成接替人学习计划"
                    ),
                ],
            }
    raise HTTPException(status_code=404, detail="交接记录不存在")

# ═══════════════════════════════════════════════
#  5. 知识图谱
# ═══════════════════════════════════════════════

@app.get("/api/knowledge-graph")
async def get_knowledge_graph():
    graph = knowledge_graph.get_graph_data() if hasattr(knowledge_graph, 'get_graph_data') else {"nodes": [], "edges": []}
    return {"success": True, "graph": graph}


@app.put("/api/knowledge-graph")
async def save_knowledge_graph(graph: Dict[str, Any] = Body(...)):
    """保存前端编辑后的知识图谱快照"""
    try:
        saved = knowledge_graph.save_graph_data(graph)
        return {"success": True, "message": "图谱已保存", "graph": saved}
    except Exception as e:
        raise _internal_http_error("保存知识图谱失败", "图谱保存失败，请稍后重试") from e


@app.post("/api/knowledge-graph/trace")
async def trace_decision(query: str = Form(...)):
    trace = knowledge_graph.trace_decision(query) if hasattr(knowledge_graph, 'trace_decision') else {}
    return {"success": True, "trace": trace}


@app.get("/api/knowledge-graph/stats")
async def graph_stats():
    graph = knowledge_graph.get_graph_data() if hasattr(knowledge_graph, 'get_graph_data') else {"nodes": [], "edges": []}
    stats = {"nodes": len(graph.get("nodes", [])), "edges": len(graph.get("edges", []))}
    return {"success": True, "stats": stats}


@app.post("/api/knowledge-graph/build")
async def build_graph():
    """手动触发图谱构建"""
    identity = _require_knowledge_permission("knowledge.manage")
    job = processing_job_service.enqueue(
        "graph_rebuild", {}, created_by=identity.user_id if identity else "user_system",
        idempotency_key="graph-rebuild", priority=65,
    )
    return {"success": True, "accepted": True, "job": job, "message": "图谱重建已进入处理队列"}


# ═══════════════════════════════════════════════
#  6. 飞书集成
# ═══════════════════════════════════════════════

@app.get("/api/feishu/webhook")
@app.post("/api/feishu/webhook")
async def feishu_webhook(request: Request):
    logger.info("收到飞书请求: method=%s", request.method)

    # GET 请求 — 飞书 URL 可用性检查
    if request.method == "GET":
        return {"success": True, "message": "feishu webhook is ready"}

    raw_body = await request.body()
    try:
        body = json.loads(raw_body.decode("utf-8"))
        if not isinstance(body, dict):
            raise ValueError("callback body must be an object")
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail="飞书回调格式无效") from exc
    from backend.feishu_bot import (
        FeishuWebhookVerificationError,
        handle_feishu_webhook,
        verify_feishu_webhook_request,
    )
    try:
        verify_feishu_webhook_request(raw_body, request.headers, body)
    except FeishuWebhookVerificationError as exc:
        logger.warning("拒绝未通过验证的飞书回调: %s", exc)
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    logger.info("飞书回调: event_type=%s, challenge=%s",
                body.get("header", {}).get("event_type", body.get("type", "N/A")),
                "yes" if body.get("challenge") else "no")
    return await handle_feishu_webhook(body)


@app.post("/api/feishu/push/knowledge")
async def push_knowledge(chat_id: str = Form(None)):
    from backend.feishu_bot import FeishuBot
    bot = FeishuBot()
    # 查询随机知识内容推送
    from backend.knowledge_base.retriever import retriever
    docs = retriever.retrieve("知识推荐", k=5)
    knowledge_items = [
        {"title": d.metadata.get("source_file", "未知"), "summary": d.page_content[:200]}
        for d in docs
    ]
    ok = await bot.push_daily_knowledge(chat_id or settings.FEISHU_DEFAULT_CHAT_ID, knowledge_items)
    return {"success": ok, "message": "推送成功" if ok else "推送失败（无知识内容或飞书未配置）"}


def _feishu_workspace_payload(chat_id: str = "", keyword: str = "") -> Dict[str, Any]:
    """构造不包含 App Secret 的飞书工作台快照。"""
    app_id = (settings.FEISHU_APP_ID or "").strip()
    masked_app_id = ""
    if app_id:
        masked_app_id = app_id if len(app_id) <= 8 else f"{app_id[:4]}••••{app_id[-4:]}"
    return {
        "success": True,
        "connection": {
            "configured": bool(app_id and settings.FEISHU_APP_SECRET),
            "app_id_masked": masked_app_id,
            "default_chat_id": settings.FEISHU_DEFAULT_CHAT_ID or "",
            "bot_name": settings.FEISHU_BOT_NAME,
            "callback_path": "/api/feishu/webhook",
            "transport": feishu_long_connection.status(),
        },
        "groups": feishu_workspace.list_groups(),
        "messages": feishu_workspace.list_messages(chat_id=chat_id, keyword=keyword, limit=30),
        "candidates": feishu_workspace.list_candidates(limit=200),
        "assets": feishu_workspace.query_assets(statuses=("published",), page_size=50)["items"],
        "audit": feishu_workspace.list_audit(limit=80),
        "summary": feishu_workspace.summary(),
        "diagnostics": feishu_workspace.diagnostics(),
    }


@app.get("/api/feishu/workspace")
async def get_feishu_workspace(chat_id: str = "", keyword: str = ""):
    """返回飞书工作台状态；不向前端泄露 App Secret。"""
    return _feishu_workspace_payload(chat_id, keyword)


@app.get("/api/feishu/content/messages")
async def get_feishu_content_messages(
    chat_id: str = "",
    keyword: str = "",
    statuses: str = "",
    exclude_statuses: str = "",
    message_type: str = "",
    date_from: str = "",
    date_to: str = "",
    page: int = 1,
    page_size: int = 20,
):
    """按内容中心当前视图分页返回飞书消息。"""
    result = feishu_workspace.query_items(
        chat_id=chat_id,
        keyword=keyword,
        content_statuses=tuple(item for item in statuses.split(",") if item),
        excluded_statuses=tuple(item for item in exclude_statuses.split(",") if item),
        message_type=message_type,
        date_from=date_from,
        date_to=date_to,
        page=page,
        page_size=page_size,
    )
    return {"success": True, **result}


@app.get("/api/feishu/content/assets")
async def get_feishu_content_assets(
    statuses: str = "published",
    keyword: str = "",
    page: int = 1,
    page_size: int = 20,
):
    """分页返回当前知识资产或撤销历史。"""
    result = feishu_workspace.query_assets(
        statuses=tuple(item for item in statuses.split(",") if item),
        keyword=keyword,
        page=page,
        page_size=page_size,
    )
    return {"success": True, **result}


@app.get("/api/feishu/events")
async def stream_feishu_workspace_events(request: Request):
    """将飞书消息和审核状态变化实时推送到工作台。"""
    async def generate():
        last_signature = None
        heartbeat = 0
        try:
            while not await request.is_disconnected():
                transport = feishu_long_connection.status()
                signature = (
                    feishu_workspace.revision(),
                    transport.get("state"),
                    transport.get("connected"),
                    transport.get("last_error"),
                )
                if signature != last_signature:
                    payload = _feishu_workspace_payload()
                    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                    yield f"retry: 2000\nevent: workspace\ndata: {data}\n\n"
                    last_signature = signature
                    heartbeat = 0
                else:
                    heartbeat += 1
                    if heartbeat >= 15:
                        yield ": keep-alive\n\n"
                        heartbeat = 0
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            return

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/feishu/connection/start")
async def start_feishu_connection():
    if not (settings.FEISHU_APP_ID and settings.FEISHU_APP_SECRET):
        raise HTTPException(status_code=400, detail="请先保存飞书 App ID 和 App Secret")
    feishu_long_connection.start(asyncio.get_running_loop())
    for _ in range(20):
        await asyncio.sleep(0.15)
        status = feishu_long_connection.status()
        if status["connected"] or status["state"] == "error":
            break
    status = feishu_long_connection.status()
    return {
        "success": status["connected"],
        "status": status,
        "message": "飞书事件长连接已建立" if status["connected"] else status.get("last_error") or "正在建立飞书事件长连接",
    }


@app.post("/api/feishu/permissions/check")
async def check_feishu_permissions():
    """验证可安全探测的飞书能力，未执行写操作的权限明确标记为待验证。"""
    from backend.feishu_bot import feishu_bot

    configured = bool((settings.FEISHU_APP_ID or "").strip() and (settings.FEISHU_APP_SECRET or "").strip())
    capabilities = [
        {
            "key": "credentials",
            "label": "应用凭证",
            "status": "configured" if configured else "missing",
            "detail": "App ID 与 App Secret 已保存" if configured else "请先保存 App ID 与 App Secret",
        }
    ]
    if not configured:
        for key, label in (("authentication", "应用鉴权"), ("chat_read", "群聊读取"), ("event_receive", "消息接收"), ("message_resource", "图片资源读取"), ("image_extraction", "图片内容识别"), ("message_send", "消息发送"), ("document_read", "文档读取")):
            capabilities.append({"key": key, "label": label, "status": "unverified", "detail": "凭证缺失，尚未验证"})
        return {"success": True, "configured": False, "verified": False, "capabilities": capabilities, "checked_at": datetime.now().isoformat(timespec="seconds")}

    authenticated = False
    auth_error = ""
    try:
        authenticated = bool(await feishu_bot._get_app_access_token())
    except Exception as exc:
        auth_error = str(exc)[:240]
    capabilities.append({
        "key": "authentication", "label": "应用鉴权",
        "status": "verified" if authenticated else "failed",
        "detail": "已通过飞书接口取得访问令牌" if authenticated else auth_error or "飞书未返回有效访问令牌",
    })

    chat_read = False
    chat_error = ""
    if authenticated:
        try:
            await feishu_bot.list_chats(page_size=1)
            chat_read = True
        except Exception as exc:
            chat_error = str(exc)[:240]
    capabilities.append({
        "key": "chat_read", "label": "群聊读取", "status": "verified" if chat_read else "failed",
        "detail": "已验证机器人群列表读取能力" if chat_read else chat_error or "群聊读取权限未验证",
    })

    transport = feishu_long_connection.status()
    diagnostics = feishu_workspace.diagnostics()
    event_verified = bool(diagnostics.get("last_event_at"))
    workspace_messages = feishu_workspace.list_items(limit=500)
    resource_verified = any(int(item.get("resource_count") or 0) > 0 for item in workspace_messages)
    local_ocr_verified = any("local_rapidocr" in str(item.get("extraction_method") or "") for item in workspace_messages)
    ocr_verified = any("feishu_ocr" in str(item.get("extraction_method") or "") for item in workspace_messages)
    vision_verified = any("vision_model" in str(item.get("extraction_method") or "") for item in workspace_messages)
    image_attempted = any(item.get("message_type") in {"image", "post"} and item.get("image_keys") for item in workspace_messages)
    capabilities.extend([
        {
            "key": "event_receive", "label": "消息接收",
            "status": "verified" if event_verified else ("connected" if transport.get("connected") else "unverified"),
            "detail": "已收到真实飞书消息事件" if event_verified else "事件通道已连接，等待真实消息验证" if transport.get("connected") else "事件通道未连接",
        },
        {
            "key": "message_resource", "label": "图片资源读取",
            "status": "verified" if resource_verified else ("failed" if image_attempted else "unverified"),
            "detail": "已成功保存飞书消息图片原件" if resource_verified else "已收到图片但资源下载未成功，请检查 im:resource 权限" if image_attempted else "发送一张图片后验证 im:resource 权限",
        },
        {
            "key": "image_extraction", "label": "图片内容识别",
            "status": "verified" if (local_ocr_verified or ocr_verified or vision_verified) else ("failed" if image_attempted else "unverified"),
            "detail": "已通过本地 RapidOCR 提取图片文字" if local_ocr_verified else "已通过飞书 OCR 提取图片文字" if ocr_verified else "已通过多模态模型提取图片内容" if vision_verified else "图片识别失败，请检查本地 OCR、飞书 OCR 权限或多模态模型" if image_attempted else "发送含文字截图后验证识别能力",
        },
        {"key": "message_send", "label": "消息发送", "status": "unverified", "detail": "为避免测试消息打扰群聊，需通过实际机器人回复验证"},
        {"key": "document_read", "label": "文档读取", "status": "unverified", "detail": "需在收到真实文件或飞书文档后验证对应权限"},
    ])
    all_verified = all(item["status"] == "verified" for item in capabilities if item["key"] in {"authentication", "chat_read", "event_receive"})
    feishu_workspace.record_audit("permissions_checked", "connection", "feishu", "核心能力已验证" if all_verified else "部分能力仍待验证", actor="管理员")
    return {
        "success": True, "configured": True, "verified": all_verified,
        "capabilities": capabilities, "checked_at": datetime.now().isoformat(timespec="seconds"),
    }


@app.get("/api/feishu/chats")
async def discover_feishu_chats():
    """发现机器人所在群聊，供管理员选择接入。"""
    from backend.feishu_bot import feishu_bot
    try:
        chats, next_page_token = await feishu_bot.list_chats(page_size=100)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    connected = {str(group.get("chat_id") or ""): group for group in feishu_workspace.list_groups()}
    items = []
    for chat in chats:
        chat_id = str(chat.get("chat_id") or "")
        if not chat_id:
            continue
        group = connected.get(chat_id)
        items.append({
            "chat_id": chat_id,
            "name": str(chat.get("name") or "未命名群聊"),
            "description": str(chat.get("description") or ""),
            "avatar": str(chat.get("avatar") or ""),
            "chat_mode": str(chat.get("chat_mode") or "group"),
            "external": bool(chat.get("external", False)),
            "connected": bool(group),
            "collection_mode": group.get("collection_mode") if group else ("archive_only" if chat.get("external", False) else "auto"),
            "synced_count": int(group.get("last_sync_count") or 0) if group else 0,
        })
    return {
        "success": True,
        "chats": items,
        "has_more": bool(next_page_token),
        "message": f"已发现 {len(items)} 个机器人所在群聊",
    }


@app.post("/api/feishu/groups")
async def save_feishu_group(payload: Dict[str, Any] = Body(...)):
    try:
        group = feishu_workspace.upsert_group(payload)
        return {"success": True, "group": group, "message": "群采集规则已保存"}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/feishu/policies/preview")
async def preview_feishu_policy(payload: Dict[str, Any] = Body(...)):
    chat_id = str(payload.get("chat_id") or "").strip()
    if not chat_id:
        raise HTTPException(status_code=400, detail="请选择需要预览规则的采集源")
    try:
        preview = feishu_workspace.preview_policy(chat_id, payload, limit=int(payload.get("limit") or 100))
        return {"success": True, "preview": preview}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/feishu/items")
async def list_feishu_items(
    chat_id: str = "",
    keyword: str = "",
    content_status: str = "",
    message_type: str = "",
    limit: int = 300,
):
    items = feishu_workspace.list_items(
        chat_id=chat_id,
        keyword=keyword,
        content_status=content_status,
        message_type=message_type,
        limit=limit,
    )
    return {"success": True, "items": items, "count": len(items)}


@app.get("/api/feishu/messages/{message_id}/resources/{resource_index}")
async def get_feishu_message_resource(message_id: str, resource_index: int):
    """以内联方式返回已归档的飞书图片，不暴露本地文件路径。"""
    message = feishu_workspace.get_message(message_id)
    if not message:
        raise HTTPException(status_code=404, detail="消息记录不存在")
    resources = message.get("resource_files") or []
    if resource_index < 0 or resource_index >= len(resources):
        raise HTTPException(status_code=404, detail="图片资源不存在")
    resource = resources[resource_index] if isinstance(resources[resource_index], dict) else {}
    stored_file = str(resource.get("stored_file") or "")
    if not stored_file or Path(stored_file).name != stored_file:
        raise HTTPException(status_code=404, detail="图片资源尚未保存")
    file_path = UPLOAD_DIR / "feishu_resources" / stored_file
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="图片文件不存在")
    return FileResponse(
        path=file_path,
        media_type=str(resource.get("content_type") or "application/octet-stream"),
        headers={"Cache-Control": "private, max-age=3600"},
    )


@app.post("/api/feishu/messages/{message_id}/enrich")
async def retry_feishu_message_enrichment(message_id: str):
    """Compatibility endpoint that still performs enrichment synchronously."""
    message = feishu_workspace.get_message(message_id)
    if not message:
        raise HTTPException(status_code=404, detail="消息记录不存在")
    if message.get("message_type") not in {"image", "post"}:
        raise HTTPException(status_code=400, detail="该消息不需要图片或富文本识别")
    try:
        raw_message = await feishu_bot.retry_enrich_message(message)
        updated = feishu_workspace.replace_enriched_message(str(message.get("chat_id") or ""), raw_message)
        if not updated:
            raise RuntimeError("消息识别结果保存失败")
        if updated.get("extraction_status") == "failed":
            raise RuntimeError(updated.get("extraction_error") or "图片识别仍未成功")
        group = feishu_workspace.get_group(str(updated.get("chat_id") or "")) or {}
        published = 0
        if group.get("collection_mode") == "auto":
            ready_ids = [
                item["candidate_id"]
                for item in feishu_workspace.list_candidates(status="ready_for_auto", chat_id=str(updated.get("chat_id") or ""))
            ]
            published = len(feishu_knowledge_service.publish_ready_candidates(ready_ids)["published"])
        return {
            "success": True,
            "record": updated,
            "published": published,
            "message": "图片与富文本内容已重新识别" + (f"，自动入库 {published} 项" if published else ""),
        }
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/feishu/messages/{message_id}/enrichments", status_code=202)
async def enqueue_feishu_message_enrichment(message_id: str):
    """Queue one existing image or rich-text message for durable OCR processing."""
    identity = _require_knowledge_permission("knowledge.publish")
    message = feishu_workspace.get_message(message_id)
    if not message:
        raise HTTPException(status_code=404, detail="消息记录不存在")
    if message.get("message_type") not in {"image", "post"}:
        raise HTTPException(status_code=400, detail="该消息不需要图片或富文本识别")
    job = processing_job_service.enqueue(
        "feishu_message_enrichment",
        {
            "message_id": message_id,
            "trigger": "manual",
            "actor_user_id": identity.user_id,
        },
        created_by=identity.user_id,
        idempotency_key=f"feishu-enrichment:{message.get('chat_id') or ''}:{message_id}",
        source_id=str(message.get("chat_id") or ""),
        priority=80,
    )
    return {
        "success": True,
        "accepted": True,
        "job": job,
        "message": "图片与富文本重新识别已进入处理任务",
    }


@app.get("/api/feishu/candidates")
async def list_feishu_candidates(status: str = "", chat_id: str = "", limit: int = 300):
    candidates = feishu_workspace.list_candidates(status=status, chat_id=chat_id, limit=limit)
    return {"success": True, "candidates": candidates, "count": len(candidates)}


@app.post("/api/feishu/candidates/batch-action")
async def batch_feishu_candidates(payload: Dict[str, Any] = Body(...)):
    """Compatibility endpoint for integrations that still require a synchronous result."""
    identity = _require_knowledge_permission("knowledge.publish")
    candidate_ids = list(dict.fromkeys(str(item).strip() for item in (payload.get("candidate_ids") or []) if str(item).strip()))
    action = str(payload.get("action") or "").strip()
    category = str(payload.get("category") or "").strip()
    if not candidate_ids:
        raise HTTPException(status_code=400, detail="请至少选择一个知识候选")
    if action not in {"approve", "exclude", "category", "retry"}:
        raise HTTPException(status_code=400, detail="不支持的批量操作")
    if action == "category" and not category:
        raise HTTPException(status_code=400, detail="请选择要应用的知识分类")

    return feishu_knowledge_service.batch_action(
        candidate_ids=candidate_ids,
        action=action,
        category=category,
        actor=identity.user_id if identity else "system",
    )


@app.post("/api/feishu/candidates/batch-actions", status_code=202)
async def enqueue_feishu_candidate_batch(payload: Dict[str, Any] = Body(...)):
    """Persist a bounded candidate batch and return before per-item processing starts."""
    identity = _require_knowledge_permission("knowledge.publish")
    candidate_ids = list(dict.fromkeys(
        str(item).strip() for item in (payload.get("candidate_ids") or []) if str(item).strip()
    ))
    action = str(payload.get("action") or "").strip()
    category = str(payload.get("category") or "").strip()
    if not candidate_ids:
        raise HTTPException(status_code=400, detail="请至少选择一个知识候选")
    if len(candidate_ids) > 100:
        raise HTTPException(status_code=400, detail="单次最多处理 100 个知识候选")
    if action not in {"approve", "exclude", "category", "retry"}:
        raise HTTPException(status_code=400, detail="不支持的批量操作")
    if action == "category" and not category:
        raise HTTPException(status_code=400, detail="请选择要应用的知识分类")

    digest = hashlib.sha256(json.dumps({
        "action": action, "category": category, "candidate_ids": sorted(candidate_ids),
    }, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    job = processing_job_service.enqueue(
        "feishu_candidate_batch",
        {
            "candidate_ids": candidate_ids,
            "action": action,
            "category": category,
            "actor_user_id": identity.user_id,
        },
        created_by=identity.user_id,
        idempotency_key=f"feishu-candidate-batch:{identity.project_id}:{digest}",
        max_attempts=3,
        priority=70,
    )
    return {
        "success": True,
        "accepted": True,
        "job": job,
        "message": f"已提交 {len(candidate_ids)} 项候选，正在后台逐项处理",
    }


@app.post("/api/feishu/assets/{asset_id}/revert")
async def revert_feishu_asset(asset_id: str, payload: Dict[str, Any] = Body(default={})):
    identity = _require_knowledge_permission("knowledge.publish")
    try:
        reason = str(payload.get("reason") or "").strip()
        if not reason:
            raise ValueError("撤销原因不能为空")
        result = feishu_knowledge_service.revert_asset(
            asset_id, reason=reason, actor=identity.user_id if identity else "system",
        )
        message = "知识资产已撤销，原始飞书记录与审计信息已保留"
        if result.get("partial_success"):
            message += "；撤销已生效，但" + "、".join(result.get("sync_warnings") or ["派生数据待修复"])
            if result.get("diagnostic_id"):
                message += f"（诊断编号：{result['diagnostic_id']}，可再次撤销重试同步）"
        return {**result, "message": message}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        diagnostic_id = uuid.uuid4().hex[:10]
        logger.exception("撤销飞书知识资产失败 [%s]: %s", diagnostic_id, asset_id)
        raise HTTPException(
            status_code=500,
            detail=f"撤销失败，请稍后重试（诊断编号：{diagnostic_id}）",
        ) from exc


@app.post("/api/feishu/assets/batch-revert")
async def batch_revert_feishu_assets(payload: Dict[str, Any] = Body(...)):
    identity = _require_knowledge_permission("knowledge.publish")
    try:
        reason = str(payload.get("reason") or "").strip()
        if not reason:
            raise ValueError("撤销原因不能为空")
        return feishu_knowledge_service.batch_revert(
            payload.get("asset_ids") or [],
            reason=reason,
            actor=identity.user_id if identity else "system",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/feishu/audit")
async def list_feishu_audit(action: str = "", limit: int = 200):
    events = feishu_workspace.list_audit(action=action, limit=limit)
    return {"success": True, "events": events, "count": len(events)}


async def _sync_feishu_group_history(chat_id: str, page_size: int, context=None) -> Dict[str, Any]:
    """Perform one authorized Feishu history sync for the persistent job runner."""
    group = feishu_workspace.get_group(chat_id)
    if not group:
        raise ProcessingJobError("请先在工作台添加该项目群", code="FEISHU_GROUP_NOT_FOUND", retryable=False)
    if group.get("collection_mode") == "off":
        raise ProcessingJobError("该群已关闭消息采集，请先启用采集规则", code="FEISHU_COLLECTION_OFF", retryable=False)
    if not (settings.FEISHU_APP_ID and settings.FEISHU_APP_SECRET):
        raise ProcessingJobError("请先保存飞书 App ID 和 App Secret", code="FEISHU_NOT_CONFIGURED", retryable=False)

    from backend.feishu_bot import feishu_bot
    messages, next_page_token = await feishu_bot.list_chat_messages(chat_id, page_size=page_size)
    if context:
        context.report("feishu_enrich", 40, f"已读取 {len(messages)} 条消息，正在提取内容")
    messages = await feishu_bot.enrich_messages(
        messages,
        skip_message_ids=feishu_workspace.existing_message_ids(),
    )
    if context:
        context.report("feishu_archive", 70, "正在归档消息并聚合知识候选")
    added = feishu_workspace.upsert_messages(chat_id, messages)
    ready_ids = [
        item["candidate_id"]
        for item in feishu_workspace.list_candidates(status="ready_for_auto", chat_id=chat_id)
    ]
    publish_result = feishu_knowledge_service.publish_ready_candidates(ready_ids) if group.get("collection_mode") == "auto" else {"published": [], "failed": []}
    if context:
        context.report("feishu_publish", 90, "正在确认自动入库结果")
    return {
        "added": added,
        "received": len(messages),
        "has_more": bool(next_page_token),
        "published": len(publish_result["published"]),
        "publish_failed": len(publish_result["failed"]),
        "message": f"已同步 {len(messages)} 条消息，新增 {added} 条记录，自动入库 {len(publish_result['published'])} 项",
    }


@app.post("/api/feishu/groups/{chat_id}/sync")
async def sync_feishu_group_messages(chat_id: str, page_size: int = Form(50)):
    """Queue authorized Feishu history synchronization and return immediately."""
    identity = get_current_identity()
    group = feishu_workspace.get_group(chat_id)
    if not group:
        raise HTTPException(status_code=404, detail="请先在工作台添加该项目群")
    if group.get("collection_mode") == "off":
        raise HTTPException(status_code=400, detail="该群已关闭消息采集，请先启用采集规则")
    if not (settings.FEISHU_APP_ID and settings.FEISHU_APP_SECRET):
        raise HTTPException(status_code=400, detail="请先保存飞书 App ID 和 App Secret")
    job = processing_job_service.enqueue(
        "feishu_group_sync", {"chat_id": chat_id, "page_size": max(1, min(page_size, 100))},
        created_by=identity.user_id if identity else "user_system",
        idempotency_key=f"feishu-sync:{chat_id}", source_id=chat_id, priority=60,
    )
    return {"success": True, "accepted": True, "job": job, "message": "飞书历史同步已进入处理队列"}


@app.post("/api/feishu/messages/{message_id}/review")
async def review_feishu_message(message_id: str, payload: Dict[str, Any] = Body(...)):
    """审核飞书消息；仅审核通过的消息会写入项目知识库。"""
    action = str(payload.get("action") or "").strip()
    if action not in {"approve", "ignore"}:
        raise HTTPException(status_code=400, detail="仅支持 approve 或 ignore 操作")
    message = feishu_workspace.get_message(message_id)
    if not message:
        raise HTTPException(status_code=404, detail="消息记录不存在")

    if action == "ignore":
        if message.get("candidate_id"):
            feishu_workspace.update_candidate(str(message["candidate_id"]), "exclude")
        result = feishu_workspace.set_review_status(message_id, "ignored")
        return {"success": True, "message": "已忽略该消息", "record": result}

    if message.get("review_status") == "approved":
        return {"success": True, "message": "该消息已入库", "record": message}

    group = feishu_workspace.get_group(str(message.get("chat_id") or "")) or {"name": "项目群"}
    category = str(payload.get("category") or group.get("default_category") or "")
    try:
        ingest_result = _ingest_feishu_message(message, group, category)
        result = feishu_workspace.set_review_status(message_id, "approved", ingest_result["filename"])
        return {
            "success": True,
            "message": "已沉淀到项目知识库",
            "record": result,
            "ingest": ingest_result,
        }
    except Exception as exc:
        raise _internal_http_error("飞书消息入库失败", "飞书消息入库失败，请稍后重试") from exc


# ═══════════════════════════════════════════════
#  7. 系统设置 & 健康检查
# ═══════════════════════════════════════════════

@app.post("/api/settings/knowledge-library/reset")
async def reset_knowledge_library():
    identity = get_current_identity()
    active_jobs = get_repository().list_processing_jobs(
        can_manage=True, status="active", page_size=100,
    )["items"]
    if active_jobs:
        raise HTTPException(status_code=409, detail="存在进行中的处理任务，请完成或取消后再重置")
    result = data_governance_service.reset_knowledge_library(actor_user_id=identity.user_id)
    feishu_history = result.get("feishu_history") or {}
    feishu_resources = feishu_history.get("resource_file_cleanup") or {}
    return {
        "success": True,
        "result": result,
        "message": (
            f"已清空 {result['assets_deleted']} 项知识资产（含待复审、已失效和已撤销项）、"
            f"{result['sources_deleted']} 个知识来源、{result['documents_deleted']} 份文档、"
            f"{result['history']['processing_jobs']} 条处理任务、"
            f"{result['history']['knowledge_tasks']} 条知识任务、"
            f"{result['history']['onboarding_learning_plans']} 个学习计划和"
            f"{result['history']['handover_cases']} 条离职交接记录、"
            f"{int(feishu_history.get('messages_deleted') or 0)} 条飞书历史消息、"
            f"{int(feishu_history.get('candidates_deleted') or 0)} 条飞书候选、"
            f"{int(feishu_history.get('assets_deleted') or 0)} 项飞书知识和"
            f"{len(feishu_resources.get('removed') or [])} 份飞书资源文件；"
            f"飞书采集群配置、账号、项目成员、分类、删除墓碑与安全审计保留，未创建备份"
        ),
    }

@app.get("/api/governance/policy")
async def get_governance_policy():
    return {"success": True, "policy": data_governance_service.policy()}


@app.put("/api/governance/policy")
async def update_governance_policy(payload: Dict[str, Any] = Body(...)):
    identity = get_current_identity()
    try:
        policy = data_governance_service.update_policy(payload, identity.user_id)
        data_governance_scheduler.wake()
        return {"success": True, "policy": policy, "message": "数据治理策略已保存"}
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/governance/retention/preview")
async def preview_governance_retention():
    return {"success": True, "preview": data_governance_service.retention_preview()}


@app.post("/api/governance/retention/apply")
async def apply_governance_retention(payload: Dict[str, Any] = Body(...)):
    identity = get_current_identity()
    if str(payload.get("confirmation") or "") != "执行数据清理":
        raise HTTPException(status_code=400, detail="请输入完整确认短语：执行数据清理")
    job = processing_job_service.enqueue(
        "data_retention",
        {"confirmation": "执行数据清理", "actor_user_id": identity.user_id},
        created_by=identity.user_id,
        idempotency_key=f"data-retention:{identity.project_id}",
        priority=80,
    )
    return {"success": True, "accepted": True, "job": job, "message": "数据清理已进入处理队列"}


@app.get("/api/governance/schedule")
async def get_governance_schedule():
    return {"success": True, **data_governance_service.schedule_status()}


@app.post("/api/governance/schedule/scan")
async def scan_governance_schedule():
    result = data_governance_service.schedule_daily_operations()
    if result.get("job_count"):
        processing_job_runner.wake()
    return {
        "success": True, **result,
        "message": f"治理调度扫描完成，新安排 {result.get('job_count', 0)} 个任务",
    }


@app.get("/api/governance/targets/preview")
async def preview_governance_target(target_type: str, target_id: str):
    try:
        preview = data_governance_service.targeted_preview(target_type, target_id)
        return {"success": True, "preview": preview}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/governance/targets/export")
async def export_governance_target(target_type: str, target_id: str):
    identity = get_current_identity()
    try:
        export = data_governance_service.targeted_export(
            target_type, target_id, actor_user_id=identity.user_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    filename = f"governance-{export['target']['type']}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    body = json.dumps(export, ensure_ascii=False, indent=2).encode("utf-8")
    return StreamingResponse(
        iter([body]), media_type="application/json; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/governance/targets/delete")
async def delete_governance_target(payload: Dict[str, Any] = Body(...)):
    identity = get_current_identity()
    target_type = str(payload.get("target_type") or "").strip().lower()
    target_id = str(payload.get("target_id") or "").strip()
    if str(payload.get("confirmation") or "") != "永久删除定向数据":
        raise HTTPException(status_code=400, detail="请输入完整确认短语：永久删除定向数据")
    if target_type == "user" and target_id == identity.user_id:
        raise HTTPException(status_code=400, detail="不能删除当前登录成员的数据")
    try:
        preview = data_governance_service.targeted_preview(target_type, target_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    target_id = str(preview["target_id"])
    job = processing_job_service.enqueue(
        "targeted_data_deletion",
        {
            "target_type": target_type, "target_id": target_id,
            "confirmation": "永久删除定向数据", "actor_user_id": identity.user_id,
        },
        created_by=identity.user_id,
        idempotency_key=f"targeted-data-deletion:{identity.project_id}:{target_type}:{target_id}",
        priority=95,
    )
    return {"success": True, "accepted": True, "job": job, "message": "定向删除已进入处理队列"}


@app.get("/api/governance/backups")
async def list_governance_backups():
    backups = data_governance_service.list_backups()
    return {"success": True, "backups": backups, "count": len(backups)}


@app.post("/api/governance/backups")
async def create_governance_backup():
    identity = get_current_identity()
    job = processing_job_service.enqueue(
        "database_backup", {"actor_user_id": identity.user_id},
        created_by=identity.user_id,
        idempotency_key=f"database-backup:{identity.project_id}",
        priority=90,
    )
    return {"success": True, "accepted": True, "job": job, "message": "数据库备份已进入处理队列"}


@app.get("/api/governance/audit")
async def list_governance_audit(
    action: str = "", actor: str = "", object_type: str = "", keyword: str = "",
    page: int = 1, page_size: int = 30,
):
    result = get_repository().list_audit_events(
        action=action, actor=actor, object_type=object_type, keyword=keyword,
        page=page, page_size=page_size,
    )
    return {"success": True, **result}


@app.get("/api/governance/audit/export")
async def export_governance_audit(
    action: str = "", actor: str = "", object_type: str = "", keyword: str = "",
):
    identity = get_current_identity()
    result = get_repository().list_audit_events(
        action=action, actor=actor, object_type=object_type, keyword=keyword,
        page=1, page_size=5000,
    )
    output = io.StringIO()
    output.write("\ufeff")
    writer = csv.writer(output)
    writer.writerow(["时间", "操作者", "动作", "对象类型", "对象 ID", "必要上下文"])
    for event in result["items"]:
        writer.writerow([
            event["created_at"], event["actor"], event["action"], event["object_type"],
            event["object_id"], json.dumps(event["detail"], ensure_ascii=False, separators=(",", ":")),
        ])
    get_repository().write_security_audit(
        identity.user_id, "governance.audit_exported", "audit_export", identity.project_id,
        {"count": len(result["items"]), "filters": {
            "action": action, "actor": actor, "object_type": object_type, "keyword": keyword,
        }},
    )
    filename = f"audit-{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
    return StreamingResponse(
        iter([output.getvalue().encode("utf-8")]), media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )

@app.get("/api/health")
async def health():
    try:
        storage_check = get_repository().database.quick_check()
        migration = get_repository().database.migration_status()
        storage_ok = bool(storage_check["ok"] and not migration["pending_versions"])
    except Exception:
        logger.exception("SQLite 健康检查失败")
        storage_ok = False
        migration = {"current_version": 0, "pending_versions": ["unknown"]}
    return {
        "success": storage_ok,
        "status": "ok" if storage_ok else "degraded",
        "storage_ok": storage_ok,
        "storage_schema_version": migration["current_version"],
        "storage_schema_pending": migration["pending_versions"],
        "llm_configured": bool((settings.OPENAI_API_KEY or "").strip()),
        "feishu_configured": bool(settings.FEISHU_APP_ID and settings.FEISHU_APP_SECRET),
        "api_key_configured": bool((settings.OPENAI_API_KEY or "").strip()),
        "feishu_app_id": settings.FEISHU_APP_ID or "",
        "feishu_app_secret_configured": bool(settings.FEISHU_APP_SECRET),
        "feishu_default_chat_id": settings.FEISHU_DEFAULT_CHAT_ID or "",
        "llm_model": settings.LLM_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL,
        "api_base_url": settings.OPENAI_BASE_URL or "https://api.openai.com/v1",
        "proxy_enabled": settings.HTTP_PROXY_ENABLED,
        "proxy_url": settings.HTTP_PROXY_URL or "",
        "verify_ssl": settings.HTTP_VERIFY_SSL,
    }


@app.get("/api/ops/overview")
async def runtime_operations_overview(window: str = "24h"):
    """Return project-scoped SLO facts, component health and actionable alerts."""
    try:
        return observability_service.overview(window)
    except Exception as exc:
        raise _internal_http_error("读取运行监控失败", "运行监控暂时不可用") from exc


@app.post("/api/ops/capacity/validate", status_code=202)
async def validate_runtime_capacity(request: Request, payload: Dict[str, Any] = Body(default={})):
    identity = request.state.identity
    message_count = max(1000, min(int(payload.get("message_count") or 10000), 50000))
    job = processing_job_service.enqueue(
        "capacity_validation",
        {"message_count": message_count, "actor_user_id": identity.user_id},
        created_by=identity.user_id,
        idempotency_key=f"capacity:{identity.project_id}:{message_count}",
        priority=40,
    )
    return {"success": True, "message": "容量验证已提交，不会写入业务数据", "job": job}


@app.post("/api/ops/maintenance", status_code=201)
async def create_runtime_maintenance(request: Request, payload: Dict[str, Any] = Body(...)):
    try:
        window = observability_service.create_maintenance_window(payload, request.state.identity.user_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"success": True, "message": "计划维护窗口已登记", "window": window}


@app.delete("/api/ops/maintenance/{window_id}")
async def cancel_runtime_maintenance(window_id: str, request: Request):
    try:
        window = observability_service.cancel_maintenance_window(window_id, request.state.identity.user_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"success": True, "message": "计划维护窗口已取消", "window": window}


@app.post("/api/ops/recovery-drills", status_code=202)
async def run_runtime_recovery_drill(request: Request):
    identity = request.state.identity
    job = processing_job_service.enqueue(
        "recovery_drill", {"actor_user_id": identity.user_id, "trigger": "manual"},
        created_by=identity.user_id,
        idempotency_key=f"recovery-drill:{identity.project_id}:{datetime.now().strftime('%Y-%m-%d')}",
        priority=45,
    )
    return {"success": True, "message": "隔离恢复演练已进入处理任务", "job": job}


@app.post("/api/settings/llm")
async def save_llm_settings(
    api_key: Optional[str] = Form(None),
    api_base_url: Optional[str] = Form(None),
    llm_model: Optional[str] = Form(None),
    llm_provider: str = Form("openai"),
):
    try:
        if api_key is not None:
            settings.OPENAI_API_KEY = api_key
        if api_base_url is not None:
            settings.OPENAI_BASE_URL = llm_service.normalize_base_url(api_base_url)
        if llm_model is not None:
            settings.LLM_MODEL = llm_model.strip()
        _save_config()
        llm_service.reinitialize()
        vector_store.reinitialize()
        return {
            "success": True,
            "message": "LLM 配置已更新",
            "api_key_configured": bool((settings.OPENAI_API_KEY or "").strip()),
            "api_base_url": settings.OPENAI_BASE_URL,
            "llm_model": settings.LLM_MODEL,
        }
    except Exception as e:
        raise _internal_http_error("保存 LLM 配置失败", "配置保存失败，请稍后重试") from e


@app.post("/api/settings/llm/test")
async def test_llm_settings():
    try:
        reply = await llm_service.test_connection()
        return {
            "success": True,
            "message": "模型连接正常",
            "reply": reply[:120],
            "api_key_configured": True,
            "api_base_url": settings.OPENAI_BASE_URL or "",
            "llm_model": settings.LLM_MODEL,
        }
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "message": "模型连接失败",
                "detail": _safe_llm_error(e),
                "api_key_configured": bool((settings.OPENAI_API_KEY or "").strip()),
                "api_base_url": settings.OPENAI_BASE_URL or "",
                "llm_model": settings.LLM_MODEL or "",
            },
        )


@app.post("/api/settings/network")
async def save_network_settings(
    proxy_enabled: bool = Form(False),
    proxy_url: str = Form(""),
    verify_ssl: bool = Form(True),
):
    try:
        settings.HTTP_PROXY_ENABLED = proxy_enabled
        settings.HTTP_PROXY_URL = proxy_url
        settings.HTTP_VERIFY_SSL = verify_ssl
        _save_config()
        llm_service.reinitialize()
        vector_store.reinitialize()
        return {"success": True, "message": "网络配置已更新"}
    except Exception as e:
        raise _internal_http_error("保存网络配置失败", "配置保存失败，请稍后重试") from e


@app.post("/api/settings/feishu")
async def save_feishu_settings(
    feishuAppId: str = Form(""),
    feishuAppSecret: Optional[str] = Form(None),
    feishuWebhookSecret: Optional[str] = Form(None),
    feishuVerificationToken: Optional[str] = Form(None),
    feishuDefaultChatId: str = Form(""),
):
    try:
        settings.FEISHU_APP_ID = feishuAppId
        if feishuAppSecret is not None:
            settings.FEISHU_APP_SECRET = feishuAppSecret
        if feishuWebhookSecret is not None:
            settings.FEISHU_WEBHOOK_SECRET = feishuWebhookSecret
        if feishuVerificationToken is not None:
            settings.FEISHU_VERIFICATION_TOKEN = feishuVerificationToken
        settings.FEISHU_DEFAULT_CHAT_ID = feishuDefaultChatId
        _save_config()
        feishu_long_connection.restart(asyncio.get_running_loop())
        return {"success": True, "message": "飞书配置已更新，正在建立事件长连接"}
    except Exception as e:
        raise _internal_http_error("保存飞书配置失败", "配置保存失败，请稍后重试") from e


@app.post("/api/restart")
async def restart_server():
    import sys, os
    os._exit(0)  # 由进程管理器自动重启


@app.get("/api/stats")
async def stats():
    docs = _get_documents()
    total_chunks = sum(d.get("chunks", 0) for d in docs)
    graph_data = knowledge_graph.get_graph_data() if hasattr(knowledge_graph, 'get_graph_data') else {}
    handovers = [_handover_view(record) for record in _load_handover_records()]
    handover_risks = [risk for handover in handovers for risk in handover.get("risks", [])]
    return {
        "total_chunks": total_chunks,
        "documents": docs,
        "knowledge_graph": {"nodes": len(graph_data.get("nodes", [])), "edges": len(graph_data.get("edges", []))},
        "project_name": "AI 知识库",
        "llm_model": settings.LLM_MODEL,
        "embedding_model": getattr(settings, 'EMBEDDING_MODEL', 'local'),
        "handover_records": len(handovers),
        "handover_risks": len(handover_risks),
    }


# ═══════════════════════════════════════════════
#  9. Tunnel 隧道管理（ngrok）
# ═══════════════════════════════════════════════

import subprocess
import threading
import time

# ngrok 进程全局引用
NGROK_PROCESS: Optional[subprocess.Popen] = None
NGROK_LOCK = threading.Lock()


def _get_ngrok_url() -> str:
    """从 ngrok API 获取公网 URL"""
    import httpx
    try:
        r = httpx.get("http://127.0.0.1:4040/api/tunnels", timeout=3)
        if r.status_code == 200:
            data = r.json()
            tunnels = data.get("tunnels", [])
            for t in tunnels:
                if t.get("public_url"):
                    return t["public_url"]
    except Exception:
        pass
    return ""


@app.post("/api/tunnel/start")
async def tunnel_start():
    """启动 ngrok tunnel"""
    global NGROK_PROCESS
    with NGROK_LOCK:
        if NGROK_PROCESS and NGROK_PROCESS.poll() is None:
            url = _get_ngrok_url()
            return {"success": True, "message": "Tunnel 已在运行中", "url": url}

        try:
            # 先清理旧进程
            subprocess.run(["taskkill", "/IM", "ngrok.exe", "/F"],
                          capture_output=True, timeout=5)
        except Exception:
            pass

        try:
            NGROK_PROCESS = subprocess.Popen(
                ["ngrok", "http", "8000", "--log=stdout"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            # 等待 ngrok 启动
            url = ""
            for _ in range(10):
                time.sleep(1)
                url = _get_ngrok_url()
                if url:
                    break

            if url:
                return {
                    "success": True,
                    "message": "Tunnel 已启动",
                    "url": url,
                }
            else:
                return {
                    "success": True,
                    "message": "Tunnel 已启动，正在获取 URL……",
                    "url": "",
                }
        except Exception as e:
            return {"success": False, "message": f"启动失败: {str(e)}", "url": ""}


@app.post("/api/tunnel/stop")
async def tunnel_stop():
    """停止 ngrok tunnel"""
    global NGROK_PROCESS
    with NGROK_LOCK:
        try:
            subprocess.run(["taskkill", "/IM", "ngrok.exe", "/F"],
                          capture_output=True, timeout=5)
        except Exception:
            pass
        NGROK_PROCESS = None
    return {"success": True, "message": "Tunnel 已停止"}


@app.get("/api/tunnel/status")
async def tunnel_status():
    """查询 tunnel 状态"""
    url = _get_ngrok_url()
    running = bool(url)
    return {
        "success": True,
        "running": running,
        "url": url,
        "message": "运行中" if running else "未启动",
    }


# ═══════════════════════════════════════════════
#  8. 静态文件服务（前端）
# ═══════════════════════════════════════════════

try:
    # 多路径搜索 frontend（开发模式 / 单文件打包 / 自定义）
    import sys as _sys
    _candidates = [
        Path("frontend"),                                          # CWD
        Path(__file__).parent.parent / "frontend",                  # 开发模式
        Path(_sys._MEIPASS) / "frontend" if getattr(_sys, 'frozen', False) else None,  # onefile
    ]
    _found = None
    for _p in _candidates:
        if _p and _p.exists() and (_p / "index.html").exists():
            _found = _p
            break
    if _found:
        app.mount("/", StaticFiles(directory=str(_found), html=True), name="frontend")
        print(f"[静态文件] 前端目录: {_found}")
except Exception:
    pass
