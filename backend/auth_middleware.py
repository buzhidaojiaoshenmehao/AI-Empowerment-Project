"""FastAPI middleware for authenticated, project-scoped API access."""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import Request
from fastapi.responses import JSONResponse

from backend.auth import CSRF_COOKIE, SESSION_COOKIE, auth_service
from backend.auth_context import reset_current_identity, set_current_identity


logger = logging.getLogger(__name__)
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
PUBLIC_PATHS = {
    "/api/health/live",
    "/api/auth/config",
    "/api/auth/login",
    "/api/auth/activate",
}


def required_permission(path: str, method: str) -> Optional[str]:
    """Resolve one authoritative permission for an API operation."""
    method = method.upper()
    if path in {"/api/auth/me", "/api/auth/logout", "/api/auth/projects"}:
        return None
    if path.startswith("/api/projects/current/members"):
        return "settings.manage" if method != "GET" else "knowledge.read"
    if path.startswith("/api/access-policies"):
        return "knowledge.manage"
    if path.startswith("/api/knowledge/assets"):
        return "knowledge.read" if method in SAFE_METHODS else "knowledge.manage"
    if path.startswith("/api/knowledge/tasks"):
        return "task.read"
    if path.startswith("/api/processing/jobs"):
        return "job.read"
    if path.startswith("/api/governance/audit"):
        return "audit.read"
    if path.startswith("/api/governance/"):
        return "settings.manage"
    if path.startswith("/api/ops/"):
        return "settings.manage"
    if path.startswith("/api/settings/") or path.startswith("/api/tunnel/") or path == "/api/restart":
        return "settings.manage"
    if path.startswith("/api/categories"):
        return "knowledge.read" if method in SAFE_METHODS else "knowledge.manage"
    if path in {"/api/documents/upload", "/api/documents/uploads"}:
        return "knowledge.upload"
    if path.startswith("/api/documents"):
        return "knowledge.read" if method in SAFE_METHODS else "knowledge.manage"
    if path == "/api/rag/feedback":
        return "knowledge.read"
    if path.startswith("/api/rag/evaluation-sets"):
        return "knowledge.read" if method in SAFE_METHODS else "knowledge.manage"
    if path.startswith("/api/search") or path.startswith("/api/chat") or path == "/api/smart-chat":
        return "knowledge.read"
    if path.startswith("/api/project/"):
        return "knowledge.read" if method in SAFE_METHODS else "knowledge.upload"
    if path.startswith("/api/knowledge-graph"):
        return "knowledge.read" if method in SAFE_METHODS or path.endswith("/trace") else "knowledge.manage"
    if path.startswith("/api/onboarding"):
        if method == "PATCH" and path.startswith("/api/onboarding/plans/") and "/items/" in path:
            return "onboarding.progress"
        return "onboarding.read" if method in SAFE_METHODS else "onboarding.manage"
    if path.startswith("/api/resignation"):
        if method in SAFE_METHODS:
            return "handover.read"
        if method in {"POST", "PUT"} and (
            path.endswith("/draft") or path.endswith("/attachments")
        ):
            return "handover.submit"
        if method == "POST" and path.endswith("/complete"):
            return "handover.manage"
        if method == "POST" and path.endswith("/accept"):
            return "handover.accept"
        if method == "POST" and (
            ("/items/" in path and path.endswith("/actions"))
            or "/inventory" in path
            or "/risks" in path
        ):
            return "handover.accept"
        return "handover.submit"
    if path.startswith("/api/feishu/audit"):
        return "audit.read"
    if path.startswith("/api/feishu"):
        if method in SAFE_METHODS:
            return "feishu.read"
        if any(fragment in path for fragment in ("/candidates/", "/assets/", "/messages/")):
            return "knowledge.publish"
        return "feishu.manage"
    if path in {"/api/health", "/api/stats"}:
        return "knowledge.read"
    return "knowledge.read"


def successful_access_action(path: str, method: str) -> Optional[str]:
    """Return a low-volume audit action without capturing request content."""
    if method == "POST" and path in {"/api/chat", "/api/smart-chat"}:
        return "access.knowledge_answer"
    if method == "GET" and path.startswith("/api/documents/preview/"):
        return "access.document_preview"
    if method == "GET" and path.startswith("/api/knowledge/assets/"):
        return "access.knowledge_asset"
    if method == "GET" and path == "/api/knowledge-graph":
        return "access.knowledge_graph"
    if method == "GET" and path.startswith("/api/feishu/messages/") and "/resources/" in path:
        return "access.feishu_resource"
    return None


async def authentication_middleware(request: Request, call_next):
    path = request.url.path
    if request.method == "OPTIONS" or not path.startswith("/api/") or path in PUBLIC_PATHS or path.startswith("/api/feishu/webhook"):
        return await call_next(request)

    session_token = request.cookies.get(SESSION_COOKIE, "")
    identity = auth_service.identity_from_token(session_token)
    if not identity:
        return JSONResponse(
            status_code=401,
            content={"success": False, "code": "AUTH_REQUIRED", "detail": "登录已失效，请重新登录"},
        )

    if request.method not in SAFE_METHODS and path not in {"/api/auth/login", "/api/auth/activate"}:
        csrf_header = request.headers.get("X-CSRF-Token", "")
        csrf_cookie = request.cookies.get(CSRF_COOKIE, "")
        if not csrf_header or not csrf_cookie or not auth_service.validate_csrf(session_token, csrf_header) or csrf_header != csrf_cookie:
            auth_service.repository.write_security_audit(
                identity.user_id, "iam.csrf_denied", "api", path,
                {"method": request.method}, project_id=identity.project_id,
            )
            return JSONResponse(
                status_code=403,
                content={"success": False, "code": "CSRF_DENIED", "detail": "安全校验失败，请刷新页面后重试"},
            )

    permission = required_permission(path, request.method)
    if permission and not identity.can(permission):
        auth_service.repository.write_security_audit(
            identity.user_id, "iam.permission_denied", "api", path,
            {"method": request.method, "permission": permission, "role": identity.role},
            project_id=identity.project_id,
        )
        return JSONResponse(
            status_code=403,
            content={
                "success": False,
                "code": "PERMISSION_DENIED",
                "detail": "当前项目角色无权执行此操作",
                "required_permission": permission,
            },
        )

    request.state.identity = identity
    context_token = set_current_identity(identity)
    try:
        response = await call_next(request)
        access_action = successful_access_action(path, request.method.upper())
        if access_action and response.status_code < 400:
            object_id = path.rsplit("/", 1)[-1] if "/" in path else path
            auth_service.repository.write_security_audit(
                identity.user_id, access_action, "api", object_id,
                {"method": request.method.upper(), "status_code": response.status_code},
                project_id=identity.project_id,
            )
        return response
    finally:
        reset_current_identity(context_token)


__all__ = ["authentication_middleware", "required_permission", "successful_access_action"]
