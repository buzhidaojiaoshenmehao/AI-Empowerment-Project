"""Local trial authentication and project-scoped authorization.

The interface is intentionally independent from FastAPI so enterprise OIDC/SSO
can replace credential verification without changing the role/permission model.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, FrozenSet, Optional

from backend.auth_context import Identity
from backend.config import settings
from backend.storage import get_repository
from backend.storage.database import DEFAULT_PROJECT_ID


SESSION_COOKIE = "ai_empowerment_session"
CSRF_COOKIE = "ai_empowerment_csrf"
PASSWORD_ITERATIONS = 600_000
SESSION_HOURS = 12
INVITATION_HOURS = 72
MAX_LOGIN_FAILURES = 5
LOCK_MINUTES = 15


ROLE_PERMISSIONS: Dict[str, FrozenSet[str]] = {
    "project_admin": frozenset({"*"}),
    "project_manager": frozenset({
        "knowledge.read", "knowledge.upload", "knowledge.manage", "knowledge.publish",
        "onboarding.read", "onboarding.manage", "onboarding.progress", "handover.read", "handover.submit",
        "handover.accept", "handover.manage", "feishu.read", "feishu.manage", "audit.read", "task.read", "task.manage",
        "job.read", "job.manage",
    }),
    "knowledge_operator": frozenset({
        "knowledge.read", "knowledge.upload", "knowledge.manage", "knowledge.publish",
        "onboarding.read", "onboarding.manage", "onboarding.progress", "handover.read", "feishu.read", "audit.read",
        "task.read", "task.manage",
        "job.read", "job.manage",
    }),
    "project_member": frozenset({
        "knowledge.read", "knowledge.upload", "onboarding.read", "onboarding.progress", "handover.read", "handover.submit",
        "handover.accept", "task.read",
        "job.read",
    }),
    "new_member": frozenset({
        "knowledge.read", "onboarding.read", "onboarding.progress", "handover.read", "handover.accept",
        "task.read", "job.read",
    }),
    "handover_participant": frozenset({
        "knowledge.read", "onboarding.read", "onboarding.progress", "handover.read", "handover.submit",
        "handover.accept", "task.read", "job.read",
    }),
}

ROLE_LABELS = {
    "project_admin": "项目管理员",
    "project_manager": "项目经理",
    "knowledge_operator": "知识运营人员",
    "project_member": "项目成员",
    "new_member": "新成员",
    "handover_participant": "交接参与人",
}


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _decode_b64(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def token_hash(token: str) -> str:
    return hashlib.sha256(str(token).encode("utf-8")).hexdigest()


def contextual_hash(value: str) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()[:24]


def _encode_password(password: str) -> str:
    password = str(password or "")
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PASSWORD_ITERATIONS)
    return f"pbkdf2_sha256${PASSWORD_ITERATIONS}${_b64(salt)}${_b64(digest)}"


def hash_password(password: str) -> str:
    password = str(password or "")
    if len(password) < 12:
        raise ValueError("密码至少需要 12 位")
    return _encode_password(password)


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, iterations, salt, expected = str(encoded).split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256", str(password).encode("utf-8"), _decode_b64(salt), int(iterations)
        )
        return hmac.compare_digest(digest, _decode_b64(expected))
    except (TypeError, ValueError):
        return False


def password_is_acceptable(password: str) -> bool:
    text = str(password or "")
    return (
        len(text) >= 12
        and any(char.isalpha() for char in text)
        and any(char.isdigit() for char in text)
        and any(not char.isalnum() for char in text)
    )


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


def parse_time(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


class AuthenticationError(ValueError):
    pass


class AuthService:
    def __init__(self, repository_provider=get_repository) -> None:
        self.repository_provider = repository_provider

    @property
    def repository(self):
        return self.repository_provider()

    def create_invitation(self, user_id: str, created_by: str = "user_system") -> Dict[str, str]:
        user = self.repository.get_user(user_id)
        if not user:
            raise ValueError("用户不存在")
        token = secrets.token_urlsafe(32)
        expires_at = iso(utc_now() + timedelta(hours=INVITATION_HOURS))
        self.repository.create_user_invitation(
            user_id, token_hash(token), expires_at, created_by=created_by,
            project_id=str(user.get("project_id") or DEFAULT_PROJECT_ID),
        )
        return {"token": token, "expires_at": expires_at, "email": str(user.get("email") or "")}

    @staticmethod
    def demo_password_for_email(email: str) -> str:
        normalized = str(email or "").strip().lower()
        return normalized.split("@", 1)[0] if "@" in normalized else ""

    def enable_demo_login(self, user_id: str, actor: str = "user_system") -> Dict[str, Any]:
        if not settings.DEMO_SIMPLE_AUTH:
            raise ValueError("演示认证模式未启用")
        user = self.repository.get_user(user_id)
        if not user:
            raise ValueError("用户不存在")
        password = self.demo_password_for_email(str(user.get("email") or ""))
        if not password:
            raise ValueError("演示成员缺少有效企业邮箱")
        updated = self.repository.set_demo_login_password(user_id, _encode_password(password))
        self.repository.write_security_audit(
            actor, "iam.demo_login_enabled", "user", user_id,
            {"email": updated.get("email", ""), "mode": "email_prefix"},
            project_id=str(updated.get("project_id") or DEFAULT_PROJECT_ID),
        )
        return updated

    def ensure_demo_accounts(self, actor: str = "startup") -> int:
        """Idempotently activate project members when local demo auth is enabled."""
        if not settings.DEMO_SIMPLE_AUTH:
            return 0
        changed = 0
        for member in self.repository.list_project_members():
            user = self.repository.get_user(str(member.get("user_id") or ""))
            if not user:
                continue
            password = self.demo_password_for_email(str(user.get("email") or ""))
            if not password:
                continue
            if user.get("status") == "active" and verify_password(password, str(user.get("password_hash") or "")):
                continue
            self.enable_demo_login(str(user["user_id"]), actor=actor)
            changed += 1
        return changed

    def activate(self, token: str, password: str) -> Dict[str, Any]:
        if settings.DEMO_SIMPLE_AUTH:
            raise AuthenticationError("演示模式无需激活，请直接使用邮箱和邮箱前缀密码登录")
        if not password_is_acceptable(password):
            raise AuthenticationError("密码至少 12 位，并同时包含字母、数字和特殊字符")
        user = self.repository.activate_invitation(token_hash(token), hash_password(password))
        if not user:
            raise AuthenticationError("激活链接无效、已使用或已过期")
        self.repository.write_security_audit(
            str(user["user_id"]), "iam.user_activated", "user", str(user["user_id"]),
            {"email": user.get("email", "")},
        )
        return {"user_id": user["user_id"], "email": user["email"], "display_name": user["display_name"]}

    def login(
        self,
        email: str,
        password: str,
        *,
        user_agent: str = "",
        remote_address: str = "",
    ) -> Dict[str, Any]:
        repository = self.repository
        user = repository.get_user_by_email(email)
        generic_error = "邮箱或密码不正确"
        if not user or user.get("user_type") != "human":
            # Keep timing closer to a real password check without maintaining a shared dummy hash.
            hashlib.pbkdf2_hmac("sha256", str(password).encode(), b"unknown-account", 50_000)
            raise AuthenticationError(generic_error)
        locked_until = parse_time(str(user.get("locked_until") or ""))
        if locked_until and locked_until > utc_now():
            raise AuthenticationError("登录尝试过多，账号已临时锁定，请稍后再试")
        if user.get("status") != "active" or not user.get("password_hash"):
            raise AuthenticationError("账号尚未激活或已停用")
        if not verify_password(password, str(user.get("password_hash") or "")):
            failures = int(user.get("failed_login_count") or 0) + 1
            next_lock = iso(utc_now() + timedelta(minutes=LOCK_MINUTES)) if failures >= MAX_LOGIN_FAILURES else ""
            repository.record_login_failure(str(user["user_id"]), failures, next_lock)
            repository.write_security_audit(
                str(user["user_id"]), "iam.login_failed", "user", str(user["user_id"]),
                {"locked": bool(next_lock)},
            )
            raise AuthenticationError(generic_error)

        memberships = [
            item for item in repository.list_project_members()
            if item.get("user_id") == user.get("user_id") and item.get("membership_status") == "active"
        ]
        if not memberships:
            raise AuthenticationError("账号尚未加入可用项目")
        membership = memberships[0]
        session_token = secrets.token_urlsafe(48)
        csrf_token = secrets.token_urlsafe(32)
        created_at = utc_now()
        repository.create_auth_session({
            "session_id": secrets.token_hex(16),
            "token_hash": token_hash(session_token),
            "csrf_hash": token_hash(csrf_token),
            "user_id": user["user_id"],
            "current_project_id": membership["project_id"],
            "expires_at": iso(created_at + timedelta(hours=SESSION_HOURS)),
            "created_at": iso(created_at),
            "user_agent_hash": contextual_hash(user_agent),
            "remote_address_hash": contextual_hash(remote_address),
        })
        repository.record_login_success(str(user["user_id"]))
        repository.write_security_audit(
            str(user["user_id"]), "iam.login_succeeded", "session", "current",
            {"project_id": membership["project_id"]},
        )
        identity = self.identity_from_token(session_token)
        return {"session_token": session_token, "csrf_token": csrf_token, "identity": identity}

    def identity_from_token(self, session_token: str) -> Optional[Identity]:
        if not session_token:
            return None
        record = self.repository.get_auth_session(token_hash(session_token))
        if not record:
            return None
        if (
            record.get("user_status") != "active"
            or record.get("membership_status") != "active"
            or record.get("project_status") != "active"
        ):
            return None
        role = str(record.get("role") or "")
        return Identity(
            user_id=str(record["user_id"]),
            email=str(record.get("email") or ""),
            display_name=str(record.get("display_name") or ""),
            organization_id=str(record.get("organization_id") or ""),
            project_id=str(record.get("current_project_id") or ""),
            project_name=str(record.get("project_name") or ""),
            role=role,
            permissions=ROLE_PERMISSIONS.get(role, frozenset()),
            session_id=str(record.get("session_id") or ""),
            duty=str(record.get("duty") or ""),
        )

    def validate_csrf(self, session_token: str, csrf_token: str) -> bool:
        record = self.repository.get_auth_session(token_hash(session_token))
        return bool(record and csrf_token and hmac.compare_digest(str(record["csrf_hash"]), token_hash(csrf_token)))

    def logout(self, session_token: str) -> None:
        if session_token:
            self.repository.delete_auth_session(token_hash(session_token))

    @staticmethod
    def identity_payload(identity: Identity) -> Dict[str, Any]:
        return {
            **asdict(identity),
            "permissions": sorted(identity.permissions),
            "role_label": ROLE_LABELS.get(identity.role, identity.role),
        }


auth_service = AuthService()


__all__ = [
    "AuthService", "AuthenticationError", "CSRF_COOKIE", "ROLE_LABELS", "ROLE_PERMISSIONS",
    "SESSION_COOKIE", "auth_service", "hash_password", "password_is_acceptable", "token_hash",
]
