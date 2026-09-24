"""Request-scoped identity used by repositories and derived projections."""
from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import FrozenSet, Optional


@dataclass(frozen=True)
class Identity:
    user_id: str
    email: str
    display_name: str
    organization_id: str
    project_id: str
    project_name: str
    role: str
    permissions: FrozenSet[str]
    session_id: str = ""
    is_system: bool = False
    duty: str = ""

    def can(self, permission: str) -> bool:
        return "*" in self.permissions or permission in self.permissions

    @property
    def cache_key(self) -> str:
        return f"{self.user_id}:{self.project_id}:{self.role}"


_identity: ContextVar[Optional[Identity]] = ContextVar("request_identity", default=None)


def get_current_identity() -> Optional[Identity]:
    return _identity.get()


def set_current_identity(identity: Optional[Identity]) -> Token:
    return _identity.set(identity)


def reset_current_identity(token: Token) -> None:
    _identity.reset(token)


__all__ = ["Identity", "get_current_identity", "set_current_identity", "reset_current_identity"]
