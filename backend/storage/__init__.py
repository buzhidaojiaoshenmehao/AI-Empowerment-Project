"""Application storage bootstrap and repository access."""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Optional

from backend.storage.database import Database, StorageProcessLock
from backend.storage.legacy import LegacyMigrator
from backend.storage.repository import StorageRepository


_lock = threading.RLock()
_repository: Optional[StorageRepository] = None
_repository_path: Optional[Path] = None
_runtime_lock: Optional[StorageProcessLock] = None
_runtime_lock_path: Optional[Path] = None


def application_base_dir() -> Path:
    """Return the runtime base directory, overridable for tests and packaging."""
    configured = (
        os.environ.get("AI_EMPOWERMENT_BASE_DIR", "")
        or os.environ.get("APP_BASE_DIR", "")
    ).strip()
    return Path(configured).expanduser().resolve() if configured else Path.cwd().resolve()


def database_path() -> Path:
    from backend.config import settings

    raw = Path(settings.DATABASE_PATH).expanduser()
    return raw.resolve() if raw.is_absolute() else (application_base_dir() / raw).resolve()


def backup_dir() -> Path:
    from backend.config import settings

    raw = Path(settings.DATABASE_BACKUP_DIR).expanduser()
    return raw.resolve() if raw.is_absolute() else (application_base_dir() / raw).resolve()


def get_repository() -> StorageRepository:
    """Return the process repository, recreating it if tests change the DB path."""
    global _repository, _repository_path
    target = database_path()
    with _lock:
        if _repository is None or _repository_path != target:
            _repository = StorageRepository(Database(target))
            _repository_path = target
        return _repository


def acquire_runtime_lock() -> bool:
    """Hold the database process lock for the lifetime of the web application."""
    global _runtime_lock, _runtime_lock_path
    target = database_path()
    with _lock:
        if _runtime_lock is not None and _runtime_lock_path == target:
            return False
        if _runtime_lock is not None:
            _runtime_lock.release()
        runtime_lock = Database(target).process_lock("application").acquire()
        _runtime_lock = runtime_lock
        _runtime_lock_path = target
        return True


def release_runtime_lock() -> None:
    global _runtime_lock, _runtime_lock_path
    with _lock:
        if _runtime_lock is not None:
            _runtime_lock.release()
        _runtime_lock = None
        _runtime_lock_path = None


def initialize_storage(run_legacy_migration: bool = True, enforce_single_process: bool = True) -> dict:
    """Initialize schema and optionally import legacy JSON exactly once."""
    lock_acquired = acquire_runtime_lock() if enforce_single_process else False
    repository = get_repository()
    try:
        repository.database.initialize()
        result = {"status": "initialized"}
        if run_legacy_migration:
            from backend.config import settings

            result = LegacyMigrator(
                repository=repository,
                base_dir=application_base_dir(),
                chroma_dir=settings.CHROMA_PERSIST_DIR,
            ).migrate_once()
        integrity = repository.database.quick_check()
        if not integrity.get("ok"):
            raise RuntimeError(f"SQLite 完整性检查失败: {integrity.get('quick_check')}")
        return {
            "database": str(repository.database.path),
            "legacy_migration": result,
            "schema": repository.database.migration_status(),
            "integrity": integrity,
        }
    except Exception:
        if lock_acquired:
            release_runtime_lock()
        raise


def reset_repository_for_tests(path: Path | str | None = None) -> None:
    """Reset the singleton. Tests may also supply a temporary database path."""
    global _repository, _repository_path
    with _lock:
        release_runtime_lock()
        _repository = None
        _repository_path = None
        if path is not None:
            from backend.config import settings

            settings.DATABASE_PATH = str(path)


__all__ = [
    "application_base_dir",
    "backup_dir",
    "database_path",
    "get_repository",
    "acquire_runtime_lock",
    "initialize_storage",
    "release_runtime_lock",
    "reset_repository_for_tests",
]
