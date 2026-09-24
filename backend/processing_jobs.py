"""Persistent, restart-safe processing jobs for the single-instance SQLite deployment."""
from __future__ import annotations

import asyncio
import inspect
import logging
import re
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional

from backend.storage import get_repository


logger = logging.getLogger(__name__)
JobHandler = Callable[["ProcessingJobContext", Dict[str, Any]], Any]
JobCleanupHandler = Callable[[Dict[str, Any]], None]
SECRET_KEYS = {"api_key", "apikey", "app_secret", "secret", "token", "password", "authorization"}


class ProcessingJobError(RuntimeError):
    def __init__(self, message: str, *, code: str = "PROCESSING_FAILED", retryable: bool = True) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class ProcessingJobCancelled(ProcessingJobError):
    def __init__(self, result: Optional[Dict[str, Any]] = None) -> None:
        super().__init__("任务已由用户取消", code="JOB_CANCELLED", retryable=False)
        self.result = result or {}


def sanitize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): ("***" if str(key).lower() in SECRET_KEYS else sanitize_value(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize_value(item) for item in value[:200]]
    if isinstance(value, tuple):
        return [sanitize_value(item) for item in value[:200]]
    if isinstance(value, str):
        text = re.sub(r"(?i)(bearer\s+)[a-z0-9._~+/=-]+", r"\1***", value)
        text = re.sub(r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*\S+", r"\1=***", text)
        return text[:4000]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)[:4000]


@dataclass
class ProcessingJobContext:
    job_id: str
    repository: Any

    def report(self, stage: str, progress: int, message: str = "") -> None:
        self.repository.update_processing_job_progress(
            self.job_id, stage=str(stage or "processing"), progress=progress,
            message=str(message or "")[:1000],
        )
        self.raise_if_cancelled()

    def raise_if_cancelled(self) -> None:
        if self.repository.processing_job_cancel_requested(self.job_id):
            raise ProcessingJobCancelled()


class ProcessingJobService:
    def __init__(self, repository_provider=get_repository) -> None:
        self.repository_provider = repository_provider
        self.handlers: Dict[str, JobHandler] = {}
        self.cleanup_handlers: Dict[str, JobCleanupHandler] = {}
        self.runner: Optional["ProcessingJobRunner"] = None

    @property
    def repository(self):
        return self.repository_provider()

    def register(self, job_type: str, handler: JobHandler) -> None:
        key = str(job_type or "").strip()
        if not key:
            raise ValueError("处理任务类型不能为空")
        self.handlers[key] = handler

    def register_cleanup(self, job_type: str, handler: JobCleanupHandler) -> None:
        self.cleanup_handlers[str(job_type or "").strip()] = handler

    def cleanup(self, job: Dict[str, Any]) -> None:
        handler = self.cleanup_handlers.get(str(job.get("job_type") or ""))
        if not handler:
            return
        try:
            handler(job)
        except Exception:
            logger.exception("Processing job %s cleanup failed", job.get("job_id"))

    def enqueue(
        self, job_type: str, payload: Optional[Dict[str, Any]] = None, *,
        created_by: str, idempotency_key: str = "", max_attempts: int = 3,
        priority: int = 50, linked_asset_id: str = "", source_id: str = "",
    ) -> Dict[str, Any]:
        key = str(job_type or "").strip()
        if key not in self.handlers:
            raise ValueError("不支持的处理任务类型")
        job = self.repository.create_or_get_processing_job({
            "job_type": key,
            "payload": sanitize_value(payload or {}),
            "created_by": created_by,
            "idempotency_key": idempotency_key,
            "max_attempts": max_attempts,
            "priority": priority,
            "linked_asset_id": linked_asset_id,
            "source_id": source_id,
        })
        if self.runner:
            self.runner.wake()
        return job

    def list_jobs(self, identity: Any, **filters: Any) -> Dict[str, Any]:
        can_manage = bool(identity.can("job.manage"))
        result = self.repository.list_processing_jobs(
            viewer_user_id=identity.user_id, can_manage=can_manage, **filters,
        )
        result["items"] = [self._decorate_job(item, identity) for item in result.get("items", [])]
        result["summary"] = self.repository.summarize_processing_jobs(
            viewer_user_id=identity.user_id, can_manage=can_manage,
        )
        return result

    def get_job(self, job_id: str, identity: Any) -> Optional[Dict[str, Any]]:
        job = self.repository.get_processing_job(
            job_id, viewer_user_id=identity.user_id, can_manage=identity.can("job.manage"),
        )
        return self._decorate_job(job, identity) if job else None

    @staticmethod
    def _decorate_job(job: Dict[str, Any], identity: Any) -> Dict[str, Any]:
        """Add user-facing identity and authoritative actions to a visible job."""
        item = dict(job)
        payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
        result = item.get("result") if isinstance(item.get("result"), dict) else {}
        item["display_name"] = next((
            str(value).strip() for value in (
                payload.get("original_filename"), result.get("filename"),
                payload.get("filename"), payload.get("display_name"), payload.get("title"),
            ) if str(value or "").strip()
        ), "")
        owns_job = str(item.get("created_by") or "") == str(identity.user_id or "")
        can_operate = bool(identity.can("job.manage") or owns_job)
        allowed_actions = []
        if can_operate and item.get("status") in {"queued", "running", "retry_wait"}:
            allowed_actions.append("cancel")
        if can_operate and item.get("status") == "failed":
            allowed_actions.append("retry")
        if (
            can_operate and item.get("status") == "cancelled"
            and item.get("job_type") != "document_ingestion"
        ):
            allowed_actions.append("retry")
        item["allowed_actions"] = allowed_actions
        return item

    def cancel(self, job_id: str, identity: Any) -> Dict[str, Any]:
        job = self.get_job(job_id, identity)
        if not job:
            raise ValueError("处理任务不存在或无权访问")
        if not identity.can("job.manage") and job.get("created_by") != identity.user_id:
            raise PermissionError("只能取消本人发起的任务")
        result = self.repository.cancel_processing_job(job_id, actor_user_id=identity.user_id)
        if result.get("status") == "cancelled":
            self.cleanup(result)
        if self.runner:
            self.runner.wake()
        return self._decorate_job(result, identity)

    def retry(self, job_id: str, identity: Any) -> Dict[str, Any]:
        job = self.get_job(job_id, identity)
        if not job:
            raise ValueError("处理任务不存在或无权访问")
        if not identity.can("job.manage") and job.get("created_by") != identity.user_id:
            raise PermissionError("只能重试本人发起的任务")
        result = self.repository.retry_processing_job(job_id, actor_user_id=identity.user_id)
        if self.runner:
            self.runner.wake()
        return self._decorate_job(result, identity)


class ProcessingJobRunner:
    def __init__(self, service: ProcessingJobService, *, poll_seconds: float = 1.0) -> None:
        self.service = service
        self.poll_seconds = max(0.05, float(poll_seconds))
        self.worker_id = f"worker_{uuid.uuid4().hex[:12]}"
        self._wake = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._stopping = False
        service.runner = self

    @property
    def running(self) -> bool:
        return bool(self._task and not self._task.done())

    def wake(self) -> None:
        self._wake.set()

    async def start(self) -> int:
        if self.running:
            return 0
        recovered = self.service.repository.recover_processing_jobs()
        self._stopping = False
        self._task = asyncio.create_task(self._run(), name="processing-job-runner")
        self.wake()
        return recovered

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
            job = self.service.repository.claim_next_processing_job(self.worker_id)
            if job:
                await self._execute(job)
                continue
            self._wake.clear()
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=self.poll_seconds)
            except asyncio.TimeoutError:
                pass

    async def _execute(self, job: Dict[str, Any]) -> None:
        job_id = str(job["job_id"])
        handler = self.service.handlers.get(str(job.get("job_type") or ""))
        if not handler:
            self.service.repository.fail_processing_job(
                job_id, error_code="HANDLER_NOT_FOUND", error_message="处理器不存在", retryable=False,
            )
            return
        context = ProcessingJobContext(job_id=job_id, repository=self.service.repository)
        try:
            context.report("starting", max(1, int(job.get("progress") or 0)), "正在准备处理")
            if inspect.iscoroutinefunction(handler):
                result = await handler(context, dict(job.get("payload") or {}))
            else:
                result = await asyncio.to_thread(handler, context, dict(job.get("payload") or {}))
                if inspect.isawaitable(result):
                    result = await result
            commit_on_cancel = bool(isinstance(result, dict) and result.pop("_job_commit_on_cancel", False))
            if not commit_on_cancel:
                context.raise_if_cancelled()
            self.service.repository.complete_processing_job(job_id, sanitize_value(result or {}))
        except ProcessingJobCancelled as exc:
            self.service.repository.mark_processing_job_cancelled(job_id, sanitize_value(exc.result))
            self.service.cleanup(job)
        except ProcessingJobError as exc:
            delay = min(300, 5 * (2 ** max(0, int(job.get("attempts") or 1) - 1)))
            self.service.repository.fail_processing_job(
                job_id, error_code=exc.code, error_message=sanitize_value(str(exc)),
                retryable=exc.retryable, retry_delay_seconds=delay,
            )
        except Exception as exc:  # noqa: BLE001 - handler boundary must persist failure
            logger.exception("Processing job %s failed", job_id)
            delay = min(300, 5 * (2 ** max(0, int(job.get("attempts") or 1) - 1)))
            self.service.repository.fail_processing_job(
                job_id, error_code="UNEXPECTED_ERROR", error_message=sanitize_value(str(exc)),
                retryable=True, retry_delay_seconds=delay,
            )


processing_job_service = ProcessingJobService()
processing_job_runner = ProcessingJobRunner(processing_job_service)


__all__ = [
    "ProcessingJobCancelled", "ProcessingJobContext", "ProcessingJobError",
    "ProcessingJobRunner", "ProcessingJobService", "processing_job_runner",
    "processing_job_service", "sanitize_value",
]
