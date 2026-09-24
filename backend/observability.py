"""Low-overhead runtime telemetry, deterministic alerts and capacity validation."""
from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import sqlite3
import tempfile
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

from backend.config import settings
from backend.data_governance import data_governance_service
from backend.storage import get_repository


logger = logging.getLogger(__name__)
LATENCY_BOUNDS_MS = (50, 100, 250, 500, 1000, 2000, 3000, 5000, 8000, 15000, 30000)
WINDOW_HOURS = {"1h": 1, "24h": 24, "7d": 24 * 7, "30d": 24 * 30}
BUSINESS_TIMEZONE = ZoneInfo("Asia/Shanghai")
BUSINESS_START_HOUR = 9
BUSINESS_END_HOUR = 18
IGNORED_PREFIXES = ("/api/ops/", "/api/auth/", "/api/health/live", "/api/feishu/events")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _minute_start(moment: Optional[datetime] = None) -> str:
    current = (moment or _utc_now()).replace(second=0, microsecond=0)
    return current.isoformat(timespec="minutes")


def _safe_route(scope: Dict[str, Any]) -> str:
    route = scope.get("route")
    template = str(getattr(route, "path", "") or "")
    if template:
        return template[:160]
    path = str(scope.get("path") or "")
    path = re.sub(r"/[0-9a-f]{16,}(?=/|$)", "/{id}", path, flags=re.I)
    path = re.sub(r"/\d+(?=/|$)", "/{id}", path)
    return path[:160]


def _metric_keys(path: str, method: str) -> List[str]:
    keys = ["api_request"]
    if method in {"POST", "PUT"} and any(fragment in path for fragment in (
        "/documents/uploads", "/documents/upload-batches", "/processing/jobs",
        "/knowledge-graph/rebuild", "/knowledge-graph/build",
        "/feishu/groups/", "/feishu/candidates/batch-actions", "/enrichments",
        "/governance/backups", "/governance/retention/apply",
    )):
        keys.append("async_acceptance")
    if path.startswith("/api/feishu/webhook"):
        keys.append("feishu_event_persistence")
    if path in {"/api/chat", "/api/smart-chat"}:
        keys.append("rag_first_content")
    return keys


class RuntimeMetricsCollector:
    """Aggregate request timings in memory and periodically flush minute buckets."""

    def __init__(self, repository_provider=get_repository, flush_seconds: float = 5.0) -> None:
        self.repository_provider = repository_provider
        self.flush_seconds = max(1.0, float(flush_seconds))
        self._lock = threading.Lock()
        self._buckets: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        self._task: Optional[asyncio.Task] = None
        self._stopping = False

    def record(self, metric_key: str, route_key: str, duration_ms: float, success: bool) -> None:
        duration = max(0.0, min(float(duration_ms), 300000.0))
        bucket = _minute_start()
        key = (str(metric_key)[:80], str(route_key)[:160], bucket)
        bound = next((item for item in LATENCY_BOUNDS_MS if duration <= item), 60000)
        with self._lock:
            item = self._buckets.setdefault(key, {
                "metric_key": key[0], "route_key": key[1], "bucket_start": key[2],
                "sample_count": 0, "success_count": 0, "error_count": 0,
                "total_duration_ms": 0.0, "max_duration_ms": 0.0, "histogram": defaultdict(int),
            })
            item["sample_count"] += 1
            item["success_count" if success else "error_count"] += 1
            item["total_duration_ms"] += duration
            item["max_duration_ms"] = max(item["max_duration_ms"], duration)
            item["histogram"][str(bound)] += 1

    def _drain(self) -> List[Dict[str, Any]]:
        with self._lock:
            values, self._buckets = list(self._buckets.values()), {}
        for item in values:
            item["histogram"] = dict(item["histogram"])
        return values

    def flush(self) -> int:
        values = self._drain()
        if not values:
            return 0
        try:
            return self.repository_provider().upsert_runtime_metric_buckets(values)
        except Exception:
            logger.exception("运行指标批量写入失败")
            # Telemetry must never block business traffic. Put the aggregates back for retry.
            with self._lock:
                for item in values:
                    key = (item["metric_key"], item["route_key"], item["bucket_start"])
                    existing = self._buckets.get(key)
                    if not existing:
                        item["histogram"] = defaultdict(int, item["histogram"])
                        self._buckets[key] = item
                        continue
                    for field in ("sample_count", "success_count", "error_count", "total_duration_ms"):
                        existing[field] += item[field]
                    existing["max_duration_ms"] = max(existing["max_duration_ms"], item["max_duration_ms"])
                    for bound, count in item["histogram"].items():
                        existing["histogram"][bound] += count
            return 0

    async def _run(self) -> None:
        while not self._stopping:
            await asyncio.sleep(self.flush_seconds)
            await asyncio.to_thread(self.flush)

    def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._stopping = False
        self._task = asyncio.create_task(self._run(), name="runtime-metrics-flush")

    async def stop(self) -> None:
        self._stopping = True
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        await asyncio.to_thread(self.flush)


runtime_metrics = RuntimeMetricsCollector()


class RuntimeObservabilityMiddleware:
    """Pure ASGI timing middleware that captures first content without request data."""

    def __init__(self, app: Any, collector: RuntimeMetricsCollector = runtime_metrics) -> None:
        self.app = app
        self.collector = collector

    async def __call__(self, scope: Dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        path = str(scope.get("path") or "")
        if not path.startswith("/api/") or any(path.startswith(prefix) for prefix in IGNORED_PREFIXES):
            await self.app(scope, receive, send)
            return
        started = time.perf_counter()
        status_code = 500
        first_content_ms: Optional[float] = None

        async def measured_send(message: Dict[str, Any]) -> None:
            nonlocal status_code, first_content_ms
            if message.get("type") == "http.response.start":
                status_code = int(message.get("status") or 500)
            elif message.get("type") == "http.response.body" and message.get("body") and first_content_ms is None:
                first_content_ms = (time.perf_counter() - started) * 1000
            await send(message)

        try:
            await self.app(scope, receive, measured_send)
        finally:
            elapsed = (time.perf_counter() - started) * 1000
            route_key = _safe_route(scope)
            success = status_code < 500
            for metric_key in _metric_keys(path, str(scope.get("method") or "GET").upper()):
                duration = first_content_ms if metric_key == "rag_first_content" and first_content_ms is not None else elapsed
                self.collector.record(metric_key, route_key, duration, success)


def _merge_metric(rows: Iterable[Dict[str, Any]], metric_key: str) -> Dict[str, Any]:
    selected = [item for item in rows if item.get("metric_key") == metric_key]
    count = sum(int(item.get("sample_count") or 0) for item in selected)
    success = sum(int(item.get("success_count") or 0) for item in selected)
    errors = sum(int(item.get("error_count") or 0) for item in selected)
    total = sum(float(item.get("total_duration_ms") or 0) for item in selected)
    maximum = max((float(item.get("max_duration_ms") or 0) for item in selected), default=0.0)
    histogram: Dict[int, int] = defaultdict(int)
    for item in selected:
        for bound, amount in dict(item.get("histogram") or {}).items():
            histogram[int(bound)] += int(amount)
    p95 = 0.0
    if count:
        threshold, cumulative = max(1, math.ceil(count * 0.95)), 0
        for bound in sorted(histogram):
            cumulative += histogram[bound]
            if cumulative >= threshold:
                p95 = float(bound)
                break
    return {
        "sample_count": count,
        "success_count": success,
        "error_count": errors,
        "availability_percent": round(success * 100 / count, 2) if count else None,
        "error_rate_percent": round(errors * 100 / count, 2) if count else None,
        "average_ms": round(total / count, 1) if count else None,
        "p95_ms": p95 if count else None,
        "max_ms": round(maximum, 1) if count else None,
    }


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("时间必须包含时区")
    return parsed.astimezone(timezone.utc)


def _document_processing_metric(events: Iterable[Dict[str, Any]], now: Optional[datetime] = None) -> Dict[str, Any]:
    current = now or _utc_now()
    items = list(events)
    within = 0
    durations = []
    for item in items:
        duration_ms = float(item.get("duration_ms") or 0)
        if item.get("status") == "running":
            try:
                duration_ms = max(0.0, (current - _parse_datetime(str(item["started_at"]))).total_seconds() * 1000)
            except (KeyError, ValueError):
                duration_ms = 0.0
        durations.append(duration_ms)
        if item.get("status") in {"succeeded", "failed"} and duration_ms <= 300000:
            within += 1
    count = len(items)
    return {
        "sample_count": count,
        "within_target_count": within,
        "completion_percent": round(within * 100 / count, 2) if count else None,
        "max_ms": round(max(durations), 1) if durations else None,
    }


class ObservabilityService:
    def __init__(self, repository_provider=get_repository) -> None:
        self.repository_provider = repository_provider

    @property
    def repository(self):
        return self.repository_provider()

    @staticmethod
    def _metric_status(metric: Dict[str, Any], target: float, field: str, minimum: int = 5) -> str:
        if int(metric.get("sample_count") or 0) < minimum or metric.get(field) is None:
            return "collecting"
        value = float(metric[field])
        higher_is_better = field in {"availability_percent", "completion_percent", "visibility_percent"}
        return "healthy" if (value >= target if higher_is_better else value <= target) else "degraded"

    def overview(self, window: str = "24h") -> Dict[str, Any]:
        runtime_metrics.flush()
        selected_window = window if window in WINDOW_HOURS else "24h"
        since = (_utc_now() - timedelta(hours=WINDOW_HOURS[selected_window])).replace(second=0, microsecond=0)
        rows = self.repository.list_runtime_metric_buckets(since.isoformat(timespec="minutes"))
        api = _merge_metric(rows, "api_request")
        acceptance = _merge_metric(rows, "async_acceptance")
        feishu = _merge_metric(rows, "feishu_event_persistence")
        rag = _merge_metric(rows, "rag_first_content")
        month_start_local = _utc_now().astimezone(BUSINESS_TIMEZONE).replace(
            day=1, hour=0, minute=0, second=0, microsecond=0,
        )
        month_start = month_start_local.astimezone(timezone.utc)
        month_rows = self.repository.list_runtime_metric_buckets(month_start.isoformat(timespec="minutes"))
        maintenance = self.repository.list_runtime_maintenance_windows(
            month_start.isoformat(), _utc_now().isoformat(), 200,
        )
        active_windows = []
        for item in maintenance:
            if item.get("status") != "scheduled":
                continue
            try:
                active_windows.append((_parse_datetime(item["starts_at"]), _parse_datetime(item["ends_at"])))
            except ValueError:
                continue
        business_rows = []
        excluded_samples = 0
        for item in month_rows:
            if item.get("metric_key") != "api_request":
                continue
            try:
                bucket = _parse_datetime(str(item.get("bucket_start") or ""))
            except ValueError:
                continue
            local = bucket.astimezone(BUSINESS_TIMEZONE)
            if local.weekday() >= 5 or not (BUSINESS_START_HOUR <= local.hour < BUSINESS_END_HOUR):
                continue
            if any(start < bucket + timedelta(minutes=1) and end > bucket for start, end in active_windows):
                excluded_samples += int(item.get("sample_count") or 0)
                continue
            business_rows.append(item)
        business_api = _merge_metric(business_rows, "api_request")
        document_metric = _document_processing_metric(
            self.repository.list_runtime_slo_events("document_processing", since.isoformat()),
        )
        failure_metric = self.repository.processing_failure_visibility(since.isoformat())
        metrics = [
            {"key": "availability", "label": "月度工作时段可用率", "target": ">= 99.5%", "unit": "%",
             "value": business_api["availability_percent"], "samples": business_api["sample_count"],
             "status": self._metric_status(business_api, 99.5, "availability_percent"),
             "excluded_samples": excluded_samples},
            {"key": "api_p95", "label": "普通接口 P95", "target": "<= 2,000 ms", "unit": "ms",
             "value": api["p95_ms"], "samples": api["sample_count"],
             "status": self._metric_status(api, 2000, "p95_ms")},
            {"key": "async_acceptance", "label": "异步受理 P95", "target": "<= 2,000 ms", "unit": "ms",
             "value": acceptance["p95_ms"], "samples": acceptance["sample_count"],
             "status": self._metric_status(acceptance, 2000, "p95_ms", 3)},
            {"key": "feishu_persistence", "label": "飞书持久化 P95", "target": "<= 3,000 ms", "unit": "ms",
             "value": feishu["p95_ms"], "samples": feishu["sample_count"],
             "status": self._metric_status(feishu, 3000, "p95_ms", 3)},
            {"key": "rag_first_content", "label": "RAG 首次内容 P95", "target": "<= 8,000 ms", "unit": "ms",
             "value": rag["p95_ms"], "samples": rag["sample_count"],
             "status": self._metric_status(rag, 8000, "p95_ms", 3)},
            {"key": "document_processing", "label": "文档 5 分钟完成/明确失败率", "target": ">= 95%", "unit": "%",
             "value": document_metric["completion_percent"], "samples": document_metric["sample_count"],
             "status": self._metric_status(document_metric, 95, "completion_percent", 3)},
            {"key": "failure_visibility", "label": "故障 1 分钟可见率", "target": "= 100%", "unit": "%",
             "value": failure_metric["visibility_percent"], "samples": failure_metric["sample_count"],
             "status": self._metric_status(failure_metric, 100, "visibility_percent", 1)},
        ]
        components = self._components()
        self._evaluate_alerts(metrics, components, api)
        alerts = self.repository.list_runtime_alerts("open", 30)
        upcoming_since = (_utc_now() - timedelta(days=7)).isoformat()
        upcoming_until = (_utc_now() + timedelta(days=30)).isoformat()
        overall = "degraded" if alerts else ("collecting" if any(item["status"] == "collecting" for item in metrics) else "healthy")
        return {
            "success": True, "window": selected_window, "since": since.isoformat(),
            "overall_status": overall, "metrics": metrics, "components": components,
            "alerts": alerts, "latest_capacity": self.repository.latest_runtime_capacity_run(),
            "latest_recovery_drill": self.repository.latest_runtime_recovery_drill(),
            "maintenance_windows": self.repository.list_runtime_maintenance_windows(upcoming_since, upcoming_until, 30),
            "methodology": {
                "availability": "按 Asia/Shanghai 当月周一至周五 09:00–18:00 统计；5xx 视为平台失败，4xx 不计故障，已登记计划维护分钟排除并单列样本。",
                "p95": "按分钟固定延迟桶近似计算；监控接口、登录与存活探针不进入统计。",
                "document_processing": "合法上传通过格式与大小校验后开始计时；5 分钟内成功或持久化明确失败均达标，超时运行项不达标。",
                "failure_visibility": "后台尝试失败与任务事件在同一 SQLite 事务持久化；按失败时间到可见事件时间计算。",
                "recovery_drill": "恢复演练复制最新有效备份到隔离临时库，校验哈希、完整性、外键和 schema；不替换在线数据库。",
                "sample_policy": "普通接口至少 5 个样本、专项指标至少 3 个样本后才判断是否达标。",
                "telemetry_loss": "异常退出时最多可能丢失一个 5 秒刷新周期的运行遥测，不影响业务数据。",
            },
        }

    def _components(self) -> List[Dict[str, Any]]:
        try:
            storage = self.repository.database.quick_check()
            migration = self.repository.database.migration_status()
            storage_status = "healthy" if storage.get("ok") and not migration.get("pending_versions") else "degraded"
        except Exception:
            logger.exception("可观测性存储检查失败")
            migration, storage_status = {"pending_versions": ["unknown"]}, "degraded"
        facts = self.repository.runtime_health_facts()
        recovery = self.repository.latest_runtime_recovery_drill()
        backups = data_governance_service.list_backups()
        regular = [item for item in backups if item.get("kind") == "regular" and item.get("valid")]
        backup_age_hours: Optional[float] = None
        if regular and regular[0].get("created_at"):
            try:
                created = datetime.fromisoformat(str(regular[0]["created_at"]).replace("Z", "+00:00"))
                backup_age_hours = round((_utc_now() - created.astimezone(timezone.utc)).total_seconds() / 3600, 1)
            except ValueError:
                backup_age_hours = None
        backup_status = "healthy" if backup_age_hours is not None and backup_age_hours <= 24 else "degraded"
        recovery_fresh = False
        if recovery and recovery.get("status") == "passed":
            try:
                recovery_fresh = _utc_now() - _parse_datetime(str(recovery["created_at"])) < timedelta(days=30)
            except (KeyError, ValueError):
                recovery_fresh = False
        recovery_status = "healthy" if recovery_fresh else "degraded"
        job_status = "degraded" if facts["failed_jobs"] else ("warning" if facts["retry_wait_jobs"] else "healthy")
        return [
            {"key": "storage", "label": "SQLite 数据库", "status": storage_status,
             "detail": "完整性正常" if storage_status == "healthy" else "完整性检查或迁移异常",
             "value": f"v{migration.get('current_version', 0)}"},
            {"key": "jobs", "label": "处理执行器", "status": job_status,
             "detail": f"失败 {facts['failed_jobs']} · 等待重试 {facts['retry_wait_jobs']}",
             "value": "可追踪"},
            {"key": "backup", "label": "数据备份", "status": backup_status,
             "detail": "暂无有效备份" if backup_age_hours is None else f"最近备份 {backup_age_hours} 小时前",
             "value": "RPO 24h"},
            {"key": "recovery", "label": "恢复演练", "status": recovery_status,
             "detail": "尚无演练记录" if not recovery else (
                 f"最近演练通过 · RTO {recovery['rto_seconds']:.2f} 秒" if recovery_status == "healthy"
                 else "最近演练未通过，请检查报告"
             ), "value": "RTO 4h"},
            {"key": "llm", "label": "模型供应商", "status": "healthy" if settings.OPENAI_API_KEY else "unconfigured",
             "detail": settings.LLM_MODEL if settings.OPENAI_API_KEY else "未配置 API Key", "value": "外部依赖"},
            {"key": "feishu", "label": "飞书事件通道", "status": "healthy" if settings.FEISHU_APP_ID and settings.FEISHU_APP_SECRET else "unconfigured",
             "detail": "凭据已配置" if settings.FEISHU_APP_ID and settings.FEISHU_APP_SECRET else "未配置", "value": "外部依赖"},
        ]

    def _evaluate_alerts(self, metrics: List[Dict[str, Any]], components: List[Dict[str, Any]], api: Dict[str, Any]) -> None:
        by_component = {item["key"]: item for item in components}
        by_metric = {item["key"]: item for item in metrics}
        rules = [
            ("storage_degraded", by_component["storage"]["status"] == "degraded", "critical", "数据库健康检查异常", "检查完整性与待执行迁移", by_component["storage"]),
            ("backup_stale", by_component["backup"]["status"] == "degraded", "warning", "有效备份超过 RPO 窗口", "在数据治理中创建并校验 SQLite 备份", by_component["backup"]),
            ("processing_failures", by_component["jobs"]["status"] == "degraded", "warning", "存在失败的后台任务", "进入处理任务中心查看错误并按需重试", by_component["jobs"]),
            ("recovery_drill_missing", by_component["recovery"]["status"] == "degraded", "warning", "恢复演练未通过或尚未执行", "基于最新有效备份运行隔离恢复演练", by_component["recovery"]),
            ("api_latency", by_metric["api_p95"]["status"] == "degraded", "warning", "普通接口 P95 超过目标", "检查慢路由、SQLite 锁等待与数据规模", by_metric["api_p95"]),
            ("document_processing_slo", by_metric["document_processing"]["status"] == "degraded", "warning", "文档 5 分钟处理目标未达标", "检查超时上传、解析失败和知识投影", by_metric["document_processing"]),
            ("failure_visibility_slo", by_metric["failure_visibility"]["status"] == "degraded", "critical", "任务故障未在 1 分钟内可见", "检查处理任务事件事务和任务中心读取链路", by_metric["failure_visibility"]),
            ("api_error_rate", api["sample_count"] >= 5 and float(api["error_rate_percent"] or 0) > 1.0, "critical", "平台 5xx 错误率超过 1%", "检查服务日志和最近变更", api),
        ]
        for rule_key, active, severity, title, summary, evidence in rules:
            self.repository.sync_runtime_alert(rule_key, active, severity=severity, title=title, summary=summary, evidence=evidence)

    def run_capacity_validation(self, context: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
        count = max(1000, min(int(payload.get("message_count") or 10000), 50000))
        context.report("capacity_prepare", 10, "正在创建隔离的 SQLite 容量样本")
        timings: List[float] = []
        with tempfile.TemporaryDirectory(prefix="ai-capacity-") as temp_dir:
            path = Path(temp_dir) / "capacity.db"
            connection = sqlite3.connect(path)
            try:
                connection.executescript(
                    "CREATE TABLE messages(id INTEGER PRIMARY KEY,chat_id TEXT,create_time INTEGER,content TEXT);"
                    "CREATE INDEX idx_messages_chat_time ON messages(chat_id,create_time DESC);"
                )
                connection.executemany(
                    "INSERT INTO messages(chat_id,create_time,content) VALUES(?,?,?)",
                    ((f"chat_{index % 20}", index, f"message {index}") for index in range(count)),
                )
                connection.commit()
                context.report("capacity_query", 55, "正在重复执行列表与详情查询")
                for index in range(60):
                    started = time.perf_counter()
                    rows = connection.execute(
                        "SELECT id,chat_id,create_time,content FROM messages WHERE chat_id=? "
                        "ORDER BY create_time DESC LIMIT 50",
                        (f"chat_{index % 20}",),
                    ).fetchall()
                    if not rows:
                        raise RuntimeError("容量样本查询结果为空")
                    connection.execute("SELECT * FROM messages WHERE id=?", (rows[0][0],)).fetchone()
                    timings.append((time.perf_counter() - started) * 1000)
            finally:
                connection.close()
        ordered = sorted(timings)
        percentile = lambda ratio: ordered[min(len(ordered) - 1, max(0, math.ceil(len(ordered) * ratio) - 1))]
        result = {
            "status": "passed" if percentile(0.95) <= 2000 else "failed",
            "message_count": count,
            "p50_ms": round(percentile(0.50), 2), "p95_ms": round(percentile(0.95), 2),
            "max_ms": round(max(ordered), 2),
            "actor_user_id": str(payload.get("actor_user_id") or "user_system"),
            "report": {
                "scope": "隔离 SQLite 列表与详情查询基准，不含 HTTP、模型和网络耗时",
                "iterations": len(timings), "target_ms": 2000,
            },
        }
        context.report("capacity_save", 90, "正在保存容量验证报告")
        saved = self.repository.save_runtime_capacity_run(result)
        self.repository.sync_runtime_alert(
            "capacity_failed", saved["status"] == "failed", severity="critical",
            title="10,000 条消息容量基线未通过", summary="检查索引、查询计划与 SQLite 锁等待", evidence=saved,
        )
        return saved

    def create_maintenance_window(self, payload: Dict[str, Any], actor_user_id: str) -> Dict[str, Any]:
        title = str(payload.get("title") or "").strip()
        if not title:
            raise ValueError("维护窗口标题不能为空")
        starts_at = _parse_datetime(str(payload.get("starts_at") or ""))
        ends_at = _parse_datetime(str(payload.get("ends_at") or ""))
        if ends_at <= starts_at:
            raise ValueError("维护结束时间必须晚于开始时间")
        if ends_at - starts_at > timedelta(days=7):
            raise ValueError("单个维护窗口不能超过 7 天")
        window = self.repository.create_runtime_maintenance_window({
            "title": title, "reason": payload.get("reason"),
            "starts_at": starts_at.isoformat(timespec="seconds"),
            "ends_at": ends_at.isoformat(timespec="seconds"),
            "created_by": actor_user_id,
        })
        self.repository.write_security_audit(
            actor_user_id, "operations.maintenance_created", "maintenance_window", window["window_id"],
            {"starts_at": window["starts_at"], "ends_at": window["ends_at"], "title": window["title"]},
        )
        return window

    def cancel_maintenance_window(self, window_id: str, actor_user_id: str) -> Dict[str, Any]:
        window = self.repository.cancel_runtime_maintenance_window(window_id, actor_user_id)
        self.repository.write_security_audit(
            actor_user_id, "operations.maintenance_cancelled", "maintenance_window", window_id,
            {"title": window.get("title", "")},
        )
        return window

    def should_schedule_recovery_drill(self) -> bool:
        backups = [item for item in data_governance_service.list_backups() if item.get("kind") == "regular" and item.get("valid")]
        if not backups:
            return False
        latest = self.repository.latest_runtime_recovery_drill()
        if not latest:
            return True
        newest_backup = backups[0]
        if str(latest.get("backup_name") or "") != str(newest_backup.get("name") or ""):
            return True
        if latest.get("status") != "passed":
            return False
        try:
            created = _parse_datetime(str(latest["created_at"]))
        except (KeyError, ValueError):
            return True
        return _utc_now() - created >= timedelta(days=30)

    def run_recovery_drill(self, context: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
        context.report("recovery_select", 10, "正在选择最新有效备份")
        backups = [item for item in data_governance_service.list_backups() if item.get("kind") == "regular" and item.get("valid")]
        started = time.perf_counter()
        backup = backups[0] if backups else {}
        report: Dict[str, Any] = {"scope": "isolated_sqlite_restore", "online_database_changed": False}
        status, error = "failed", ""
        rpo_hours = 0.0
        try:
            if not backup:
                raise RuntimeError("没有可用于演练的有效常规备份")
            created = _parse_datetime(str(backup.get("created_at") or ""))
            rpo_hours = max(0.0, (_utc_now() - created).total_seconds() / 3600)
            root = Path(data_governance_service.backup_dir_provider()).expanduser().resolve()
            source_path = root / str(backup["name"])
            expected_hash = str(backup.get("sha256") or "")
            if expected_hash and self.repository.database.file_hash(source_path) != expected_hash:
                raise RuntimeError("演练前备份哈希复核失败")
            context.report("recovery_restore", 35, "正在隔离临时库中恢复备份")
            with tempfile.TemporaryDirectory(prefix="ai-recovery-drill-") as temp_dir:
                target_path = Path(temp_dir) / "restored.db"
                source = sqlite3.connect(str(source_path))
                target = sqlite3.connect(str(target_path))
                try:
                    source.backup(target)
                    target.commit()
                finally:
                    target.close()
                    source.close()
                isolated_database = self.repository.database.__class__(target_path)
                isolated_database.initialize()
                isolated_check = isolated_database.quick_check()
                isolated_migration = isolated_database.migration_status()
                quick = isolated_check["quick_check"]
                foreign = isolated_check["foreign_key_errors"]
                schema_version = int(isolated_migration["current_version"])
                pending_versions = list(isolated_migration["pending_versions"])
                with isolated_database.transaction() as connection:
                    project_count = int(connection.execute("SELECT COUNT(*) FROM projects").fetchone()[0])
            report.update({
                "sha256_verified": bool(expected_hash), "quick_check": quick,
                "foreign_key_errors": len(foreign), "schema_version": schema_version,
                "schema_pending_versions": pending_versions,
                "project_count": project_count,
            })
            if quick != ["ok"] or foreign or pending_versions or project_count < 1:
                raise RuntimeError("隔离恢复后的数据库完整性校验未通过")
            status = "passed" if rpo_hours <= 24 else "failed"
            if status == "failed":
                error = "备份已超过 24 小时 RPO"
        except Exception as exc:  # Persist a failed drill instead of hiding it in a failed job.
            error = str(exc)[:1000]
            report["error"] = error
        rto_seconds = max(0.0, time.perf_counter() - started)
        if rto_seconds > 4 * 3600:
            status, error = "failed", "隔离恢复耗时超过 4 小时 RTO"
        report.update({"rpo_target_hours": 24, "rto_target_seconds": 14400, "error": error})
        context.report("recovery_record", 90, "正在保存恢复演练报告")
        saved = self.repository.save_runtime_recovery_drill({
            "status": status, "backup_name": backup.get("name", ""),
            "backup_created_at": backup.get("created_at", ""), "rpo_hours": round(rpo_hours, 2),
            "rto_seconds": round(rto_seconds, 3), "report": report,
            "actor_user_id": str(payload.get("actor_user_id") or "user_system"),
        })
        self.repository.sync_runtime_alert(
            "recovery_drill_missing", status != "passed", severity="warning",
            title="恢复演练未通过或尚未执行", summary=error or "请检查恢复演练报告", evidence=saved,
        )
        return saved


observability_service = ObservabilityService()


__all__ = [
    "RuntimeMetricsCollector", "RuntimeObservabilityMiddleware", "ObservabilityService",
    "observability_service", "runtime_metrics",
]
