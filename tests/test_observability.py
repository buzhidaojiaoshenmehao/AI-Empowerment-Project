import asyncio
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from backend.observability import (
    ObservabilityService,
    RuntimeMetricsCollector,
    RuntimeObservabilityMiddleware,
)
from backend.storage.database import Database
from backend.storage.repository import StorageRepository


class _Context:
    def __init__(self):
        self.events = []

    def report(self, stage, progress, message):
        self.events.append((stage, progress, message))


class ObservabilityRepositoryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repository = StorageRepository(Database(Path(self.temp.name) / "app.db"))

    def tearDown(self):
        self.temp.cleanup()

    def test_schema_aggregation_and_alert_resolution(self):
        self.repository.upsert_runtime_metric_buckets([{
            "metric_key": "api_request", "route_key": "/api/documents",
            "bucket_start": "2026-08-24T10:00+00:00", "sample_count": 3,
            "success_count": 2, "error_count": 1, "total_duration_ms": 750,
            "max_duration_ms": 500, "histogram": {"100": 1, "500": 2},
        }])
        self.repository.upsert_runtime_metric_buckets([{
            "metric_key": "api_request", "route_key": "/api/documents",
            "bucket_start": "2026-08-24T10:00+00:00", "sample_count": 2,
            "success_count": 2, "error_count": 0, "total_duration_ms": 250,
            "max_duration_ms": 200, "histogram": {"250": 2},
        }])
        rows = self.repository.list_runtime_metric_buckets("2026-08-24T09:00+00:00")
        self.assertEqual(rows[0]["sample_count"], 5)
        self.assertEqual(rows[0]["success_count"], 4)
        self.assertEqual(sum(rows[0]["histogram"].values()), 5)

        created = self.repository.sync_runtime_alert(
            "api_error", True, severity="critical", title="错误率异常",
            summary="检查服务", evidence={"rate": 20},
        )
        self.assertEqual(created["status"], "open")
        self.assertEqual(len(self.repository.list_runtime_alerts("open")), 1)
        resolved = self.repository.sync_runtime_alert(
            "api_error", False, severity="critical", title="错误率异常",
        )
        self.assertEqual(resolved["status"], "resolved")
        self.assertEqual(self.repository.list_runtime_alerts("open"), [])

    def test_collector_batches_without_request_content(self):
        collector = RuntimeMetricsCollector(lambda: self.repository, flush_seconds=60)
        collector.record("api_request", "/api/documents/{id}", 125, True)
        collector.record("api_request", "/api/documents/{id}", 2250, False)
        self.assertEqual(collector.flush(), 1)
        rows = self.repository.list_runtime_metric_buckets("2000-01-01T00:00+00:00")
        self.assertEqual(rows[0]["route_key"], "/api/documents/{id}")
        serialized = str(rows)
        self.assertNotIn("prompt", serialized.lower())
        self.assertNotIn("password", serialized.lower())

    def test_capacity_validation_is_isolated_and_persistent(self):
        service = ObservabilityService(lambda: self.repository)
        context = _Context()
        with self.repository.database.transaction() as connection:
            before = int(connection.execute("SELECT COUNT(*) FROM feishu_messages").fetchone()[0])
        result = service.run_capacity_validation(context, {
            "message_count": 10000, "actor_user_id": "user_admin",
        })
        with self.repository.database.transaction() as connection:
            after = int(connection.execute("SELECT COUNT(*) FROM feishu_messages").fetchone()[0])
        self.assertEqual(before, after)
        self.assertEqual(result["message_count"], 10000)
        self.assertEqual(result["report"]["iterations"], 60)
        self.assertEqual(self.repository.latest_runtime_capacity_run()["run_id"], result["run_id"])
        self.assertTrue(any(item[0] == "capacity_query" for item in context.events))

    def test_overview_marks_low_sample_metrics_as_collecting(self):
        service = ObservabilityService(lambda: self.repository)
        with patch("backend.observability.data_governance_service.list_backups", return_value=[]):
            overview = service.overview("24h")
        self.assertEqual(overview["overall_status"], "degraded")  # missing backup is actionable
        self.assertTrue(all(item["status"] == "collecting" for item in overview["metrics"]))
        self.assertTrue(any(item["rule_key"] == "backup_stale" for item in overview["alerts"]))

    def test_monthly_business_availability_excludes_registered_maintenance(self):
        fixed_now = datetime(2026, 8, 26, 8, 0, tzinfo=timezone.utc)
        self.repository.upsert_runtime_metric_buckets([
            {
                "metric_key": "api_request", "route_key": "/api/documents",
                "bucket_start": "2026-08-03T01:30+00:00", "sample_count": 5,
                "success_count": 0, "error_count": 5, "total_duration_ms": 500,
                "max_duration_ms": 100, "histogram": {"100": 5},
            },
            {
                "metric_key": "api_request", "route_key": "/api/documents",
                "bucket_start": "2026-08-03T03:00+00:00", "sample_count": 5,
                "success_count": 5, "error_count": 0, "total_duration_ms": 500,
                "max_duration_ms": 100, "histogram": {"100": 5},
            },
        ])
        self.repository.create_runtime_maintenance_window({
            "title": "数据库维护", "starts_at": "2026-08-03T01:00:00+00:00",
            "ends_at": "2026-08-03T02:00:00+00:00", "created_by": "admin",
        })
        service = ObservabilityService(lambda: self.repository)
        with patch("backend.observability._utc_now", return_value=fixed_now), \
                patch("backend.observability.data_governance_service.list_backups", return_value=[]):
            overview = service.overview("30d")
        availability = next(item for item in overview["metrics"] if item["key"] == "availability")
        self.assertEqual(availability["value"], 100.0)
        self.assertEqual(availability["samples"], 5)
        self.assertEqual(availability["excluded_samples"], 5)

    def test_document_and_failure_visibility_metrics_are_persistent(self):
        event_id = self.repository.start_runtime_slo_event("document_processing", "upload-1")
        completed = self.repository.complete_runtime_slo_event(event_id, succeeded=False, detail={"error_type": "ValueError"})
        self.assertEqual(completed["status"], "failed")
        self.assertNotIn("secret", str(completed))

        job = self.repository.create_or_get_processing_job({
            "job_type": "test", "created_by": "admin", "max_attempts": 1,
        })
        self.repository.claim_next_processing_job("worker")
        self.repository.fail_processing_job(
            job["job_id"], error_code="TEST_FAILED", error_message="明确错误", retryable=False,
        )
        visibility = self.repository.processing_failure_visibility("2000-01-01T00:00:00+00:00")
        self.assertEqual(visibility["sample_count"], 1)
        self.assertEqual(visibility["visibility_percent"], 100.0)

    def test_recovery_drill_restores_only_to_isolated_database(self):
        backup = self.repository.database.backup(Path(self.temp.name) / "backups", "drill-source")
        backup_item = {
            "name": Path(backup["path"]).name, "created_at": backup["created_at"],
            "sha256": backup["sha256"], "valid": True, "kind": "regular",
        }
        service = ObservabilityService(lambda: self.repository)
        context = _Context()
        with patch("backend.observability.data_governance_service.list_backups", return_value=[backup_item]), \
                patch("backend.observability.data_governance_service.backup_dir_provider", return_value=Path(self.temp.name) / "backups"):
            result = service.run_recovery_drill(context, {"actor_user_id": "admin"})
            self.assertFalse(service.should_schedule_recovery_drill())
        self.assertEqual(result["status"], "passed")
        self.assertFalse(result["report"]["online_database_changed"])
        self.assertEqual(result["report"]["quick_check"], ["ok"])
        self.assertEqual(self.repository.latest_runtime_recovery_drill()["drill_id"], result["drill_id"])
        audit = self.repository.list_audit_events(action="operations.recovery_drill_completed")
        self.assertEqual(audit["total"], 1)
        self.assertEqual(audit["items"][0]["actor"], "admin")
        self.assertEqual(audit["items"][0]["object_id"], result["drill_id"])
        self.assertEqual(audit["items"][0]["detail"]["status"], "passed")
        self.assertTrue(audit["items"][0]["detail"]["sha256_verified"])
        self.assertFalse(audit["items"][0]["detail"]["online_database_changed"])
        self.assertTrue(any(stage == "recovery_restore" for stage, _, _ in context.events))

    def test_recovery_drill_report_rolls_back_when_audit_write_fails(self):
        with patch.object(
            self.repository,
            "_write_audit_connection",
            side_effect=RuntimeError("audit unavailable"),
        ), self.assertRaisesRegex(RuntimeError, "audit unavailable"):
            self.repository.save_runtime_recovery_drill({
                "status": "passed",
                "backup_name": "backup.db",
                "backup_created_at": "2026-08-28T11:08:21+00:00",
                "rpo_hours": 0.05,
                "rto_seconds": 0.014,
                "report": {"sha256_verified": True, "online_database_changed": False},
                "actor_user_id": "admin",
            })
        self.assertIsNone(self.repository.latest_runtime_recovery_drill())
        self.assertEqual(
            self.repository.list_audit_events(action="operations.recovery_drill_completed")["total"],
            0,
        )


class ObservabilityMiddlewareTest(unittest.IsolatedAsyncioTestCase):
    async def test_middleware_records_route_timing_without_body(self):
        recorded = []

        class Collector:
            def record(self, *args):
                recorded.append(args)

        async def app(scope, receive, send):
            scope["route"] = type("Route", (), {"path": "/api/chat"})()
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"private answer", "more_body": False})

        middleware = RuntimeObservabilityMiddleware(app, Collector())
        messages = []

        async def receive():
            return {"type": "http.request", "body": b"secret prompt", "more_body": False}

        async def send(message):
            messages.append(message)

        await middleware({"type": "http", "path": "/api/chat", "method": "POST"}, receive, send)
        self.assertEqual([item[0] for item in recorded], ["api_request", "rag_first_content"])
        self.assertTrue(all(item[1] == "/api/chat" for item in recorded))
        self.assertNotIn("secret prompt", str(recorded))
        self.assertNotIn("private answer", str(recorded))


if __name__ == "__main__":
    unittest.main()
