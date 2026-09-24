"""Central SQLite repositories. Business modules must not contain SQL."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from backend.handover_policy import item_review_policy
from backend.storage.database import DEFAULT_ORGANIZATION_ID, DEFAULT_PROJECT_ID, Database, utc_now


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _json_load(value: Any, default: Any) -> Any:
    if not value:
        return default
    try:
        parsed = json.loads(str(value))
        return parsed
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def _stable_id(prefix: str, *parts: Any) -> str:
    raw = ":".join(str(part or "") for part in parts)
    return f"{prefix}_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:24]}"


def _like_literal(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


_SENSITIVE_AUDIT_KEYS = {
    "api_key", "apikey", "app_secret", "secret", "token", "password",
    "authorization", "cookie", "session", "content", "prompt", "context",
}


DEFAULT_HANDOVER_ITEMS = (
    ("project_context", "项目背景与目标", "说明项目目标、当前阶段、关键范围与成功标准。"),
    ("scope_boundary", "职责范围与边界", "列明负责事项、不负责事项、权限边界与接替后的责任归属。"),
    ("current_work", "在办事项与优先级", "列明进行中工作、当前进度、下一步、截止时间与依赖。"),
    ("key_decisions", "关键决策与技术约束", "沉淀重要决策、选择依据、不能轻易改变的约束及历史背景。"),
    ("operating_procedures", "日常流程与操作手册", "提供例行流程、发布步骤、常用工具和异常处理路径。"),
    ("risk_inventory", "风险、故障与应急路径", "说明已知风险、常见故障、影响范围、缓解方案和升级路径。"),
    ("stakeholders", "关键联系人与协作接口", "列明上下游、业务与技术联系人，以及协作约定。"),
    ("access_transfer", "账号、权限与资源移交", "只记录授权范围与申请路径，不得填写口令、密钥等敏感值。"),
)


def _redact_audit_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): ("***" if str(key).lower() in _SENSITIVE_AUDIT_KEYS else _redact_audit_value(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_audit_value(item) for item in value[:100]]
    if isinstance(value, str):
        return value[:1000]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)[:1000]


class StorageRepository:
    """Repository facade for all authoritative business state."""

    def __init__(self, database: Database, project_id: str = DEFAULT_PROJECT_ID) -> None:
        self.database = database
        self.project_id = project_id
        self.database.initialize()
        self.backfill_knowledge_assets()
        self.backfill_handover_governance()

    def project_name(self) -> str:
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT name FROM projects WHERE project_id=?", (self.project_id,)
            ).fetchone()
            return str(row["name"] if row else "默认项目")

    # RAG quality evaluation and answer feedback -------------------
    @staticmethod
    def _rag_evaluation_case_payload(row: Any) -> Dict[str, Any]:
        payload = dict(row)
        payload["expected_points"] = _json_load(payload.pop("expected_points_json", "[]"), [])
        payload["allowed_asset_ids"] = _json_load(payload.pop("allowed_asset_ids_json", "[]"), [])
        payload["allowed_source_files"] = _json_load(payload.pop("allowed_source_files_json", "[]"), [])
        payload["expect_no_answer"] = bool(payload.get("expect_no_answer"))
        return payload

    @staticmethod
    def _rag_evaluation_run_payload(row: Any) -> Dict[str, Any]:
        payload = dict(row)
        payload["strategy"] = _json_load(payload.pop("strategy_json", "{}"), {})
        payload["metrics"] = _json_load(payload.pop("metrics_json", "{}"), {})
        return payload

    @staticmethod
    def _rag_evaluation_result_payload(row: Any) -> Dict[str, Any]:
        payload = dict(row)
        payload["no_answer"] = bool(payload.get("no_answer"))
        payload["passed"] = bool(payload.get("passed"))
        payload["citations"] = _json_load(payload.pop("citations_json", "[]"), [])
        payload["matched_points"] = _json_load(payload.pop("matched_points_json", "[]"), [])
        return payload

    def create_rag_evaluation_set(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        now = utc_now()
        evaluation_set_id = str(payload.get("evaluation_set_id") or f"rages_{uuid.uuid4().hex}")
        with self.database.transaction(write=True) as connection:
            connection.execute(
                """
                INSERT INTO rag_evaluation_sets(
                    evaluation_set_id,project_id,name,version,description,status,min_case_target,
                    created_by,created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    evaluation_set_id, self.project_id, str(payload.get("name") or "").strip(),
                    int(payload.get("version") or 1), str(payload.get("description") or "").strip(),
                    str(payload.get("status") or "draft"), int(payload.get("min_case_target") or 30),
                    str(payload.get("created_by") or "user_system"), now, now,
                ),
            )
            self._write_audit_connection(
                connection,
                f"audit_{uuid.uuid4().hex}",
                str(payload.get("created_by") or "user_system"),
                "rag.evaluation_set_created",
                "rag_evaluation_set",
                evaluation_set_id,
                {
                    "name": str(payload.get("name") or "").strip()[:200],
                    "version": int(payload.get("version") or 1),
                    "status": str(payload.get("status") or "draft"),
                    "min_case_target": int(payload.get("min_case_target") or 30),
                },
                project_id=self.project_id,
                created_at=now,
            )
        return self.get_rag_evaluation_set(evaluation_set_id) or {}

    def add_rag_evaluation_case(self, evaluation_set_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        now = utc_now()
        evaluation_case_id = str(payload.get("evaluation_case_id") or f"ragec_{uuid.uuid4().hex}")
        with self.database.transaction(write=True) as connection:
            parent = connection.execute(
                "SELECT status FROM rag_evaluation_sets WHERE evaluation_set_id=? AND project_id=?",
                (evaluation_set_id, self.project_id),
            ).fetchone()
            if not parent:
                raise ValueError("评测集不存在")
            if str(parent["status"]) == "archived":
                raise ValueError("归档评测集不能新增问题")
            position = payload.get("position")
            if position is None:
                row = connection.execute(
                    "SELECT COALESCE(MAX(position),-1)+1 AS next_position FROM rag_evaluation_cases WHERE evaluation_set_id=?",
                    (evaluation_set_id,),
                ).fetchone()
                position = int(row["next_position"] or 0)
            connection.execute(
                """
                INSERT INTO rag_evaluation_cases(
                    evaluation_case_id,evaluation_set_id,project_id,question,scenario,
                    expected_points_json,allowed_asset_ids_json,allowed_source_files_json,
                    expect_no_answer,position,created_by,created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    evaluation_case_id, evaluation_set_id, self.project_id,
                    str(payload.get("question") or "").strip(), str(payload.get("scenario") or "fact"),
                    _json_dump(payload.get("expected_points") or []),
                    _json_dump(payload.get("allowed_asset_ids") or []),
                    _json_dump(payload.get("allowed_source_files") or []),
                    1 if payload.get("expect_no_answer") else 0, int(position),
                    str(payload.get("created_by") or "user_system"), now, now,
                ),
            )
            connection.execute(
                "UPDATE rag_evaluation_sets SET updated_at=? WHERE evaluation_set_id=?",
                (now, evaluation_set_id),
            )
            row = connection.execute(
                "SELECT * FROM rag_evaluation_cases WHERE evaluation_case_id=?",
                (evaluation_case_id,),
            ).fetchone()
        return self._rag_evaluation_case_payload(row)

    def list_rag_evaluation_sets(self) -> List[Dict[str, Any]]:
        with self.database.transaction() as connection:
            rows = connection.execute(
                """
                SELECT rs.*,
                    (SELECT COUNT(*) FROM rag_evaluation_cases rc
                     WHERE rc.evaluation_set_id=rs.evaluation_set_id) AS case_count,
                    (SELECT COUNT(*) FROM rag_evaluation_runs rr
                     WHERE rr.evaluation_set_id=rs.evaluation_set_id AND rr.status='completed') AS run_count
                FROM rag_evaluation_sets rs
                WHERE rs.project_id=?
                ORDER BY rs.updated_at DESC,rs.evaluation_set_id
                """,
                (self.project_id,),
            ).fetchall()
            result = []
            for row in rows:
                item = dict(row)
                item["ready"] = int(item.get("case_count") or 0) >= int(item.get("min_case_target") or 30)
                result.append(item)
            return result

    def get_rag_evaluation_set(self, evaluation_set_id: str) -> Optional[Dict[str, Any]]:
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM rag_evaluation_sets WHERE evaluation_set_id=? AND project_id=?",
                (evaluation_set_id, self.project_id),
            ).fetchone()
            if not row:
                return None
            cases = [
                self._rag_evaluation_case_payload(item)
                for item in connection.execute(
                    "SELECT * FROM rag_evaluation_cases WHERE evaluation_set_id=? ORDER BY position,evaluation_case_id",
                    (evaluation_set_id,),
                )
            ]
            runs = [
                self._rag_evaluation_run_payload(item)
                for item in connection.execute(
                    "SELECT * FROM rag_evaluation_runs WHERE evaluation_set_id=? ORDER BY started_at DESC,evaluation_run_id DESC LIMIT 20",
                    (evaluation_set_id,),
                )
            ]
        payload = dict(row)
        payload.update(
            cases=cases, runs=runs, case_count=len(cases),
            ready=len(cases) >= int(payload.get("min_case_target") or 30),
        )
        return payload

    def save_rag_evaluation_run(
        self, evaluation_set_id: str, *, strategy: Dict[str, Any], metrics: Dict[str, Any],
        results: Sequence[Dict[str, Any]], created_by: str,
    ) -> Dict[str, Any]:
        now = utc_now()
        evaluation_run_id = f"rager_{uuid.uuid4().hex}"
        with self.database.transaction(write=True) as connection:
            parent = connection.execute(
                "SELECT 1 FROM rag_evaluation_sets WHERE evaluation_set_id=? AND project_id=?",
                (evaluation_set_id, self.project_id),
            ).fetchone()
            if not parent:
                raise ValueError("评测集不存在")
            connection.execute(
                """
                INSERT INTO rag_evaluation_runs(
                    evaluation_run_id,evaluation_set_id,project_id,status,strategy_json,metrics_json,
                    created_by,started_at,completed_at
                ) VALUES(?,?,?,'completed',?,?,?,?,?)
                """,
                (
                    evaluation_run_id, evaluation_set_id, self.project_id,
                    _json_dump(strategy), _json_dump(metrics), created_by, now, now,
                ),
            )
            for item in results:
                connection.execute(
                    """
                    INSERT INTO rag_evaluation_results(
                        evaluation_result_id,evaluation_run_id,evaluation_case_id,project_id,no_answer,
                        citation_validity_rate,allowed_source_precision,expected_evidence_coverage,passed,
                        citations_json,matched_points_json,detail,created_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        f"rageres_{uuid.uuid4().hex}", evaluation_run_id,
                        str(item.get("evaluation_case_id") or ""), self.project_id,
                        1 if item.get("no_answer") else 0,
                        float(item.get("citation_validity_rate") or 0),
                        float(item.get("allowed_source_precision") or 0),
                        float(item.get("expected_evidence_coverage") or 0),
                        1 if item.get("passed") else 0,
                        _json_dump(item.get("citations") or []),
                        _json_dump(item.get("matched_points") or []),
                        str(item.get("detail") or "")[:1000], now,
                    ),
                )
            self._write_audit_connection(
                connection,
                f"audit_{uuid.uuid4().hex}",
                created_by,
                "rag.evaluation_run_completed",
                "rag_evaluation_run",
                evaluation_run_id,
                {
                    "evaluation_set_id": evaluation_set_id,
                    "case_count": int(metrics.get("case_count") or 0),
                    "passed_count": int(metrics.get("passed_count") or 0),
                    "pass_rate": float(metrics.get("pass_rate") or 0),
                    "citation_validity_rate": float(metrics.get("citation_validity_rate") or 0),
                    "permission_leak_count": int(metrics.get("permission_leak_count") or 0),
                    "no_answer_accuracy": float(metrics.get("no_answer_accuracy") or 0),
                    "ready_case_target_met": bool(metrics.get("ready_case_target_met")),
                    "evaluation_contract": str(strategy.get("evaluation_contract") or ""),
                    "text_normalization": str(strategy.get("text_normalization") or ""),
                },
                project_id=self.project_id,
                created_at=now,
            )
            row = connection.execute(
                "SELECT * FROM rag_evaluation_runs WHERE evaluation_run_id=?",
                (evaluation_run_id,),
            ).fetchone()
            result_rows = connection.execute(
                "SELECT * FROM rag_evaluation_results WHERE evaluation_run_id=? ORDER BY evaluation_result_id",
                (evaluation_run_id,),
            ).fetchall()
        run = self._rag_evaluation_run_payload(row)
        run["results"] = [self._rag_evaluation_result_payload(item) for item in result_rows]
        return run

    def record_rag_answer_feedback(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        now = utc_now()
        feedback_id = str(payload.get("feedback_id") or f"ragfb_{uuid.uuid4().hex}")
        with self.database.transaction(write=True) as connection:
            connection.execute(
                """
                INSERT INTO rag_answer_feedback(
                    feedback_id,project_id,answer_id,user_id,question,helpful,reason,note,
                    citations_json,knowledge_task_id,created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(project_id,answer_id,user_id) DO UPDATE SET
                    question=excluded.question,helpful=excluded.helpful,reason=excluded.reason,
                    note=excluded.note,citations_json=excluded.citations_json,updated_at=excluded.updated_at
                """,
                (
                    feedback_id, self.project_id, str(payload.get("answer_id") or ""),
                    str(payload.get("user_id") or "user_system"), str(payload.get("question") or "")[:2000],
                    1 if payload.get("helpful") else 0, str(payload.get("reason") or ""),
                    str(payload.get("note") or "")[:1000], _json_dump(payload.get("citations") or []),
                    str(payload.get("knowledge_task_id") or ""), now, now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM rag_answer_feedback WHERE project_id=? AND answer_id=? AND user_id=?",
                (self.project_id, str(payload.get("answer_id") or ""), str(payload.get("user_id") or "user_system")),
            ).fetchone()
        result = dict(row)
        result["helpful"] = bool(result.get("helpful"))
        result["citations"] = _json_load(result.pop("citations_json", "[]"), [])
        return result

    def link_rag_feedback_task(self, feedback_id: str, task_id: str) -> None:
        with self.database.transaction(write=True) as connection:
            connection.execute(
                "UPDATE rag_answer_feedback SET knowledge_task_id=?,updated_at=? WHERE feedback_id=? AND project_id=?",
                (task_id, utc_now(), feedback_id, self.project_id),
            )

    # Persistent processing jobs ------------------------------------
    @staticmethod
    def _processing_job_payload(row: Any) -> Dict[str, Any]:
        payload = dict(row)
        payload["payload"] = _json_load(payload.pop("payload_json", "{}"), {})
        payload["result"] = _json_load(payload.pop("result_json", "{}"), {})
        payload["cancel_requested"] = bool(payload.get("cancel_requested"))
        return payload

    @staticmethod
    def _future_time(seconds: int) -> str:
        return (datetime.now(timezone.utc) + timedelta(seconds=max(0, seconds))).isoformat(timespec="seconds")

    def _append_processing_event(
        self, connection: sqlite3.Connection, job_id: str, *, event_type: str,
        from_status: str = "", to_status: str = "", stage: str = "",
        progress: int = 0, message: str = "", data: Optional[Dict[str, Any]] = None,
    ) -> None:
        connection.execute(
            """
            INSERT INTO processing_job_events(
                event_id,job_id,project_id,event_type,from_status,to_status,stage,
                progress,message,data_json,created_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                f"pje_{uuid.uuid4().hex}", job_id, self.project_id, event_type,
                from_status, to_status, stage, max(0, min(int(progress), 100)),
                str(message or "")[:1000], _json_dump(data or {}), utc_now(),
            ),
        )

    def create_or_get_processing_job(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        now = utc_now()
        with self.database.transaction(write=True) as connection:
            return self._create_or_get_processing_job_connection(connection, payload, now=now)

    def _create_or_get_processing_job_connection(
        self, connection: sqlite3.Connection, payload: Dict[str, Any], *, now: str = "",
    ) -> Dict[str, Any]:
        """Create or reuse a processing job inside the caller's transaction."""
        now = now or utc_now()
        idempotency_key = str(payload.get("idempotency_key") or "").strip()
        if idempotency_key:
            existing = connection.execute(
                """
                SELECT * FROM processing_jobs
                WHERE project_id=? AND idempotency_key=?
                  AND status IN ('queued','running','retry_wait')
                """,
                (self.project_id, idempotency_key),
            ).fetchone()
            if existing:
                result = self._processing_job_payload(existing)
                result["created"] = False
                return result
        job_id = str(payload.get("job_id") or f"job_{uuid.uuid4().hex}")
        connection.execute(
            """
            INSERT INTO processing_jobs(
                job_id,project_id,job_type,idempotency_key,status,stage,progress,priority,
                payload_json,max_attempts,available_at,created_by,linked_asset_id,source_id,
                created_at,updated_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                job_id, self.project_id, str(payload.get("job_type") or "").strip(),
                idempotency_key, "queued", "queued", 0,
                max(0, min(int(payload.get("priority") or 50), 100)),
                _json_dump(payload.get("payload") or {}),
                max(1, min(int(payload.get("max_attempts") or 3), 20)),
                str(payload.get("available_at") or now),
                str(payload.get("created_by") or "user_system"),
                str(payload.get("linked_asset_id") or ""), str(payload.get("source_id") or ""),
                now, now,
            ),
        )
        self._append_processing_event(
            connection, job_id, event_type="created", to_status="queued",
            stage="queued", message="任务已进入处理队列",
        )
        row = connection.execute("SELECT * FROM processing_jobs WHERE job_id=?", (job_id,)).fetchone()
        result = self._processing_job_payload(row)
        result["created"] = True
        return result

    def list_processing_jobs(
        self, *, viewer_user_id: str = "", can_manage: bool = False, status: str = "",
        job_type: str = "", keyword: str = "", page: int = 1, page_size: int = 50,
    ) -> Dict[str, Any]:
        where = ["project_id=?"]
        params: List[Any] = [self.project_id]
        if not can_manage:
            where.append("created_by=?")
            params.append(viewer_user_id)
        if status == "active":
            where.append("status IN ('queued','running','retry_wait')")
        elif status:
            where.append("status=?")
            params.append(status)
        if job_type:
            where.append("job_type=?")
            params.append(job_type)
        if keyword:
            literal = f"%{_like_literal(keyword.lower())}%"
            where.append(
                "(lower(job_id) LIKE ? ESCAPE '\\' OR lower(job_type) LIKE ? ESCAPE '\\' "
                "OR lower(source_id) LIKE ? ESCAPE '\\' OR lower(payload_json) LIKE ? ESCAPE '\\' "
                "OR lower(result_json) LIKE ? ESCAPE '\\')"
            )
            params.extend((literal, literal, literal, literal, literal))
        size = max(1, min(int(page_size), 100))
        current_page = max(1, int(page))
        clause = " AND ".join(where)
        with self.database.transaction() as connection:
            total = int(connection.execute(
                f"SELECT COUNT(*) AS count FROM processing_jobs WHERE {clause}", tuple(params)
            ).fetchone()["count"])
            rows = connection.execute(
                f"""
                SELECT * FROM processing_jobs WHERE {clause}
                ORDER BY CASE status WHEN 'running' THEN 0 WHEN 'queued' THEN 1
                    WHEN 'retry_wait' THEN 2 WHEN 'failed' THEN 3 ELSE 4 END,
                    priority DESC,updated_at DESC
                LIMIT ? OFFSET ?
                """,
                tuple(params + [size, (current_page - 1) * size]),
            ).fetchall()
        return {
            "items": [self._processing_job_payload(row) for row in rows],
            "total": total, "page": current_page, "page_size": size,
            "pages": max(1, (total + size - 1) // size),
        }

    def summarize_processing_jobs(self, *, viewer_user_id: str = "", can_manage: bool = False) -> Dict[str, int]:
        where = "project_id=?"
        params: List[Any] = [self.project_id]
        if not can_manage:
            where += " AND created_by=?"
            params.append(viewer_user_id)
        with self.database.transaction() as connection:
            rows = connection.execute(
                f"SELECT status,COUNT(*) AS count FROM processing_jobs WHERE {where} GROUP BY status",
                tuple(params),
            ).fetchall()
        result = {key: 0 for key in ("queued", "running", "retry_wait", "succeeded", "failed", "cancelled")}
        result.update({str(row["status"]): int(row["count"]) for row in rows})
        result["active"] = result["queued"] + result["running"] + result["retry_wait"]
        return result

    def get_processing_job(
        self, job_id: str, *, viewer_user_id: str = "", can_manage: bool = False,
    ) -> Optional[Dict[str, Any]]:
        where = "job_id=? AND project_id=?"
        params: List[Any] = [job_id, self.project_id]
        if not can_manage:
            where += " AND created_by=?"
            params.append(viewer_user_id)
        with self.database.transaction() as connection:
            row = connection.execute(
                f"SELECT * FROM processing_jobs WHERE {where}", tuple(params)
            ).fetchone()
            if not row:
                return None
            result = self._processing_job_payload(row)
            result["attempt_history"] = [dict(item) for item in connection.execute(
                "SELECT * FROM processing_job_attempts WHERE job_id=? ORDER BY attempt_number DESC",
                (job_id,),
            ).fetchall()]
            result["events"] = []
            for event in connection.execute(
                "SELECT * FROM processing_job_events WHERE job_id=? ORDER BY created_at,rowid",
                (job_id,),
            ).fetchall():
                item = dict(event)
                item["data"] = _json_load(item.pop("data_json", "{}"), {})
                result["events"].append(item)
            return result

    def recover_processing_jobs(self) -> int:
        now = utc_now()
        with self.database.transaction(write=True) as connection:
            rows = connection.execute(
                "SELECT job_id,status,stage,progress,attempts,max_attempts FROM processing_jobs WHERE project_id=? AND status='running'",
                (self.project_id,),
            ).fetchall()
            for row in rows:
                job_id = str(row["job_id"])
                can_retry = int(row["attempts"]) < int(row["max_attempts"])
                next_status = "retry_wait" if can_retry else "failed"
                connection.execute(
                    """
                    UPDATE processing_jobs SET status=?,stage=?,available_at=?,locked_at='',locked_by='',
                        heartbeat_at='',error_code='WORKER_INTERRUPTED',last_error=?,updated_at=? WHERE job_id=?
                    """,
                    (next_status, "retry_wait" if can_retry else "failed", now,
                     "服务重启导致上一次执行中断", now, job_id),
                )
                connection.execute(
                    """
                    UPDATE processing_job_attempts SET status='interrupted',error_code='WORKER_INTERRUPTED',
                        error_message='服务重启导致执行中断',completed_at=?
                    WHERE job_id=? AND status='running'
                    """,
                    (now, job_id),
                )
                self._append_processing_event(
                    connection, job_id, event_type="recovered", from_status="running",
                    to_status=next_status, stage=str(row["stage"]), progress=int(row["progress"]),
                    message="服务重启后恢复任务",
                )
        return len(rows)

    def claim_next_processing_job(self, worker_id: str) -> Optional[Dict[str, Any]]:
        now = utc_now()
        with self.database.transaction(write=True) as connection:
            row = connection.execute(
                """
                SELECT * FROM processing_jobs
                WHERE project_id=? AND status IN ('queued','retry_wait') AND available_at<=?
                ORDER BY priority DESC,created_at LIMIT 1
                """,
                (self.project_id, now),
            ).fetchone()
            if not row:
                return None
            job_id = str(row["job_id"])
            attempt = int(row["attempts"]) + 1
            connection.execute(
                """
                UPDATE processing_jobs SET status='running',stage='starting',attempts=?,locked_at=?,
                    locked_by=?,heartbeat_at=?,started_at=CASE WHEN started_at='' THEN ? ELSE started_at END,
                    updated_at=? WHERE job_id=?
                """,
                (attempt, now, worker_id, now, now, now, job_id),
            )
            connection.execute(
                """
                INSERT INTO processing_job_attempts(
                    attempt_id,job_id,project_id,attempt_number,worker_id,status,started_at
                ) VALUES(?,?,?,?,?,'running',?)
                """,
                (f"pja_{uuid.uuid4().hex}", job_id, self.project_id, attempt, worker_id, now),
            )
            self._append_processing_event(
                connection, job_id, event_type="started", from_status=str(row["status"]),
                to_status="running", stage="starting", progress=int(row["progress"]),
                message=f"开始第 {attempt} 次执行",
            )
            claimed = connection.execute("SELECT * FROM processing_jobs WHERE job_id=?", (job_id,)).fetchone()
        return self._processing_job_payload(claimed)

    def update_processing_job_progress(self, job_id: str, *, stage: str, progress: int, message: str = "") -> None:
        now = utc_now()
        value = max(0, min(int(progress), 99))
        with self.database.transaction(write=True) as connection:
            row = connection.execute(
                "SELECT status,stage,progress FROM processing_jobs WHERE job_id=? AND project_id=?",
                (job_id, self.project_id),
            ).fetchone()
            if not row or row["status"] != "running":
                return
            connection.execute(
                "UPDATE processing_jobs SET stage=?,progress=?,heartbeat_at=?,updated_at=? WHERE job_id=?",
                (stage, value, now, now, job_id),
            )
            if stage != row["stage"] or message:
                self._append_processing_event(
                    connection, job_id, event_type="progress", from_status="running", to_status="running",
                    stage=stage, progress=value, message=message,
                )

    def processing_job_cancel_requested(self, job_id: str) -> bool:
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT cancel_requested,status FROM processing_jobs WHERE job_id=? AND project_id=?",
                (job_id, self.project_id),
            ).fetchone()
        return bool(row and (row["cancel_requested"] or row["status"] == "cancelled"))

    def complete_processing_job(self, job_id: str, result: Dict[str, Any]) -> None:
        now = utc_now()
        with self.database.transaction(write=True) as connection:
            row = connection.execute("SELECT status,attempts FROM processing_jobs WHERE job_id=?", (job_id,)).fetchone()
            if not row or row["status"] != "running":
                return
            connection.execute(
                """
                UPDATE processing_jobs SET status='succeeded',stage='completed',progress=100,
                    result_json=?,error_code='',last_error='',locked_at='',locked_by='',heartbeat_at='',
                    completed_at=?,updated_at=? WHERE job_id=?
                """,
                (_json_dump(result or {}), now, now, job_id),
            )
            connection.execute(
                "UPDATE processing_job_attempts SET status='succeeded',completed_at=? WHERE job_id=? AND attempt_number=?",
                (now, job_id, int(row["attempts"])),
            )
            self._append_processing_event(
                connection, job_id, event_type="succeeded", from_status="running",
                to_status="succeeded", stage="completed", progress=100, message="任务处理完成",
            )

    def fail_processing_job(
        self, job_id: str, *, error_code: str, error_message: str,
        retryable: bool = True, retry_delay_seconds: int = 5,
    ) -> str:
        now = utc_now()
        sanitized = str(error_message or "处理失败")[:1000]
        with self.database.transaction(write=True) as connection:
            row = connection.execute(
                "SELECT status,attempts,max_attempts,stage,progress FROM processing_jobs WHERE job_id=?",
                (job_id,),
            ).fetchone()
            if not row or row["status"] != "running":
                return str(row["status"] if row else "missing")
            next_status = "retry_wait" if retryable and int(row["attempts"]) < int(row["max_attempts"]) else "failed"
            available = self._future_time(retry_delay_seconds) if next_status == "retry_wait" else now
            connection.execute(
                """
                UPDATE processing_jobs SET status=?,stage=?,available_at=?,error_code=?,last_error=?,
                    locked_at='',locked_by='',heartbeat_at='',completed_at=?,updated_at=? WHERE job_id=?
                """,
                (next_status, "retry_wait" if next_status == "retry_wait" else "failed", available,
                 str(error_code or "PROCESSING_FAILED")[:80], sanitized,
                 now if next_status == "failed" else "", now, job_id),
            )
            connection.execute(
                """
                UPDATE processing_job_attempts SET status=?,error_code=?,error_message=?,completed_at=?
                WHERE job_id=? AND attempt_number=?
                """,
                (next_status, str(error_code or "PROCESSING_FAILED")[:80], sanitized, now,
                 job_id, int(row["attempts"])),
            )
            self._append_processing_event(
                connection, job_id, event_type=next_status, from_status="running", to_status=next_status,
                stage=str(row["stage"]), progress=int(row["progress"]), message=sanitized,
                data={"available_at": available},
            )
        return next_status

    def cancel_processing_job(self, job_id: str, *, actor_user_id: str) -> Dict[str, Any]:
        now = utc_now()
        with self.database.transaction(write=True) as connection:
            row = connection.execute(
                "SELECT * FROM processing_jobs WHERE job_id=? AND project_id=?", (job_id, self.project_id)
            ).fetchone()
            if not row:
                raise ValueError("处理任务不存在")
            status = str(row["status"])
            if status in {"succeeded", "failed", "cancelled"}:
                raise ValueError("当前状态不能取消")
            if status == "running":
                connection.execute(
                    "UPDATE processing_jobs SET cancel_requested=1,updated_at=? WHERE job_id=?", (now, job_id)
                )
                self._append_processing_event(
                    connection, job_id, event_type="cancel_requested", from_status="running",
                    to_status="running", stage=str(row["stage"]), progress=int(row["progress"]),
                    message="已请求停止正在运行的任务", data={"actor_user_id": actor_user_id},
                )
            else:
                connection.execute(
                    """
                    UPDATE processing_jobs SET status='cancelled',stage='cancelled',cancel_requested=1,
                        completed_at=?,updated_at=? WHERE job_id=?
                    """,
                    (now, now, job_id),
                )
                self._append_processing_event(
                    connection, job_id, event_type="cancelled", from_status=status,
                    to_status="cancelled", stage="cancelled", progress=int(row["progress"]),
                    message="任务已取消", data={"actor_user_id": actor_user_id},
                )
        return self.get_processing_job(job_id, can_manage=True) or {}

    def mark_processing_job_cancelled(self, job_id: str, result: Optional[Dict[str, Any]] = None) -> None:
        now = utc_now()
        with self.database.transaction(write=True) as connection:
            row = connection.execute("SELECT status,attempts,stage,progress FROM processing_jobs WHERE job_id=?", (job_id,)).fetchone()
            if not row or row["status"] != "running":
                return
            connection.execute(
                """
                UPDATE processing_jobs SET status='cancelled',stage='cancelled',result_json=?,
                    locked_at='',locked_by='',heartbeat_at='',completed_at=?,updated_at=? WHERE job_id=?
                """,
                (_json_dump(result or {}), now, now, job_id),
            )
            connection.execute(
                "UPDATE processing_job_attempts SET status='cancelled',completed_at=? WHERE job_id=? AND attempt_number=?",
                (now, job_id, int(row["attempts"])),
            )
            self._append_processing_event(
                connection, job_id, event_type="cancelled", from_status="running",
                to_status="cancelled", stage="cancelled", progress=int(row["progress"]),
                message="任务已按取消请求停止",
            )

    def retry_processing_job(self, job_id: str, *, actor_user_id: str) -> Dict[str, Any]:
        now = utc_now()
        with self.database.transaction(write=True) as connection:
            row = connection.execute(
                "SELECT * FROM processing_jobs WHERE job_id=? AND project_id=?", (job_id, self.project_id)
            ).fetchone()
            if not row:
                raise ValueError("处理任务不存在")
            if row["status"] not in {"failed", "cancelled"}:
                raise ValueError("只有失败或已取消任务可以重试")
            max_attempts = min(20, max(int(row["max_attempts"]), int(row["attempts"]) + 3))
            connection.execute(
                """
                UPDATE processing_jobs SET status='queued',stage='queued',progress=0,max_attempts=?,
                    available_at=?,cancel_requested=0,error_code='',last_error='',completed_at='',updated_at=?
                WHERE job_id=?
                """,
                (max_attempts, now, now, job_id),
            )
            self._append_processing_event(
                connection, job_id, event_type="manual_retry", from_status=str(row["status"]),
                to_status="queued", stage="queued", progress=0, message="已人工重新加入队列",
                data={"actor_user_id": actor_user_id},
            )
        return self.get_processing_job(job_id, can_manage=True) or {}

    # Project data governance ---------------------------------------
    @staticmethod
    def _governance_policy_payload(row: Any) -> Dict[str, Any]:
        payload = dict(row) if row else {}
        for key in ("daily_backup_enabled", "daily_retention_enabled"):
            if key in payload:
                payload[key] = bool(payload[key])
        return payload

    def get_data_governance_policy(self) -> Dict[str, Any]:
        now = utc_now()
        with self.database.transaction(write=True) as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO data_governance_policies(
                    project_id,raw_message_retention_days,updated_by,created_at,updated_at
                ) VALUES(?, 90, 'user_system', ?, ?)
                """,
                (self.project_id, now, now),
            )
            row = connection.execute(
                "SELECT * FROM data_governance_policies WHERE project_id=?", (self.project_id,)
            ).fetchone()
        return self._governance_policy_payload(row)

    def update_data_governance_policy(self, changes: Dict[str, Any], *, actor_user_id: str) -> Dict[str, Any]:
        current = self.get_data_governance_policy()
        ranges = {
            "raw_message_retention_days": (7, 3650),
            "access_audit_retention_days": (30, 3650),
            "processing_history_retention_days": (7, 3650),
            "backup_retention_count": (1, 100),
        }
        updates: Dict[str, Any] = {}
        for key, (minimum, maximum) in ranges.items():
            if key not in changes:
                continue
            value = int(changes[key])
            if value < minimum or value > maximum:
                raise ValueError(f"{key} 必须在 {minimum} 到 {maximum} 之间")
            updates[key] = value
        if "default_sensitivity" in changes:
            value = str(changes.get("default_sensitivity") or "").strip()
            if value not in {"public", "internal", "confidential", "restricted"}:
                raise ValueError("不支持的默认敏感级别")
            updates["default_sensitivity"] = value
        for key in ("daily_backup_enabled", "daily_retention_enabled"):
            if key in changes:
                value = changes.get(key)
                if not isinstance(value, bool) and not (isinstance(value, int) and value in (0, 1)):
                    raise ValueError(f"{key} 必须是布尔值")
                updates[key] = int(bool(value))
        if "schedule_time" in changes:
            value = str(changes.get("schedule_time") or "").strip()
            try:
                parsed = datetime.strptime(value, "%H:%M")
            except ValueError as exc:
                raise ValueError("schedule_time 必须使用 HH:MM 格式") from exc
            updates["schedule_time"] = parsed.strftime("%H:%M")
        if "timezone" in changes:
            value = str(changes.get("timezone") or "").strip()
            try:
                ZoneInfo(value)
            except (ValueError, ZoneInfoNotFoundError) as exc:
                raise ValueError("timezone 必须是有效的 IANA 时区") from exc
            updates["timezone"] = value
        if not updates:
            return current
        now = utc_now()
        assignments = ",".join(f"{key}=?" for key in updates)
        with self.database.transaction(write=True) as connection:
            connection.execute(
                f"UPDATE data_governance_policies SET {assignments},updated_by=?,updated_at=? WHERE project_id=?",
                tuple(updates.values()) + (actor_user_id, now, self.project_id),
            )
            self._write_audit_connection(
                connection, f"audit_{uuid.uuid4().hex}", actor_user_id,
                "governance.policy_updated", "data_governance_policy", self.project_id,
                {"changes": updates}, project_id=self.project_id, created_at=now,
            )
            row = connection.execute(
                "SELECT * FROM data_governance_policies WHERE project_id=?", (self.project_id,)
            ).fetchone()
        return self._governance_policy_payload(row)

    def schedule_daily_governance_jobs(
        self, *, local_date: str, scheduled_for: str, operations: Sequence[str],
    ) -> Dict[str, Any]:
        try:
            datetime.strptime(str(local_date), "%Y-%m-%d")
            datetime.fromisoformat(str(scheduled_for).replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("治理调度日期或时间格式不正确") from exc
        normalized = [
            item for item in dict.fromkeys(str(value or "").strip() for value in operations)
            if item in {"database_backup", "data_retention"}
        ]
        jobs: List[Dict[str, Any]] = []
        existing_operations: List[str] = []
        now = utc_now()
        with self.database.transaction(write=True) as connection:
            for operation in normalized:
                existing = connection.execute(
                    """
                    SELECT gsi.*,pj.status AS processing_status
                    FROM governance_schedule_intents gsi
                    JOIN processing_jobs pj ON pj.job_id=gsi.processing_job_id
                    WHERE gsi.project_id=? AND gsi.operation=? AND gsi.local_date=?
                    """,
                    (self.project_id, operation, local_date),
                ).fetchone()
                if existing:
                    existing_operations.append(operation)
                    continue
                intent_id = _stable_id("governance_schedule", self.project_id, operation, local_date)
                payload = {
                    "actor_user_id": "user_system", "trigger": "daily_governance_schedule",
                    "schedule_local_date": local_date, "scheduled_for": scheduled_for,
                }
                if operation == "data_retention":
                    payload["confirmation"] = "执行数据清理"
                job = self._create_or_get_processing_job_connection(
                    connection,
                    {
                        "job_type": operation, "payload": payload, "created_by": "user_system",
                        "idempotency_key": f"daily-governance:{self.project_id}:{operation}:{local_date}",
                        "max_attempts": 5, "priority": 65 if operation == "database_backup" else 55,
                    },
                    now=now,
                )
                connection.execute(
                    """
                    INSERT INTO governance_schedule_intents(
                        intent_id,project_id,operation,local_date,scheduled_for,processing_job_id,created_at
                    ) VALUES(?,?,?,?,?,?,?)
                    """,
                    (intent_id, self.project_id, operation, local_date, scheduled_for, job["job_id"], now),
                )
                jobs.append(job)
            if jobs:
                self._write_audit_connection(
                    connection, f"audit_{uuid.uuid4().hex}", "user_system",
                    "governance.daily_jobs_scheduled", "governance_schedule", local_date,
                    {"operations": [job["job_type"] for job in jobs], "scheduled_for": scheduled_for},
                    project_id=self.project_id, created_at=now,
                )
        return {
            "local_date": local_date, "scheduled_for": scheduled_for,
            "jobs": jobs, "job_count": len(jobs), "existing_operations": existing_operations,
        }

    def list_governance_schedule_intents(self, *, limit: int = 30) -> List[Dict[str, Any]]:
        with self.database.transaction() as connection:
            return [dict(row) for row in connection.execute(
                """
                SELECT gsi.*,pj.job_type,pj.status AS processing_status,pj.stage AS processing_stage,
                       pj.progress AS processing_progress,pj.attempts AS processing_attempts,
                       pj.last_error AS processing_error,pj.completed_at,pj.updated_at
                FROM governance_schedule_intents gsi
                JOIN processing_jobs pj ON pj.job_id=gsi.processing_job_id
                WHERE gsi.project_id=?
                ORDER BY gsi.local_date DESC,gsi.operation,gsi.created_at DESC
                LIMIT ?
                """,
                (self.project_id, max(1, min(int(limit), 200))),
            )]

    def list_audit_events(
        self, *, action: str = "", actor: str = "", object_type: str = "",
        keyword: str = "", page: int = 1, page_size: int = 50,
    ) -> Dict[str, Any]:
        where = ["project_id=?"]
        params: List[Any] = [self.project_id]
        for column, value in (("action", action), ("actor", actor), ("object_type", object_type)):
            if value:
                where.append(f"{column}=?")
                params.append(str(value))
        if keyword:
            literal = f"%{_like_literal(keyword.lower())}%"
            where.append(
                "(lower(action) LIKE ? ESCAPE '\\' OR lower(actor) LIKE ? ESCAPE '\\' "
                "OR lower(object_type) LIKE ? ESCAPE '\\' OR lower(object_id) LIKE ? ESCAPE '\\')"
            )
            params.extend((literal, literal, literal, literal))
        size = max(1, min(int(page_size), 5000))
        current_page = max(1, int(page))
        clause = " AND ".join(where)
        with self.database.transaction() as connection:
            total = int(connection.execute(
                f"SELECT COUNT(*) FROM audit_events WHERE {clause}", tuple(params)
            ).fetchone()[0])
            rows = connection.execute(
                f"SELECT * FROM audit_events WHERE {clause} ORDER BY created_at DESC,rowid DESC LIMIT ? OFFSET ?",
                tuple(params + [size, (current_page - 1) * size]),
            ).fetchall()
        items = []
        for row in rows:
            item = dict(row)
            item["detail"] = _redact_audit_value(_json_load(item.pop("data_json", "{}"), {}))
            items.append(item)
        return {
            "items": items, "total": total, "page": current_page, "page_size": size,
            "pages": max(1, (total + size - 1) // size),
        }

    @staticmethod
    def _retention_cutoff(days: int) -> str:
        return (datetime.now(timezone.utc) - timedelta(days=int(days))).isoformat(timespec="seconds")

    def preview_data_retention(self) -> Dict[str, Any]:
        policy = self.get_data_governance_policy()
        raw_cutoff = self._retention_cutoff(policy["raw_message_retention_days"])
        audit_cutoff = self._retention_cutoff(policy["access_audit_retention_days"])
        job_cutoff = self._retention_cutoff(policy["processing_history_retention_days"])
        with self.database.transaction() as connection:
            raw_messages = int(connection.execute(
                """
                SELECT COUNT(*) FROM feishu_messages fm
                WHERE fm.project_id=? AND COALESCE(NULLIF(fm.create_time,''),fm.updated_at)<?
                  AND json_extract(fm.data_json,'$.retention_redacted') IS NOT 1
                  AND NOT EXISTS(SELECT 1 FROM feishu_candidate_messages fcm WHERE fcm.message_id=fm.message_id)
                  AND NOT EXISTS(SELECT 1 FROM feishu_asset_messages fam WHERE fam.message_id=fm.message_id)
                """,
                (self.project_id, raw_cutoff),
            ).fetchone()[0])
            access_audits = int(connection.execute(
                "SELECT COUNT(*) FROM audit_events WHERE project_id=? AND action LIKE 'access.%' AND created_at<?",
                (self.project_id, audit_cutoff),
            ).fetchone()[0])
            terminal_jobs = int(connection.execute(
                """
                SELECT COUNT(*) FROM processing_jobs
                WHERE project_id=? AND status IN ('succeeded','cancelled')
                  AND completed_at<>'' AND completed_at<?
                """,
                (self.project_id, job_cutoff),
            ).fetchone()[0])
            retained_assets = int(connection.execute(
                "SELECT COUNT(*) FROM knowledge_assets WHERE project_id=? AND status IN ('expired','revoked','deleted')",
                (self.project_id,),
            ).fetchone()[0])
        return {
            "policy": policy,
            "raw_message_redaction_count": raw_messages,
            "access_audit_expiry_count": access_audits,
            "processing_job_expiry_count": terminal_jobs,
            "retained_asset_history_count": retained_assets,
            "total_actions": raw_messages + access_audits + terminal_jobs,
        }

    def apply_data_retention(self, *, actor_user_id: str) -> Dict[str, Any]:
        preview = self.preview_data_retention()
        policy = preview["policy"]
        raw_cutoff = self._retention_cutoff(policy["raw_message_retention_days"])
        audit_cutoff = self._retention_cutoff(policy["access_audit_retention_days"])
        job_cutoff = self._retention_cutoff(policy["processing_history_retention_days"])
        now = utc_now()
        run_id = f"retention_{uuid.uuid4().hex}"
        with self.database.transaction(write=True) as connection:
            message_rows = connection.execute(
                """
                SELECT fm.message_id,fm.chat_id,fm.message_type,fm.create_time
                FROM feishu_messages fm
                WHERE fm.project_id=? AND COALESCE(NULLIF(fm.create_time,''),fm.updated_at)<?
                  AND json_extract(fm.data_json,'$.retention_redacted') IS NOT 1
                  AND NOT EXISTS(SELECT 1 FROM feishu_candidate_messages fcm WHERE fcm.message_id=fm.message_id)
                  AND NOT EXISTS(SELECT 1 FROM feishu_asset_messages fam WHERE fam.message_id=fm.message_id)
                """,
                (self.project_id, raw_cutoff),
            ).fetchall()
            for row in message_rows:
                connection.execute(
                    "UPDATE feishu_messages SET content_hash='',data_json=?,updated_at=? WHERE message_id=?",
                    (_json_dump({
                        "retention_redacted": True, "message_id": row["message_id"],
                        "chat_id": row["chat_id"], "message_type": row["message_type"],
                        "create_time": row["create_time"], "redacted_at": now,
                    }), now, row["message_id"]),
                )
            access_deleted = connection.execute(
                "DELETE FROM audit_events WHERE project_id=? AND action LIKE 'access.%' AND created_at<?",
                (self.project_id, audit_cutoff),
            ).rowcount
            jobs_deleted = connection.execute(
                """
                DELETE FROM processing_jobs WHERE project_id=? AND status IN ('succeeded','cancelled')
                  AND completed_at<>'' AND completed_at<?
                """,
                (self.project_id, job_cutoff),
            ).rowcount
            result = {
                "raw_messages_redacted": len(message_rows),
                "access_audits_deleted": max(0, int(access_deleted)),
                "processing_jobs_deleted": max(0, int(jobs_deleted)),
                "asset_histories_retained": int(preview["retained_asset_history_count"]),
            }
            connection.execute(
                """
                INSERT INTO data_retention_runs(run_id,project_id,status,actor_user_id,policy_json,result_json,created_at)
                VALUES(?,?, 'succeeded', ?,?,?,?)
                """,
                (run_id, self.project_id, actor_user_id, _json_dump(policy), _json_dump(result), now),
            )
            self._write_audit_connection(
                connection, f"audit_{uuid.uuid4().hex}", actor_user_id,
                "governance.retention_applied", "data_retention_run", run_id,
                result, project_id=self.project_id, created_at=now,
            )
        return {"run_id": run_id, **result}

    @staticmethod
    def _validate_governance_target(target_type: str, target_id: str) -> tuple[str, str]:
        normalized_type = str(target_type or "").strip().lower()
        normalized_id = str(target_id or "").strip()
        if normalized_type not in {"source", "user", "asset"}:
            raise ValueError("定向治理类型必须是 source、user 或 asset")
        if not normalized_id or len(normalized_id) > 255:
            raise ValueError("定向治理对象标识不能为空且不能超过 255 个字符")
        return normalized_type, normalized_id

    def _targeted_governance_scope(
        self, connection: sqlite3.Connection, target_type: str, target_id: str,
    ) -> Dict[str, Any]:
        target_type, target_id = self._validate_governance_target(target_type, target_id)
        target: Dict[str, Any]
        if target_type == "source":
            row = connection.execute(
                "SELECT * FROM knowledge_sources WHERE project_id=? AND source_id=?",
                (self.project_id, target_id),
            ).fetchone()
            if not row:
                raise ValueError("知识来源不存在")
            target = dict(row)
            source_rows = [row]
            asset_rows = connection.execute(
                "SELECT * FROM knowledge_assets WHERE project_id=? AND primary_source_id=? ORDER BY asset_id",
                (self.project_id, target_id),
            ).fetchall()
        elif target_type == "asset":
            row = connection.execute(
                "SELECT * FROM knowledge_assets WHERE project_id=? AND asset_id=?",
                (self.project_id, target_id),
            ).fetchone()
            if not row:
                alias = connection.execute(
                    "SELECT asset_id FROM knowledge_asset_aliases WHERE project_id=? AND alias_value=? LIMIT 1",
                    (self.project_id, target_id),
                ).fetchone()
                row = connection.execute(
                    "SELECT * FROM knowledge_assets WHERE project_id=? AND asset_id=?",
                    (self.project_id, str(alias["asset_id"]) if alias else ""),
                ).fetchone()
            if not row:
                raise ValueError("知识资产不存在")
            target_id = str(row["asset_id"])
            target = dict(row)
            asset_rows = [row]
            source_rows = connection.execute(
                "SELECT * FROM knowledge_sources WHERE project_id=? AND source_id=?",
                (self.project_id, row["primary_source_id"]),
            ).fetchall()
        else:
            row = connection.execute(
                """
                SELECT u.*,pm.role AS project_role,pm.status AS membership_status
                FROM users u JOIN project_memberships pm ON pm.user_id=u.user_id
                WHERE pm.project_id=? AND u.user_id=?
                """,
                (self.project_id, target_id),
            ).fetchone()
            if not row:
                raise ValueError("项目成员不存在")
            target = dict(row)
            source_rows = connection.execute(
                "SELECT * FROM knowledge_sources WHERE project_id=? AND owner_user_id=? ORDER BY source_id",
                (self.project_id, target_id),
            ).fetchall()
            asset_rows = connection.execute(
                """
                SELECT DISTINCT ka.* FROM knowledge_assets ka
                LEFT JOIN knowledge_sources ks ON ks.source_id=ka.primary_source_id
                WHERE ka.project_id=? AND (ka.owner_user_id=? OR ks.owner_user_id=?)
                ORDER BY ka.asset_id
                """,
                (self.project_id, target_id, target_id),
            ).fetchall()

        source_ids = {str(row["source_id"]) for row in source_rows}
        asset_ids = {str(row["asset_id"]) for row in asset_rows}
        document_where = []
        document_params: List[Any] = [self.project_id]
        if asset_ids:
            placeholders = ",".join("?" for _ in asset_ids)
            document_where.append(f"d.asset_id IN ({placeholders})")
            document_params.extend(sorted(asset_ids))
            document_where.append(
                f"EXISTS(SELECT 1 FROM asset_version_documents avd JOIN asset_versions av "
                f"ON av.version_id=avd.version_id WHERE avd.document_id=d.document_id "
                f"AND av.asset_id IN ({placeholders}))"
            )
            document_params.extend(sorted(asset_ids))
        if target_type in {"source", "user"} and source_ids:
            placeholders = ",".join("?" for _ in source_ids)
            document_where.append(f"d.source_id IN ({placeholders})")
            document_params.extend(sorted(source_ids))
        if target_type == "user":
            document_where.append("d.owner_user_id=?")
            document_params.append(target_id)
        document_rows = []
        if document_where:
            document_rows = connection.execute(
                "SELECT DISTINCT d.* FROM documents d WHERE d.project_id=? AND ("
                + " OR ".join(document_where) + ") ORDER BY d.document_id",
                tuple(document_params),
            ).fetchall()
        document_ids = {str(row["document_id"]) for row in document_rows}

        version_rows = []
        if asset_ids:
            placeholders = ",".join("?" for _ in asset_ids)
            version_rows = connection.execute(
                f"SELECT * FROM asset_versions WHERE project_id=? AND asset_id IN ({placeholders}) "
                "ORDER BY asset_id,version_no",
                (self.project_id, *sorted(asset_ids)),
            ).fetchall()
        version_ids = {str(row["version_id"]) for row in version_rows}

        message_ids: set[str] = set()
        if asset_ids:
            placeholders = ",".join("?" for _ in asset_ids)
            for row in connection.execute(
                f"""
                SELECT DISTINCT fam.message_id FROM feishu_asset_messages fam
                JOIN feishu_assets fa ON fa.asset_id=fam.asset_id AND fa.project_id=?
                LEFT JOIN knowledge_asset_aliases kaa ON kaa.project_id=fa.project_id
                    AND kaa.alias_type='feishu_asset' AND kaa.alias_value=fa.asset_id
                WHERE kaa.asset_id IN ({placeholders}) OR fa.asset_id IN ({placeholders})
                """,
                (self.project_id, *sorted(asset_ids), *sorted(asset_ids)),
            ):
                message_ids.add(str(row["message_id"]))
        candidate_ids = {
            str(row["external_key"])
            for row in source_rows
            if str(row["source_type"]) == "feishu_conversation" and row["external_key"]
        }
        if candidate_ids:
            placeholders = ",".join("?" for _ in candidate_ids)
            for row in connection.execute(
                f"SELECT message_id FROM feishu_candidate_messages WHERE candidate_id IN ({placeholders})",
                tuple(sorted(candidate_ids)),
            ):
                message_ids.add(str(row["message_id"]))
        if target_type == "user":
            for row in connection.execute(
                """
                SELECT message_id FROM feishu_messages
                WHERE project_id=? AND (
                    json_extract(data_json,'$.sender')=?
                    OR json_extract(data_json,'$.sender.id')=?
                    OR json_extract(data_json,'$.sender.open_id')=?
                )
                """,
                (self.project_id, target_id, target_id, target_id),
            ):
                message_ids.add(str(row["message_id"]))

        return {
            "target_type": target_type,
            "target_id": target_id,
            "target": target,
            "sources": [dict(row) for row in source_rows],
            "assets": [dict(row) for row in asset_rows],
            "versions": [dict(row) for row in version_rows],
            "documents": [dict(row) for row in document_rows],
            "source_ids": sorted(source_ids),
            "asset_ids": sorted(asset_ids),
            "version_ids": sorted(version_ids),
            "document_ids": sorted(document_ids),
            "message_ids": sorted(message_ids),
        }

    @staticmethod
    def _targeted_preview_payload(scope: Dict[str, Any], chunk_count: int) -> Dict[str, Any]:
        documents = list(scope["documents"])
        active_assets = sum(
            1 for row in scope["assets"] if str(row.get("status") or "") not in {"deleted", "revoked"}
        )
        return {
            "target_type": scope["target_type"],
            "target_id": scope["target_id"],
            "target_label": str(
                scope["target"].get("display_name")
                or scope["target"].get("title")
                or scope["target"].get("user_id")
                or scope["target_id"]
            ),
            "source_count": len(scope["sources"]),
            "asset_count": len(scope["assets"]),
            "active_asset_count": active_assets,
            "version_count": len(scope["versions"]),
            "document_count": len(documents),
            "chunk_count": int(chunk_count),
            "message_count": len(scope["message_ids"]),
            "file_count": len({str(row.get("stored_file") or "") for row in documents if row.get("stored_file")}),
        }

    def preview_targeted_governance(self, target_type: str, target_id: str) -> Dict[str, Any]:
        with self.database.transaction() as connection:
            scope = self._targeted_governance_scope(connection, target_type, target_id)
            chunk_count = 0
            if scope["document_ids"]:
                placeholders = ",".join("?" for _ in scope["document_ids"])
                chunk_count = int(connection.execute(
                    f"SELECT COUNT(*) FROM document_chunks WHERE document_id IN ({placeholders})",
                    tuple(scope["document_ids"]),
                ).fetchone()[0])
        return self._targeted_preview_payload(scope, chunk_count)

    @staticmethod
    def _governance_export_row(row: Dict[str, Any]) -> Dict[str, Any]:
        item = dict(row)
        for key in list(item):
            if key.endswith("_json"):
                item[key[:-5]] = _json_load(item.pop(key), {} if key != "required_asset_ids_json" else [])
        return item

    def export_targeted_governance(
        self, target_type: str, target_id: str, *, actor_user_id: str,
    ) -> Dict[str, Any]:
        with self.database.transaction() as connection:
            scope = self._targeted_governance_scope(connection, target_type, target_id)
            document_ids = scope["document_ids"]
            chunks = []
            if document_ids:
                placeholders = ",".join("?" for _ in document_ids)
                chunks = [dict(row) for row in connection.execute(
                    f"SELECT * FROM document_chunks WHERE document_id IN ({placeholders}) "
                    "ORDER BY document_id,chunk_index,chunk_id",
                    tuple(document_ids),
                )]
            messages = []
            if scope["message_ids"]:
                placeholders = ",".join("?" for _ in scope["message_ids"])
                messages = [dict(row) for row in connection.execute(
                    f"SELECT * FROM feishu_messages WHERE project_id=? AND message_id IN ({placeholders}) "
                    "ORDER BY create_time,message_id",
                    (self.project_id, *scope["message_ids"]),
                )]
            user_records: Dict[str, Any] = {}
            if scope["target_type"] == "user":
                user_records = {
                    "profile": dict(connection.execute(
                        "SELECT * FROM users WHERE user_id=?", (scope["target_id"],)
                    ).fetchone()),
                    "membership": dict(connection.execute(
                        "SELECT * FROM project_memberships WHERE project_id=? AND user_id=?",
                        (self.project_id, scope["target_id"]),
                    ).fetchone()),
                    "external_identities": [dict(row) for row in connection.execute(
                        "SELECT provider,external_user_id,verified_email,created_at,updated_at "
                        "FROM user_external_identities WHERE user_id=? ORDER BY provider",
                        (scope["target_id"],),
                    )],
                    "knowledge_tasks": [dict(row) for row in connection.execute(
                        "SELECT * FROM knowledge_tasks WHERE project_id=? AND (assignee_user_id=? OR reporter_user_id=?) "
                        "ORDER BY created_at,task_id",
                        (self.project_id, scope["target_id"], scope["target_id"]),
                    )],
                    "onboarding_plans": [dict(row) for row in connection.execute(
                        "SELECT * FROM onboarding_learning_plans WHERE project_id=? AND (user_id=? OR manager_user_id=?) "
                        "ORDER BY created_at,plan_id",
                        (self.project_id, scope["target_id"], scope["target_id"]),
                    )],
                }
            payload = {
                "schema_version": 1,
                "exported_at": utc_now(),
                "project_id": self.project_id,
                "target": {"type": scope["target_type"], "id": scope["target_id"]},
                "records": {
                    "sources": [self._governance_export_row(row) for row in scope["sources"]],
                    "assets": [self._governance_export_row(row) for row in scope["assets"]],
                    "versions": [self._governance_export_row(row) for row in scope["versions"]],
                    "documents": [self._governance_export_row(row) for row in scope["documents"]],
                    "chunks": [self._governance_export_row(row) for row in chunks],
                    "feishu_messages": [self._governance_export_row(row) for row in messages],
                    "user": {
                        key: (
                            [self._governance_export_row(item) for item in value]
                            if isinstance(value, list) else self._governance_export_row(value)
                        )
                        for key, value in user_records.items()
                    },
                },
            }
        counts = {
            key: len(value) for key, value in payload["records"].items()
            if isinstance(value, list)
        }
        self.write_security_audit(
            actor_user_id, "governance.target_exported", scope["target_type"], scope["target_id"],
            {"counts": counts}, project_id=self.project_id,
        )
        return payload

    def apply_targeted_governance_deletion(
        self, target_type: str, target_id: str, *, actor_user_id: str,
    ) -> Dict[str, Any]:
        target_type, target_id = self._validate_governance_target(target_type, target_id)
        if target_type == "user" and target_id in {actor_user_id, "user_system"}:
            raise ValueError("不能通过定向治理删除当前操作者或系统用户")
        now = utc_now()
        run_id = f"target_delete_{uuid.uuid4().hex}"
        with self.database.transaction(write=True) as connection:
            scope = self._targeted_governance_scope(connection, target_type, target_id)
            legacy_feishu_rows: List[Dict[str, Any]] = []
            if scope["asset_ids"]:
                placeholders = ",".join("?" for _ in scope["asset_ids"])
                legacy_feishu_rows = [dict(row) for row in connection.execute(
                    f"""
                    SELECT DISTINCT fa.asset_id,fa.candidate_id,fa.status,fa.data_json
                    FROM feishu_assets fa
                    LEFT JOIN knowledge_asset_aliases kaa
                      ON kaa.project_id=fa.project_id AND kaa.alias_type='feishu_asset'
                     AND kaa.alias_value=fa.asset_id
                    WHERE fa.project_id=? AND (
                        fa.asset_id IN ({placeholders}) OR kaa.asset_id IN ({placeholders})
                    )
                    """,
                    (self.project_id, *scope["asset_ids"], *scope["asset_ids"]),
                )]
            legacy_feishu_asset_ids = sorted({
                str(row["asset_id"]) for row in legacy_feishu_rows if row.get("asset_id")
            })
            legacy_candidate_ids = sorted({
                str(row["candidate_id"]) for row in legacy_feishu_rows if row.get("candidate_id")
            })
            pending_run = connection.execute(
                """
                SELECT run_id,result_json FROM targeted_deletion_runs
                WHERE project_id=? AND target_type=? AND target_id=? AND status='succeeded'
                ORDER BY created_at DESC,run_id DESC LIMIT 1
                """,
                (self.project_id, scope["target_type"], scope["target_id"]),
            ).fetchone()
            if pending_run:
                pending_result = _json_load(pending_run["result_json"], {})
                if pending_result.get("file_cleanup_complete") is False:
                    return {
                        "run_id": str(pending_run["run_id"]),
                        "target_type": scope["target_type"], "target_id": scope["target_id"],
                        **pending_result,
                    }
            document_ids = scope["document_ids"]
            chunk_count = 0
            if document_ids:
                placeholders = ",".join("?" for _ in document_ids)
                chunk_count = int(connection.execute(
                    f"SELECT COUNT(*) FROM document_chunks WHERE document_id IN ({placeholders})",
                    tuple(document_ids),
                ).fetchone()[0])
            preview = self._targeted_preview_payload(scope, chunk_count)
            changed_assets = [
                row for row in scope["assets"]
                if str(row.get("status") or "") != "deleted"
                or not bool(_json_load(row.get("metadata_json"), {}).get("targeted_deleted"))
            ]
            changed_versions = [
                row for row in scope["versions"]
                if str(row.get("status") or "") != "revoked" or bool(row.get("content_hash"))
            ]
            changed_documents = [
                row for row in scope["documents"]
                if str(row.get("status") or "") != "deleted"
                or not bool(_json_load(row.get("metadata_json"), {}).get("targeted_deleted"))
            ]
            changed_messages = []
            if scope["message_ids"]:
                placeholders = ",".join("?" for _ in scope["message_ids"])
                changed_messages = [dict(row) for row in connection.execute(
                    f"SELECT message_id,data_json FROM feishu_messages WHERE project_id=? "
                    f"AND message_id IN ({placeholders})",
                    (self.project_id, *scope["message_ids"]),
                ) if not bool(_json_load(row["data_json"], {}).get("targeted_deleted"))]
            tombstone = _json_dump({
                "targeted_deleted": True, "target_type": scope["target_type"],
                "run_id": run_id, "deleted_at": now,
            })
            stored_files = sorted({
                str(row.get("stored_file") or "") for row in changed_documents
                if row.get("stored_file") and not str(row.get("stored_file")).startswith("deleted/")
            })
            if scope["message_ids"]:
                placeholders = ",".join("?" for _ in scope["message_ids"])
                connection.execute(
                    f"UPDATE feishu_messages SET content_hash='',data_json=?,content_status='revoked',updated_at=? "
                    f"WHERE project_id=? AND message_id IN ({placeholders})",
                    (tombstone, now, self.project_id, *scope["message_ids"]),
                )
            legacy_assets_reverted = 0
            if legacy_feishu_asset_ids:
                placeholders = ",".join("?" for _ in legacy_feishu_asset_ids)
                legacy_assets_reverted = int(connection.execute(
                    f"UPDATE feishu_assets SET status='reverted',stored_file='',data_json=?,updated_at=? "
                    f"WHERE project_id=? AND asset_id IN ({placeholders}) "
                    "AND (status NOT IN ('reverted','deleted') "
                    "OR COALESCE(json_extract(data_json,'$.targeted_deleted'),0) IS NOT 1)",
                    (tombstone, now, self.project_id, *legacy_feishu_asset_ids),
                ).rowcount)
            legacy_candidates_reverted = 0
            if legacy_candidate_ids:
                placeholders = ",".join("?" for _ in legacy_candidate_ids)
                legacy_candidates_reverted = int(connection.execute(
                    f"UPDATE feishu_candidates SET status='reverted',data_json=?,updated_at=? "
                    f"WHERE project_id=? AND candidate_id IN ({placeholders}) "
                    "AND (status NOT IN ('reverted','deleted') "
                    "OR COALESCE(json_extract(data_json,'$.targeted_deleted'),0) IS NOT 1)",
                    (tombstone, now, self.project_id, *legacy_candidate_ids),
                ).rowcount)
            if document_ids:
                placeholders = ",".join("?" for _ in document_ids)
                connection.execute(
                    f"DELETE FROM document_chunks WHERE document_id IN ({placeholders})",
                    tuple(document_ids),
                )
                for document_id in document_ids:
                    connection.execute(
                        """
                        UPDATE documents SET source_file='[已删除原件]',stored_file=?,status='deleted',
                            content_hash='',metadata_json=?,owner_user_id='',access_policy_id='',
                            acl_revision=acl_revision+1,updated_at=? WHERE document_id=? AND project_id=?
                        """,
                        (f"deleted/{document_id}", tombstone, now, document_id, self.project_id),
                    )
            if scope["version_ids"]:
                placeholders = ",".join("?" for _ in scope["version_ids"])
                connection.execute(
                    f"UPDATE asset_projection_states SET status='obsolete',last_error='定向治理删除',updated_at=? "
                    f"WHERE version_id IN ({placeholders})",
                    (now, *scope["version_ids"]),
                )
                connection.execute(
                    f"UPDATE asset_versions SET status='revoked',content_hash='',summary='',failure_reason='',"
                    f"metadata_json=?,updated_at=? WHERE version_id IN ({placeholders})",
                    (tombstone, now, *scope["version_ids"]),
                )
            for asset in changed_assets:
                asset_id = str(asset["asset_id"])
                revision = int(asset.get("lifecycle_revision") or 0) + 1
                connection.execute(
                    """
                    UPDATE knowledge_assets SET asset_key=?,title='[已删除资产]',topic='',summary='',
                        owner_user_id='',access_policy_id='',status='deleted',review_due_at='',
                        revoked_at=?,lifecycle_revision=?,updated_by=?,metadata_json=?,updated_at=?
                    WHERE project_id=? AND asset_id=?
                    """,
                    (f"deleted:{asset_id}", now, revision, actor_user_id, tombstone, now, self.project_id, asset_id),
                )
                connection.execute(
                    "DELETE FROM knowledge_asset_aliases WHERE project_id=? AND asset_id=?",
                    (self.project_id, asset_id),
                )
                event_id = _stable_id("event", "knowledge_asset", asset_id, revision, "targeted_deleted")
                connection.execute(
                    """
                    INSERT OR IGNORE INTO domain_outbox(
                        event_id,project_id,aggregate_type,aggregate_id,aggregate_revision,event_type,
                        payload_json,available_at,created_at,updated_at
                    ) VALUES(?,?,?,?,?,'knowledge.asset_deleted',?,?,?,?)
                    """,
                    (event_id, self.project_id, "knowledge_asset", asset_id, revision,
                     _json_dump({"asset_id": asset_id, "reason": "targeted_governance", "run_id": run_id}),
                     now, now, now),
                )
            source_ids_to_delete = set(scope["source_ids"]) if target_type in {"source", "user"} else set()
            if target_type == "asset":
                for source_id in scope["source_ids"]:
                    remaining = int(connection.execute(
                        "SELECT COUNT(*) FROM knowledge_assets WHERE project_id=? AND primary_source_id=? "
                        "AND status NOT IN ('deleted','revoked')",
                        (self.project_id, source_id),
                    ).fetchone()[0])
                    if remaining == 0:
                        source_ids_to_delete.add(source_id)
            changed_source_ids = {
                str(row["source_id"]) for row in scope["sources"]
                if str(row.get("source_id")) in source_ids_to_delete
                and (
                    str(row.get("status") or "") != "deleted"
                    or not bool(_json_load(row.get("metadata_json"), {}).get("targeted_deleted"))
                )
            }
            for source_id in sorted(changed_source_ids):
                connection.execute(
                    """
                    UPDATE knowledge_sources SET external_key=?,display_name='[已删除来源]',owner_user_id='',
                        access_policy_id='',status='deleted',metadata_json=?,updated_at=?
                    WHERE project_id=? AND source_id=?
                    """,
                    (f"deleted:{source_id}", tombstone, now, self.project_id, source_id),
                )
            if scope["asset_ids"]:
                placeholders = ",".join("?" for _ in scope["asset_ids"])
                node_ids = [str(row["node_id"]) for row in connection.execute(
                    f"SELECT node_id FROM graph_nodes WHERE project_id=? "
                    f"AND json_extract(data_json,'$.asset_id') IN ({placeholders})",
                    (self.project_id, *scope["asset_ids"]),
                )]
                if node_ids:
                    node_placeholders = ",".join("?" for _ in node_ids)
                    connection.execute(
                        f"DELETE FROM graph_edges WHERE project_id=? AND "
                        f"(source_node_id IN ({node_placeholders}) OR target_node_id IN ({node_placeholders}))",
                        (self.project_id, *node_ids, *node_ids),
                    )
                    connection.execute(
                        f"DELETE FROM graph_nodes WHERE project_id=? AND node_id IN ({node_placeholders})",
                        (self.project_id, *node_ids),
                    )
                connection.execute(
                    f"UPDATE onboarding_learning_items SET status='blocked',evidence='',updated_at=? "
                    f"WHERE asset_id IN ({placeholders}) AND status<>'completed'",
                    (now, *scope["asset_ids"]),
                )
                connection.execute(
                    f"UPDATE handover_items SET validity_status='invalid',updated_at=? "
                    f"WHERE project_id=? AND asset_id IN ({placeholders})",
                    (now, self.project_id, *scope["asset_ids"]),
                )
            if target_type == "user":
                connection.execute(
                    "DELETE FROM access_policy_users WHERE user_id=? AND access_policy_id IN "
                    "(SELECT access_policy_id FROM access_policies WHERE project_id=?)",
                    (scope["target_id"], self.project_id),
                )
                connection.execute("DELETE FROM auth_sessions WHERE user_id=?", (scope["target_id"],))
                connection.execute("DELETE FROM user_external_identities WHERE user_id=?", (scope["target_id"],))
                connection.execute(
                    "UPDATE project_memberships SET status='disabled',updated_at=? WHERE project_id=? AND user_id=?",
                    (now, self.project_id, scope["target_id"]),
                )
                connection.execute(
                    """
                    UPDATE users SET display_name='[已删除用户]',email='',email_normalized='',status='disabled',
                        password_hash='',password_changed_at=?,failed_login_count=0,locked_until='',updated_at=?
                    WHERE user_id=?
                    """,
                    (now, now, scope["target_id"]),
                )
                connection.execute(
                    "UPDATE onboarding_learning_plans SET status='cancelled',updated_at=? "
                    "WHERE project_id=? AND user_id=? AND status='active'",
                    (now, self.project_id, scope["target_id"]),
                )
            result = {
                "sources_deleted": len(changed_source_ids),
                "assets_deleted": len(changed_assets),
                "versions_revoked": len(changed_versions),
                "documents_deleted": len(changed_documents),
                "chunks_deleted": chunk_count,
                "messages_redacted": len(changed_messages),
                "legacy_feishu_assets_reverted": legacy_assets_reverted,
                "legacy_feishu_candidates_reverted": legacy_candidates_reverted,
                "files": stored_files,
                "file_cleanup_complete": False,
            }
            connection.execute(
                """
                INSERT INTO targeted_deletion_runs(
                    run_id,project_id,target_type,target_id,status,actor_user_id,preview_json,result_json,created_at
                ) VALUES(?,?,?,?, 'succeeded', ?,?,?,?)
                """,
                (run_id, self.project_id, scope["target_type"], scope["target_id"], actor_user_id,
                 _json_dump(preview), _json_dump(result), now),
            )
            self._write_audit_connection(
                connection, f"audit_{uuid.uuid4().hex}", actor_user_id,
                "governance.target_deleted", scope["target_type"], scope["target_id"],
                {"run_id": run_id, **result, "files": len(stored_files)},
                project_id=self.project_id, created_at=now,
            )
        return {"run_id": run_id, "target_type": scope["target_type"], "target_id": scope["target_id"], **result}

    def complete_targeted_file_cleanup(
        self, run_id: str, *, removed: Sequence[str], actor_user_id: str,
    ) -> Dict[str, Any]:
        now = utc_now()
        with self.database.transaction(write=True) as connection:
            row = connection.execute(
                "SELECT * FROM targeted_deletion_runs WHERE project_id=? AND run_id=?",
                (self.project_id, str(run_id or "")),
            ).fetchone()
            if not row:
                raise ValueError("定向删除运行不存在")
            result = _json_load(row["result_json"], {})
            if result.get("file_cleanup_complete") is True:
                return result
            expected = sorted(set(str(item) for item in result.get("files", []) if str(item)))
            removed_items = sorted(set(str(item) for item in removed if str(item)))
            result["files"] = len(expected)
            result["files_removed"] = len(removed_items)
            result["file_cleanup_complete"] = True
            connection.execute(
                "UPDATE targeted_deletion_runs SET result_json=? WHERE run_id=?",
                (_json_dump(result), row["run_id"]),
            )
            self._write_audit_connection(
                connection, f"audit_{uuid.uuid4().hex}", actor_user_id,
                "governance.target_files_removed", str(row["target_type"]), str(row["target_id"]),
                {"run_id": row["run_id"], "expected": len(expected), "removed": len(removed_items)},
                project_id=self.project_id, created_at=now,
            )
        return result

    # Knowledge task workflow ----------------------------------------
    @staticmethod
    def _knowledge_task_payload(row: Any) -> Dict[str, Any]:
        payload = dict(row)
        payload["metadata"] = _json_load(payload.pop("metadata_json", "{}"), {})
        payload["overdue"] = bool(
            payload.get("due_at")
            and payload.get("due_at") < utc_now()
            and payload.get("status") not in {"completed", "cancelled"}
        )
        return payload

    def create_or_merge_knowledge_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        now = utc_now()
        task_id = str(payload.get("task_id") or f"kt_{uuid.uuid4().hex}")
        dedupe_key = str(payload.get("dedupe_key") or "").strip()
        assignee = str(payload.get("assignee_user_id") or "").strip()
        reporter = str(payload.get("reporter_user_id") or "user_system").strip() or "user_system"
        with self.database.transaction(write=True) as connection:
            if assignee:
                member = connection.execute(
                    "SELECT 1 FROM project_memberships WHERE project_id=? AND user_id=? AND status='active'",
                    (self.project_id, assignee),
                ).fetchone()
                if not member:
                    raise ValueError("任务负责人不是当前项目的有效成员")
            existing = None
            if dedupe_key:
                existing = connection.execute(
                    """
                    SELECT * FROM knowledge_tasks
                    WHERE project_id=? AND dedupe_key=?
                      AND status IN ('unassigned','open','in_progress','pending_acceptance')
                    """,
                    (self.project_id, dedupe_key),
                ).fetchone()
            if existing:
                task_id = str(existing["task_id"])
                refresh_asset_publish_binding = bool(
                    str(payload.get("task_type") or "") == "asset_publish"
                    and str(payload.get("source_type") or "") == "asset_version"
                )
                reopen_submitted_task = bool(
                    str(existing["status"]) == "pending_acceptance"
                    and (
                        (
                            str(payload.get("task_type") or "") == "handover_gap"
                            and str(payload.get("source_type") or "") == "handover_item"
                        )
                        or (
                            str(payload.get("task_type") or "") == "asset_review"
                            and str(payload.get("source_type") or "") == "knowledge_asset"
                        )
                        or refresh_asset_publish_binding
                    )
                )
                event_status = "in_progress" if reopen_submitted_task else str(existing["status"])
                if refresh_asset_publish_binding:
                    effective_assignee = assignee or str(existing["assignee_user_id"] or "")
                    connection.execute(
                        """
                        UPDATE knowledge_tasks SET occurrence_count=occurrence_count+1,
                            status=?,completion_evidence='',title=?,description=?,source_key=?,
                            linked_asset_id=?,assignee_user_id=?,metadata_json=?,
                            last_occurrence_at=?,updated_at=? WHERE task_id=?
                        """,
                        (
                            event_status, str(payload.get("title") or "").strip(),
                            str(payload.get("description") or "").strip(),
                            str(payload.get("source_key") or ""),
                            str(payload.get("linked_asset_id") or ""), effective_assignee,
                            _json_dump(payload.get("metadata") or {}), now, now, task_id,
                        ),
                    )
                    event_type = "reopened" if reopen_submitted_task else "merged"
                elif reopen_submitted_task:
                    connection.execute(
                        """
                        UPDATE knowledge_tasks SET occurrence_count=occurrence_count+1,
                            status='in_progress',completion_evidence='',
                            last_occurrence_at=?,updated_at=? WHERE task_id=?
                        """,
                        (now, now, task_id),
                    )
                    event_type = "reopened"
                else:
                    connection.execute(
                        """
                        UPDATE knowledge_tasks SET occurrence_count=occurrence_count+1,
                            last_occurrence_at=?,updated_at=? WHERE task_id=?
                        """,
                        (now, now, task_id),
                    )
                    event_type = "merged"
                created = False
            else:
                status = "open" if assignee else "unassigned"
                event_status = status
                connection.execute(
                    """
                    INSERT INTO knowledge_tasks(
                        task_id,project_id,task_type,title,description,source_type,source_key,dedupe_key,
                        linked_asset_id,role_key,handover_id,priority,assignee_user_id,reporter_user_id,
                        due_at,status,last_occurrence_at,metadata_json,created_at,updated_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        task_id, self.project_id, str(payload.get("task_type") or "knowledge_gap"),
                        str(payload.get("title") or "").strip(), str(payload.get("description") or "").strip(),
                        str(payload.get("source_type") or "manual"), str(payload.get("source_key") or ""),
                        dedupe_key, str(payload.get("linked_asset_id") or ""), str(payload.get("role_key") or ""),
                        str(payload.get("handover_id") or ""), str(payload.get("priority") or "medium"),
                        assignee, reporter, str(payload.get("due_at") or ""), status, now,
                        _json_dump(payload.get("metadata") or {}), now, now,
                    ),
                )
                event_type = "created"
                created = True
            connection.execute(
                """
                INSERT INTO knowledge_task_events(
                    event_id,task_id,project_id,actor_user_id,event_type,to_status,note,data_json,created_at
                ) VALUES(?,?,?,?,?,?,?,?,?)
                """,
                (
                    f"kte_{uuid.uuid4().hex}", task_id, self.project_id, reporter, event_type,
                    event_status,
                    str(payload.get("reason") or payload.get("description") or "")[:1000],
                    _json_dump({"source_type": payload.get("source_type", "manual")}), now,
                ),
            )
        task = self.get_knowledge_task(task_id, can_manage=True) or {}
        task["created"] = created
        return task

    def list_knowledge_tasks(
        self, *, viewer_user_id: str = "", can_manage: bool = False,
        status: str = "", task_type: str = "", assignee_user_id: str = "",
        keyword: str = "", limit: int = 200,
    ) -> List[Dict[str, Any]]:
        where = ["kt.project_id=?"]
        params: List[Any] = [self.project_id]
        if not can_manage:
            where.append("(kt.assignee_user_id=? OR kt.reporter_user_id=?)")
            params.extend((viewer_user_id, viewer_user_id))
        if status:
            where.append("kt.status=?")
            params.append(status)
        if task_type:
            where.append("kt.task_type=?")
            params.append(task_type)
        if assignee_user_id:
            where.append("kt.assignee_user_id=?")
            params.append(assignee_user_id)
        if keyword:
            literal = f"%{_like_literal(keyword.lower())}%"
            where.append("(lower(kt.title) LIKE ? ESCAPE '\\' OR lower(kt.description) LIKE ? ESCAPE '\\')")
            params.extend((literal, literal))
        params.append(max(1, min(int(limit), 500)))
        with self.database.transaction() as connection:
            rows = connection.execute(
                f"""
                SELECT kt.*,au.display_name AS assignee_name,ru.display_name AS reporter_name
                FROM knowledge_tasks kt
                LEFT JOIN users au ON au.user_id=kt.assignee_user_id
                LEFT JOIN users ru ON ru.user_id=kt.reporter_user_id
                WHERE {' AND '.join(where)}
                ORDER BY CASE kt.status
                    WHEN 'pending_acceptance' THEN 0 WHEN 'in_progress' THEN 1
                    WHEN 'open' THEN 2 WHEN 'unassigned' THEN 3 ELSE 4 END,
                    CASE kt.priority WHEN 'critical' THEN 0 WHEN 'high' THEN 1
                        WHEN 'medium' THEN 2 ELSE 3 END,
                    CASE WHEN kt.due_at='' THEN 1 ELSE 0 END,kt.due_at,kt.updated_at DESC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
        return [self._knowledge_task_payload(row) for row in rows]

    def get_knowledge_task(
        self, task_id: str, *, viewer_user_id: str = "", can_manage: bool = False,
    ) -> Optional[Dict[str, Any]]:
        where = "kt.task_id=? AND kt.project_id=?"
        params: List[Any] = [task_id, self.project_id]
        if not can_manage:
            where += " AND (kt.assignee_user_id=? OR kt.reporter_user_id=?)"
            params.extend((viewer_user_id, viewer_user_id))
        with self.database.transaction() as connection:
            row = connection.execute(
                f"""
                SELECT kt.*,au.display_name AS assignee_name,ru.display_name AS reporter_name
                FROM knowledge_tasks kt
                LEFT JOIN users au ON au.user_id=kt.assignee_user_id
                LEFT JOIN users ru ON ru.user_id=kt.reporter_user_id
                WHERE {where}
                """,
                tuple(params),
            ).fetchone()
            if not row:
                return None
            task = self._knowledge_task_payload(row)
            task["events"] = [
                {**dict(item), "data": _json_load(item["data_json"], {})}
                for item in connection.execute(
                    "SELECT * FROM knowledge_task_events WHERE task_id=? ORDER BY created_at,rowid",
                    (task_id,),
                )
            ]
            for item in task["events"]:
                item.pop("data_json", None)
            for event in reversed(task["events"]):
                writeback = event.get("data", {}).get("writeback")
                if isinstance(writeback, dict):
                    task["writeback"] = dict(writeback)
                    processing_job_id = str(writeback.get("processing_job_id") or "")
                    if processing_job_id:
                        processing_job = connection.execute(
                            """
                            SELECT job_id,job_type,status,stage,progress,attempts,max_attempts,
                                   error_code,last_error,available_at,completed_at,updated_at
                            FROM processing_jobs WHERE project_id=? AND job_id=?
                            """,
                            (self.project_id, processing_job_id),
                        ).fetchone()
                        if processing_job:
                            task["writeback"]["processing_job"] = dict(processing_job)
                    break
            task["notifications"] = [
                dict(item) for item in connection.execute(
                    """
                    SELECT tn.*,pj.status AS processing_status,pj.attempts AS processing_attempts,
                           pj.max_attempts AS processing_max_attempts,pj.available_at AS processing_available_at
                    FROM task_notifications tn
                    LEFT JOIN processing_jobs pj ON pj.job_id=tn.processing_job_id
                    WHERE tn.task_id=?
                    ORDER BY COALESCE(NULLIF(tn.attempted_at,''),tn.scheduled_at,tn.created_at) DESC,tn.rowid DESC
                    LIMIT 20
                    """,
                    (task_id,),
                )
            ]
            return task

    def update_knowledge_task(
        self, task_id: str, *, actor_user_id: str, event_type: str,
        to_status: str = "", assignee_user_id: Optional[str] = None,
        evidence: Optional[str] = None, note: str = "", due_at: Optional[str] = None,
        expected_status: str = "", linked_asset_id: Optional[str] = None,
        metadata_updates: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        now = utc_now()
        writeback: Optional[Dict[str, Any]] = None
        with self.database.transaction(write=True) as connection:
            task = connection.execute(
                "SELECT * FROM knowledge_tasks WHERE task_id=? AND project_id=?",
                (task_id, self.project_id),
            ).fetchone()
            if not task:
                raise ValueError("知识任务不存在")
            if expected_status and str(task["status"]) != expected_status:
                raise ValueError("任务状态已变化，请刷新后重试")
            assignments = ["updated_at=?"]
            params: List[Any] = [now]
            if to_status:
                assignments.append("status=?")
                params.append(to_status)
                if to_status == "completed":
                    assignments.extend(("completed_at=?", "accepted_at=?"))
                    params.extend((now, now))
            if assignee_user_id is not None:
                if assignee_user_id:
                    member = connection.execute(
                        "SELECT 1 FROM project_memberships WHERE project_id=? AND user_id=? AND status='active'",
                        (self.project_id, assignee_user_id),
                    ).fetchone()
                    if not member:
                        raise ValueError("任务负责人不是当前项目的有效成员")
                assignments.append("assignee_user_id=?")
                params.append(assignee_user_id)
            if evidence is not None:
                assignments.append("completion_evidence=?")
                params.append(evidence)
            if linked_asset_id is not None:
                assignments.append("linked_asset_id=?")
                params.append(str(linked_asset_id or ""))
            if metadata_updates is not None:
                metadata = _json_load(task["metadata_json"], {})
                metadata.update(metadata_updates)
                assignments.append("metadata_json=?")
                params.append(_json_dump(metadata))
            if due_at is not None:
                assignments.append("due_at=?")
                params.append(due_at)
            params.append(task_id)
            connection.execute(
                f"UPDATE knowledge_tasks SET {','.join(assignments)} WHERE task_id=?",
                tuple(params),
            )
            if to_status == "completed":
                completed_task = connection.execute(
                    "SELECT * FROM knowledge_tasks WHERE task_id=? AND project_id=?",
                    (task_id, self.project_id),
                ).fetchone()
                writeback = self._writeback_completed_handover_task_connection(
                    connection, completed_task, actor_user_id=actor_user_id, now=now,
                ) or self._writeback_completed_asset_review_task_connection(
                    connection, completed_task, actor_user_id=actor_user_id, now=now,
                ) or self._writeback_completed_asset_publish_task_connection(
                    connection, completed_task, actor_user_id=actor_user_id, now=now,
                )
            event_data: Dict[str, Any] = {
                "assignee_user_id": assignee_user_id, "due_at": due_at,
                "linked_asset_id": linked_asset_id,
            }
            if metadata_updates:
                event_data["metadata_updates"] = metadata_updates
            if writeback:
                event_data["writeback"] = writeback
            connection.execute(
                """
                INSERT INTO knowledge_task_events(
                    event_id,task_id,project_id,actor_user_id,event_type,from_status,to_status,note,data_json,created_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    f"kte_{uuid.uuid4().hex}", task_id, self.project_id, actor_user_id,
                    event_type, str(task["status"]), to_status or str(task["status"]), note[:2000],
                    _json_dump(event_data), now,
                ),
            )
        updated = self.get_knowledge_task(task_id, can_manage=True) or {}
        if writeback and not isinstance(updated.get("writeback"), dict):
            updated["writeback"] = writeback
        return updated

    def _writeback_completed_handover_task_connection(
        self, connection: sqlite3.Connection, task: sqlite3.Row, *,
        actor_user_id: str, now: str,
    ) -> Optional[Dict[str, Any]]:
        """Re-submit the exact rejected handover item linked to a completed gap task.

        The task and handover item are updated in the caller's transaction.  This
        deliberately stops at ``submitted``: only the recipient or a handover
        manager may perform the independent item acceptance.
        """
        if str(task["task_type"] or "") != "handover_gap" or str(task["source_type"] or "") != "handover_item":
            return None

        handover_id = str(task["handover_id"] or "").strip()
        item_id = str(task["source_key"] or "").strip()
        result: Dict[str, Any] = {
            "type": "handover_item_resubmit", "status": "skipped", "applied": False,
            "handover_id": handover_id, "item_id": item_id,
        }
        if not handover_id or not item_id:
            return {
                **result, "reason": "invalid_reference",
                "message": "任务已验收，但关联交接项标识不完整，未自动更新",
            }

        case = connection.execute(
            "SELECT status FROM handover_cases WHERE project_id=? AND handover_id=?",
            (self.project_id, handover_id),
        ).fetchone()
        item = connection.execute(
            "SELECT * FROM handover_items WHERE project_id=? AND handover_id=? AND item_id=?",
            (self.project_id, handover_id, item_id),
        ).fetchone()
        if not case or not item:
            return {
                **result, "reason": "target_missing",
                "message": "任务已验收，但关联交接记录已不存在，未自动更新",
            }
        if str(item["knowledge_task_id"] or "") != str(task["task_id"]):
            return {
                **result, "reason": "link_mismatch",
                "message": "任务已验收，但交接项已关联其他补充任务，未自动更新",
            }
        if str(case["status"] or "") == "completed":
            return {
                **result, "reason": "handover_closed",
                "message": "任务已验收；关联交接已关闭，未再修改封存状态",
            }

        current = str(item["status"] or "")
        if current != "rejected":
            messages = {
                "submitted": "任务已验收；关联交接项已由责任人重新提交，无需重复更新",
                "accepted": "任务已验收；关联交接项已经验收通过，无需重复更新",
                "pending": "任务已验收，但关联交接项尚未处于退回状态，未自动更新",
            }
            return {
                **result, "reason": f"item_{current or 'unknown'}",
                "message": messages.get(current, "任务已验收，但关联交接项状态已变化，未自动更新"),
            }

        evidence = str(task["completion_evidence"] or "").strip()
        if not evidence:
            return {
                **result, "reason": "missing_evidence",
                "message": "任务已验收，但缺少可回写的完成证据，交接项仍保持退回状态",
            }
        submitted_by = str(task["assignee_user_id"] or actor_user_id or "user_system")
        connection.execute(
            """
            UPDATE handover_items
            SET status='submitted',validity_status='unknown',evidence=?,rejection_reason='',
                submitted_by=?,submitted_at=?,reviewed_by='',reviewed_at='',
                review_mode='',proxy_reason='',
                revision=revision+1,updated_at=?
            WHERE project_id=? AND handover_id=? AND item_id=? AND status='rejected'
              AND knowledge_task_id=?
            """,
            (
                evidence[:4000], submitted_by, now, now, self.project_id,
                handover_id, item_id, str(task["task_id"]),
            ),
        )
        if connection.execute("SELECT changes()").fetchone()[0] != 1:
            return {
                **result, "reason": "concurrent_change",
                "message": "任务已验收，但交接项状态同时发生变化，请刷新后核对",
            }
        self._write_audit_connection(
            connection, f"audit_{uuid.uuid4().hex}", actor_user_id,
            "handover.item_resubmitted_from_task", "handover_item", item_id,
            {
                "handover_id": handover_id, "knowledge_task_id": str(task["task_id"]),
                "from_status": current, "to_status": "submitted", "has_evidence": True,
                "submitted_by": submitted_by,
            },
            project_id=self.project_id, created_at=now,
        )
        return {
            **result, "status": "applied", "applied": True, "reason": "",
            "message": "任务已验收，关联交接项已重新提交，等待接替人验收",
        }

    def _writeback_completed_asset_review_task_connection(
        self, connection: sqlite3.Connection, task: sqlite3.Row, *,
        actor_user_id: str, now: str,
    ) -> Optional[Dict[str, Any]]:
        """Confirm the exact review-due asset linked to an accepted review task."""
        if str(task["task_type"] or "") != "asset_review" or str(task["source_type"] or "") != "knowledge_asset":
            return None

        source_asset_id = str(task["source_key"] or "").strip()
        linked_asset_id = str(task["linked_asset_id"] or "").strip()
        result: Dict[str, Any] = {
            "type": "asset_review_confirm", "status": "skipped", "applied": False,
            "asset_id": source_asset_id or linked_asset_id,
        }
        if not source_asset_id or not linked_asset_id or source_asset_id != linked_asset_id:
            return {
                **result, "reason": "invalid_reference",
                "message": "任务已验收，但关联资产标识不完整或不一致，未自动确认复审",
            }

        asset = connection.execute(
            "SELECT * FROM knowledge_assets WHERE project_id=? AND asset_id=?",
            (self.project_id, source_asset_id),
        ).fetchone()
        if not asset:
            return {
                **result, "reason": "target_missing",
                "message": "任务已验收，但关联知识资产已不存在，未自动确认复审",
            }
        current = str(asset["status"] or "")
        if current != "review_due":
            messages = {
                "active": "任务已验收；关联知识资产已由治理人员确认有效，无需重复更新",
                "expired": "任务已验收；关联知识资产已失效，未恢复为生效状态",
                "revoked": "任务已验收；关联知识资产已撤销，未恢复为生效状态",
                "deleted": "任务已验收；关联知识资产已删除，未恢复为生效状态",
            }
            return {
                **result, "reason": f"asset_{current or 'unknown'}",
                "message": messages.get(current, "任务已验收，但关联知识资产不处于待复审状态，未自动更新"),
            }

        version_id = str(asset["current_version_id"] or "").strip()
        version = connection.execute(
            "SELECT status FROM asset_versions WHERE project_id=? AND asset_id=? AND version_id=?",
            (self.project_id, source_asset_id, version_id),
        ).fetchone() if version_id else None
        if not version or str(version["status"] or "") != "active":
            return {
                **result, "reason": "current_version_not_active",
                "message": "任务已验收，但关联资产没有可确认的生效版本，未自动更新",
            }

        revision = int(asset["lifecycle_revision"] or 0) + 1
        connection.execute(
            """
            UPDATE knowledge_assets
            SET status='active',review_due_at='',lifecycle_revision=?,updated_by=?,updated_at=?
            WHERE project_id=? AND asset_id=? AND status='review_due'
              AND lifecycle_revision=? AND current_version_id=?
            """,
            (
                revision, actor_user_id, now, self.project_id, source_asset_id,
                int(asset["lifecycle_revision"] or 0), version_id,
            ),
        )
        if connection.execute("SELECT changes()").fetchone()[0] != 1:
            return {
                **result, "reason": "concurrent_change",
                "message": "任务已验收，但关联资产状态同时发生变化，请刷新后核对",
            }
        connection.execute(
            """
            UPDATE asset_projection_states SET status='repair_required',
                desired_revision=desired_revision+1,last_error='知识有效性状态已更新',updated_at=?
            WHERE version_id=? AND projection_type='readiness' AND status<>'obsolete'
            """,
            (now, version_id),
        )
        event_type = "knowledge.asset_active"
        event_id = _stable_id("event", "knowledge_asset", source_asset_id, revision, event_type)
        connection.execute(
            """
            INSERT OR IGNORE INTO domain_outbox(
                event_id,project_id,aggregate_type,aggregate_id,aggregate_revision,event_type,
                payload_json,available_at,created_at,updated_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?)
            """,
            (
                event_id, self.project_id, "knowledge_asset", source_asset_id, revision, event_type,
                _json_dump({
                    "asset_id": source_asset_id, "version_id": version_id,
                    "reason": "知识资产复审任务验收通过", "knowledge_task_id": str(task["task_id"]),
                }), now, now, now,
            ),
        )
        processing_job = self._create_or_get_processing_job_connection(
            connection,
            {
                "job_type": "projection_repair",
                "payload": {
                    "asset_id": source_asset_id,
                    "actor": actor_user_id or "user_system",
                    "force": False,
                    "reason": "asset_review_completed",
                    "knowledge_task_id": str(task["task_id"]),
                    "asset_revision": revision,
                },
                "created_by": str(task["assignee_user_id"] or actor_user_id or "user_system"),
                "idempotency_key": f"projection-repair:asset-review:{source_asset_id}:{revision}",
                "max_attempts": 3,
                "priority": 80,
                "linked_asset_id": source_asset_id,
                "source_id": str(asset["primary_source_id"] or ""),
            },
            now=now,
        )
        self._write_audit_connection(
            connection, f"audit_{uuid.uuid4().hex}", actor_user_id,
            event_type, "knowledge_asset", source_asset_id,
            {
                "from": current, "to": "active", "revision": revision,
                "version_id": version_id, "knowledge_task_id": str(task["task_id"]),
            },
            project_id=self.project_id, created_at=now,
        )
        return {
            **result, "status": "applied", "applied": True, "reason": "",
            "version_id": version_id, "processing_job_id": processing_job["job_id"],
            "processing_job_created": bool(processing_job.get("created")),
            "message": "任务已验收，关联知识资产已确认继续有效，岗位就绪度刷新已进入处理队列",
        }

    def _writeback_completed_asset_publish_task_connection(
        self, connection: sqlite3.Connection, task: sqlite3.Row, *,
        actor_user_id: str, now: str,
    ) -> Optional[Dict[str, Any]]:
        """Publish only the exact ready version authorized by an accepted task."""
        if str(task["task_type"] or "") != "asset_publish" or str(task["source_type"] or "") != "asset_version":
            return None

        asset_id = str(task["linked_asset_id"] or "").strip()
        source_version_id = str(task["source_key"] or "").strip()
        metadata = _json_load(task["metadata_json"], {})
        metadata_version_id = str(metadata.get("linked_version_id") or "").strip()
        result: Dict[str, Any] = {
            "type": "asset_version_publish", "status": "skipped", "applied": False,
            "asset_id": asset_id, "version_id": source_version_id or metadata_version_id,
        }
        try:
            expected_revision = int(metadata.get("asset_revision"))
        except (TypeError, ValueError):
            expected_revision = -1
        if (
            not asset_id or not source_version_id or source_version_id != metadata_version_id
            or expected_revision < 1
        ):
            return {
                **result, "reason": "invalid_reference",
                "message": "任务已验收，但关联资产版本或生命周期标识不完整，未自动发布",
            }

        asset = connection.execute(
            "SELECT * FROM knowledge_assets WHERE project_id=? AND asset_id=?",
            (self.project_id, asset_id),
        ).fetchone()
        version = connection.execute(
            "SELECT * FROM asset_versions WHERE project_id=? AND asset_id=? AND version_id=?",
            (self.project_id, asset_id, source_version_id),
        ).fetchone()
        if not asset or not version:
            return {
                **result, "reason": "target_missing",
                "message": "任务已验收，但关联资产或版本已不存在，未自动发布",
            }

        asset_status = str(asset["status"] or "")
        if asset_status != "pending_review":
            messages = {
                "active": "任务已验收；关联资产已由治理人员发布或恢复，无需重复更新",
                "draft": "任务已验收；关联资产已退回草稿，未自动发布",
                "review_due": "任务已验收；关联资产已进入复审流程，未使用旧发布任务覆盖",
                "expired": "任务已验收；关联资产已失效，未恢复为生效状态",
                "revoked": "任务已验收；关联资产已撤销，未恢复为生效状态",
                "deleted": "任务已验收；关联资产已删除，未恢复为生效状态",
            }
            return {
                **result, "reason": f"asset_{asset_status or 'unknown'}",
                "message": messages.get(asset_status, "任务已验收，但关联资产不处于待审核状态，未自动发布"),
            }
        if int(asset["lifecycle_revision"] or 0) != expected_revision:
            return {
                **result, "reason": "lifecycle_changed",
                "message": "任务已验收，但资产生命周期已变化，请由治理人员核对后重新提交审核",
            }
        if str(version["status"] or "") != "ready":
            return {
                **result, "reason": f"version_{str(version['status'] or 'unknown')}",
                "message": "任务已验收，但关联版本已不再处于可发布状态，未自动发布",
            }
        newer = connection.execute(
            """
            SELECT version_id,status FROM asset_versions
            WHERE project_id=? AND asset_id=? AND version_no>?
              AND status IN ('preparing','ready','active')
            ORDER BY version_no DESC LIMIT 1
            """,
            (self.project_id, asset_id, int(version["version_no"] or 0)),
        ).fetchone()
        if newer:
            return {
                **result, "reason": "newer_version_exists",
                "newer_version_id": str(newer["version_id"]),
                "message": "任务已验收，但资产已有更新候选版本，旧任务未自动发布，请重新提交最新版本审核",
            }

        connection.execute("SAVEPOINT asset_publish_task_writeback")
        try:
            published = self.publish_asset_version(
                asset_id, source_version_id, actor_user_id,
                expected_revision=expected_revision, _connection=connection,
                _trigger={
                    "knowledge_task_id": str(task["task_id"]),
                    "trigger": "accepted_asset_publish_task",
                },
            )
        except ValueError as exc:
            connection.execute("ROLLBACK TO SAVEPOINT asset_publish_task_writeback")
            connection.execute("RELEASE SAVEPOINT asset_publish_task_writeback")
            return {
                **result, "reason": "publish_gate_failed",
                "message": f"任务已验收，但版本未通过发布门槛：{str(exc)[:300]}",
            }
        else:
            connection.execute("RELEASE SAVEPOINT asset_publish_task_writeback")

        processing_job = self._create_or_get_processing_job_connection(
            connection,
            {
                "job_type": "projection_repair",
                "payload": {
                    "asset_id": asset_id, "actor": actor_user_id or "user_system",
                    "force": False, "reason": "asset_publish_task_completed",
                    "knowledge_task_id": str(task["task_id"]),
                    "asset_revision": int(published.get("lifecycle_revision") or expected_revision + 1),
                },
                "created_by": str(task["assignee_user_id"] or actor_user_id or "user_system"),
                "idempotency_key": (
                    f"projection-repair:asset-publish:{asset_id}:"
                    f"{int(published.get('lifecycle_revision') or expected_revision + 1)}"
                ),
                "max_attempts": 3, "priority": 80, "linked_asset_id": asset_id,
                "source_id": str(asset["primary_source_id"] or ""),
            },
            now=now,
        )
        return {
            **result, "status": "applied", "applied": True, "reason": "",
            "processing_job_id": str(processing_job["job_id"]),
            "processing_job_created": bool(processing_job.get("created")),
            "message": "任务已验收，关联知识版本已发布生效，派生投影刷新已进入处理队列",
        }

    def record_task_notification(
        self, task_id: str, *, channel: str, target_id: str, status: str, error: str = "",
    ) -> Dict[str, Any]:
        now = utc_now()
        notification_id = f"ktn_{uuid.uuid4().hex}"
        with self.database.transaction(write=True) as connection:
            connection.execute(
                """
                INSERT INTO task_notifications(
                    notification_id,task_id,project_id,channel,target_id,status,error,attempted_at,created_at,
                    notification_kind,scheduled_at,sent_at,attempt_count
                ) VALUES(?,?,?,?,?,?,?,?,?,'manual',?,?,1)
                """,
                (
                    notification_id, task_id, self.project_id, channel, target_id, status,
                    error[:1000], now, now, now, now if status == "sent" else "",
                ),
            )
        return {"notification_id": notification_id, "status": status, "error": error[:1000]}

    @staticmethod
    def _task_notification_policy_payload(row: Any) -> Dict[str, Any]:
        if not row:
            return {
                "enabled": False, "target_chat_id": "", "timezone": "Asia/Shanghai",
                "quiet_start": "22:00", "quiet_end": "08:00",
                "remind_before_hours": 24, "reminder_interval_hours": 24,
                "escalation_after_hours": 24, "escalation_interval_hours": 24,
                "updated_by": "", "created_at": "", "updated_at": "",
            }
        payload = dict(row)
        payload["enabled"] = bool(payload.get("enabled"))
        return payload

    def get_task_notification_policy(self) -> Dict[str, Any]:
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM knowledge_task_notification_policies WHERE project_id=?",
                (self.project_id,),
            ).fetchone()
        return self._task_notification_policy_payload(row)

    def update_task_notification_policy(
        self, policy: Dict[str, Any], *, actor_user_id: str,
    ) -> Dict[str, Any]:
        now = utc_now()
        with self.database.transaction(write=True) as connection:
            existing = connection.execute(
                "SELECT created_at FROM knowledge_task_notification_policies WHERE project_id=?",
                (self.project_id,),
            ).fetchone()
            connection.execute(
                """
                INSERT INTO knowledge_task_notification_policies(
                    project_id,enabled,target_chat_id,timezone,quiet_start,quiet_end,
                    remind_before_hours,reminder_interval_hours,escalation_after_hours,
                    escalation_interval_hours,updated_by,created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(project_id) DO UPDATE SET
                    enabled=excluded.enabled,target_chat_id=excluded.target_chat_id,
                    timezone=excluded.timezone,quiet_start=excluded.quiet_start,
                    quiet_end=excluded.quiet_end,remind_before_hours=excluded.remind_before_hours,
                    reminder_interval_hours=excluded.reminder_interval_hours,
                    escalation_after_hours=excluded.escalation_after_hours,
                    escalation_interval_hours=excluded.escalation_interval_hours,
                    updated_by=excluded.updated_by,updated_at=excluded.updated_at
                """,
                (
                    self.project_id, int(bool(policy.get("enabled"))),
                    str(policy.get("target_chat_id") or ""), str(policy.get("timezone") or "Asia/Shanghai"),
                    str(policy.get("quiet_start") or "22:00"), str(policy.get("quiet_end") or "08:00"),
                    int(policy.get("remind_before_hours") or 24),
                    int(policy.get("reminder_interval_hours") or 24),
                    int(policy.get("escalation_after_hours") or 24),
                    int(policy.get("escalation_interval_hours") or 24),
                    actor_user_id, str(existing["created_at"] if existing else now), now,
                ),
            )
            self._write_audit_connection(
                connection, f"audit_{uuid.uuid4().hex}", actor_user_id,
                "knowledge_task.notification_policy_updated", "knowledge_task_notification_policy",
                self.project_id,
                {
                    "enabled": bool(policy.get("enabled")),
                    "has_target_chat": bool(str(policy.get("target_chat_id") or "")),
                    "timezone": str(policy.get("timezone") or "Asia/Shanghai"),
                    "quiet_start": str(policy.get("quiet_start") or "22:00"),
                    "quiet_end": str(policy.get("quiet_end") or "08:00"),
                    "remind_before_hours": int(policy.get("remind_before_hours") or 24),
                    "reminder_interval_hours": int(policy.get("reminder_interval_hours") or 24),
                    "escalation_after_hours": int(policy.get("escalation_after_hours") or 24),
                    "escalation_interval_hours": int(policy.get("escalation_interval_hours") or 24),
                },
                project_id=self.project_id, created_at=now,
            )
            row = connection.execute(
                "SELECT * FROM knowledge_task_notification_policies WHERE project_id=?",
                (self.project_id,),
            ).fetchone()
        return self._task_notification_policy_payload(row)

    def list_task_reminder_candidates(self, *, due_before: str, limit: int = 500) -> List[Dict[str, Any]]:
        with self.database.transaction() as connection:
            return [dict(row) for row in connection.execute(
                """
                SELECT kt.*,au.display_name AS assignee_name,ru.display_name AS reporter_name,
                       MAX(CASE WHEN tn.notification_kind<>'manual' THEN tn.scheduled_at ELSE '' END)
                           AS last_auto_notification_at
                FROM knowledge_tasks kt
                LEFT JOIN users au ON au.user_id=kt.assignee_user_id
                LEFT JOIN users ru ON ru.user_id=kt.reporter_user_id
                LEFT JOIN task_notifications tn ON tn.task_id=kt.task_id
                WHERE kt.project_id=? AND kt.status IN ('unassigned','open','in_progress','pending_acceptance')
                  AND kt.due_at<>'' AND kt.due_at<=?
                GROUP BY kt.task_id
                ORDER BY kt.due_at,CASE kt.priority
                    WHEN 'critical' THEN 0 WHEN 'high' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END,kt.task_id
                LIMIT ?
                """,
                (self.project_id, due_before, max(1, min(int(limit), 1000))),
            )]

    def create_task_notification_digest(
        self, entries: Sequence[Dict[str, Any]], *, target_id: str, available_at: str,
    ) -> Optional[Dict[str, Any]]:
        if not entries or not str(target_id or "").strip():
            return None
        now = utc_now()
        created_entries: List[Dict[str, Any]] = []
        with self.database.transaction(write=True) as connection:
            for entry in entries:
                task_id = str(entry.get("task_id") or "")
                dedupe_key = str(entry.get("dedupe_key") or "")
                current = connection.execute(
                    """
                    SELECT status FROM knowledge_tasks WHERE project_id=? AND task_id=?
                      AND status IN ('unassigned','open','in_progress','pending_acceptance')
                    """,
                    (self.project_id, task_id),
                ).fetchone()
                if not current or not dedupe_key:
                    continue
                notification_id = _stable_id("ktn", self.project_id, dedupe_key)
                connection.execute(
                    """
                    INSERT OR IGNORE INTO task_notifications(
                        notification_id,task_id,project_id,channel,target_id,status,error,
                        attempted_at,created_at,notification_kind,dedupe_key,scheduled_at,
                        escalation_level
                    ) VALUES(?,?,?,?,?,'pending','','',?,?,?,?,?)
                    """,
                    (
                        notification_id, task_id, self.project_id, "feishu", str(target_id), now,
                        str(entry.get("notification_kind") or "due_soon"), dedupe_key,
                        str(available_at or now), max(0, int(entry.get("escalation_level") or 0)),
                    ),
                )
                if connection.execute("SELECT changes()").fetchone()[0] == 1:
                    created_entries.append({**entry, "notification_id": notification_id})
            if not created_entries:
                return None
            notification_ids = sorted(str(item["notification_id"]) for item in created_entries)
            digest_key = _stable_id("task_reminder_digest", *notification_ids)
            processing_job = self._create_or_get_processing_job_connection(
                connection,
                {
                    "job_type": "knowledge_task_notification",
                    "payload": {
                        "notification_ids": notification_ids,
                        "target_chat_id": str(target_id),
                        "trigger": "scheduled_task_reminder",
                    },
                    "created_by": "user_system", "idempotency_key": digest_key,
                    "max_attempts": 4, "priority": 60, "available_at": str(available_at or now),
                },
                now=now,
            )
            placeholders = ",".join("?" for _ in notification_ids)
            connection.execute(
                f"UPDATE task_notifications SET processing_job_id=? WHERE notification_id IN ({placeholders})",
                (str(processing_job["job_id"]), *notification_ids),
            )
        return {
            "processing_job": processing_job,
            "notifications": created_entries,
            "notification_count": len(created_entries),
        }

    def task_notification_delivery_items(self, notification_ids: Sequence[str]) -> List[Dict[str, Any]]:
        ids = [str(item) for item in notification_ids if str(item)]
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        with self.database.transaction() as connection:
            return [dict(row) for row in connection.execute(
                f"""
                SELECT tn.*,kt.title,kt.description,kt.status AS task_status,kt.priority,kt.due_at,
                       kt.assignee_user_id,au.display_name AS assignee_name,
                       kt.reporter_user_id,ru.display_name AS reporter_name
                FROM task_notifications tn
                JOIN knowledge_tasks kt ON kt.task_id=tn.task_id AND kt.project_id=tn.project_id
                LEFT JOIN users au ON au.user_id=kt.assignee_user_id
                LEFT JOIN users ru ON ru.user_id=kt.reporter_user_id
                WHERE tn.project_id=? AND tn.notification_id IN ({placeholders})
                ORDER BY tn.escalation_level DESC,kt.due_at,kt.task_id
                """,
                (self.project_id, *ids),
            )]

    def update_task_notification_deliveries(
        self, notification_ids: Sequence[str], *, status: str, error: str = "",
        increment_attempt: bool = True,
    ) -> int:
        ids = [str(item) for item in notification_ids if str(item)]
        if not ids:
            return 0
        placeholders = ",".join("?" for _ in ids)
        now = utc_now()
        with self.database.transaction(write=True) as connection:
            connection.execute(
                f"""
                UPDATE task_notifications SET status=?,error=?,attempted_at=?,
                    sent_at=CASE WHEN ?='sent' THEN ? ELSE sent_at END,
                    attempt_count=attempt_count+? WHERE project_id=?
                    AND notification_id IN ({placeholders})
                """,
                (
                    status, str(error or "")[:1000], now, status, now,
                    int(bool(increment_attempt)), self.project_id, *ids,
                ),
            )
            return int(connection.execute("SELECT changes()").fetchone()[0])

    # Unified knowledge assets -----------------------------------------
    def backfill_knowledge_assets(self, force: bool = False) -> Dict[str, int]:
        """Idempotently adopt legacy documents and Feishu assets into schema v4."""
        marker = f"knowledge_asset_v4_backfill:{self.project_id}"
        with self.database.transaction() as connection:
            missing_documents = int(connection.execute(
                """
                SELECT COUNT(*) FROM documents d
                LEFT JOIN asset_version_documents avd ON avd.document_id=d.document_id
                WHERE d.project_id=? AND avd.document_id IS NULL
                  AND d.status NOT IN ('deleted','revoked')
                  AND COALESCE(json_extract(d.metadata_json,'$.targeted_deleted'),0) IS NOT 1
                """,
                (self.project_id,),
            ).fetchone()[0])
            missing_feishu = int(connection.execute(
                """
                SELECT COUNT(*) FROM feishu_assets fa
                LEFT JOIN knowledge_asset_aliases kaa
                    ON kaa.project_id=fa.project_id AND kaa.alias_type='feishu_asset' AND kaa.alias_value=fa.asset_id
                WHERE fa.project_id=? AND kaa.asset_id IS NULL
                  AND fa.status NOT IN ('reverted','deleted')
                  AND COALESCE(json_extract(fa.data_json,'$.targeted_deleted'),0) IS NOT 1
                """,
                (self.project_id,),
            ).fetchone()[0])
            missing = missing_documents + missing_feishu
            marked = connection.execute(
                "SELECT 1 FROM system_state WHERE state_key=?", (marker,)
            ).fetchone() is not None
        if not force and not missing and marked:
            return {"documents": 0, "assets": 0, "versions": 0, "aliases": 0}

        counts = {"documents": 0, "assets": 0, "versions": 0, "aliases": 0}
        now = utc_now()
        with self.database.transaction(write=True) as connection:
            rows = connection.execute(
                """
                SELECT d.* FROM documents d
                LEFT JOIN asset_version_documents avd ON avd.document_id=d.document_id
                WHERE d.project_id=? AND avd.document_id IS NULL
                  AND d.status NOT IN ('deleted','revoked')
                  AND COALESCE(json_extract(d.metadata_json,'$.targeted_deleted'),0) IS NOT 1
                ORDER BY d.created_at,d.document_id
                """,
                (self.project_id,),
            ).fetchall()
            grouped: Dict[str, List[sqlite3.Row]] = {}
            for row in rows:
                source_type = str(row["source_type"] or "document")
                external_key = str(row["source_file"] or row["stored_file"] or row["document_id"])
                if row["handover_id"]:
                    source_type = "handover_document"
                    external_key = f"{row['handover_id']}:{external_key}"
                metadata = _json_load(row["metadata_json"], {})
                if source_type.startswith("feishu"):
                    external_key = str(metadata.get("feishu_candidate_id") or external_key)
                source_id = str(row["source_id"] or _stable_id(
                    "source", row["organization_id"], self.project_id, source_type, external_key,
                ))
                existing_source = connection.execute(
                    "SELECT source_id FROM knowledge_sources WHERE project_id=? AND source_type=? AND external_key=?",
                    (self.project_id, source_type, external_key),
                ).fetchone()
                if existing_source:
                    source_id = str(existing_source["source_id"])
                else:
                    connection.execute(
                        """
                        INSERT INTO knowledge_sources(
                            source_id,organization_id,project_id,source_type,external_key,display_name,
                            owner_user_id,visibility,sensitivity_level,access_policy_id,status,
                            metadata_json,created_at,updated_at
                        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            source_id, str(row["organization_id"] or DEFAULT_ORGANIZATION_ID), self.project_id,
                            source_type, external_key, str(row["source_file"] or row["stored_file"]),
                            str(row["owner_user_id"] or ""), str(row["visibility"] or "project"),
                            str(row["sensitivity_level"] or "internal"), str(row["access_policy_id"] or ""),
                            "active" if row["status"] not in {"revoked", "deleted"} else str(row["status"]),
                            _json_dump({"backfilled": True}), str(row["created_at"] or now), now,
                        ),
                    )
                asset_id = str(row["asset_id"] or _stable_id("asset", self.project_id, source_id))
                asset_key = f"{source_type}:{external_key}"
                existing_asset = connection.execute(
                    "SELECT asset_id FROM knowledge_assets WHERE project_id=? AND asset_key=?",
                    (self.project_id, asset_key),
                ).fetchone()
                if existing_asset:
                    asset_id = str(existing_asset["asset_id"])
                elif not connection.execute(
                    "SELECT 1 FROM knowledge_assets WHERE asset_id=?", (asset_id,)
                ).fetchone():
                    connection.execute(
                        """
                        INSERT INTO knowledge_assets(
                            asset_id,organization_id,project_id,primary_source_id,asset_key,title,
                            owner_user_id,sensitivity_level,visibility,access_policy_id,status,
                            created_by,updated_by,created_at,updated_at,metadata_json
                        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            asset_id, str(row["organization_id"] or DEFAULT_ORGANIZATION_ID), self.project_id,
                            source_id, asset_key, str(row["source_file"] or row["stored_file"]),
                            str(row["owner_user_id"] or ""), str(row["sensitivity_level"] or "internal"),
                            str(row["visibility"] or "project"), str(row["access_policy_id"] or ""), "draft",
                            str(row["uploader"] or "migration"), "migration", str(row["created_at"] or now), now,
                            _json_dump({"backfilled": True}),
                        ),
                    )
                    counts["assets"] += 1
                connection.execute(
                    "UPDATE documents SET organization_id=?,source_id=?,asset_id=?,updated_at=? WHERE document_id=?",
                    (str(row["organization_id"] or DEFAULT_ORGANIZATION_ID), source_id, asset_id, now, row["document_id"]),
                )
                grouped.setdefault(asset_id, []).append(row)

            for asset_id, documents in grouped.items():
                existing_max = int(connection.execute(
                    "SELECT COALESCE(MAX(version_no),0) FROM asset_versions WHERE asset_id=?", (asset_id,)
                ).fetchone()[0])
                active_candidates = [row for row in documents if str(row["status"] or "active") == "active"]
                current_row = active_candidates[-1] if active_candidates else None
                for offset, row in enumerate(documents, start=1):
                    version_no = existing_max + offset
                    version_id = str(row["version_id"] or _stable_id(
                        "version", asset_id, row["content_hash"] or row["stored_file"], version_no,
                    ))
                    collision = connection.execute(
                        "SELECT asset_id FROM asset_versions WHERE version_id=?", (version_id,)
                    ).fetchone()
                    if collision and str(collision["asset_id"]) != asset_id:
                        version_id = _stable_id("version", asset_id, row["document_id"], version_no)
                    version_status = "active" if current_row is not None and row["document_id"] == current_row["document_id"] else "superseded"
                    if current_row is None:
                        version_status = "revoked" if str(row["status"]) in {"revoked", "deleted"} else "failed"
                    connection.execute(
                        """
                        INSERT INTO asset_versions(
                            version_id,asset_id,project_id,version_no,status,content_hash,summary,
                            supersedes_version_id,valid_from,valid_until,created_by,published_by,
                            published_at,metadata_json,created_at,updated_at
                        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        ON CONFLICT(version_id) DO NOTHING
                        """,
                        (
                            version_id, asset_id, self.project_id, version_no, version_status,
                            str(row["content_hash"] or ""), "", str(row["supersedes_version_id"] or ""),
                            str(row["valid_from"] or ""), str(row["valid_until"] or ""),
                            str(row["uploader"] or "migration"), str(row["uploader"] or "migration") if version_status == "active" else "",
                            str(row["registered_at"] or now) if version_status == "active" else "",
                            _json_dump({"backfilled": True}), str(row["created_at"] or now), now,
                        ),
                    )
                    connection.execute(
                        "INSERT OR IGNORE INTO asset_version_documents(version_id,document_id,created_at) VALUES(?,?,?)",
                        (version_id, row["document_id"], now),
                    )
                    connection.execute(
                        "UPDATE documents SET version_id=?,status=?,updated_at=? WHERE document_id=?",
                        (version_id, "active" if version_status == "active" else version_status, now, row["document_id"]),
                    )
                    for projection in ("vector", "graph", "context", "readiness"):
                        connection.execute(
                            """
                            INSERT OR IGNORE INTO asset_projection_states(
                                version_id,projection_type,status,desired_revision,applied_revision,updated_at
                            ) VALUES(?,?,?,1,1,?)
                            """,
                            (version_id, projection, "ready" if version_status == "active" else "obsolete", now),
                        )
                    categories = connection.execute(
                        "SELECT category_id,is_primary FROM document_categories WHERE document_id=?",
                        (row["document_id"],),
                    ).fetchall()
                    for category in categories:
                        connection.execute(
                            "INSERT OR IGNORE INTO asset_categories(asset_id,category_id,is_primary,created_at) VALUES(?,?,?,?)",
                            (asset_id, category["category_id"], category["is_primary"], now),
                        )
                    for alias_type, alias_value in (
                        ("document_id", row["document_id"]),
                        ("stored_file", row["stored_file"]),
                        ("source_file", row["source_file"]),
                    ):
                        if alias_value:
                            connection.execute(
                                "INSERT OR REPLACE INTO knowledge_asset_aliases(project_id,alias_type,alias_value,asset_id,created_at) VALUES(?,?,?,?,?)",
                                (self.project_id, alias_type, str(alias_value), asset_id, now),
                            )
                            counts["aliases"] += 1
                    counts["documents"] += 1
                    counts["versions"] += 1
                if current_row is not None:
                    current = connection.execute(
                        "SELECT version_id FROM asset_version_documents WHERE document_id=?",
                        (current_row["document_id"],),
                    ).fetchone()
                    connection.execute(
                        """
                        UPDATE knowledge_assets SET status='active',current_version_id=?,published_at=?,
                            lifecycle_revision=MAX(lifecycle_revision,1),updated_by='migration',updated_at=?
                        WHERE asset_id=?
                        """,
                        (str(current["version_id"]), str(current_row["registered_at"] or now), now, asset_id),
                    )
                else:
                    connection.execute(
                        "UPDATE knowledge_assets SET status='revoked',revoked_at=?,updated_by='migration',updated_at=? WHERE asset_id=?",
                        (now, now, asset_id),
                    )

            feishu_rows = connection.execute(
                "SELECT asset_id,stored_file,data_json FROM feishu_assets WHERE project_id=? "
                "AND status NOT IN ('reverted','deleted') "
                "AND COALESCE(json_extract(data_json,'$.targeted_deleted'),0) IS NOT 1",
                (self.project_id,),
            ).fetchall()
            for row in feishu_rows:
                data = _json_load(row["data_json"], {})
                stored_file = str(row["stored_file"] or data.get("stored_file") or "")
                mapped = connection.execute(
                    """
                    SELECT d.asset_id FROM documents d
                    JOIN asset_version_documents avd ON avd.document_id=d.document_id
                    WHERE d.project_id=? AND (d.stored_file=? OR d.source_file=?) LIMIT 1
                    """,
                    (self.project_id, stored_file, str(data.get("source_file") or "")),
                ).fetchone()
                if mapped:
                    connection.execute(
                        "INSERT OR REPLACE INTO knowledge_asset_aliases(project_id,alias_type,alias_value,asset_id,created_at) VALUES(?,?,?,?,?)",
                        (self.project_id, "feishu_asset", str(row["asset_id"]), str(mapped["asset_id"]), now),
                    )
                    counts["aliases"] += 1
                    continue
                legacy_asset_id = str(row["asset_id"])
                candidate_id = str(data.get("candidate_id") or legacy_asset_id)
                source_type = "feishu_conversation"
                external_key = candidate_id
                source = connection.execute(
                    "SELECT source_id FROM knowledge_sources WHERE project_id=? AND source_type=? AND external_key=?",
                    (self.project_id, source_type, external_key),
                ).fetchone()
                source_id = str(source["source_id"]) if source else _stable_id(
                    "source", DEFAULT_ORGANIZATION_ID, self.project_id, source_type, external_key,
                )
                if not source:
                    connection.execute(
                        """
                        INSERT INTO knowledge_sources(
                            source_id,organization_id,project_id,source_type,external_key,display_name,
                            status,metadata_json,created_at,updated_at
                        ) VALUES(?,?,?,?,?,?, 'active',?,?,?)
                        """,
                        (source_id, DEFAULT_ORGANIZATION_ID, self.project_id, source_type, external_key,
                         str(data.get("source_file") or stored_file or candidate_id),
                         _json_dump({"backfilled": True, "repair_required": True}), now, now),
                    )
                asset_key = f"{source_type}:{external_key}"
                existing = connection.execute(
                    "SELECT asset_id FROM knowledge_assets WHERE project_id=? AND asset_key=?",
                    (self.project_id, asset_key),
                ).fetchone()
                canonical_id = str(existing["asset_id"]) if existing else legacy_asset_id
                if not existing:
                    if connection.execute("SELECT 1 FROM knowledge_assets WHERE asset_id=?", (canonical_id,)).fetchone():
                        canonical_id = _stable_id("asset", self.project_id, asset_key)
                    connection.execute(
                        """
                        INSERT INTO knowledge_assets(
                            asset_id,organization_id,project_id,primary_source_id,asset_key,title,status,
                            created_by,updated_by,created_at,updated_at,metadata_json
                        ) VALUES(?,?,?,?,?,?, 'draft','migration','migration',?,?,?)
                        """,
                        (canonical_id, DEFAULT_ORGANIZATION_ID, self.project_id, source_id, asset_key,
                         str(data.get("source_file") or stored_file or candidate_id), now, now,
                         _json_dump({"backfilled": True, "repair_required": True,
                                     "legacy_status": str(data.get("status") or "published")})),
                    )
                    counts["assets"] += 1
                connection.execute(
                    "INSERT OR REPLACE INTO knowledge_asset_aliases(project_id,alias_type,alias_value,asset_id,created_at) VALUES(?,?,?,?,?)",
                    (self.project_id, "feishu_asset", legacy_asset_id, canonical_id, now),
                )
                counts["aliases"] += 1
            connection.execute(
                "INSERT OR REPLACE INTO system_state(state_key,value_text,updated_at) VALUES(?,?,?)",
                (marker, "complete", now),
            )
        return counts

    def _asset_visible(
        self,
        asset: Dict[str, Any],
        identity: Any,
        connection: Optional[sqlite3.Connection] = None,
    ) -> bool:
        if identity is None or getattr(identity, "is_system", False):
            return True
        if str(getattr(identity, "project_id", "")) != str(asset.get("project_id") or ""):
            return False
        visibility = str(asset.get("visibility") or "project")
        if visibility in {"project", "organization"}:
            return True
        user_id = str(getattr(identity, "user_id", ""))
        if visibility == "private":
            return str(asset.get("owner_user_id") or "") == user_id
        if visibility != "restricted" or not asset.get("access_policy_id"):
            return False
        role = str(getattr(identity, "role", ""))
        def allowed(db: sqlite3.Connection) -> bool:
            return db.execute(
                """
                SELECT 1 WHERE EXISTS(
                    SELECT 1 FROM access_policy_users WHERE access_policy_id=? AND user_id=?
                ) OR EXISTS(
                    SELECT 1 FROM access_policy_roles WHERE access_policy_id=? AND role=?
                )
                """,
                (asset["access_policy_id"], user_id, asset["access_policy_id"], role),
            ).fetchone() is not None
        if connection is not None:
            return allowed(connection)
        with self.database.transaction() as db:
            return allowed(db)

    def resolve_asset_id(self, value: str) -> str:
        value = str(value or "").strip()
        if not value:
            return ""
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT asset_id FROM knowledge_assets WHERE project_id=? AND asset_id=?",
                (self.project_id, value),
            ).fetchone()
            if row:
                return str(row["asset_id"])
            row = connection.execute(
                "SELECT asset_id FROM knowledge_asset_aliases WHERE project_id=? AND alias_value=? ORDER BY alias_type LIMIT 1",
                (self.project_id, value),
            ).fetchone()
            return str(row["asset_id"]) if row else ""

    def add_asset_alias(self, asset_id: str, alias_type: str, alias_value: str) -> None:
        if not alias_value:
            return
        with self.database.transaction(write=True) as connection:
            if not connection.execute(
                "SELECT 1 FROM knowledge_assets WHERE asset_id=? AND project_id=?", (asset_id, self.project_id)
            ).fetchone():
                raise ValueError("知识资产不存在")
            connection.execute(
                "INSERT OR REPLACE INTO knowledge_asset_aliases(project_id,alias_type,alias_value,asset_id,created_at) VALUES(?,?,?,?,?)",
                (self.project_id, str(alias_type), str(alias_value), asset_id, utc_now()),
            )

    def _asset_dict(self, connection: sqlite3.Connection, row: sqlite3.Row) -> Dict[str, Any]:
        result = {**dict(row), "metadata": _json_load(row["metadata_json"], {})}
        result.pop("metadata_json", None)
        result["categories"] = [
            str(item["name"])
            for item in connection.execute(
                """
                SELECT c.name FROM asset_categories ac JOIN categories c ON c.category_id=ac.category_id
                WHERE ac.asset_id=? ORDER BY ac.is_primary DESC,c.sort_order,c.category_id
                """,
                (row["asset_id"],),
            )
        ]
        result["applicable_roles"] = [
            {"role": str(item["role_key"]), "requirement_level": str(item["requirement_level"]), "weight": int(item["weight"])}
            for item in connection.execute(
                "SELECT role_key,requirement_level,weight FROM asset_applicable_roles WHERE asset_id=? ORDER BY role_key",
                (row["asset_id"],),
            )
        ]
        return result

    @staticmethod
    def _policy_principals(
        connection: sqlite3.Connection, policy_id: str,
    ) -> tuple[set[str], set[str]]:
        if not policy_id:
            return set(), set()
        users = {
            str(row["user_id"])
            for row in connection.execute(
                "SELECT user_id FROM access_policy_users WHERE access_policy_id=?",
                (policy_id,),
            )
        }
        roles = {
            str(row["role"])
            for row in connection.execute(
                "SELECT role FROM access_policy_roles WHERE access_policy_id=?",
                (policy_id,),
            )
        }
        return users, roles

    def _user_allowed_by_policy(
        self, connection: sqlite3.Connection, user_id: str, policy_id: str,
    ) -> bool:
        if not user_id or not policy_id:
            return False
        users, roles = self._policy_principals(connection, policy_id)
        if user_id in users:
            return True
        if not roles:
            return False
        placeholders = ",".join("?" for _ in roles)
        return connection.execute(
            f"""
            SELECT 1 FROM project_memberships
            WHERE project_id=? AND user_id=? AND status='active'
              AND role IN ({placeholders}) LIMIT 1
            """,
            (self.project_id, user_id, *sorted(roles)),
        ).fetchone() is not None

    def _acl_is_within_source(
        self, connection: sqlite3.Connection, *, asset_visibility: str, asset_policy_id: str, asset_owner_user_id: str,
        source_visibility: str, source_policy_id: str, source_owner_user_id: str,
    ) -> bool:
        """Prove every subject allowed by the asset is also allowed by its source."""
        rank = {"private": 0, "restricted": 1, "project": 2, "organization": 3}
        if asset_visibility not in rank or source_visibility not in rank:
            return False
        if rank[asset_visibility] > rank[source_visibility]:
            return False
        if asset_visibility == "private":
            if not asset_owner_user_id:
                return False
            if source_visibility == "private":
                return asset_owner_user_id == source_owner_user_id
            if source_visibility == "restricted":
                return self._user_allowed_by_policy(
                    connection, asset_owner_user_id, source_policy_id,
                )
            return True
        if asset_visibility == "restricted":
            if not asset_policy_id:
                return False
            if source_visibility != "restricted":
                return True
            if asset_policy_id == source_policy_id:
                return True
            asset_users, asset_roles = self._policy_principals(connection, asset_policy_id)
            source_users, source_roles = self._policy_principals(connection, source_policy_id)
            if not asset_roles.issubset(source_roles):
                return False
            return all(
                user_id in source_users
                or self._user_allowed_by_policy(connection, user_id, source_policy_id)
                for user_id in asset_users
            )
        return True

    def _version_dict(
        self, connection: sqlite3.Connection, row: sqlite3.Row, identity: Any = None,
        asset: Optional[Dict[str, Any]] = None, source: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        item = {**dict(row), "metadata": _json_load(row["metadata_json"], {})}
        item.pop("metadata_json", None)
        documents = []
        for document in connection.execute(
                """
                SELECT d.*,(SELECT COUNT(*) FROM document_chunks ch WHERE ch.document_id=d.document_id) AS chunks
                FROM asset_version_documents avd JOIN documents d ON d.document_id=avd.document_id
                WHERE avd.version_id=? ORDER BY d.document_id
                """,
                (row["version_id"],),
            ):
            document_data = self._document_row(connection, document)
            if identity is not None and not (
                self._asset_visible(document_data, identity, connection)
                and (asset is None or self._asset_visible(asset, identity, connection))
                and (source is None or self._asset_visible(source, identity, connection))
            ):
                continue
            documents.append(document_data)
        item["documents"] = documents
        item["projections"] = {
            str(state["projection_type"]): dict(state)
            for state in connection.execute(
                "SELECT * FROM asset_projection_states WHERE version_id=? ORDER BY projection_type",
                (row["version_id"],),
            )
        }
        return item

    def get_asset_detail(self, asset_id: str, identity: Any = None) -> Optional[Dict[str, Any]]:
        asset = self.get_asset(asset_id, identity=identity)
        if not asset:
            return None
        with self.database.transaction() as connection:
            source_row = connection.execute(
                "SELECT * FROM knowledge_sources WHERE source_id=?", (asset["primary_source_id"],)
            ).fetchone()
            source = None
            if source_row:
                source = {**dict(source_row), "metadata": _json_load(source_row["metadata_json"], {})}
                source.pop("metadata_json", None)
            version_row = None
            if asset.get("current_version_id"):
                version_row = connection.execute(
                    "SELECT * FROM asset_versions WHERE version_id=?", (asset["current_version_id"],)
                ).fetchone()
            current_version = self._version_dict(
                connection, version_row, identity=identity, asset=asset, source=source,
            ) if version_row else None
        return {
            **asset,
            "summary": str(asset.get("summary") or ""),
            "source": source,
            "current_version": current_version,
            "projections": (current_version or {}).get("projections", {}),
            "documents": (current_version or {}).get("documents", []),
        }

    def get_asset(self, asset_id: str, identity: Any = None) -> Optional[Dict[str, Any]]:
        resolved = self.resolve_asset_id(asset_id)
        if not resolved:
            return None
        with self.database.transaction() as connection:
            row = connection.execute("SELECT * FROM knowledge_assets WHERE asset_id=?", (resolved,)).fetchone()
            if not row:
                return None
            asset = self._asset_dict(connection, row)
            source_row = connection.execute(
                "SELECT * FROM knowledge_sources WHERE source_id=? AND project_id=?",
                (asset["primary_source_id"], self.project_id),
            ).fetchone()
            if not source_row:
                return None
            source = {**dict(source_row), "metadata": _json_load(source_row["metadata_json"], {})}
            source.pop("metadata_json", None)
            visible = self._asset_visible(asset, identity, connection) and self._asset_visible(
                source, identity, connection,
            )
        return asset if visible else None

    def get_source(self, source_id: str, identity: Any = None) -> Optional[Dict[str, Any]]:
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM knowledge_sources WHERE source_id=? AND project_id=?",
                (str(source_id or ""), self.project_id),
            ).fetchone()
            if not row:
                return None
            source = {**dict(row), "metadata": _json_load(row["metadata_json"], {})}
            source.pop("metadata_json", None)
        return source if self._asset_visible(source, identity) else None

    def list_assets(
        self, identity: Any = None, *, include_inactive: bool = False,
        include_deleted: bool = False,
    ) -> List[Dict[str, Any]]:
        if identity is None:
            try:
                from backend.auth_context import get_current_identity
                identity = get_current_identity()
            except Exception:
                identity = None
        with self.database.transaction() as connection:
            if include_inactive or include_deleted:
                deleted_clause = "" if include_deleted else " AND status<>'deleted'"
                rows = connection.execute(
                    f"SELECT * FROM knowledge_assets WHERE project_id=?{deleted_clause} "
                    "ORDER BY updated_at DESC,asset_id",
                    (self.project_id,),
                ).fetchall()
            else:
                now = utc_now()
                rows = connection.execute(
                    """
                    SELECT ka.* FROM knowledge_assets ka
                    JOIN asset_versions av ON av.version_id=ka.current_version_id
                    WHERE ka.project_id=? AND ka.status IN ('active','review_due') AND av.status='active'
                    AND (av.valid_from='' OR av.valid_from<=?)
                    AND (av.valid_until='' OR av.valid_until>?)
                    ORDER BY ka.updated_at DESC,ka.asset_id
                    """,
                    (self.project_id, now, now),
                ).fetchall()
            candidate_assets = [self._asset_dict(connection, row) for row in rows]
            candidate_source_ids = sorted({
                str(asset["primary_source_id"])
                for asset in candidate_assets if asset.get("primary_source_id")
            })
            sources: Dict[str, Dict[str, Any]] = {}
            if candidate_source_ids:
                placeholders = ",".join("?" for _ in candidate_source_ids)
                for source_row in connection.execute(
                    f"SELECT * FROM knowledge_sources WHERE source_id IN ({placeholders})",
                    tuple(candidate_source_ids),
                ):
                    source = {**dict(source_row), "metadata": _json_load(source_row["metadata_json"], {})}
                    source.pop("metadata_json", None)
                    sources[str(source_row["source_id"])] = source
            assets = []
            for asset in candidate_assets:
                source = sources.get(str(asset.get("primary_source_id") or ""))
                if source and self._asset_visible(asset, identity, connection) and self._asset_visible(
                    source, identity, connection,
                ):
                    assets.append(asset)
            if not assets:
                return []

            # The knowledge page needs source and current-version identity in the
            # list response. Load the related rows in batches so the UI does not
            # create an N+1 request pattern as the asset count grows.
            version_ids = sorted({str(asset["current_version_id"]) for asset in assets if asset.get("current_version_id")})
            assets_by_version = {
                str(asset["current_version_id"]): asset
                for asset in assets if asset.get("current_version_id")
            }
            versions: Dict[str, Dict[str, Any]] = {}

            if version_ids:
                placeholders = ",".join("?" for _ in version_ids)
                documents_by_version: Dict[str, List[Dict[str, Any]]] = {version_id: [] for version_id in version_ids}
                for document_row in connection.execute(
                    f"""
                    SELECT avd.version_id,d.*,
                           (SELECT COUNT(*) FROM document_chunks ch WHERE ch.document_id=d.document_id) AS chunks
                    FROM asset_version_documents avd
                    JOIN documents d ON d.document_id=avd.document_id
                    WHERE avd.version_id IN ({placeholders})
                    ORDER BY avd.version_id,d.document_id
                    """,
                    tuple(version_ids),
                ):
                    document = self._document_row(connection, document_row)
                    version_id = str(document_row["version_id"])
                    owning_asset = assets_by_version.get(version_id)
                    owning_source = sources.get(str((owning_asset or {}).get("primary_source_id") or ""))
                    if identity is None or (
                        owning_asset and owning_source
                        and self._asset_visible(document, identity, connection)
                        and self._asset_visible(owning_asset, identity, connection)
                        and self._asset_visible(owning_source, identity, connection)
                    ):
                        documents_by_version[version_id].append(document)

                projections_by_version: Dict[str, Dict[str, Dict[str, Any]]] = {
                    version_id: {} for version_id in version_ids
                }
                for projection_row in connection.execute(
                    f"SELECT * FROM asset_projection_states WHERE version_id IN ({placeholders}) ORDER BY version_id,projection_type",
                    tuple(version_ids),
                ):
                    projections_by_version[str(projection_row["version_id"])][str(projection_row["projection_type"])] = dict(projection_row)

                for version_row in connection.execute(
                    f"SELECT * FROM asset_versions WHERE version_id IN ({placeholders})",
                    tuple(version_ids),
                ):
                    version = {**dict(version_row), "metadata": _json_load(version_row["metadata_json"], {})}
                    version.pop("metadata_json", None)
                    version_id = str(version_row["version_id"])
                    version["documents"] = documents_by_version.get(version_id, [])
                    version["projections"] = projections_by_version.get(version_id, {})
                    versions[version_id] = version

            for asset in assets:
                current_version = versions.get(str(asset.get("current_version_id") or ""))
                asset["source"] = sources.get(str(asset.get("primary_source_id") or ""))
                asset["current_version"] = current_version
                asset["projections"] = (current_version or {}).get("projections", {})
                asset["documents"] = (current_version or {}).get("documents", [])
            return assets

    def list_asset_versions(self, asset_id: str, identity: Any = None) -> List[Dict[str, Any]]:
        asset = self.get_asset(asset_id, identity=identity)
        if not asset:
            return []
        source = self.get_source(str(asset["primary_source_id"]), identity=identity)
        if not source:
            return []
        with self.database.transaction() as connection:
            return [
                self._version_dict(connection, row, identity=identity, asset=asset, source=source)
                for row in connection.execute(
                "SELECT * FROM asset_versions WHERE asset_id=? ORDER BY version_no DESC", (asset["asset_id"],)
                )
            ]

    def get_asset_version(
        self, asset_id: str, version_id: str, identity: Any = None,
    ) -> Optional[Dict[str, Any]]:
        """Return one historical version after source, asset and document ACL intersection."""
        asset = self.get_asset(asset_id, identity=identity)
        if not asset:
            return None
        source = self.get_source(str(asset["primary_source_id"]), identity=identity)
        if not source:
            return None
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM asset_versions WHERE version_id=? AND asset_id=? AND project_id=?",
                (str(version_id or ""), asset["asset_id"], self.project_id),
            ).fetchone()
            if not row:
                return None
            return self._version_dict(
                connection, row, identity=identity, asset=asset, source=source,
            )

    def summarize_asset_processing(self, asset_ids: Sequence[str]) -> Dict[str, int]:
        """Return lifecycle/projection counters for an already-authorized asset set."""
        normalized = sorted({str(item) for item in asset_ids if str(item)})
        if not normalized:
            return {
                "failed_version_count": 0,
                "repair_required_count": 0,
                "processing_count": 0,
            }
        placeholders = ",".join("?" for _ in normalized)
        with self.database.transaction() as connection:
            failed_versions = int(connection.execute(
                f"SELECT COUNT(*) FROM asset_versions WHERE asset_id IN ({placeholders}) AND status='failed'",
                tuple(normalized),
            ).fetchone()[0])
            repair_required = int(connection.execute(
                f"""
                SELECT COUNT(DISTINCT ka.asset_id)
                FROM knowledge_assets ka
                JOIN asset_projection_states aps ON aps.version_id=ka.current_version_id
                WHERE ka.asset_id IN ({placeholders}) AND aps.status='repair_required'
                """,
                tuple(normalized),
            ).fetchone()[0])
            processing_versions = {
                str(row["asset_id"])
                for row in connection.execute(
                    f"""
                    SELECT DISTINCT asset_id FROM asset_versions
                    WHERE asset_id IN ({placeholders}) AND status IN ('preparing','ready')
                    """,
                    tuple(normalized),
                )
            }
            processing_projections = {
                str(row["asset_id"])
                for row in connection.execute(
                    f"""
                    SELECT DISTINCT ka.asset_id
                    FROM knowledge_assets ka
                    JOIN asset_projection_states aps ON aps.version_id=ka.current_version_id
                    WHERE ka.asset_id IN ({placeholders}) AND aps.status IN ('pending','building')
                    """,
                    tuple(normalized),
                )
            }
        return {
            "failed_version_count": failed_versions,
            "repair_required_count": repair_required,
            "processing_count": len(processing_versions | processing_projections),
        }

    def prepare_asset_version(
        self, *, source_type: str, external_key: str, title: str, content_hash: str,
        actor: str, asset_id: str = "", source_id: str = "", categories: Sequence[str] = (),
        applicable_roles: Sequence[str] = (), visibility: str = "project", sensitivity_level: str = "internal",
        access_policy_id: str = "", owner_user_id: str = "", summary: str = "",
        valid_from: str = "", valid_until: str = "", review_due_at: str = "", metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        source_type = str(source_type or "document").strip()
        external_key = str(external_key or title).strip()
        if not external_key or not title:
            raise ValueError("知识来源标识和标题不能为空")
        if visibility not in {"project", "organization", "restricted", "private"}:
            raise ValueError("不支持的可见范围")
        now = utc_now()
        with self.database.transaction(write=True) as connection:
            explicit_asset = None
            if asset_id:
                explicit_asset = connection.execute(
                    "SELECT * FROM knowledge_assets WHERE project_id=? AND asset_id=?",
                    (self.project_id, asset_id),
                ).fetchone()
                if not explicit_asset:
                    alias = connection.execute(
                        "SELECT asset_id FROM knowledge_asset_aliases WHERE project_id=? AND alias_value=? LIMIT 1",
                        (self.project_id, asset_id),
                    ).fetchone()
                    if alias:
                        explicit_asset = connection.execute(
                            "SELECT * FROM knowledge_assets WHERE project_id=? AND asset_id=?",
                            (self.project_id, alias["asset_id"]),
                        ).fetchone()

            source = None
            if explicit_asset:
                source = connection.execute(
                    "SELECT * FROM knowledge_sources WHERE source_id=? AND project_id=?",
                    (explicit_asset["primary_source_id"], self.project_id),
                ).fetchone()
                if source_id and source and str(source["source_id"]) != str(source_id):
                    raise ValueError("指定资产与知识来源不匹配")
            elif source_id:
                source = connection.execute(
                    "SELECT * FROM knowledge_sources WHERE source_id=? AND project_id=?", (source_id, self.project_id)
                ).fetchone()
                if not source:
                    raise ValueError("指定的知识来源不存在")
            if not source:
                source = connection.execute(
                    "SELECT * FROM knowledge_sources WHERE project_id=? AND source_type=? AND external_key=?",
                    (self.project_id, source_type, external_key),
                ).fetchone()

            if source:
                source_id = str(source["source_id"])
                if not self._acl_is_within_source(
                    connection,
                    asset_visibility=visibility,
                    asset_policy_id=access_policy_id,
                    asset_owner_user_id=owner_user_id,
                    source_visibility=str(source["visibility"]),
                    source_policy_id=str(source["access_policy_id"]),
                    source_owner_user_id=str(source["owner_user_id"]),
                ):
                    raise ValueError("知识资产权限范围不得宽于知识来源")
            else:
                source_id = str(source_id or _stable_id(
                    "source", DEFAULT_ORGANIZATION_ID, self.project_id, source_type, external_key,
                ))
                # Permanent deletion keeps the old source ID as an audit tombstone while
                # releasing its external key. Re-uploading the same source must create a
                # fresh identity instead of colliding with that retained primary key.
                while connection.execute(
                    "SELECT 1 FROM knowledge_sources WHERE source_id=?", (source_id,),
                ).fetchone():
                    source_id = _stable_id(
                        "source", DEFAULT_ORGANIZATION_ID, self.project_id,
                        source_type, external_key, uuid.uuid4().hex,
                    )
                connection.execute(
                    """
                    INSERT INTO knowledge_sources(
                        source_id,organization_id,project_id,source_type,external_key,display_name,owner_user_id,
                        visibility,sensitivity_level,access_policy_id,status,metadata_json,created_at,updated_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,'active',?,?,?)
                    """,
                    (source_id, DEFAULT_ORGANIZATION_ID, self.project_id, source_type, external_key, title,
                     owner_user_id, visibility, sensitivity_level, access_policy_id, _json_dump(metadata or {}), now, now),
                )
            asset_key = f"{source_type}:{external_key}"
            existing = explicit_asset or connection.execute(
                "SELECT * FROM knowledge_assets WHERE project_id=? AND asset_key=?", (self.project_id, asset_key)
            ).fetchone()
            if existing:
                asset_id = str(existing["asset_id"])
                if str(existing["status"]) in {"revoked", "deleted"}:
                    raise ValueError("知识资产已撤销或删除，不能直接覆盖发布")
            else:
                asset_id = str(asset_id or _stable_id("asset", self.project_id, asset_key))
                if connection.execute("SELECT 1 FROM knowledge_assets WHERE asset_id=?", (asset_id,)).fetchone():
                    asset_id = _stable_id("asset", self.project_id, asset_key, uuid.uuid4().hex)
                connection.execute(
                    """
                    INSERT INTO knowledge_assets(
                        asset_id,organization_id,project_id,primary_source_id,asset_key,title,summary,owner_user_id,
                        sensitivity_level,visibility,access_policy_id,status,review_due_at,created_by,updated_by,
                        created_at,updated_at,metadata_json
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,'draft',?,?,?,?,?,?)
                    """,
                    (asset_id, DEFAULT_ORGANIZATION_ID, self.project_id, source_id, asset_key, title, summary,
                     owner_user_id, sensitivity_level, visibility, access_policy_id, review_due_at, actor, actor,
                     now, now, _json_dump(metadata or {})),
                )
            publication_metadata = {
                "title": title,
                "summary": summary,
                "owner_user_id": owner_user_id,
                "visibility": visibility,
                "sensitivity_level": sensitivity_level,
                "access_policy_id": access_policy_id,
                "review_due_at": review_due_at,
                "categories": list(dict.fromkeys(str(item).strip() for item in categories if str(item).strip())),
                "applicable_roles": list(dict.fromkeys(str(item).strip() for item in applicable_roles if str(item).strip())),
                "source": {
                    "display_name": title,
                    "external_key": external_key,
                    "owner_user_id": owner_user_id,
                    "visibility": visibility,
                    "sensitivity_level": sensitivity_level,
                    "access_policy_id": access_policy_id,
                    "metadata": metadata or {},
                },
            }
            duplicate = connection.execute(
                """
                SELECT av.* FROM asset_versions av
                WHERE av.asset_id=? AND av.content_hash=? AND av.status IN ('preparing','ready','active')
                AND (
                    av.status<>'active' OR EXISTS(
                        SELECT 1 FROM asset_version_documents avd
                        JOIN document_chunks ch ON ch.document_id=avd.document_id
                        WHERE avd.version_id=av.version_id
                    )
                )
                ORDER BY av.version_no DESC LIMIT 1
                """,
                (asset_id, content_hash),
            ).fetchone()
            if duplicate:
                duplicate_metadata = _json_load(duplicate["metadata_json"], {})
                if duplicate_metadata.get("publication_metadata") == publication_metadata:
                    return {"asset_id": asset_id, "source_id": source_id, "version_id": str(duplicate["version_id"]),
                            "version_no": int(duplicate["version_no"]), "status": str(duplicate["status"]), "duplicate": True}
            next_no = int(connection.execute(
                "SELECT COALESCE(MAX(version_no),0)+1 FROM asset_versions WHERE asset_id=?", (asset_id,)
            ).fetchone()[0])
            previous = str(existing["current_version_id"] or "") if existing else ""
            version_id = _stable_id("version", asset_id, next_no, content_hash or uuid.uuid4().hex)
            connection.execute(
                """
                INSERT INTO asset_versions(
                    version_id,asset_id,project_id,version_no,status,content_hash,summary,supersedes_version_id,
                    valid_from,valid_until,review_due_at,created_by,metadata_json,created_at,updated_at
                ) VALUES(?,?,?,?,'preparing',?,?,?,?,?,?,?,?,?,?)
                """,
                (version_id, asset_id, self.project_id, next_no, content_hash, summary, previous, valid_from,
                 valid_until, review_due_at, actor, _json_dump({
                     **(metadata or {}), "publication_metadata": publication_metadata,
                 }), now, now),
            )
            for projection in ("vector", "graph", "context", "readiness"):
                connection.execute(
                    "INSERT INTO asset_projection_states(version_id,projection_type,status,updated_at) VALUES(?,?, 'pending',?)",
                    (version_id, projection, now),
                )
            # Existing online metadata remains untouched until the ready version
            # becomes active in the publication transaction.
            for alias_type, alias in (("external_key", external_key), (source_type, external_key)):
                connection.execute(
                    "INSERT OR REPLACE INTO knowledge_asset_aliases(project_id,alias_type,alias_value,asset_id,created_at) VALUES(?,?,?,?,?)",
                    (self.project_id, alias_type, alias, asset_id, now),
                )
            self._write_audit_connection(
                connection, f"audit_{uuid.uuid4().hex}", actor, "knowledge.version_prepared", "knowledge_asset",
                asset_id, {"version_id": version_id, "version_no": next_no, "source_type": source_type},
                project_id=self.project_id, created_at=now,
            )
            return {"asset_id": asset_id, "source_id": source_id, "version_id": version_id,
                    "version_no": next_no, "status": "preparing", "duplicate": False}

    def set_asset_projection(self, version_id: str, projection_type: str, status: str, error: str = "") -> Dict[str, Any]:
        if projection_type not in {"vector", "graph", "context", "readiness"}:
            raise ValueError("不支持的知识投影类型")
        if status not in {"pending", "building", "ready", "repair_required", "obsolete"}:
            raise ValueError("不支持的知识投影状态")
        with self.database.transaction(write=True) as connection:
            cursor = connection.execute(
                """
                UPDATE asset_projection_states SET status=?,applied_revision=CASE WHEN ?='ready' THEN desired_revision ELSE applied_revision END,
                    attempts=attempts+CASE WHEN ?='building' THEN 1 ELSE 0 END,
                    last_error=?,updated_at=? WHERE version_id=? AND projection_type=?
                """,
                (status, status, status, str(error or "")[:1000], utc_now(), version_id, projection_type),
            )
            if not cursor.rowcount:
                raise ValueError("知识资产版本不存在")
        return {"version_id": version_id, "projection_type": projection_type, "status": status}

    def asset_projection_status(self, asset_id: str, identity: Any = None) -> Optional[Dict[str, Any]]:
        detail = self.get_asset_detail(asset_id, identity=identity)
        if not detail:
            return None
        projections = detail.get("projections") or {}
        return {
            "asset_id": detail["asset_id"],
            "version_id": str(detail.get("current_version_id") or ""),
            "status": "healthy" if projections and all(
                item.get("status") == "ready" for item in projections.values()
            ) else "repair_required",
            "projections": projections,
        }

    def current_projection_targets(self, asset_id: str = "", *, include_ready: bool = False) -> List[Dict[str, Any]]:
        params: List[Any] = [self.project_id]
        scope = ""
        if asset_id:
            resolved = self.resolve_asset_id(asset_id)
            if not resolved:
                return []
            scope = " AND ka.asset_id=?"
            params.append(resolved)
        status_scope = "" if include_ready else " AND aps.status<>'ready'"
        with self.database.transaction() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT ka.asset_id,ka.current_version_id AS version_id,aps.projection_type,
                           aps.status,aps.desired_revision,aps.applied_revision,aps.attempts,aps.last_error
                    FROM knowledge_assets ka
                    JOIN asset_versions av ON av.version_id=ka.current_version_id AND av.status='active'
                    JOIN asset_projection_states aps ON aps.version_id=av.version_id
                    WHERE ka.project_id=? AND ka.status IN ('active','review_due')
                    """ + scope + status_scope + " ORDER BY aps.projection_type,ka.asset_id",
                    tuple(params),
                )
            ]

    def count_current_projection_targets(self, asset_id: str = "") -> int:
        """Count current active projection states that still need processing."""
        params: List[Any] = [self.project_id]
        scope = ""
        if asset_id:
            resolved = self.resolve_asset_id(asset_id)
            if not resolved:
                return 0
            scope = " AND ka.asset_id=?"
            params.append(resolved)
        with self.database.transaction() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM knowledge_assets ka
                JOIN asset_versions av ON av.version_id=ka.current_version_id AND av.status='active'
                JOIN asset_projection_states aps ON aps.version_id=av.version_id
                WHERE ka.project_id=? AND ka.status IN ('active','review_due')
                  AND aps.status<>'ready'
                """ + scope,
                tuple(params),
            ).fetchone()
        return int(row["count"] if row else 0)

    def readiness_assets(self, identity: Any = None) -> List[Dict[str, Any]]:
        """Adapt authoritative current assets to the readiness calculator contract."""
        assets = []
        for asset in self.list_assets(identity=identity):
            detail = self.get_asset_detail(str(asset["asset_id"]), identity=identity)
            if not detail or not detail.get("current_version"):
                continue
            version = detail["current_version"]
            documents = version.get("documents") or []
            source = detail.get("source") or {}
            vector_state = (version.get("projections") or {}).get("vector", {})
            refs = [
                str(document.get("source_file") or document.get("stored_file") or "")
                for document in documents
                if document.get("source_file") or document.get("stored_file")
            ]
            assets.append({
                "asset_id": detail["asset_id"],
                "version_id": version["version_id"],
                "title": detail.get("title") or "",
                "status": detail.get("status") or "",
                "asset_status": detail.get("status") or "",
                "version_status": version.get("status") or "",
                "is_current_version": True,
                "valid_from": version.get("valid_from") or "",
                "valid_until": version.get("valid_until") or "",
                "review_due_at": version.get("review_due_at") or detail.get("review_due_at") or "",
                "authority_score": float(detail.get("authority_level") or 0) / 100.0,
                "topics": list(detail.get("categories") or []),
                "categories": list(detail.get("categories") or []),
                "applicable_roles": [
                    str(item.get("role") or "") if isinstance(item, dict) else str(item)
                    for item in detail.get("applicable_roles") or []
                ],
                "source_id": source.get("source_id") or detail.get("primary_source_id") or "",
                "source_refs": refs,
                "source_file": refs[0] if refs else source.get("display_name") or "",
                "projection_status": vector_state.get("status") or "pending_refresh",
            })
        return assets

    def mark_asset_version_ready(self, version_id: str) -> Dict[str, Any]:
        with self.database.transaction(write=True) as connection:
            version = connection.execute("SELECT * FROM asset_versions WHERE version_id=?", (version_id,)).fetchone()
            if not version:
                raise ValueError("知识资产版本不存在")
            projections = {str(row["projection_type"]): str(row["status"]) for row in connection.execute(
                "SELECT projection_type,status FROM asset_projection_states WHERE version_id=?", (version_id,)
            )}
            if projections.get("vector") != "ready":
                raise ValueError("向量索引未成功，版本不能发布")
            blocked = [name for name in ("graph", "context", "readiness") if projections.get(name) not in {"ready", "repair_required"}]
            if blocked:
                raise ValueError("知识投影尚未完成: " + "、".join(blocked))
            if str(version["status"]) == "preparing":
                connection.execute("UPDATE asset_versions SET status='ready',updated_at=? WHERE version_id=?", (utc_now(), version_id))
            elif str(version["status"]) not in {"ready", "active"}:
                raise ValueError("当前版本状态不能进入待发布")
        return {"version_id": version_id, "status": "ready"}

    def publish_asset_version(
        self, asset_id: str, version_id: str, actor: str,
        expected_revision: Optional[int] = None, *,
        _connection: Optional[sqlite3.Connection] = None,
        _trigger: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        now = utc_now()
        transaction = nullcontext(_connection) if _connection is not None else self.database.transaction(write=True)
        with transaction as connection:
            asset = connection.execute("SELECT * FROM knowledge_assets WHERE asset_id=? AND project_id=?", (asset_id, self.project_id)).fetchone()
            version = connection.execute("SELECT * FROM asset_versions WHERE version_id=? AND asset_id=?", (version_id, asset_id)).fetchone()
            if not asset or not version:
                raise ValueError("知识资产或版本不存在")
            if expected_revision is not None and int(asset["lifecycle_revision"]) != int(expected_revision):
                raise ValueError("知识资产已被其他操作更新，请刷新后重试")
            if str(asset["current_version_id"]) == version_id and str(version["status"]) == "active":
                return self._asset_dict(connection, asset)
            if str(asset["status"]) not in {"draft", "pending_review", "active", "review_due"}:
                raise ValueError(f"资产状态 {asset['status']} 不允许执行 publish")
            if str(version["status"]) != "ready":
                raise ValueError("版本尚未达到可发布状态")
            vector = connection.execute(
                "SELECT status FROM asset_projection_states WHERE version_id=? AND projection_type='vector'", (version_id,)
            ).fetchone()
            if not vector or str(vector["status"]) != "ready":
                raise ValueError("向量索引未成功，版本不能发布")
            version_metadata = _json_load(version["metadata_json"], {})
            publication = dict(version_metadata.get("publication_metadata") or {})
            source_update = dict(publication.get("source") or {})
            target_visibility = str(publication.get("visibility") or asset["visibility"] or "project")
            target_policy = str(publication.get("access_policy_id") or "")
            target_owner = str(publication.get("owner_user_id") or "")
            source = connection.execute(
                "SELECT * FROM knowledge_sources WHERE source_id=? AND project_id=?",
                (asset["primary_source_id"], self.project_id),
            ).fetchone()
            if not source:
                raise ValueError("知识来源不存在")
            if not self._acl_is_within_source(
                connection,
                asset_visibility=target_visibility,
                asset_policy_id=target_policy,
                asset_owner_user_id=target_owner,
                source_visibility=str(source["visibility"]),
                source_policy_id=str(source["access_policy_id"]),
                source_owner_user_id=str(source["owner_user_id"]),
            ):
                raise ValueError("知识资产权限范围不得宽于知识来源")
            if target_visibility == "private" and not target_owner:
                raise ValueError("私有知识资产必须指定负责人")
            if target_visibility == "restricted" and not target_policy:
                raise ValueError("受限知识资产必须指定访问策略")
            old_version = str(asset["current_version_id"] or "")
            if old_version:
                connection.execute("UPDATE asset_versions SET status='superseded',updated_at=? WHERE version_id=? AND status='active'", (now, old_version))
                connection.execute(
                    "UPDATE documents SET status='superseded',updated_at=? WHERE document_id IN (SELECT document_id FROM asset_version_documents WHERE version_id=?)",
                    (now, old_version),
                )
                connection.execute("UPDATE asset_projection_states SET status='obsolete',updated_at=? WHERE version_id=?", (now, old_version))
            event_payload = {
                "asset_id": asset_id, "version_id": version_id,
                "superseded_version_id": old_version,
            }
            if _trigger:
                event_payload.update(_trigger)
            connection.execute(
                "UPDATE asset_versions SET status='active',published_by=?,published_at=?,updated_at=? WHERE version_id=?",
                (actor, now, now, version_id),
            )
            connection.execute(
                "UPDATE documents SET status='active',valid_from=CASE WHEN valid_from='' THEN ? ELSE valid_from END,updated_at=? WHERE document_id IN (SELECT document_id FROM asset_version_documents WHERE version_id=?)",
                (now, now, version_id),
            )
            revision = int(asset["lifecycle_revision"] or 0) + 1
            review_due = str(publication.get("review_due_at") or version["review_due_at"] or asset["review_due_at"] or "")
            source_metadata = source_update.get("metadata")
            connection.execute(
                """
                UPDATE knowledge_sources SET display_name=?,external_key=?,owner_user_id=?,visibility=?,
                    sensitivity_level=?,access_policy_id=?,status='active',metadata_json=?,updated_at=?
                WHERE source_id=? AND project_id=?
                """,
                (
                    str(source_update.get("display_name") or publication.get("title") or asset["title"]),
                    str(source_update.get("external_key") or ""),
                    target_owner, target_visibility,
                    str(publication.get("sensitivity_level") or asset["sensitivity_level"]), target_policy,
                    _json_dump(source_metadata if isinstance(source_metadata, dict) else {}), now,
                    asset["primary_source_id"], self.project_id,
                ),
            )
            connection.execute(
                """
                UPDATE knowledge_assets SET status='active',current_version_id=?,title=?,summary=?,owner_user_id=?,
                    sensitivity_level=?,visibility=?,access_policy_id=?,review_due_at=?,published_at=?,
                    revoked_at='',lifecycle_revision=?,updated_by=?,updated_at=? WHERE asset_id=?
                """,
                (
                    version_id, str(publication.get("title") or asset["title"]),
                    str(publication.get("summary") or version["summary"] or asset["summary"]), target_owner,
                    str(publication.get("sensitivity_level") or asset["sensitivity_level"]), target_visibility,
                    target_policy, review_due, now, revision, actor, now, asset_id,
                ),
            )
            permission_values = (
                target_owner, str(publication.get("sensitivity_level") or asset["sensitivity_level"]),
                target_visibility, target_policy, now, version_id,
            )
            connection.execute(
                """
                UPDATE documents SET owner_user_id=?,sensitivity_level=?,visibility=?,access_policy_id=?,
                    acl_revision=acl_revision+1,updated_at=?
                WHERE document_id IN (SELECT document_id FROM asset_version_documents WHERE version_id=?)
                """,
                permission_values,
            )
            connection.execute("DELETE FROM asset_categories WHERE asset_id=?", (asset_id,))
            for index, name in enumerate(publication.get("categories") or []):
                category_id = self._ensure_category(connection, str(name), index)
                connection.execute(
                    "INSERT INTO asset_categories(asset_id,category_id,is_primary,created_at) VALUES(?,?,?,?)",
                    (asset_id, category_id, int(index == 0), now),
                )
            connection.execute("DELETE FROM asset_applicable_roles WHERE asset_id=?", (asset_id,))
            for role in publication.get("applicable_roles") or []:
                connection.execute(
                    "INSERT INTO asset_applicable_roles(asset_id,role_key,created_at) VALUES(?,?,?)",
                    (asset_id, str(role), now),
                )
            event_id = _stable_id("event", "knowledge_asset", asset_id, revision, "activated")
            connection.execute(
                """
                INSERT OR IGNORE INTO domain_outbox(
                    event_id,project_id,aggregate_type,aggregate_id,aggregate_revision,event_type,
                    payload_json,available_at,created_at,updated_at
                ) VALUES(?,?,?,?,?,'knowledge.asset_activated',?,?,?,?)
                """,
                (event_id, self.project_id, "knowledge_asset", asset_id, revision,
                 _json_dump(event_payload), now, now, now),
            )
            audit_detail = {
                "version_id": version_id, "superseded_version_id": old_version,
                "revision": revision,
            }
            if _trigger:
                audit_detail.update(_trigger)
            self._write_audit_connection(
                connection, f"audit_{uuid.uuid4().hex}", actor, "knowledge.asset_activated", "knowledge_asset",
                asset_id, audit_detail,
                project_id=self.project_id, created_at=now,
            )
            row = connection.execute("SELECT * FROM knowledge_assets WHERE asset_id=?", (asset_id,)).fetchone()
            return self._asset_dict(connection, row)

    def fail_asset_version(self, version_id: str, reason: str, actor: str = "system") -> None:
        now = utc_now()
        with self.database.transaction(write=True) as connection:
            row = connection.execute("SELECT asset_id,status FROM asset_versions WHERE version_id=?", (version_id,)).fetchone()
            if not row or str(row["status"]) in {"active", "superseded", "revoked"}:
                return
            connection.execute(
                "UPDATE asset_versions SET status='failed',failure_reason=?,updated_at=? WHERE version_id=?",
                (str(reason or "")[:1000], now, version_id),
            )
            connection.execute(
                "UPDATE documents SET status='failed',updated_at=? WHERE document_id IN (SELECT document_id FROM asset_version_documents WHERE version_id=?)",
                (now, version_id),
            )
            self._write_audit_connection(
                connection, f"audit_{uuid.uuid4().hex}", actor, "knowledge.version_failed", "knowledge_asset",
                str(row["asset_id"]), {"version_id": version_id, "reason": str(reason or "")[:500]},
                project_id=self.project_id, created_at=now,
            )

    def patch_asset_metadata(self, asset_id: str, changes: Dict[str, Any], actor: str) -> Dict[str, Any]:
        allowed = {"title", "topic", "summary", "owner_user_id", "authority_level", "sensitivity_level",
                   "visibility", "access_policy_id", "review_due_at"}
        unknown = sorted(set(changes) - allowed)
        if unknown:
            raise ValueError("不支持的资产字段: " + "、".join(unknown))
        updates = dict(changes)
        if not updates:
            raise ValueError("没有可更新的资产字段")
        if "visibility" in updates and updates["visibility"] not in {"project", "organization", "restricted", "private"}:
            raise ValueError("不支持的可见范围")
        with self.database.transaction(write=True) as connection:
            current = connection.execute(
                "SELECT * FROM knowledge_assets WHERE asset_id=? AND project_id=?", (asset_id, self.project_id)
            ).fetchone()
            if not current:
                raise ValueError("知识资产不存在")
            source = connection.execute(
                "SELECT * FROM knowledge_sources WHERE source_id=?", (current["primary_source_id"],)
            ).fetchone()
            if not source:
                raise ValueError("知识来源不存在")
            target_visibility = str(updates.get("visibility", current["visibility"]))
            target_policy = str(updates.get("access_policy_id", current["access_policy_id"]))
            target_owner = str(updates.get("owner_user_id", current["owner_user_id"]))
            if not self._acl_is_within_source(
                connection,
                asset_visibility=target_visibility,
                asset_policy_id=target_policy,
                asset_owner_user_id=target_owner,
                source_visibility=str(source["visibility"]),
                source_policy_id=str(source["access_policy_id"]),
                source_owner_user_id=str(source["owner_user_id"]),
            ):
                raise ValueError("知识资产权限范围不得宽于知识来源")
            assignments = [f"{key}=?" for key in updates]
            params = list(updates.values()) + [actor, utc_now(), asset_id, self.project_id]
            cursor = connection.execute(
                f"UPDATE knowledge_assets SET {','.join(assignments)},lifecycle_revision=lifecycle_revision+1,updated_by=?,updated_at=? WHERE asset_id=? AND project_id=?",
                tuple(params),
            )
            row = connection.execute("SELECT * FROM knowledge_assets WHERE asset_id=?", (asset_id,)).fetchone()
            permission_changes = {
                key: updates[key] for key in (
                    "owner_user_id", "sensitivity_level", "visibility", "access_policy_id",
                ) if key in updates
            }
            if permission_changes:
                document_assignments = [f"{key}=?" for key in permission_changes]
                document_params = list(permission_changes.values()) + [utc_now(), asset_id]
                connection.execute(
                    f"""
                    UPDATE documents SET {','.join(document_assignments)},acl_revision=acl_revision+1,updated_at=?
                    WHERE document_id IN (
                        SELECT avd.document_id FROM asset_version_documents avd
                        JOIN asset_versions av ON av.version_id=avd.version_id WHERE av.asset_id=?
                    )
                    """,
                    tuple(document_params),
                )
            current_version = str(row["current_version_id"] or "")
            if current_version:
                connection.execute(
                    """
                    UPDATE asset_projection_states SET status='repair_required',
                        desired_revision=desired_revision+1,last_error='资产元数据已更新',updated_at=?
                    WHERE version_id=? AND status<>'obsolete'
                    """,
                    (utc_now(), current_version),
                )
            self._write_audit_connection(
                connection, f"audit_{uuid.uuid4().hex}", actor, "knowledge.asset_metadata_updated", "knowledge_asset",
                asset_id, {"fields": sorted(updates)}, project_id=self.project_id,
            )
            return self._asset_dict(connection, row)

    def transition_asset(self, asset_id: str, action: str, actor: str, reason: str = "", expected_revision: Optional[int] = None) -> Dict[str, Any]:
        action = str(action or "").strip().lower()
        action = {"request_review": "submit_review"}.get(action, action)
        transitions = {
            "submit_review": ({"draft"}, "pending_review"),
            "return_to_draft": ({"pending_review"}, "draft"),
            "mark_review_due": ({"active"}, "review_due"),
            "confirm_valid": ({"review_due"}, "active"),
            "expire": ({"active", "review_due"}, "expired"),
            "revoke": ({"draft", "pending_review", "active", "review_due", "expired"}, "revoked"),
            # Legacy hard-delete remains explicit for existing document APIs.
            "delete": ({"draft", "pending_review", "revoked", "expired"}, "deleted"),
        }
        if action not in transitions:
            raise ValueError("不支持的生命周期操作")
        if action == "revoke" and not str(reason or "").strip():
            raise ValueError("撤销原因不能为空")
        now = utc_now()
        with self.database.transaction(write=True) as connection:
            asset = connection.execute("SELECT * FROM knowledge_assets WHERE asset_id=? AND project_id=?", (asset_id, self.project_id)).fetchone()
            if not asset:
                raise ValueError("知识资产不存在")
            if expected_revision is not None and int(asset["lifecycle_revision"]) != int(expected_revision):
                raise ValueError("知识资产已被其他操作更新，请刷新后重试")
            allowed_sources, target = transitions[action]
            if str(asset["status"]) == target:
                return self._asset_dict(connection, asset)
            if str(asset["status"]) not in allowed_sources:
                raise ValueError(f"资产状态 {asset['status']} 不允许执行 {action}")
            if action == "submit_review":
                ready_version = connection.execute(
                    """
                    SELECT version_id FROM asset_versions
                    WHERE project_id=? AND asset_id=? AND status='ready'
                    ORDER BY version_no DESC LIMIT 1
                    """,
                    (self.project_id, asset_id),
                ).fetchone()
                if not ready_version:
                    raise ValueError("没有可审核的 ready 版本，请先完成版本处理")
            revision = int(asset["lifecycle_revision"] or 0) + 1
            current_version = str(asset["current_version_id"] or "")
            if action in {"revoke", "delete", "expire"}:
                # Authoritative state changes first; every retrieval path consults this state.
                if current_version:
                    connection.execute(
                        "UPDATE asset_versions SET status='revoked',updated_at=? WHERE version_id=? AND status='active'",
                        (now, current_version),
                    )
                    connection.execute(
                        "UPDATE documents SET status=?,updated_at=? WHERE document_id IN (SELECT document_id FROM asset_version_documents WHERE version_id=?)",
                        ("deleted" if action == "delete" else "revoked", now, current_version),
                    )
                    connection.execute(
                        "UPDATE asset_projection_states SET status='obsolete',updated_at=? WHERE version_id=?",
                        (now, current_version),
                    )
                connection.execute(
                    "UPDATE knowledge_sources SET status=?,updated_at=? WHERE source_id=?",
                    ("deleted" if action == "delete" else "revoked", now, asset["primary_source_id"]),
                )
            connection.execute(
                """
                UPDATE knowledge_assets SET status=?,revoked_at=?,review_due_at=?,lifecycle_revision=?,
                    updated_by=?,updated_at=? WHERE asset_id=?
                """,
                (
                    target, now if action in {"revoke", "delete"} else str(asset["revoked_at"] or ""),
                    (
                        "" if action == "confirm_valid"
                        else now if action == "mark_review_due"
                        else str(asset["review_due_at"] or "")
                    ),
                    revision, actor, now, asset_id,
                ),
            )
            if current_version and action in {"mark_review_due", "confirm_valid"}:
                connection.execute(
                    """
                    UPDATE asset_projection_states SET status='repair_required',
                        desired_revision=desired_revision+1,last_error='知识有效性状态已更新',updated_at=?
                    WHERE version_id=? AND projection_type='readiness' AND status<>'obsolete'
                    """,
                    (now, current_version),
                )
            event_type = f"knowledge.asset_{target}"
            event_id = _stable_id("event", "knowledge_asset", asset_id, revision, event_type)
            connection.execute(
                """
                INSERT OR IGNORE INTO domain_outbox(
                    event_id,project_id,aggregate_type,aggregate_id,aggregate_revision,event_type,
                    payload_json,available_at,created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?)
                """,
                (event_id, self.project_id, "knowledge_asset", asset_id, revision, event_type,
                 _json_dump({"asset_id": asset_id, "version_id": current_version, "reason": reason}), now, now, now),
            )
            self._write_audit_connection(
                connection, f"audit_{uuid.uuid4().hex}", actor, event_type, "knowledge_asset", asset_id,
                {"from": str(asset["status"]), "to": target, "reason": reason, "revision": revision},
                project_id=self.project_id, created_at=now,
            )
            row = connection.execute("SELECT * FROM knowledge_assets WHERE asset_id=?", (asset_id,)).fetchone()
            return self._asset_dict(connection, row)

    # Identity and access ------------------------------------------------
    def bind_external_identity(
        self, provider: str, external_user_id: str, user_id: str, verified_email: str = "",
    ) -> Dict[str, Any]:
        provider = str(provider or "").strip().lower()
        external_user_id = str(external_user_id or "").strip()
        if not provider or not external_user_id:
            raise ValueError("外部身份提供方和用户标识不能为空")
        now = utc_now()
        with self.database.transaction(write=True) as connection:
            user = connection.execute(
                "SELECT * FROM users WHERE user_id=? AND status IN ('active','invited')", (user_id,)
            ).fetchone()
            membership = connection.execute(
                "SELECT * FROM project_memberships WHERE project_id=? AND user_id=? AND status='active'",
                (self.project_id, user_id),
            ).fetchone()
            if not user or not membership:
                raise ValueError("只能绑定当前项目的有效成员")
            conflict = connection.execute(
                "SELECT user_id FROM user_external_identities WHERE provider=? AND external_user_id=?",
                (provider, external_user_id),
            ).fetchone()
            if conflict and str(conflict["user_id"]) != user_id:
                raise ValueError("该外部账号已绑定其他成员")
            connection.execute(
                """
                INSERT INTO user_external_identities(
                    provider,external_user_id,user_id,organization_id,verified_email,created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?)
                ON CONFLICT(provider,external_user_id) DO UPDATE SET
                    user_id=excluded.user_id,organization_id=excluded.organization_id,
                    verified_email=excluded.verified_email,updated_at=excluded.updated_at
                """,
                (
                    provider, external_user_id, user_id, str(user["organization_id"]),
                    str(verified_email or user["email"] or "").strip().lower(), now, now,
                ),
            )
            self._write_audit_connection(
                connection, f"audit_{uuid.uuid4().hex}", user_id,
                "security.external_identity_bound", "user", user_id,
                {"provider": provider, "external_user_id": external_user_id},
                project_id=self.project_id, created_at=now,
            )
        return self.resolve_external_identity(provider, external_user_id) or {}

    def resolve_external_identity(
        self, provider: str, external_user_id: str, *, verified_email: str = "",
    ) -> Optional[Dict[str, Any]]:
        provider = str(provider or "").strip().lower()
        external_user_id = str(external_user_id or "").strip()
        if not provider or not external_user_id:
            return None
        with self.database.transaction() as connection:
            row = connection.execute(
                """
                SELECT u.user_id,u.email,u.display_name,u.organization_id,u.status,
                       m.project_id,m.role,m.status AS membership_status,p.name AS project_name
                FROM user_external_identities x
                JOIN users u ON u.user_id=x.user_id
                JOIN project_memberships m ON m.user_id=u.user_id AND m.project_id=?
                JOIN projects p ON p.project_id=m.project_id
                WHERE x.provider=? AND x.external_user_id=?
                  AND u.status IN ('active','invited') AND m.status='active' AND p.status='active'
                """,
                (self.project_id, provider, external_user_id),
            ).fetchone()
        if row:
            return dict(row)
        email = str(verified_email or "").strip().lower()
        if not email:
            return None
        user = self.get_user_by_email(email)
        if not user:
            return None
        with self.database.transaction() as connection:
            membership = connection.execute(
                "SELECT 1 FROM project_memberships WHERE project_id=? AND user_id=? AND status='active'",
                (self.project_id, user["user_id"]),
            ).fetchone()
        if not membership or str(user.get("status") or "") not in {"active", "invited"}:
            return None
        return self.bind_external_identity(provider, external_user_id, str(user["user_id"]), email)

    def identity_can_access_scope(
        self, identity: Any, *, visibility: str, access_policy_id: str = "", owner_user_id: str = "",
    ) -> bool:
        return self._asset_visible({
            "project_id": self.project_id,
            "visibility": visibility,
            "access_policy_id": access_policy_id,
            "owner_user_id": owner_user_id,
        }, identity)

    def provision_project_member(
        self,
        email: str,
        display_name: str,
        role: str,
        *,
        status: str = "invited",
        granted_by: str = "user_system",
        duty: str = "",
    ) -> Dict[str, Any]:
        """Create or update a user and grant the least-privilege project role."""
        normalized = str(email or "").strip().lower()
        if not normalized or "@" not in normalized:
            raise ValueError("邮箱格式不正确")
        normalized_duty = str(duty or "").strip()
        user_id = _stable_id("user", normalized)
        now = utc_now()
        with self.database.transaction(write=True) as connection:
            existing = connection.execute(
                "SELECT user_id,status,password_hash FROM users WHERE email_normalized=?",
                (normalized,),
            ).fetchone()
            if existing:
                user_id = str(existing["user_id"])
                effective_status = "active" if existing["password_hash"] else status
                connection.execute(
                    """
                    UPDATE users SET display_name=?,email=?,organization_id=?,user_type='human',
                        duty=?,status=?,updated_at=? WHERE user_id=?
                    """,
                    (display_name.strip(), normalized, DEFAULT_ORGANIZATION_ID, normalized_duty,
                     effective_status, now, user_id),
                )
            else:
                connection.execute(
                    """
                    INSERT INTO users(
                        user_id,display_name,user_type,created_at,updated_at,organization_id,
                        email,email_normalized,status,duty
                    ) VALUES(?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        user_id, display_name.strip(), "human", now, now, DEFAULT_ORGANIZATION_ID,
                        normalized, normalized, status, normalized_duty,
                    ),
                )
            connection.execute(
                """
                INSERT INTO project_memberships(
                    project_id,user_id,role,created_at,status,granted_by,updated_at
                ) VALUES(?,?,?,?,?,?,?)
                ON CONFLICT(project_id,user_id) DO UPDATE SET
                    role=excluded.role,status='active',granted_by=excluded.granted_by,
                    updated_at=excluded.updated_at
                """,
                (self.project_id, user_id, role, now, "active", granted_by, now),
            )
            connection.execute(
                """
                UPDATE documents SET owner_user_id=?,updated_at=?
                WHERE project_id=? AND owner_user_id='' AND lower(trim(uploader)) IN (?,?)
                """,
                (user_id, now, self.project_id, normalized, display_name.strip().lower()),
            )
        return self.get_user(user_id, project_id=self.project_id) or {}

    def get_user(self, user_id: str, project_id: str = "") -> Optional[Dict[str, Any]]:
        project_id = project_id or self.project_id
        with self.database.transaction() as connection:
            row = connection.execute(
                """
                SELECT u.*,m.project_id,m.role,m.status AS membership_status,
                       p.name AS project_name,p.status AS project_status
                FROM users u
                LEFT JOIN project_memberships m ON m.user_id=u.user_id AND m.project_id=?
                LEFT JOIN projects p ON p.project_id=m.project_id
                WHERE u.user_id=?
                """,
                (project_id, user_id),
            ).fetchone()
            return dict(row) if row else None

    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        normalized = str(email or "").strip().lower()
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE email_normalized=?",
                (normalized,),
            ).fetchone()
            return dict(row) if row else None

    def set_demo_login_password(self, user_id: str, password_hash: str) -> Dict[str, Any]:
        """Activate one local demo member and invalidate obsolete auth material."""
        now = utc_now()
        with self.database.transaction(write=True) as connection:
            cursor = connection.execute(
                """
                UPDATE users SET password_hash=?,password_changed_at=?,status='active',
                    failed_login_count=0,locked_until='',updated_at=?
                WHERE user_id=? AND user_type='human'
                """,
                (password_hash, now, now, user_id),
            )
            if not cursor.rowcount:
                raise ValueError("演示成员不存在")
            connection.execute(
                "UPDATE user_invitations SET consumed_at=? WHERE user_id=? AND consumed_at=''",
                (now, user_id),
            )
            connection.execute("DELETE FROM auth_sessions WHERE user_id=?", (user_id,))
        return self.get_user(user_id, project_id=self.project_id) or {}

    def list_project_members(self, project_id: str = "") -> List[Dict[str, Any]]:
        project_id = project_id or self.project_id
        with self.database.transaction() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT u.user_id,u.display_name,u.email,u.status,u.last_login_at,u.duty,
                           m.project_id,m.role,m.status AS membership_status,m.created_at,m.updated_at
                    FROM project_memberships m
                    JOIN users u ON u.user_id=m.user_id
                    WHERE m.project_id=? AND u.user_type='human'
                    ORDER BY CASE m.role WHEN 'project_admin' THEN 0 WHEN 'project_manager' THEN 1
                                      WHEN 'knowledge_operator' THEN 2 ELSE 3 END,
                             u.display_name,u.email
                    """,
                    (project_id,),
                )
            ]

    def update_project_membership(
        self, user_id: str, *, role: str = "", status: str = "", duty: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not role and not status and duty is None:
            raise ValueError("至少需要更新角色、成员状态或职能")
        assignments = []
        params: List[Any] = []
        if role:
            assignments.append("role=?")
            params.append(role)
        if status:
            assignments.append("status=?")
            params.append(status)
        assignments.append("updated_at=?")
        params.append(utc_now())
        params.extend((self.project_id, user_id))
        with self.database.transaction(write=True) as connection:
            cursor = connection.execute(
                f"UPDATE project_memberships SET {','.join(assignments)} WHERE project_id=? AND user_id=?",
                tuple(params),
            )
            if not cursor.rowcount:
                raise ValueError("项目成员不存在")
            if duty is not None:
                connection.execute(
                    "UPDATE users SET duty=?,updated_at=? WHERE user_id=?",
                    (str(duty).strip(), utc_now(), user_id),
                )
            if status and status != "active":
                connection.execute(
                    "DELETE FROM auth_sessions WHERE user_id=? AND current_project_id=?",
                    (user_id, self.project_id),
                )
        return self.get_user(user_id, project_id=self.project_id) or {}

    # Onboarding role templates and learning plans ------------------
    @staticmethod
    def _onboarding_template_payload(connection: sqlite3.Connection, row: Any) -> Dict[str, Any]:
        payload = dict(row)
        payload["topics"] = []
        for topic_row in connection.execute(
            "SELECT * FROM onboarding_template_topics WHERE template_id=? ORDER BY position,topic_key",
            (payload["template_id"],),
        ):
            topic = dict(topic_row)
            topic["required_asset_ids"] = _json_load(topic.pop("required_asset_ids_json", "[]"), [])
            payload["topics"].append(topic)
        return payload

    def list_onboarding_templates(self, *, include_inactive: bool = False) -> List[Dict[str, Any]]:
        where = "project_id=?" + ("" if include_inactive else " AND status='active'")
        with self.database.transaction() as connection:
            rows = connection.execute(
                f"SELECT * FROM onboarding_role_templates WHERE {where} ORDER BY name,role_key",
                (self.project_id,),
            ).fetchall()
            return [self._onboarding_template_payload(connection, row) for row in rows]

    def get_onboarding_template(self, template_id: str = "", *, role_key: str = "") -> Optional[Dict[str, Any]]:
        if not template_id and not role_key:
            return None
        with self.database.transaction() as connection:
            if template_id:
                row = connection.execute(
                    "SELECT * FROM onboarding_role_templates WHERE project_id=? AND template_id=?",
                    (self.project_id, template_id),
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT * FROM onboarding_role_templates WHERE project_id=? AND role_key=?",
                    (self.project_id, role_key),
                ).fetchone()
            return self._onboarding_template_payload(connection, row) if row else None

    def upsert_onboarding_template(self, payload: Dict[str, Any], actor: str) -> Dict[str, Any]:
        role_key = str(payload.get("role_key") or payload.get("name") or "").strip()
        name = str(payload.get("name") or role_key).strip()
        topics = payload.get("topics") or []
        if not role_key or not name:
            raise ValueError("岗位标识和岗位名称不能为空")
        if not isinstance(topics, list) or not topics:
            raise ValueError("岗位模板至少需要一个知识主题")
        template_status = str(payload.get("status") or "active").strip().lower()
        if template_status not in {"active", "inactive"}:
            raise ValueError("岗位模板状态无效")
        normalized_topics = []
        seen = set()
        for index, item in enumerate(topics):
            if not isinstance(item, dict):
                raise ValueError("知识主题格式无效")
            topic_key = str(item.get("topic_key") or item.get("topic_id") or "").strip()
            label = str(item.get("label") or topic_key).strip()
            if not topic_key or not label:
                raise ValueError("知识主题标识和名称不能为空")
            if topic_key in seen:
                raise ValueError(f"知识主题重复：{label}")
            seen.add(topic_key)
            try:
                weight = float(item.get("weight") or 1)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{label} 的主题权重无效") from exc
            if weight <= 0 or weight > 100:
                raise ValueError(f"{label} 的主题权重必须大于 0 且不超过 100")
            required_assets = [
                str(value).strip() for value in item.get("required_asset_ids") or [] if str(value).strip()
            ]
            normalized_topics.append({
                "topic_key": topic_key, "label": label, "weight": weight,
                "required_asset_ids": list(dict.fromkeys(required_assets)),
                "practice_task": str(item.get("practice_task") or "").strip(),
                "owner_user_id": str(item.get("owner_user_id") or "").strip(),
                "completion_standard": str(item.get("completion_standard") or "").strip(),
                "position": index,
            })
        now = utc_now()
        requested_id = str(payload.get("template_id") or "").strip()
        with self.database.transaction(write=True) as connection:
            existing = None
            if requested_id:
                existing = connection.execute(
                    "SELECT * FROM onboarding_role_templates WHERE project_id=? AND template_id=?",
                    (self.project_id, requested_id),
                ).fetchone()
            if existing is None:
                existing = connection.execute(
                    "SELECT * FROM onboarding_role_templates WHERE project_id=? AND role_key=?",
                    (self.project_id, role_key),
                ).fetchone()
            template_id = str(existing["template_id"] if existing else requested_id or f"ort_{uuid.uuid4().hex}")
            revision = int(existing["revision"] if existing else 0) + 1
            if existing:
                connection.execute(
                    """
                    UPDATE onboarding_role_templates SET role_key=?,name=?,focus=?,status=?,revision=?,
                        updated_by=?,updated_at=? WHERE template_id=? AND project_id=?
                    """,
                    (
                        role_key, name, str(payload.get("focus") or "").strip(),
                        template_status, revision, actor, now,
                        template_id, self.project_id,
                    ),
                )
                connection.execute("DELETE FROM onboarding_template_topics WHERE template_id=?", (template_id,))
                action = "onboarding.template_updated"
            else:
                connection.execute(
                    """
                    INSERT INTO onboarding_role_templates(
                        template_id,project_id,role_key,name,focus,status,revision,
                        created_by,updated_by,created_at,updated_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        template_id, self.project_id, role_key, name,
                        str(payload.get("focus") or "").strip(), template_status,
                        revision, actor, actor, now, now,
                    ),
                )
                action = "onboarding.template_created"
            for item in normalized_topics:
                topic_id = _stable_id("ort_topic", template_id, item["topic_key"])
                connection.execute(
                    """
                    INSERT INTO onboarding_template_topics(
                        topic_id,template_id,topic_key,label,weight,required_asset_ids_json,
                        practice_task,owner_user_id,completion_standard,position,created_at,updated_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        topic_id, template_id, item["topic_key"], item["label"], item["weight"],
                        _json_dump(item["required_asset_ids"]), item["practice_task"],
                        item["owner_user_id"], item["completion_standard"], item["position"], now, now,
                    ),
                )
            self._write_audit_connection(
                connection, f"audit_{uuid.uuid4().hex}", actor, action,
                "onboarding_template", template_id,
                {"role_key": role_key, "revision": revision, "topic_count": len(normalized_topics)},
                project_id=self.project_id, created_at=now,
            )
            row = connection.execute(
                "SELECT * FROM onboarding_role_templates WHERE template_id=?", (template_id,),
            ).fetchone()
            return self._onboarding_template_payload(connection, row)

    @staticmethod
    def _onboarding_plan_payload(connection: sqlite3.Connection, row: Any) -> Dict[str, Any]:
        payload = dict(row)
        user = connection.execute(
            "SELECT display_name,email FROM users WHERE user_id=?", (payload["user_id"],),
        ).fetchone()
        payload["user_name"] = str(user["display_name"] if user else "")
        payload["user_email"] = str(user["email"] if user else "")
        template = connection.execute(
            "SELECT role_key,name FROM onboarding_role_templates WHERE template_id=?",
            (payload["template_id"],),
        ).fetchone()
        payload["role_key"] = str(template["role_key"] if template else "")
        if template and not payload.get("role_name"):
            payload["role_name"] = str(template["name"] or "")
        payload["items"] = [dict(item) for item in connection.execute(
            "SELECT * FROM onboarding_learning_items WHERE plan_id=? ORDER BY position,item_id",
            (payload["plan_id"],),
        )]
        return payload

    def create_onboarding_plan(
        self, payload: Dict[str, Any], items: List[Dict[str, Any]], actor: str,
    ) -> Dict[str, Any]:
        user_id = str(payload.get("user_id") or "").strip()
        template_id = str(payload.get("template_id") or "").strip()
        if not user_id or not template_id:
            raise ValueError("学习成员和岗位模板不能为空")
        now = utc_now()
        with self.database.transaction(write=True) as connection:
            member = connection.execute(
                """
                SELECT 1 FROM project_memberships m JOIN users u ON u.user_id=m.user_id
                WHERE m.project_id=? AND m.user_id=? AND m.status='active' AND u.status='active'
                """,
                (self.project_id, user_id),
            ).fetchone()
            template = connection.execute(
                "SELECT * FROM onboarding_role_templates WHERE project_id=? AND template_id=? AND status='active'",
                (self.project_id, template_id),
            ).fetchone()
            if not member:
                raise ValueError("学习成员不是当前项目的有效成员")
            if not template:
                raise ValueError("岗位模板不存在或已停用")
            existing = connection.execute(
                """
                SELECT * FROM onboarding_learning_plans
                WHERE project_id=? AND user_id=? AND template_id=? AND status='active'
                """,
                (self.project_id, user_id, template_id),
            ).fetchone()
            if existing:
                result = self._onboarding_plan_payload(connection, existing)
                result["created"] = False
                return result
            plan_id = str(payload.get("plan_id") or f"olp_{uuid.uuid4().hex}")
            connection.execute(
                """
                INSERT INTO onboarding_learning_plans(
                    plan_id,project_id,user_id,template_id,template_revision,role_name,status,
                    target_date,manager_user_id,source_type,source_key,created_by,started_at,
                    created_at,updated_at
                ) VALUES(?,?,?,?,?,?,'active',?,?,?,?,?,?,?,?)
                """,
                (
                    plan_id, self.project_id, user_id, template_id, int(template["revision"]),
                    str(template["name"]), str(payload.get("target_date") or ""),
                    str(payload.get("manager_user_id") or actor), str(payload.get("source_type") or "manual"),
                    str(payload.get("source_key") or ""), actor, now, now, now,
                ),
            )
            for index, item in enumerate(items):
                item_status = str(item.get("status") or "pending").strip().lower()
                if item_status not in {"pending", "completed", "blocked"}:
                    raise ValueError("学习任务初始状态无效")
                connection.execute(
                    """
                    INSERT INTO onboarding_learning_items(
                        item_id,plan_id,topic_key,item_type,title,description,asset_id,version_id,
                        weight,status,position,created_at,updated_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        str(item.get("item_id") or f"oli_{uuid.uuid4().hex}"), plan_id,
                        str(item.get("topic_key") or ""), str(item.get("item_type") or "reading"),
                        str(item.get("title") or "学习任务"), str(item.get("description") or ""),
                        str(item.get("asset_id") or ""), str(item.get("version_id") or ""),
                        max(0.01, float(item.get("weight") or 1)),
                        item_status, index, now, now,
                    ),
                )
            self._write_audit_connection(
                connection, f"audit_{uuid.uuid4().hex}", actor, "onboarding.plan_created",
                "onboarding_plan", plan_id,
                {"user_id": user_id, "template_id": template_id, "item_count": len(items)},
                project_id=self.project_id, created_at=now,
            )
            row = connection.execute(
                "SELECT * FROM onboarding_learning_plans WHERE plan_id=?", (plan_id,),
            ).fetchone()
            result = self._onboarding_plan_payload(connection, row)
            result["created"] = True
            return result

    def add_onboarding_reading_items(
        self, plan_id: str, items: List[Dict[str, Any]], actor: str,
    ) -> Dict[str, Any]:
        """Idempotently attach newly visible handover assets to an active plan."""
        now = utc_now()
        with self.database.transaction(write=True) as connection:
            plan = connection.execute(
                "SELECT * FROM onboarding_learning_plans WHERE project_id=? AND plan_id=?",
                (self.project_id, plan_id),
            ).fetchone()
            if not plan:
                raise ValueError("学习计划不存在")
            if str(plan["status"]) != "active":
                raise ValueError("只能向进行中的学习计划补充交接资料")
            existing_asset_ids = {
                str(row["asset_id"])
                for row in connection.execute(
                    """
                    SELECT asset_id FROM onboarding_learning_items
                    WHERE plan_id=? AND item_type='reading' AND asset_id<>''
                    """,
                    (plan_id,),
                )
            }
            position_row = connection.execute(
                "SELECT COALESCE(MAX(position),-1) AS position FROM onboarding_learning_items WHERE plan_id=?",
                (plan_id,),
            ).fetchone()
            position = int(position_row["position"] if position_row else -1) + 1
            added_asset_ids: List[str] = []
            for item in items:
                asset_id = str(item.get("asset_id") or "").strip()
                if not asset_id or asset_id in existing_asset_ids:
                    continue
                current = connection.execute(
                    """
                    SELECT ka.asset_id,av.version_id
                    FROM knowledge_assets ka
                    JOIN asset_versions av ON av.version_id=ka.current_version_id
                    WHERE ka.project_id=? AND ka.asset_id=?
                      AND ka.status IN ('active','review_due') AND av.status='active'
                    """,
                    (self.project_id, asset_id),
                ).fetchone()
                if not current:
                    continue
                connection.execute(
                    """
                    INSERT INTO onboarding_learning_items(
                        item_id,plan_id,topic_key,item_type,title,description,asset_id,version_id,
                        weight,status,position,created_at,updated_at
                    ) VALUES(?,?,?,'reading',?,?,?,?,?,'pending',?,?,?)
                    """,
                    (
                        str(item.get("item_id") or f"oli_{uuid.uuid4().hex}"), plan_id,
                        str(item.get("topic_key") or "交接文档"),
                        str(item.get("title") or "阅读交接资料"),
                        str(item.get("description") or "阅读并核对本次交接范围。"),
                        asset_id, str(current["version_id"]),
                        max(0.01, float(item.get("weight") or 1)), position, now, now,
                    ),
                )
                existing_asset_ids.add(asset_id)
                added_asset_ids.append(asset_id)
                position += 1
            if added_asset_ids:
                connection.execute(
                    "UPDATE onboarding_learning_plans SET updated_at=? WHERE plan_id=?",
                    (now, plan_id),
                )
                self._write_audit_connection(
                    connection, f"audit_{uuid.uuid4().hex}", actor,
                    "onboarding.handover_scope_linked", "onboarding_plan", plan_id,
                    {"asset_ids": added_asset_ids, "added_count": len(added_asset_ids)},
                    project_id=self.project_id, created_at=now,
                )
            row = connection.execute(
                "SELECT * FROM onboarding_learning_plans WHERE plan_id=?", (plan_id,),
            ).fetchone()
            result = self._onboarding_plan_payload(connection, row)
            result["added_asset_ids"] = added_asset_ids
            result["added_count"] = len(added_asset_ids)
            return result

    def refresh_onboarding_reading_items(
        self, plan_id: str, topic_matches: Dict[str, Dict[str, Any]], actor: str,
        *, trigger: str = "manual",
    ) -> Dict[str, Any]:
        """Relink blocked reading items to current authorized assets atomically."""
        now = utc_now()
        with self.database.transaction(write=True) as connection:
            plan = connection.execute(
                "SELECT * FROM onboarding_learning_plans WHERE project_id=? AND plan_id=?",
                (self.project_id, plan_id),
            ).fetchone()
            if not plan:
                raise ValueError("学习计划不存在")
            if str(plan["status"]) != "active":
                raise ValueError("只能刷新进行中的学习计划")

            rows = [dict(row) for row in connection.execute(
                """
                SELECT * FROM onboarding_learning_items
                WHERE plan_id=? AND item_type='reading'
                ORDER BY position,item_id
                """,
                (plan_id,),
            )]
            by_topic: Dict[str, List[Dict[str, Any]]] = {}
            for row in rows:
                by_topic.setdefault(str(row.get("topic_key") or ""), []).append(row)

            unblocked = relinked = added = removed_stale = newly_blocked = 0
            matched_asset_ids: List[str] = []
            changed = False
            for topic_key, topic_rows in by_topic.items():
                match = dict(topic_matches.get(topic_key) or {})
                raw_candidates = list(match.get("candidates") or [])
                candidates: List[Dict[str, Any]] = []
                seen_assets = set()
                for candidate in raw_candidates:
                    asset_id = str(candidate.get("asset_id") or "").strip()
                    version_id = str(candidate.get("version_id") or "").strip()
                    if not asset_id or not version_id or asset_id in seen_assets:
                        continue
                    current = connection.execute(
                        """
                        SELECT ka.asset_id,ka.current_version_id
                        FROM knowledge_assets ka
                        JOIN asset_versions av ON av.version_id=ka.current_version_id
                        WHERE ka.project_id=? AND ka.asset_id=? AND ka.current_version_id=?
                          AND ka.status IN ('active','review_due') AND av.status='active'
                        """,
                        (self.project_id, asset_id, version_id),
                    ).fetchone()
                    if not current:
                        continue
                    seen_assets.add(asset_id)
                    candidates.append(dict(candidate))

                candidate_versions = {
                    (str(item["asset_id"]), str(item["version_id"])) for item in candidates
                }
                protected = [row for row in topic_rows if str(row.get("status")) == "completed"]
                pending_valid = [
                    row for row in topic_rows
                    if str(row.get("status")) == "pending"
                    and (
                        str(row.get("asset_id") or ""), str(row.get("version_id") or "")
                    ) in candidate_versions
                ]
                refreshable = [row for row in topic_rows if str(row.get("status")) == "blocked"]
                invalid_pending = [
                    row for row in topic_rows
                    if str(row.get("status")) == "pending" and row not in pending_valid
                ]
                refreshable.extend(invalid_pending)

                represented = {
                    str(row.get("asset_id") or "")
                    for row in [*protected, *pending_valid]
                    if str(row.get("asset_id") or "")
                }
                available = [item for item in candidates if str(item["asset_id"]) not in represented]

                paired_count = min(len(refreshable), len(available))
                for row, candidate in zip(refreshable, available):
                    previous_asset_id = str(row.get("asset_id") or "")
                    connection.execute(
                        """
                        UPDATE onboarding_learning_items
                        SET title=?,description=?,asset_id=?,version_id=?,weight=?,status='pending',
                            evidence='',completed_by='',completed_at='',updated_at=?
                        WHERE plan_id=? AND item_id=?
                        """,
                        (
                            str(candidate.get("title") or f"阅读：{match.get('label') or topic_key}"),
                            str(candidate.get("description") or "阅读并核对当前有效知识资料。"),
                            str(candidate["asset_id"]), str(candidate["version_id"]),
                            max(0.01, float(candidate.get("weight") or row.get("weight") or 1)),
                            now, plan_id, str(row["item_id"]),
                        ),
                    )
                    unblocked += 1
                    relinked += int(bool(previous_asset_id) and previous_asset_id != str(candidate["asset_id"]))
                    matched_asset_ids.append(str(candidate["asset_id"]))
                    changed = True

                remaining_candidates = available[paired_count:]
                if remaining_candidates:
                    position_row = connection.execute(
                        "SELECT COALESCE(MAX(position),-1) AS position FROM onboarding_learning_items WHERE plan_id=?",
                        (plan_id,),
                    ).fetchone()
                    position = int(position_row["position"] if position_row else -1) + 1
                    for candidate in remaining_candidates:
                        connection.execute(
                            """
                            INSERT INTO onboarding_learning_items(
                                item_id,plan_id,topic_key,item_type,title,description,asset_id,version_id,
                                weight,status,position,created_at,updated_at
                            ) VALUES(?,?,?,'reading',?,?,?,?,?,'pending',?,?,?)
                            """,
                            (
                                f"oli_{uuid.uuid4().hex}", plan_id, topic_key,
                                str(candidate.get("title") or f"阅读：{match.get('label') or topic_key}"),
                                str(candidate.get("description") or "阅读并核对当前有效知识资料。"),
                                str(candidate["asset_id"]), str(candidate["version_id"]),
                                max(0.01, float(candidate.get("weight") or 1)), position, now, now,
                            ),
                        )
                        matched_asset_ids.append(str(candidate["asset_id"]))
                        position += 1
                        added += 1
                        changed = True

                unused = refreshable[paired_count:]
                has_satisfied_item = bool(protected or pending_valid or candidates)
                if unused and has_satisfied_item:
                    item_ids = [str(row["item_id"]) for row in unused]
                    placeholders = ",".join("?" for _ in item_ids)
                    connection.execute(
                        f"DELETE FROM onboarding_learning_items WHERE plan_id=? AND item_id IN ({placeholders})",
                        (plan_id, *item_ids),
                    )
                    removed_stale += len(item_ids)
                    changed = True
                elif unused:
                    keeper, *extras = unused
                    label = str(match.get("label") or topic_key or "必需知识")
                    if str(keeper.get("status")) != "blocked":
                        newly_blocked += 1
                    if (
                        str(keeper.get("status")) != "blocked"
                        or str(keeper.get("asset_id") or "")
                        or str(keeper.get("title") or "") != f"待补充资料：{label}"
                    ):
                        connection.execute(
                            """
                            UPDATE onboarding_learning_items
                            SET title=?,description=?,asset_id='',version_id='',status='blocked',
                                evidence='',completed_by='',completed_at='',updated_at=?
                            WHERE plan_id=? AND item_id=?
                            """,
                            (
                                f"待补充资料：{label}",
                                "当前权限范围内没有可用资料，需先完成知识补充任务。",
                                now, plan_id, str(keeper["item_id"]),
                            ),
                        )
                        changed = True
                    if extras:
                        item_ids = [str(row["item_id"]) for row in extras]
                        placeholders = ",".join("?" for _ in item_ids)
                        connection.execute(
                            f"DELETE FROM onboarding_learning_items WHERE plan_id=? AND item_id IN ({placeholders})",
                            (plan_id, *item_ids),
                        )
                        removed_stale += len(item_ids)
                        changed = True

            if changed:
                connection.execute(
                    "UPDATE onboarding_learning_plans SET updated_at=? WHERE plan_id=?",
                    (now, plan_id),
                )
                self._write_audit_connection(
                    connection, f"audit_{uuid.uuid4().hex}", actor,
                    "onboarding.plan_materials_refreshed", "onboarding_plan", plan_id,
                    {
                        "trigger": str(trigger or "manual")[:80],
                        "unblocked_count": unblocked,
                        "relinked_count": relinked,
                        "added_count": added,
                        "removed_stale_count": removed_stale,
                        "newly_blocked_count": newly_blocked,
                        "matched_asset_ids": list(dict.fromkeys(matched_asset_ids)),
                    },
                    project_id=self.project_id, created_at=now,
                )

            row = connection.execute(
                "SELECT * FROM onboarding_learning_plans WHERE plan_id=?", (plan_id,),
            ).fetchone()
            result = self._onboarding_plan_payload(connection, row)
            result["refresh"] = {
                "changed": changed,
                "unblocked_count": unblocked,
                "relinked_count": relinked,
                "added_count": added,
                "removed_stale_count": removed_stale,
                "newly_blocked_count": newly_blocked,
                "remaining_blocked_count": sum(
                    str(item.get("status")) == "blocked" for item in result.get("items") or []
                ),
                "matched_asset_ids": list(dict.fromkeys(matched_asset_ids)),
            }
            return result

    def list_onboarding_plans(
        self, *, user_id: str = "", manager_user_id: str = "", status: str = "",
    ) -> List[Dict[str, Any]]:
        where = ["project_id=?"]
        params: List[Any] = [self.project_id]
        if user_id:
            where.append("user_id=?")
            params.append(user_id)
        if manager_user_id:
            where.append("manager_user_id=?")
            params.append(manager_user_id)
        if status:
            where.append("status=?")
            params.append(status)
        with self.database.transaction() as connection:
            rows = connection.execute(
                f"SELECT * FROM onboarding_learning_plans WHERE {' AND '.join(where)} ORDER BY updated_at DESC",
                tuple(params),
            ).fetchall()
            return [self._onboarding_plan_payload(connection, row) for row in rows]

    def get_onboarding_plan(self, plan_id: str) -> Optional[Dict[str, Any]]:
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM onboarding_learning_plans WHERE project_id=? AND plan_id=?",
                (self.project_id, plan_id),
            ).fetchone()
            return self._onboarding_plan_payload(connection, row) if row else None

    def update_onboarding_item(
        self, plan_id: str, item_id: str, *, status: str, evidence: str,
        actor: str, complete_plan: bool = False,
    ) -> Dict[str, Any]:
        if status not in {"pending", "completed", "blocked"}:
            raise ValueError("不支持的学习任务状态")
        now = utc_now()
        with self.database.transaction(write=True) as connection:
            row = connection.execute(
                """
                SELECT oli.* FROM onboarding_learning_items oli
                JOIN onboarding_learning_plans olp ON olp.plan_id=oli.plan_id
                WHERE olp.project_id=? AND oli.plan_id=? AND oli.item_id=?
                """,
                (self.project_id, plan_id, item_id),
            ).fetchone()
            if not row:
                raise ValueError("学习任务不存在")
            connection.execute(
                """
                UPDATE onboarding_learning_items SET status=?,evidence=?,completed_by=?,completed_at=?,updated_at=?
                WHERE plan_id=? AND item_id=?
                """,
                (
                    status, str(evidence or "")[:2000], actor if status == "completed" else "",
                    now if status == "completed" else "", now, plan_id, item_id,
                ),
            )
            if row["item_type"] != "manager_confirmation" and status != "completed":
                connection.execute(
                    """
                    UPDATE onboarding_learning_items
                    SET status='pending',evidence='',completed_by='',completed_at='',updated_at=?
                    WHERE plan_id=? AND item_type='manager_confirmation' AND status='completed'
                    """,
                    (now, plan_id),
                )
            if complete_plan:
                connection.execute(
                    "UPDATE onboarding_learning_plans SET status='completed',completed_at=?,updated_at=? WHERE plan_id=?",
                    (now, now, plan_id),
                )
            else:
                connection.execute(
                    "UPDATE onboarding_learning_plans SET status='active',completed_at='',updated_at=? WHERE plan_id=?",
                    (now, plan_id),
                )
            self._write_audit_connection(
                connection, f"audit_{uuid.uuid4().hex}", actor, "onboarding.item_updated",
                "onboarding_plan", plan_id,
                {"item_id": item_id, "item_type": row["item_type"], "status": status},
                project_id=self.project_id, created_at=now,
            )
            plan = connection.execute(
                "SELECT * FROM onboarding_learning_plans WHERE plan_id=?", (plan_id,),
            ).fetchone()
            return self._onboarding_plan_payload(connection, plan)

    def create_access_policy(
        self,
        name: str,
        *,
        user_ids: Sequence[str] = (),
        roles: Sequence[str] = (),
        created_by: str = "user_system",
    ) -> Dict[str, Any]:
        policy_id = _stable_id("policy", self.project_id, name)
        now = utc_now()
        with self.database.transaction(write=True) as connection:
            connection.execute(
                """
                INSERT INTO access_policies(
                    access_policy_id,project_id,name,policy_type,status,created_by,created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?,?)
                ON CONFLICT(project_id,name) DO UPDATE SET status='active',updated_at=excluded.updated_at
                """,
                (policy_id, self.project_id, name.strip(), "restricted", "active", created_by, now, now),
            )
            connection.execute("DELETE FROM access_policy_users WHERE access_policy_id=?", (policy_id,))
            connection.execute("DELETE FROM access_policy_roles WHERE access_policy_id=?", (policy_id,))
            for user_id in dict.fromkeys(str(item) for item in user_ids if str(item)):
                connection.execute(
                    "INSERT INTO access_policy_users(access_policy_id,user_id,created_at) VALUES(?,?,?)",
                    (policy_id, user_id, now),
                )
            for role in dict.fromkeys(str(item) for item in roles if str(item)):
                connection.execute(
                    "INSERT INTO access_policy_roles(access_policy_id,role,created_at) VALUES(?,?,?)",
                    (policy_id, role, now),
                )
        return {"access_policy_id": policy_id, "name": name, "user_ids": list(user_ids), "roles": list(roles)}

    def set_document_access(
        self,
        filename: str,
        visibility: str,
        access_policy_id: str = "",
        owner_user_id: str = "",
    ) -> Dict[str, Any]:
        if visibility not in {"project", "organization", "restricted", "private"}:
            raise ValueError("不支持的文档可见范围")
        if visibility == "restricted" and not access_policy_id:
            raise ValueError("受限文档必须绑定访问策略")
        now = utc_now()
        with self.database.transaction(write=True) as connection:
            rows = self._document_rows_for_delete(connection, filename, False)
            if not rows:
                raise ValueError("文档不存在")
            row = rows[0]
            connection.execute(
                """
                UPDATE documents SET visibility=?,access_policy_id=?,owner_user_id=CASE WHEN ?<>'' THEN ? ELSE owner_user_id END,
                    acl_revision=acl_revision+1,updated_at=? WHERE document_id=?
                """,
                (visibility, access_policy_id if visibility == "restricted" else "", owner_user_id, owner_user_id, now, row["document_id"]),
            )
        return self.get_document(str(row["document_id"])) or {
            "document_id": row["document_id"], "visibility": visibility, "access_policy_id": access_policy_id,
        }

    def create_user_invitation(
        self,
        user_id: str,
        token_hash: str,
        expires_at: str,
        *,
        created_by: str = "user_system",
        project_id: str = "",
    ) -> str:
        project_id = project_id or self.project_id
        invitation_id = _stable_id("invite", user_id, token_hash)
        now = utc_now()
        with self.database.transaction(write=True) as connection:
            membership = connection.execute(
                "SELECT 1 FROM project_memberships WHERE project_id=? AND user_id=? AND status='active'",
                (project_id, user_id),
            ).fetchone()
            if not membership:
                raise ValueError("用户不是当前项目的有效成员")
            connection.execute(
                "UPDATE user_invitations SET consumed_at=? WHERE user_id=? AND consumed_at=''",
                (now, user_id),
            )
            connection.execute(
                """
                INSERT INTO user_invitations(
                    invitation_id,user_id,project_id,token_hash,expires_at,created_by,created_at
                ) VALUES(?,?,?,?,?,?,?)
                """,
                (invitation_id, user_id, project_id, token_hash, expires_at, created_by, now),
            )
        return invitation_id

    def activate_invitation(self, token_hash: str, password_hash: str) -> Optional[Dict[str, Any]]:
        now = utc_now()
        with self.database.transaction(write=True) as connection:
            invitation = connection.execute(
                """
                SELECT * FROM user_invitations
                WHERE token_hash=? AND consumed_at='' AND expires_at>?
                """,
                (token_hash, now),
            ).fetchone()
            if not invitation:
                return None
            connection.execute(
                """
                UPDATE users SET password_hash=?,password_changed_at=?,status='active',
                    failed_login_count=0,locked_until='',updated_at=? WHERE user_id=?
                """,
                (password_hash, now, now, invitation["user_id"]),
            )
            connection.execute(
                "UPDATE user_invitations SET consumed_at=? WHERE invitation_id=?",
                (now, invitation["invitation_id"]),
            )
            connection.execute("DELETE FROM auth_sessions WHERE user_id=?", (invitation["user_id"],))
            row = connection.execute(
                "SELECT * FROM users WHERE user_id=?", (invitation["user_id"],)
            ).fetchone()
            return dict(row) if row else None

    def record_login_failure(self, user_id: str, failed_count: int, locked_until: str = "") -> None:
        with self.database.transaction(write=True) as connection:
            connection.execute(
                "UPDATE users SET failed_login_count=?,locked_until=?,updated_at=? WHERE user_id=?",
                (failed_count, locked_until, utc_now(), user_id),
            )

    def record_login_success(self, user_id: str) -> None:
        now = utc_now()
        with self.database.transaction(write=True) as connection:
            connection.execute(
                """
                UPDATE users SET failed_login_count=0,locked_until='',last_login_at=?,updated_at=?
                WHERE user_id=?
                """,
                (now, now, user_id),
            )

    def create_auth_session(self, session: Dict[str, Any]) -> None:
        with self.database.transaction(write=True) as connection:
            connection.execute("DELETE FROM auth_sessions WHERE expires_at<=?", (utc_now(),))
            connection.execute(
                """
                INSERT INTO auth_sessions(
                    session_id,token_hash,csrf_hash,user_id,current_project_id,expires_at,
                    last_seen_at,created_at,user_agent_hash,remote_address_hash
                ) VALUES(?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    session["session_id"], session["token_hash"], session["csrf_hash"],
                    session["user_id"], session["current_project_id"], session["expires_at"],
                    session["created_at"], session["created_at"],
                    session.get("user_agent_hash", ""), session.get("remote_address_hash", ""),
                ),
            )

    def get_auth_session(self, token_hash: str) -> Optional[Dict[str, Any]]:
        with self.database.transaction() as connection:
            row = connection.execute(
                """
                SELECT s.*,u.display_name,u.email,u.email_normalized,u.status AS user_status,u.duty,
                       m.role,m.status AS membership_status,p.name AS project_name,
                       p.status AS project_status,p.organization_id
                FROM auth_sessions s
                JOIN users u ON u.user_id=s.user_id
                JOIN project_memberships m ON m.user_id=u.user_id AND m.project_id=s.current_project_id
                JOIN projects p ON p.project_id=s.current_project_id
                WHERE s.token_hash=? AND s.expires_at>?
                """,
                (token_hash, utc_now()),
            ).fetchone()
            return dict(row) if row else None

    def touch_auth_session(self, session_id: str) -> None:
        with self.database.transaction(write=True) as connection:
            connection.execute(
                "UPDATE auth_sessions SET last_seen_at=? WHERE session_id=?",
                (utc_now(), session_id),
            )

    def delete_auth_session(self, token_hash: str) -> None:
        with self.database.transaction(write=True) as connection:
            connection.execute("DELETE FROM auth_sessions WHERE token_hash=?", (token_hash,))

    def write_security_audit(
        self,
        actor: str,
        action: str,
        object_type: str,
        object_id: str,
        detail: Optional[Dict[str, Any]] = None,
        *,
        project_id: str = "",
    ) -> None:
        project_id = project_id or self.project_id
        now = utc_now()
        audit_id = f"audit_{uuid.uuid4().hex}"
        with self.database.transaction(write=True) as connection:
            self._write_audit_connection(
                connection, audit_id, actor, action, object_type, object_id,
                detail or {}, project_id=project_id, created_at=now,
            )

    @staticmethod
    def _write_audit_connection(
        connection: sqlite3.Connection,
        audit_id: str,
        actor: str,
        action: str,
        object_type: str,
        object_id: str,
        detail: Dict[str, Any],
        *,
        project_id: str,
        created_at: str = "",
    ) -> None:
        connection.execute(
            """
            INSERT INTO audit_events(
                audit_id,project_id,actor,action,object_type,object_id,created_at,data_json
            ) VALUES(?,?,?,?,?,?,?,?)
            """,
            (
                audit_id, project_id, actor or "anonymous", action, object_type,
                object_id, created_at or utc_now(), _json_dump({
                    **detail,
                    "audit_id": audit_id,
                    "actor": actor or "anonymous",
                    "action": action,
                    "object_type": object_type,
                    "object_id": object_id,
                    "created_at": created_at or utc_now(),
                }),
            ),
        )

    def accessible_document_ids(self, identity: Any = None) -> set[str]:
        """Return authorized documents from the current active, unexpired version only."""
        if identity is None:
            try:
                from backend.auth_context import get_current_identity

                identity = get_current_identity()
            except Exception:
                identity = None
        if getattr(identity, "project_id", "") != self.project_id:
            if identity is not None and not getattr(identity, "is_system", False):
                return set()
        user_id = str(getattr(identity, "user_id", ""))
        role = str(getattr(identity, "role", ""))
        # Management permissions authorize lifecycle operations, not content
        # visibility. Human administrators and operators must still be named by
        # a private/restricted source policy before its content can enter RAG,
        # graph, preview, or any other read path. Only trusted background jobs
        # may bypass user ACLs.
        privileged = identity is None or getattr(identity, "is_system", False)
        now = utc_now()
        lifecycle = """
            d.project_id=? AND d.status='active'
            AND (d.valid_from='' OR d.valid_from<=?)
            AND (d.valid_until='' OR d.valid_until>?)
            AND (
                avd.document_id IS NULL
                OR (
                    av.version_id=ka.current_version_id AND av.status='active'
                    AND ka.status IN ('active','review_due') AND ks.status='active'
                    AND (av.valid_from='' OR av.valid_from<=?)
                    AND (av.valid_until='' OR av.valid_until>?)
                )
            )
        """
        with self.database.transaction() as connection:
            if privileged:
                rows = connection.execute(
                    """
                    SELECT DISTINCT d.document_id FROM documents d
                    LEFT JOIN asset_version_documents avd ON avd.document_id=d.document_id
                    LEFT JOIN asset_versions av ON av.version_id=avd.version_id
                    LEFT JOIN knowledge_assets ka ON ka.asset_id=av.asset_id
                    LEFT JOIN knowledge_sources ks ON ks.source_id=ka.primary_source_id
                    WHERE
                    """ + lifecycle,
                    (self.project_id, now, now, now, now),
                )
                return {str(row["document_id"]) for row in rows}
            rows = connection.execute(
                """
                SELECT DISTINCT d.document_id
                FROM documents d
                LEFT JOIN asset_version_documents avd ON avd.document_id=d.document_id
                LEFT JOIN asset_versions av ON av.version_id=avd.version_id
                LEFT JOIN knowledge_assets ka ON ka.asset_id=av.asset_id
                LEFT JOIN knowledge_sources ks ON ks.source_id=ka.primary_source_id
                WHERE
                """ + lifecycle + """ AND (
                    d.visibility IN ('project','organization')
                    OR (d.visibility='private' AND d.owner_user_id=?)
                    OR (d.visibility='restricted' AND (
                        EXISTS(SELECT 1 FROM access_policy_users pu WHERE pu.access_policy_id=d.access_policy_id AND pu.user_id=?)
                        OR EXISTS(SELECT 1 FROM access_policy_roles pr WHERE pr.access_policy_id=d.access_policy_id AND pr.role=?)
                    ))
                ) AND (avd.document_id IS NULL OR (
                    (ka.visibility IN ('project','organization')
                     OR (ka.visibility='private' AND ka.owner_user_id=?)
                     OR (ka.visibility='restricted' AND (
                        EXISTS(SELECT 1 FROM access_policy_users pu WHERE pu.access_policy_id=ka.access_policy_id AND pu.user_id=?)
                        OR EXISTS(SELECT 1 FROM access_policy_roles pr WHERE pr.access_policy_id=ka.access_policy_id AND pr.role=?)
                     )))
                    AND (ks.visibility IN ('project','organization')
                     OR (ks.visibility='private' AND ks.owner_user_id=?)
                     OR (ks.visibility='restricted' AND (
                        EXISTS(SELECT 1 FROM access_policy_users pu WHERE pu.access_policy_id=ks.access_policy_id AND pu.user_id=?)
                        OR EXISTS(SELECT 1 FROM access_policy_roles pr WHERE pr.access_policy_id=ks.access_policy_id AND pr.role=?)
                     )))
                ))
                """,
                (self.project_id, now, now, now, now,
                 user_id, user_id, role, user_id, user_id, role, user_id, user_id, role),
            )
            return {str(row["document_id"]) for row in rows}

    def accessible_source_files(self, identity: Any = None) -> set[str]:
        document_ids = self.accessible_document_ids(identity)
        if not document_ids:
            return set()
        placeholders = ",".join("?" for _ in document_ids)
        with self.database.transaction() as connection:
            rows = connection.execute(
                f"SELECT source_file,stored_file FROM documents WHERE document_id IN ({placeholders})",
                tuple(document_ids),
            )
            return {
                str(value)
                for row in rows
                for value in (row["source_file"], row["stored_file"])
                if value
            }

    # Categories ---------------------------------------------------------
    def list_categories(self) -> List[str]:
        with self.database.transaction() as connection:
            return [
                row["name"]
                for row in connection.execute(
                    "SELECT name FROM categories WHERE project_id=? ORDER BY sort_order,category_id",
                    (self.project_id,),
                )
            ]

    def replace_categories(self, names: Sequence[str]) -> List[str]:
        normalized = list(dict.fromkeys(str(name).strip() for name in names if str(name).strip()))
        with self.database.transaction(write=True) as connection:
            existing = {
                row["name"]: row["category_id"]
                for row in connection.execute("SELECT category_id,name FROM categories WHERE project_id=?", (self.project_id,))
            }
            referenced = {
                row["name"]
                for row in connection.execute(
                    """
                    SELECT DISTINCT c.name FROM categories c
                    JOIN document_categories dc ON dc.category_id=c.category_id
                    WHERE c.project_id=?
                    """,
                    (self.project_id,),
                )
            }
            now = utc_now()
            for index, name in enumerate(normalized):
                self._ensure_category(connection, name, index)
            for name, category_id in existing.items():
                if name not in normalized and name not in referenced:
                    connection.execute("DELETE FROM categories WHERE category_id=?", (category_id,))
            connection.execute(
                "UPDATE categories SET sort_order=sort_order+? WHERE project_id=? AND name NOT IN ({})".format(
                    ",".join("?" for _ in normalized) or "''"
                ),
                (len(normalized), self.project_id, *normalized),
            )
            connection.execute(
                "UPDATE projects SET updated_at=? WHERE project_id=?", (now, self.project_id)
            )
        return self.list_categories()

    def add_category(self, name: str) -> List[str]:
        name = name.strip()
        with self.database.transaction(write=True) as connection:
            count = connection.execute(
                "SELECT COUNT(*) FROM categories WHERE project_id=?", (self.project_id,)
            ).fetchone()[0]
            self._ensure_category(connection, name, count)
        return self.list_categories()

    def rename_category(self, old_name: str, new_name: str) -> List[str]:
        with self.database.transaction(write=True) as connection:
            existing = connection.execute(
                "SELECT category_id FROM categories WHERE project_id=? AND name=?",
                (self.project_id, old_name),
            ).fetchone()
            if not existing:
                raise KeyError("原分类不存在")
            if connection.execute(
                "SELECT 1 FROM categories WHERE project_id=? AND name=?",
                (self.project_id, new_name),
            ).fetchone():
                raise ValueError("新分类名称已存在")
            now = utc_now()
            connection.execute(
                "UPDATE categories SET name=?,updated_at=? WHERE category_id=?",
                (new_name, now, existing["category_id"]),
            )
            connection.execute(
                "UPDATE documents SET primary_category=?,updated_at=? WHERE project_id=? AND primary_category=?",
                (new_name, now, self.project_id, old_name),
            )
        return self.list_categories()

    def delete_category(self, name: str) -> List[str]:
        with self.database.transaction(write=True) as connection:
            row = connection.execute(
                "SELECT category_id FROM categories WHERE project_id=? AND name=?",
                (self.project_id, name),
            ).fetchone()
            if not row:
                raise KeyError("分类不存在")
            affected = [
                item["document_id"] for item in connection.execute(
                    "SELECT document_id FROM document_categories WHERE category_id=?", (row["category_id"],)
                )
            ]
            connection.execute("DELETE FROM document_categories WHERE category_id=?", (row["category_id"],))
            fallback_id = None
            if affected:
                if name == "未分类":
                    raise ValueError("未分类仍被文档使用，不能删除")
                fallback_id = self._ensure_category(connection, "未分类", 9999)
            now = utc_now()
            for document_id in affected:
                replacement = connection.execute(
                    "SELECT category_id FROM document_categories WHERE document_id=? ORDER BY is_primary DESC,category_id LIMIT 1",
                    (document_id,),
                ).fetchone()
                replacement_id = int(replacement["category_id"]) if replacement else int(fallback_id)
                if not replacement:
                    connection.execute(
                        "INSERT INTO document_categories(document_id,category_id,is_primary) VALUES(?,?,1)",
                        (document_id, replacement_id),
                    )
                replacement_name = connection.execute(
                    "SELECT name FROM categories WHERE category_id=?", (replacement_id,)
                ).fetchone()["name"]
                connection.execute(
                    "UPDATE documents SET primary_category=?,updated_at=? WHERE document_id=? AND primary_category=?",
                    (replacement_name, now, document_id, name),
                )
            connection.execute("DELETE FROM categories WHERE category_id=?", (row["category_id"],))
        return self.list_categories()

    def _ensure_category(self, connection: sqlite3.Connection, name: str, sort_order: int = 0) -> int:
        name = str(name or "未分类").strip() or "未分类"
        now = utc_now()
        connection.execute(
            """
            INSERT INTO categories(project_id,name,sort_order,created_at,updated_at)
            VALUES(?,?,?,?,?)
            ON CONFLICT(project_id,name) DO UPDATE SET updated_at=excluded.updated_at
            """,
            (self.project_id, name, sort_order, now, now),
        )
        return int(connection.execute(
            "SELECT category_id FROM categories WHERE project_id=? AND name=?",
            (self.project_id, name),
        ).fetchone()[0])

    # Documents ----------------------------------------------------------
    def upsert_document_chunks(self, documents: Sequence[Any], vector_ids: Sequence[str] = ()) -> Optional[dict]:
        if not documents:
            return None
        first_meta = dict(getattr(documents[0], "metadata", {}) or {})
        stored_file = str(first_meta.get("stored_file") or first_meta.get("source_file") or "").strip()
        source_file = str(first_meta.get("source_file") or stored_file).strip()
        if not stored_file:
            return None
        document = {
            **first_meta,
            "stored_file": stored_file,
            "source_file": source_file,
            "categories": first_meta.get("categories") or ([first_meta.get("category")] if first_meta.get("category") else []),
        }
        chunks = []
        for index, item in enumerate(documents):
            item_meta = dict(getattr(item, "metadata", {}) or {})
            item_stored_file = str(item_meta.get("stored_file") or item_meta.get("source_file") or "").strip()
            if item_stored_file and item_stored_file != stored_file:
                raise ValueError("一次知识块写入只能属于同一个文档版本")
            chunks.append({
                "text": str(getattr(item, "page_content", "") or ""),
                "metadata": item_meta,
                "vector_id": str(vector_ids[index]) if index < len(vector_ids) else "",
                "chunk_index": index,
            })
        document["content_hash"] = str(document.get("content_hash") or hashlib.sha256(
            "\n".join(item["text"] for item in chunks).encode("utf-8")
        ).hexdigest())
        with self.database.transaction(write=True) as connection:
            return self._upsert_document_connection(connection, document, chunks)

    def upsert_document_metadata(self, doc_id: str, metadata: Dict[str, Any]) -> dict:
        stored_file = str(metadata.get("stored_file") or doc_id or metadata.get("source_file") or "").strip()
        if not stored_file:
            raise ValueError("文档存储文件名不能为空")
        # Projection writers must be able to enrich a preparing document without
        # bypassing the public lifecycle gate or resetting its ACL/asset fields.
        with self.database.transaction() as connection:
            row = connection.execute(
                """
                SELECT d.*,(SELECT COUNT(*) FROM document_chunks ch WHERE ch.document_id=d.document_id) AS chunks
                FROM documents d WHERE d.project_id=? AND (d.document_id=? OR d.stored_file=?) LIMIT 1
                """,
                (self.project_id, doc_id, stored_file),
            ).fetchone()
            existing = self._document_row(connection, row) if row else None
        merged = {**(existing or {}), **metadata, "stored_file": stored_file}
        with self.database.transaction(write=True) as connection:
            return self._upsert_document_connection(connection, merged, None)

    def _upsert_document_connection(
        self,
        connection: sqlite3.Connection,
        document: Dict[str, Any],
        chunks: Optional[Sequence[Dict[str, Any]]],
    ) -> dict:
        stored_file = str(document.get("stored_file") or document.get("name") or document.get("source_file") or "").strip()
        source_file = str(document.get("source_file") or stored_file).strip()
        document_id = str(document.get("document_id") or _stable_id("doc", self.project_id, stored_file))
        categories = document.get("categories") or ([document.get("category")] if document.get("category") else [])
        categories = list(dict.fromkeys(str(item).strip() for item in categories if str(item).strip())) or ["未分类"]
        primary = str(document.get("primary_category") or document.get("category") or categories[0])
        organization_id = str(document.get("organization_id") or DEFAULT_ORGANIZATION_ID)
        source_type = str(document.get("source_type") or "document")
        content_hash = str(document.get("content_hash") or "")
        source_id = str(document.get("source_id") or _stable_id(
            "source", organization_id, self.project_id, source_type, source_file
        ))
        asset_id = str(document.get("asset_id") or _stable_id("asset", self.project_id, source_id))
        version_id = str(document.get("version_id") or _stable_id(
            "version", asset_id, content_hash or stored_file
        ))
        owner_user_id = str(document.get("owner_user_id") or "")
        if not owner_user_id:
            try:
                from backend.auth_context import get_current_identity

                identity = get_current_identity()
                if identity and not getattr(identity, "is_system", False):
                    owner_user_id = str(getattr(identity, "user_id", ""))
            except Exception:
                pass
        now = utc_now()
        previous = connection.execute("SELECT metadata_json,created_at FROM documents WHERE document_id=?", (document_id,)).fetchone()
        previous_metadata = _json_load(previous["metadata_json"], {}) if previous else {}
        metadata = {**previous_metadata, **document, "categories": categories, "category": primary}
        connection.execute(
            """
            INSERT INTO documents(
                document_id,project_id,source_file,stored_file,primary_category,uploader,role,handover_id,
                source_type,status,content_hash,upload_date,registered_at,metadata_json,created_at,updated_at,
                organization_id,source_id,asset_id,version_id,supersedes_version_id,visibility,
                sensitivity_level,access_policy_id,acl_revision,valid_from,valid_until,owner_user_id
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(document_id) DO UPDATE SET
                source_file=excluded.source_file,stored_file=excluded.stored_file,
                primary_category=excluded.primary_category,uploader=excluded.uploader,role=excluded.role,
                handover_id=excluded.handover_id,source_type=excluded.source_type,status=excluded.status,
                content_hash=excluded.content_hash,upload_date=excluded.upload_date,
                registered_at=excluded.registered_at,metadata_json=excluded.metadata_json,updated_at=excluded.updated_at,
                organization_id=excluded.organization_id,source_id=excluded.source_id,asset_id=excluded.asset_id,
                version_id=excluded.version_id,supersedes_version_id=excluded.supersedes_version_id,
                visibility=excluded.visibility,sensitivity_level=excluded.sensitivity_level,
                access_policy_id=excluded.access_policy_id,acl_revision=excluded.acl_revision,
                valid_from=excluded.valid_from,valid_until=excluded.valid_until,
                owner_user_id=excluded.owner_user_id
            """,
            (
                document_id, self.project_id, source_file, stored_file, primary,
                str(document.get("uploader") or ""), str(document.get("role") or ""),
                str(document.get("handover_id") or ""), source_type,
                str(document.get("status") or "active"), content_hash,
                str(document.get("upload_date") or ""), str(document.get("registered_at") or now),
                _json_dump(metadata), previous["created_at"] if previous else str(document.get("created_at") or now), now,
                organization_id, source_id, asset_id, version_id,
                str(document.get("supersedes_version_id") or ""), str(document.get("visibility") or "project"),
                str(document.get("sensitivity_level") or "internal"), str(document.get("access_policy_id") or ""),
                int(document.get("acl_revision") or 0), str(document.get("valid_from") or ""),
                str(document.get("valid_until") or ""), owner_user_id,
            ),
        )
        connection.execute("DELETE FROM document_categories WHERE document_id=?", (document_id,))
        for index, category in enumerate(categories):
            category_id = self._ensure_category(connection, category, index)
            connection.execute(
                "INSERT INTO document_categories(document_id,category_id,is_primary) VALUES(?,?,?)",
                (document_id, category_id, int(category == primary or (index == 0 and primary not in categories))),
            )
        managed_version = connection.execute(
            "SELECT asset_id FROM asset_versions WHERE version_id=?", (version_id,)
        ).fetchone()
        if managed_version:
            if str(managed_version["asset_id"]) != asset_id:
                raise ValueError("文档版本与知识资产不匹配")
            connection.execute(
                "INSERT OR REPLACE INTO asset_version_documents(version_id,document_id,document_role,created_at) VALUES(?,?,?,?)",
                (version_id, document_id, str(document.get("document_role") or "authoritative"), now),
            )
            for alias_type, alias_value in (
                ("document_id", document_id), ("stored_file", stored_file), ("source_file", source_file),
            ):
                connection.execute(
                    "INSERT OR REPLACE INTO knowledge_asset_aliases(project_id,alias_type,alias_value,asset_id,created_at) VALUES(?,?,?,?,?)",
                    (self.project_id, alias_type, alias_value, asset_id, now),
                )
        persisted_chunk_ids: List[str] = []
        if chunks is not None:
            connection.execute("DELETE FROM document_chunks WHERE document_id=?", (document_id,))
            for index, chunk in enumerate(chunks):
                chunk_index = int(chunk.get("chunk_index", index))
                text = str(chunk.get("text") or chunk.get("text_content") or "")
                chunk_id = str(chunk.get("chunk_id") or _stable_id("chunk", document_id, chunk_index, hashlib.sha256(text.encode()).hexdigest()))
                connection.execute(
                    """
                    INSERT INTO document_chunks(chunk_id,document_id,chunk_index,vector_id,text_content,metadata_json,created_at)
                    VALUES(?,?,?,?,?,?,?)
                    """,
                    (chunk_id, document_id, chunk_index, str(chunk.get("vector_id") or ""), text,
                     _json_dump(chunk.get("metadata") or {}), str(chunk.get("created_at") or now)),
                )
                persisted_chunk_ids.append(chunk_id)
        else:
            persisted_chunk_ids = [
                str(row["chunk_id"])
                for row in connection.execute(
                    "SELECT chunk_id FROM document_chunks WHERE document_id=? ORDER BY chunk_index,chunk_id",
                    (document_id,),
                )
            ]
        return {
            "document_id": document_id,
            "stored_file": stored_file,
            "source_file": source_file,
            "chunk_ids": persisted_chunk_ids,
        }

    def list_documents(self) -> List[Dict[str, Any]]:
        allowed = self.accessible_document_ids()
        if not allowed:
            return []
        placeholders = ",".join("?" for _ in allowed)
        with self.database.transaction() as connection:
            rows = connection.execute(
                f"""
                SELECT d.*,(SELECT COUNT(*) FROM document_chunks ch WHERE ch.document_id=d.document_id) AS chunks
                FROM documents d WHERE d.project_id=? AND d.status <> 'deleted'
                AND d.document_id IN ({placeholders})
                ORDER BY d.registered_at DESC,d.document_id
                """,
                (self.project_id, *allowed),
            ).fetchall()
            return [self._document_row(connection, row) for row in rows]

    def get_document(self, filename: str) -> Optional[Dict[str, Any]]:
        allowed = self.accessible_document_ids()
        if not allowed:
            return None
        placeholders = ",".join("?" for _ in allowed)
        with self.database.transaction() as connection:
            row = connection.execute(
                f"""
                SELECT d.*,(SELECT COUNT(*) FROM document_chunks ch WHERE ch.document_id=d.document_id) AS chunks
                FROM documents d WHERE d.project_id=? AND (d.document_id=? OR d.stored_file=? OR d.source_file=?)
                AND d.document_id IN ({placeholders})
                ORDER BY CASE WHEN d.stored_file=? THEN 0 ELSE 1 END LIMIT 1
                """,
                (self.project_id, filename, filename, filename, *allowed, filename),
            ).fetchone()
            return self._document_row(connection, row) if row else None

    def _document_row(self, connection: sqlite3.Connection, row: sqlite3.Row) -> Dict[str, Any]:
        metadata = _json_load(row["metadata_json"], {})
        categories = [
            item["name"]
            for item in connection.execute(
                """
                SELECT c.name FROM categories c JOIN document_categories dc ON dc.category_id=c.category_id
                WHERE dc.document_id=? ORDER BY dc.is_primary DESC,c.sort_order,c.category_id
                """,
                (row["document_id"],),
            )
        ]
        return {
            **metadata,
            "document_id": row["document_id"],
            "name": row["stored_file"],
            "source_file": row["source_file"],
            "stored_file": row["stored_file"],
            "categories": categories,
            "category": row["primary_category"],
            "uploader": row["uploader"],
            "role": row["role"],
            "handover_id": row["handover_id"],
            "source_type": row["source_type"],
            "status": row["status"],
            "content_hash": row["content_hash"],
            "upload_date": row["upload_date"] or row["registered_at"],
            "registered_at": row["registered_at"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "chunks": int(row["chunks"] or 0),
            "organization_id": row["organization_id"],
            "project_id": row["project_id"],
            "source_id": row["source_id"],
            "asset_id": row["asset_id"],
            "version_id": row["version_id"],
            "supersedes_version_id": row["supersedes_version_id"],
            "visibility": row["visibility"],
            "sensitivity_level": row["sensitivity_level"],
            "access_policy_id": row["access_policy_id"],
            "acl_revision": int(row["acl_revision"] or 0),
            "valid_from": row["valid_from"],
            "valid_until": row["valid_until"],
            "owner_user_id": row["owner_user_id"],
        }

    def _document_rows_for_delete(
        self, connection: sqlite3.Connection, filename: str, all_versions: bool,
    ) -> List[sqlite3.Row]:
        projection = "d.*,(SELECT COUNT(*) FROM document_chunks ch WHERE ch.document_id=d.document_id) AS chunks"
        if all_versions:
            stored_suffix = f"%\\_{_like_literal(filename)}"
            return connection.execute(
                f"""
                SELECT {projection} FROM documents d WHERE d.project_id=? AND (
                    d.source_file=? OR d.stored_file=? OR d.stored_file LIKE ? ESCAPE '\\'
                ) ORDER BY d.document_id
                """,
                (self.project_id, filename, filename, stored_suffix),
            ).fetchall()

        rows = connection.execute(
            f"""
            SELECT {projection} FROM documents d
            WHERE d.project_id=? AND (d.document_id=? OR d.stored_file=?)
            ORDER BY d.document_id
            """,
            (self.project_id, filename, filename),
        ).fetchall()
        if not rows:
            rows = connection.execute(
                f"SELECT {projection} FROM documents d WHERE d.project_id=? AND d.source_file=? ORDER BY d.document_id",
                (self.project_id, filename),
            ).fetchall()
            if len(rows) > 1:
                raise ValueError("该文件存在多个版本，请使用具体版本的存储标识删除")
        return rows

    def _document_snapshot_connection(self, connection: sqlite3.Connection, row: sqlite3.Row) -> dict:
        chunks = [
            {
                "chunk_id": item["chunk_id"],
                "chunk_index": int(item["chunk_index"]),
                "vector_id": item["vector_id"],
                "text": item["text_content"],
                "metadata": _json_load(item["metadata_json"], {}),
                "created_at": item["created_at"],
            }
            for item in connection.execute(
                """
                SELECT * FROM document_chunks WHERE document_id=?
                ORDER BY chunk_index,chunk_id
                """,
                (row["document_id"],),
            )
        ]
        return {"document": self._document_row(connection, row), "chunks": chunks}

    def delete_document_with_snapshot(self, filename: str, all_versions: bool = False) -> List[dict]:
        """Atomically capture and delete document versions for compensation."""
        with self.database.transaction(write=True) as connection:
            rows = self._document_rows_for_delete(connection, filename, all_versions)
            snapshots = [self._document_snapshot_connection(connection, row) for row in rows]
            for row in rows:
                connection.execute("DELETE FROM documents WHERE document_id=?", (row["document_id"],))
            return snapshots

    def restore_document_snapshots(self, snapshots: Sequence[Dict[str, Any]]) -> int:
        """Restore documents removed by a failed derived-index operation."""
        with self.database.transaction(write=True) as connection:
            for snapshot in snapshots:
                self._upsert_document_connection(
                    connection,
                    dict(snapshot.get("document") or {}),
                    list(snapshot.get("chunks") or []),
                )
        return len(snapshots)

    def delete_document(self, filename: str) -> int:
        """Delete one unambiguous document version."""
        return len(self.delete_document_with_snapshot(filename, all_versions=False))

    def delete_documents_by_source(self, filename: str) -> int:
        """Explicitly delete every version belonging to one source."""
        return len(self.delete_document_with_snapshot(filename, all_versions=True))

    def document_chunks(self) -> List[Dict[str, Any]]:
        """Return projection inventory, including staged managed versions.

        Retrieval still calls ``active_chunk_ids`` immediately before returning
        results, so preparing versions can be embedded without becoming visible.
        """
        with self.database.transaction() as connection:
            return [
                {
                    "chunk_id": row["chunk_id"], "document_id": row["document_id"],
                    "chunk_index": row["chunk_index"], "vector_id": row["vector_id"],
                    "text": row["text_content"], "metadata": _json_load(row["metadata_json"], {}),
                }
                for row in connection.execute(
                    """
                    SELECT ch.* FROM document_chunks ch
                    JOIN documents d ON d.document_id=ch.document_id
                    LEFT JOIN asset_version_documents avd ON avd.document_id=d.document_id
                    LEFT JOIN asset_versions av ON av.version_id=avd.version_id
                    LEFT JOIN knowledge_assets ka ON ka.asset_id=av.asset_id
                    LEFT JOIN knowledge_sources ks ON ks.source_id=ka.primary_source_id
                    WHERE d.project_id=?
                    AND (d.valid_from='' OR d.valid_from<=?)
                    AND (d.valid_until='' OR d.valid_until>?)
                    AND (
                        (avd.document_id IS NULL AND d.status='active')
                        OR (
                            avd.document_id IS NOT NULL AND d.status IN ('preparing','ready','active')
                            AND av.status IN ('preparing','ready','active') AND ks.status='active'
                            AND (av.valid_from='' OR av.valid_from<=?)
                            AND (av.valid_until='' OR av.valid_until>?)
                        )
                    )
                    ORDER BY d.document_id,ch.chunk_index
                    """,
                    (self.project_id, utc_now(), utc_now(), utc_now(), utc_now()),
                )
            ]

    def active_chunk_ids(self, chunk_ids: Sequence[str] = ()) -> set[str]:
        """Return currently retrievable chunk IDs, optionally limited to candidates."""
        allowed_documents = self.accessible_document_ids()
        if not allowed_documents:
            return set()
        requested = list(dict.fromkeys(str(item) for item in chunk_ids if str(item)))
        now = utc_now()
        document_scope = ",".join("?" for _ in allowed_documents)
        params: List[Any] = [self.project_id, now, now, *allowed_documents]
        scope = f" AND d.document_id IN ({document_scope})"
        if requested:
            scope += " AND ch.chunk_id IN (" + ",".join("?" for _ in requested) + ")"
            params.extend(requested)
        with self.database.transaction() as connection:
            rows = connection.execute(
                """
                SELECT ch.chunk_id FROM document_chunks ch
                JOIN documents d ON d.document_id=ch.document_id
                WHERE d.project_id=? AND d.status='active'
                AND (d.valid_from='' OR d.valid_from<=?)
                AND (d.valid_until='' OR d.valid_until>?)
                """ + scope,
                tuple(params),
            ).fetchall()
        return {str(row["chunk_id"]) for row in rows}

    def active_document_sources(self) -> List[str]:
        """Return source aliases that are currently eligible for retrieval."""
        allowed_documents = self.accessible_document_ids()
        if not allowed_documents:
            return []
        placeholders = ",".join("?" for _ in allowed_documents)
        now = utc_now()
        with self.database.transaction() as connection:
            rows = connection.execute(
                f"""
                SELECT source_file,stored_file FROM documents
                WHERE project_id=? AND status='active'
                AND (valid_from='' OR valid_from<=?)
                AND (valid_until='' OR valid_until>?)
                AND document_id IN ({placeholders})
                ORDER BY document_id
                """,
                (self.project_id, now, now, *allowed_documents),
            ).fetchall()
        return list(dict.fromkeys(
            str(value).strip()
            for row in rows
            for value in (row["source_file"], row["stored_file"])
            if str(value or "").strip()
        ))

    def active_knowledge_source_ids(self) -> List[str]:
        """Return every active source that contributes project knowledge."""
        with self.database.transaction() as connection:
            rows = connection.execute(
                """
                SELECT source_id FROM knowledge_sources
                WHERE project_id=? AND status='active'
                UNION
                SELECT source_id FROM documents
                WHERE project_id=? AND status='active' AND TRIM(source_id)<>''
                ORDER BY source_id
                """,
                (self.project_id, self.project_id),
            ).fetchall()
        return [str(row["source_id"]) for row in rows if str(row["source_id"] or "")]

    def resettable_knowledge_source_ids(self) -> List[str]:
        """Return every non-tombstoned source included in a project library reset.

        A project-level reset is intentionally broader than retrieval. Revoked,
        expired and review-due assets must not remain in the ordinary library
        after an administrator has confirmed that all knowledge data is to be
        cleared. Existing ``deleted`` rows are already content-free tombstones
        and remain solely for governance audit history.
        """
        with self.database.transaction() as connection:
            rows = connection.execute(
                """
                SELECT source_id FROM knowledge_sources
                WHERE project_id=? AND status<>'deleted'
                UNION
                SELECT source_id FROM documents
                WHERE project_id=? AND status<>'deleted' AND TRIM(source_id)<>''
                ORDER BY source_id
                """,
                (self.project_id, self.project_id),
            ).fetchall()
        return [str(row["source_id"]) for row in rows if str(row["source_id"] or "")]

    @staticmethod
    def _project_feishu_file_references_connection(connection: sqlite3.Connection, project_id: str) -> Dict[str, List[str]]:
        resource_files = set()
        for row in connection.execute(
            "SELECT data_json FROM feishu_messages WHERE project_id=?", (project_id,)
        ):
            message = _json_load(row["data_json"], {})
            for resource in message.get("resource_files", []) if isinstance(message, dict) else []:
                if isinstance(resource, dict) and str(resource.get("stored_file") or "").strip():
                    resource_files.add(str(resource["stored_file"]).strip())
        asset_files = {
            str(row["stored_file"] or "").strip()
            for row in connection.execute(
                "SELECT stored_file FROM feishu_assets WHERE project_id=?", (project_id,)
            )
            if str(row["stored_file"] or "").strip()
        }
        return {"asset_files": sorted(asset_files), "resource_files": sorted(resource_files)}

    def project_feishu_file_references(self) -> Dict[str, List[str]]:
        """Return only the files that are currently attributable to this project."""
        with self.database.transaction() as connection:
            return self._project_feishu_file_references_connection(connection, self.project_id)

    def clear_project_feishu_history(self) -> Dict[str, Any]:
        """Clear persisted Feishu content while retaining collection-source configuration.

        ``feishu_groups`` represents an administrator's authorized collection sources
        and policies, not collected history. Retaining it lets collection resume after
        a library reset without silently expanding access or requiring re-authorization.
        Security audit records are likewise intentionally retained outside this method.
        """
        with self.database.transaction(write=True) as connection:
            file_references = self._project_feishu_file_references_connection(connection, self.project_id)
            counts = {
                "groups_retained": int(connection.execute(
                    "SELECT COUNT(*) FROM feishu_groups WHERE project_id=?", (self.project_id,)
                ).fetchone()[0]),
                "messages_deleted": int(connection.execute(
                    "SELECT COUNT(*) FROM feishu_messages WHERE project_id=?", (self.project_id,)
                ).fetchone()[0]),
                "candidates_deleted": int(connection.execute(
                    "SELECT COUNT(*) FROM feishu_candidates WHERE project_id=?", (self.project_id,)
                ).fetchone()[0]),
                "assets_deleted": int(connection.execute(
                    "SELECT COUNT(*) FROM feishu_assets WHERE project_id=?", (self.project_id,)
                ).fetchone()[0]),
                "diagnostics_deleted": int(connection.execute(
                    "SELECT COUNT(*) FROM feishu_diagnostics WHERE project_id=?", (self.project_id,)
                ).fetchone()[0]),
            }
            # Junction rows must be removed while their project-scoped parents still
            # exist. The order is explicit rather than relying on SQLite FK settings.
            connection.execute(
                "DELETE FROM feishu_candidate_messages WHERE candidate_id IN "
                "(SELECT candidate_id FROM feishu_candidates WHERE project_id=?)",
                (self.project_id,),
            )
            connection.execute(
                "DELETE FROM feishu_asset_messages WHERE asset_id IN "
                "(SELECT asset_id FROM feishu_assets WHERE project_id=?)",
                (self.project_id,),
            )
            connection.execute("DELETE FROM feishu_assets WHERE project_id=?", (self.project_id,))
            connection.execute("DELETE FROM feishu_candidates WHERE project_id=?", (self.project_id,))
            connection.execute("DELETE FROM feishu_messages WHERE project_id=?", (self.project_id,))
            connection.execute("DELETE FROM feishu_diagnostics WHERE project_id=?", (self.project_id,))

            revision_row = connection.execute(
                "SELECT value_text FROM system_state WHERE state_key=?", (f"feishu_revision:{self.project_id}",)
            ).fetchone()
            next_revision = int(revision_row["value_text"] if revision_row else 0) + 1
            connection.execute(
                """
                INSERT INTO system_state(state_key,value_text,updated_at) VALUES(?,?,?)
                ON CONFLICT(state_key) DO UPDATE SET value_text=excluded.value_text,updated_at=excluded.updated_at
                """,
                (f"feishu_revision:{self.project_id}", str(next_revision), utc_now()),
            )
        return {
            **counts,
            **file_references,
            "revision": next_revision,
        }

    def clear_project_demo_history(self) -> Dict[str, int]:
        """Clear resettable project histories while preserving accounts and configuration."""
        with self.database.transaction(write=True) as connection:
            counts = {
                "processing_jobs": int(connection.execute(
                    "SELECT COUNT(*) FROM processing_jobs WHERE project_id=?", (self.project_id,)
                ).fetchone()[0]),
                "knowledge_tasks": int(connection.execute(
                    "SELECT COUNT(*) FROM knowledge_tasks WHERE project_id=?", (self.project_id,)
                ).fetchone()[0]),
                "handover_cases": int(connection.execute(
                    "SELECT COUNT(*) FROM handover_cases WHERE project_id=?", (self.project_id,)
                ).fetchone()[0]),
                "onboarding_learning_plans": int(connection.execute(
                    "SELECT COUNT(*) FROM onboarding_learning_plans WHERE project_id=?", (self.project_id,)
                ).fetchone()[0]),
            }
            # Snapshots are immutable during normal operation. A project reset is the
            # explicit administrative exception and restores the guard immediately.
            connection.execute("DROP TRIGGER IF EXISTS trg_handover_snapshots_no_delete")
            connection.execute(
                "DELETE FROM handover_snapshots WHERE project_id=?", (self.project_id,)
            )
            connection.execute(
                """
                CREATE TRIGGER IF NOT EXISTS trg_handover_snapshots_no_delete
                BEFORE DELETE ON handover_snapshots
                BEGIN
                    SELECT RAISE(ABORT, 'handover snapshot is immutable');
                END
                """
            )
            connection.execute("DELETE FROM handover_cases WHERE project_id=?", (self.project_id,))
            connection.execute("DELETE FROM onboarding_learning_plans WHERE project_id=?", (self.project_id,))
            connection.execute("DELETE FROM knowledge_tasks WHERE project_id=?", (self.project_id,))
            connection.execute("DELETE FROM governance_schedule_intents WHERE project_id=?", (self.project_id,))
            connection.execute("DELETE FROM processing_jobs WHERE project_id=?", (self.project_id,))
        return counts

    # Handovers ----------------------------------------------------------
    def list_handovers(self) -> List[Dict[str, Any]]:
        with self.database.transaction() as connection:
            rows = connection.execute(
                "SELECT * FROM handover_cases WHERE project_id=? ORDER BY created_at DESC", (self.project_id,)
            ).fetchall()
            return [self._handover_row(connection, row) for row in rows]

    def get_handover(self, handover_id: str) -> Optional[Dict[str, Any]]:
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM handover_cases WHERE project_id=? AND handover_id=?",
                (self.project_id, str(handover_id or "")),
            ).fetchone()
            return self._handover_row(connection, row) if row else None

    def save_handover(
        self, record: Dict[str, Any], *, actor: str = "user_system", action: str = "handover.updated",
    ) -> Dict[str, Any]:
        handover_id = str(record.get("id") or record.get("handover_id") or "").strip()
        if not handover_id:
            raise ValueError("交接记录缺少 ID")
        with self.database.transaction(write=True) as connection:
            self._upsert_handover_connection(connection, record)
            self._write_audit_connection(
                connection, f"audit_{uuid.uuid4().hex}", actor, action,
                "handover", handover_id,
                {
                    "status": str(record.get("status") or ""),
                    "recipient_user_id": str(record.get("recipient_user_id") or ""),
                    "learning_plan_status": str(record.get("learning_plan_status") or ""),
                    "learning_plan_id": str(record.get("learning_plan_id") or ""),
                },
                project_id=self.project_id, created_at=utc_now(),
            )
            row = connection.execute(
                "SELECT * FROM handover_cases WHERE project_id=? AND handover_id=?",
                (self.project_id, handover_id),
            ).fetchone()
            return self._handover_row(connection, row)

    def cancel_handover(self, handover_id: str, *, actor: str) -> Dict[str, Any]:
        """Remove an unaccepted handover and its workflow rows, preserving documents."""
        handover_id = str(handover_id or "").strip()
        with self.database.transaction(write=True) as connection:
            case = connection.execute(
                "SELECT * FROM handover_cases WHERE project_id=? AND handover_id=?",
                (self.project_id, handover_id),
            ).fetchone()
            if not case:
                raise ValueError("交接记录不存在")
            if connection.execute(
                "SELECT 1 FROM handover_snapshots WHERE handover_id=?", (handover_id,)
            ).fetchone():
                raise ValueError("已封存的交接不能取消")
            record = self._handover_row(connection, case)
            connection.execute(
                "DELETE FROM handover_cases WHERE project_id=? AND handover_id=?",
                (self.project_id, handover_id),
            )
            self._write_audit_connection(
                connection, f"audit_{uuid.uuid4().hex}", actor, "handover.cancelled",
                "handover", handover_id,
                {"status": str(record.get("status") or ""), "document_count": len(record.get("files") or [])},
                project_id=self.project_id, created_at=utc_now(),
            )
            return record

    def list_handover_asset_ids(self, handover_id: str) -> List[str]:
        with self.database.transaction() as connection:
            return [
                str(row["asset_id"])
                for row in connection.execute(
                    """
                    SELECT DISTINCT asset_id FROM documents
                    WHERE project_id=? AND handover_id=? AND status<>'deleted' AND asset_id<>''
                    ORDER BY asset_id
                    """,
                    (self.project_id, str(handover_id or "")),
                )
            ]

    def list_handover_asset_versions(self, handover_id: str) -> List[Dict[str, str]]:
        with self.database.transaction() as connection:
            rows = [
                {
                    "asset_id": str(row["asset_id"] or ""),
                    "version_id": str(row["version_id"] or ""),
                    "source_file": str(row["source_file"] or ""),
                    "stored_file": str(row["stored_file"] or ""),
                }
                for row in connection.execute(
                    """
                    SELECT DISTINCT asset_id,version_id,source_file,stored_file FROM documents
                    WHERE project_id=? AND handover_id=? AND status<>'deleted' AND asset_id<>''
                    ORDER BY source_file,asset_id,version_id
                    """,
                    (self.project_id, str(handover_id or "")),
                )
            ]
            known = {(item["asset_id"], item["version_id"]) for item in rows}
            for row in connection.execute(
                """
                SELECT ka.asset_id,ka.current_version_id AS version_id,
                       ks.display_name AS source_file,'' AS stored_file
                FROM handover_inventory_entries hi
                JOIN knowledge_assets ka ON ka.asset_id=hi.reference_id
                JOIN knowledge_sources ks ON ks.source_id=ka.primary_source_id
                WHERE hi.project_id=? AND hi.handover_id=?
                  AND hi.source_type='knowledge_asset' AND hi.status='included'
                  AND ka.status NOT IN ('revoked','deleted')
                ORDER BY ka.title,ka.asset_id
                """,
                (self.project_id, str(handover_id or "")),
            ):
                key = (str(row["asset_id"] or ""), str(row["version_id"] or ""))
                if key in known:
                    continue
                rows.append({key_name: str(row[key_name] or "") for key_name in ("asset_id", "version_id", "source_file", "stored_file")})
                known.add(key)
            return rows

    @staticmethod
    def _normalize_handover_evidence_refs(value: Any) -> List[Dict[str, str]]:
        """Keep only stable evidence identifiers; presentation data is always reloaded."""
        if not isinstance(value, list):
            return []
        refs: List[Dict[str, str]] = []
        seen = set()
        for entry in value:
            if not isinstance(entry, dict):
                continue
            asset_id = str(entry.get("asset_id") or "").strip()
            version_id = str(entry.get("version_id") or "").strip()
            if not asset_id or not version_id or (asset_id, version_id) in seen:
                continue
            source_kind = str(entry.get("source_kind") or "existing").strip().lower()
            if source_kind not in {"existing", "uploaded", "legacy"}:
                source_kind = "existing"
            refs.append({
                "asset_id": asset_id,
                "version_id": version_id,
                "source_kind": source_kind,
            })
            seen.add((asset_id, version_id))
        return refs

    def _handover_evidence_reference_payload(
        self, reference: Dict[str, Any], *, identity: Any = None,
    ) -> Dict[str, Any]:
        """Resolve an evidence pointer without leaking a later-restricted asset title."""
        asset_id = str(reference.get("asset_id") or "")
        version_id = str(reference.get("version_id") or "")
        asset = self.get_asset(asset_id, identity=identity)
        if not asset:
            return {
                "asset_id": asset_id, "version_id": version_id,
                "source_kind": str(reference.get("source_kind") or "existing"),
                "available": False, "title": "资料已不可访问",
                "source_file": "", "version_status": "unavailable",
            }
        version = self.get_asset_version(asset_id, version_id, identity=identity)
        if not version:
            return {
                "asset_id": asset_id, "version_id": version_id,
                "source_kind": str(reference.get("source_kind") or "existing"),
                "available": False, "title": "资料版本已不可访问",
                "source_file": "", "version_status": "unavailable",
            }
        documents = list(version.get("documents") or [])
        return {
            "asset_id": asset_id,
            "version_id": version_id,
            "source_kind": str(reference.get("source_kind") or "existing"),
            "available": True,
            "title": str(asset.get("title") or "未命名资料"),
            "source_file": str((documents[0] if documents else {}).get("source_file") or ""),
            "version_status": str(version.get("status") or ""),
            "categories": list(asset.get("categories") or []),
        }

    def list_handover_item_evidence_assets(
        self, handover_id: str, item_id: str, *, identity: Any = None,
    ) -> List[Dict[str, Any]]:
        with self.database.transaction() as connection:
            rows = connection.execute(
                """
                SELECT asset_id,version_id,source_kind,attached_by,attached_at
                FROM handover_item_evidence_assets
                WHERE project_id=? AND handover_id=? AND item_id=?
                ORDER BY attached_at,evidence_id
                """,
                (self.project_id, str(handover_id or ""), str(item_id or "")),
            ).fetchall()
        return [
            {
                **self._handover_evidence_reference_payload(dict(row), identity=identity),
                "attached_by": str(row["attached_by"] or ""),
                "attached_at": str(row["attached_at"] or ""),
            }
            for row in rows
        ]

    def get_handover_item_draft(
        self, handover_id: str, item_id: str, user_id: str,
    ) -> Dict[str, Any]:
        with self.database.transaction() as connection:
            row = connection.execute(
                """
                SELECT evidence,asset_refs_json,created_at,updated_at
                FROM handover_item_drafts
                WHERE project_id=? AND handover_id=? AND item_id=? AND user_id=?
                """,
                (self.project_id, str(handover_id or ""), str(item_id or ""), str(user_id or "")),
            ).fetchone()
        if not row:
            return {"evidence": "", "asset_refs": [], "created_at": "", "updated_at": ""}
        return {
            "evidence": str(row["evidence"] or ""),
            "asset_refs": self._normalize_handover_evidence_refs(_json_load(row["asset_refs_json"], [])),
            "created_at": str(row["created_at"] or ""),
            "updated_at": str(row["updated_at"] or ""),
        }

    def save_handover_item_draft(
        self, handover_id: str, item_id: str, user_id: str, *,
        evidence: str, asset_refs: Any,
    ) -> Dict[str, Any]:
        now = utc_now()
        refs = self._normalize_handover_evidence_refs(asset_refs)
        if len(refs) > 10:
            raise ValueError("每个必交项最多关联 10 份资料")
        with self.database.transaction(write=True) as connection:
            if not connection.execute(
                "SELECT 1 FROM handover_items WHERE project_id=? AND handover_id=? AND item_id=?",
                (self.project_id, str(handover_id or ""), str(item_id or "")),
            ).fetchone():
                raise ValueError("交接项不存在")
            connection.execute(
                """
                INSERT INTO handover_item_drafts(
                    handover_id,item_id,project_id,user_id,evidence,asset_refs_json,created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?,?)
                ON CONFLICT(handover_id,item_id,user_id) DO UPDATE SET
                    evidence=excluded.evidence,asset_refs_json=excluded.asset_refs_json,updated_at=excluded.updated_at
                """,
                (
                    str(handover_id or ""), str(item_id or ""), self.project_id, str(user_id or ""),
                    str(evidence or "")[:4000], _json_dump(refs), now, now,
                ),
            )
        return self.get_handover_item_draft(handover_id, item_id, user_id)

    def clear_handover_item_draft(
        self, connection: sqlite3.Connection, handover_id: str, item_id: str, user_id: str,
    ) -> None:
        connection.execute(
            "DELETE FROM handover_item_drafts WHERE project_id=? AND handover_id=? AND item_id=? AND user_id=?",
            (self.project_id, handover_id, item_id, user_id),
        )

    def _replace_handover_item_evidence_assets_connection(
        self, connection: sqlite3.Connection, handover_id: str, item_id: str,
        refs: Any, *, actor: str, now: str,
    ) -> List[Dict[str, str]]:
        normalized = self._normalize_handover_evidence_refs(refs)
        if len(normalized) > 10:
            raise ValueError("每个必交项最多关联 10 份资料")
        connection.execute(
            "DELETE FROM handover_item_evidence_assets WHERE project_id=? AND handover_id=? AND item_id=?",
            (self.project_id, handover_id, item_id),
        )
        for reference in normalized:
            connection.execute(
                """
                INSERT INTO handover_item_evidence_assets(
                    evidence_id,handover_id,item_id,project_id,asset_id,version_id,source_kind,attached_by,attached_at
                ) VALUES(?,?,?,?,?,?,?,?,?)
                """,
                (
                    f"hie_{uuid.uuid4().hex}", handover_id, item_id, self.project_id,
                    reference["asset_id"], reference["version_id"], reference["source_kind"], actor, now,
                ),
            )
        return normalized

    def backfill_handover_governance(self) -> int:
        """Idempotently add the governed checklist to legacy handover cases."""
        with self.database.transaction(write=True) as connection:
            rows = connection.execute(
                "SELECT * FROM handover_cases WHERE project_id=?", (self.project_id,)
            ).fetchall()
            inserted = 0
            for row in rows:
                data = _json_load(row["data_json"], {})
                inserted += self._ensure_handover_items_connection(
                    connection,
                    str(row["handover_id"]),
                    owner_user_id=str(data.get("departing_user_id") or data.get("created_by_user_id") or ""),
                    recipient_user_id=str(data.get("recipient_user_id") or ""),
                    due_at=str(row["due_date"] or ""),
                    role_template_id=str(data.get("role_template_id") or ""),
                )
                departing_user_id = str(data.get("departing_user_id") or "")
                has_inventory = connection.execute(
                    "SELECT 1 FROM handover_inventory_entries WHERE handover_id=? LIMIT 1",
                    (str(row["handover_id"]),),
                ).fetchone()
                if departing_user_id and not has_inventory and str(row["status"] or "") != "completed":
                    self._refresh_handover_inventory_connection(
                        connection, str(row["handover_id"]), departing_user_id,
                        actor=str(data.get("created_by_user_id") or "user_system"),
                    )
            return inserted

    def _ensure_handover_items_connection(
        self, connection: sqlite3.Connection, handover_id: str, *,
        owner_user_id: str = "", recipient_user_id: str = "", due_at: str = "",
        role_template_id: str = "",
    ) -> int:
        now = utc_now()
        before = connection.total_changes
        for position, (item_key, title, description) in enumerate(DEFAULT_HANDOVER_ITEMS, 1):
            connection.execute(
                """
                INSERT OR IGNORE INTO handover_items(
                    item_id,handover_id,project_id,item_key,item_type,title,description,required,
                    owner_user_id,recipient_user_id,due_at,status,position,created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    _stable_id("hi", handover_id, item_key), handover_id, self.project_id,
                    item_key, "knowledge", title, description, 1, owner_user_id,
                    recipient_user_id, due_at, "pending", position, now, now,
                ),
            )
        if role_template_id and not connection.execute(
            "SELECT 1 FROM handover_items WHERE handover_id=? AND item_type='role_topic' LIMIT 1",
            (handover_id,),
        ).fetchone():
            template = connection.execute(
                "SELECT * FROM onboarding_role_templates WHERE project_id=? AND template_id=? AND status='active'",
                (self.project_id, role_template_id),
            ).fetchone()
            if template:
                topics = connection.execute(
                    "SELECT * FROM onboarding_template_topics WHERE template_id=? ORDER BY position,topic_key",
                    (role_template_id,),
                ).fetchall()
                for offset, topic in enumerate(topics, len(DEFAULT_HANDOVER_ITEMS) + 1):
                    standard = str(topic["completion_standard"] or "").strip()
                    practice = str(topic["practice_task"] or "").strip()
                    details = [f"岗位模板“{template['name']}”要求交接该知识主题。"]
                    if standard:
                        details.append(f"验收标准：{standard}")
                    if practice:
                        details.append(f"接替实践：{practice}")
                    item_key = f"role_topic:{role_template_id}:{topic['topic_key']}"
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO handover_items(
                            item_id,handover_id,project_id,item_key,item_type,title,description,required,
                            owner_user_id,recipient_user_id,due_at,status,position,created_at,updated_at
                        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            _stable_id("hi", handover_id, item_key), handover_id, self.project_id,
                            item_key, "role_topic", f"岗位知识：{topic['label']}", "\n".join(details), 1,
                            str(topic["owner_user_id"] or owner_user_id), recipient_user_id, due_at,
                            "pending", offset, now, now,
                        ),
                    )
        return connection.total_changes - before

    def get_handover_item(self, handover_id: str, item_id: str) -> Optional[Dict[str, Any]]:
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM handover_items WHERE project_id=? AND handover_id=? AND item_id=?",
                (self.project_id, str(handover_id or ""), str(item_id or "")),
            ).fetchone()
            return self._handover_item_payload(row) if row else None

    @staticmethod
    def _handover_inventory_payload(row: sqlite3.Row) -> Dict[str, Any]:
        payload = dict(row)
        payload["metadata"] = _json_load(payload.pop("metadata_json", "{}"), {})
        return payload

    def _refresh_handover_inventory_connection(
        self, connection: sqlite3.Connection, handover_id: str,
        departing_user_id: str, *, actor: str,
    ) -> Dict[str, int]:
        """Discover deterministic ownership/participation facts without resetting reviews."""
        now = utc_now()
        discovered: List[Dict[str, Any]] = []
        asset_ids = set()
        assets = connection.execute(
            """
            SELECT ka.*,ks.source_type,ks.display_name AS source_name
            FROM knowledge_assets ka
            JOIN knowledge_sources ks ON ks.source_id=ka.primary_source_id
            WHERE ka.project_id=? AND ka.status NOT IN ('revoked','deleted')
              AND (
                ka.owner_user_id=? OR ka.created_by=? OR ka.updated_by=?
                OR EXISTS(
                    SELECT 1 FROM documents d
                    WHERE d.project_id=ka.project_id AND d.asset_id=ka.asset_id
                      AND d.owner_user_id=? AND d.status<>'deleted'
                )
              )
            ORDER BY ka.title,ka.asset_id
            """,
            (self.project_id, departing_user_id, departing_user_id, departing_user_id, departing_user_id),
        ).fetchall()
        for asset in assets:
            asset_id = str(asset["asset_id"])
            asset_ids.add(asset_id)
            if str(asset["owner_user_id"] or "") == departing_user_id:
                relation_type, relation_detail = "owner", "离职人员是当前知识责任人"
            elif str(asset["created_by"] or "") == departing_user_id:
                relation_type, relation_detail = "creator", "离职人员创建了该知识资产"
            elif str(asset["updated_by"] or "") == departing_user_id:
                relation_type, relation_detail = "maintainer", "离职人员是最近维护人"
            else:
                relation_type, relation_detail = "uploader", "离职人员上传或持有该资产来源"
            discovered.append({
                "source_type": "knowledge_asset", "reference_id": asset_id,
                "version_id": str(asset["current_version_id"] or ""),
                "title": str(asset["title"] or asset["source_name"] or asset_id),
                "relation_type": relation_type, "relation_detail": relation_detail,
                "metadata": {
                    "asset_status": str(asset["status"] or ""),
                    "source_type": str(asset["source_type"] or ""),
                    "topic": str(asset["topic"] or ""),
                },
            })

        tasks = connection.execute(
            """
            SELECT * FROM knowledge_tasks
            WHERE project_id=? AND status NOT IN ('completed','cancelled')
              AND (assignee_user_id=? OR reporter_user_id=?)
            ORDER BY updated_at DESC,task_id
            """,
            (self.project_id, departing_user_id, departing_user_id),
        ).fetchall()
        for task in tasks:
            assignee = str(task["assignee_user_id"] or "") == departing_user_id
            discovered.append({
                "source_type": "knowledge_task", "reference_id": str(task["task_id"]),
                "version_id": "", "title": str(task["title"] or task["task_id"]),
                "relation_type": "assignee" if assignee else "reporter",
                "relation_detail": "离职人员是未完成任务经办人" if assignee else "离职人员报告了该未完成任务",
                "metadata": {
                    "task_type": str(task["task_type"] or ""), "status": str(task["status"] or ""),
                    "priority": str(task["priority"] or ""), "due_at": str(task["due_at"] or ""),
                },
            })

        for node in connection.execute(
            "SELECT * FROM graph_nodes WHERE project_id=? ORDER BY node_id LIMIT 1000",
            (self.project_id,),
        ).fetchall():
            data = _json_load(node["data_json"], {})
            node_asset_id = str(data.get("asset_id") or "")
            user_fields = {
                str(data.get(key) or "")
                for key in ("owner_user_id", "created_by", "updated_by", "decision_maker_user_id")
            }
            if departing_user_id not in user_fields and node_asset_id not in asset_ids:
                continue
            relation_type = "decision_participant" if str(data.get("decision_maker_user_id") or "") == departing_user_id else "asset_topic"
            discovered.append({
                "source_type": "graph_topic", "reference_id": str(node["node_id"]),
                "version_id": str(data.get("version_id") or ""),
                "title": str(data.get("label") or data.get("title") or data.get("name") or node["node_id"]),
                "relation_type": relation_type,
                "relation_detail": "离职人员参与该决策主题" if relation_type == "decision_participant" else "该图谱主题来源于离职人员关联资产",
                "metadata": {"asset_id": node_asset_id, "source": str(node["source"] or "")},
            })

        before = connection.total_changes
        for item in discovered:
            inventory_id = _stable_id(
                "hinv", handover_id, item["source_type"], item["reference_id"], item["version_id"],
            )
            connection.execute(
                """
                INSERT INTO handover_inventory_entries(
                    inventory_id,handover_id,project_id,source_type,reference_id,version_id,title,
                    relation_type,relation_detail,status,metadata_json,created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,'pending',?,?,?)
                ON CONFLICT(handover_id,source_type,reference_id,version_id) DO UPDATE SET
                    title=excluded.title,relation_type=excluded.relation_type,
                    relation_detail=excluded.relation_detail,metadata_json=excluded.metadata_json,
                    updated_at=excluded.updated_at
                """,
                (
                    inventory_id, handover_id, self.project_id, item["source_type"],
                    item["reference_id"], item["version_id"], item["title"],
                    item["relation_type"], item["relation_detail"], _json_dump(item["metadata"]), now, now,
                ),
            )
        changed = connection.total_changes - before
        if discovered:
            self._write_audit_connection(
                connection, f"audit_{uuid.uuid4().hex}", actor, "handover.inventory_refreshed",
                "handover", handover_id,
                {"discovered_count": len(discovered), "changed_count": changed},
                project_id=self.project_id, created_at=now,
            )
        return {"discovered_count": len(discovered), "changed_count": changed}

    def refresh_handover_inventory(
        self, handover_id: str, departing_user_id: str, *, actor: str,
    ) -> Dict[str, Any]:
        with self.database.transaction(write=True) as connection:
            case = connection.execute(
                "SELECT status FROM handover_cases WHERE project_id=? AND handover_id=?",
                (self.project_id, handover_id),
            ).fetchone()
            if not case:
                raise ValueError("交接记录不存在")
            if str(case["status"] or "") == "completed":
                raise ValueError("交接已关闭，不能刷新知识盘点")
            summary = self._refresh_handover_inventory_connection(
                connection, handover_id, departing_user_id, actor=actor,
            )
            rows = connection.execute(
                "SELECT * FROM handover_inventory_entries WHERE handover_id=? ORDER BY source_type,title,inventory_id",
                (handover_id,),
            ).fetchall()
            return {**summary, "entries": [self._handover_inventory_payload(row) for row in rows]}

    def update_handover_inventory(
        self, handover_id: str, inventory_id: str, action: str,
        payload: Dict[str, Any], *, actor: str,
    ) -> Dict[str, Any]:
        now = utc_now()
        with self.database.transaction(write=True) as connection:
            case = connection.execute(
                "SELECT status FROM handover_cases WHERE project_id=? AND handover_id=?",
                (self.project_id, handover_id),
            ).fetchone()
            if not case:
                raise ValueError("交接记录不存在")
            if str(case["status"] or "") == "completed":
                raise ValueError("交接已关闭，不能修改知识盘点")
            row = connection.execute(
                "SELECT * FROM handover_inventory_entries WHERE handover_id=? AND inventory_id=?",
                (handover_id, inventory_id),
            ).fetchone()
            if not row:
                raise ValueError("知识盘点项不存在")
            if action == "include":
                target, reason = "included", ""
            elif action == "exclude":
                target = "excluded"
                reason = str(payload.get("reason") or "").strip()
                if not reason:
                    raise ValueError("排除盘点项必须说明原因")
            elif action == "reset":
                target, reason = "pending", ""
            else:
                raise ValueError("不支持的知识盘点操作")
            connection.execute(
                """
                UPDATE handover_inventory_entries
                SET status=?,exclusion_reason=?,reviewed_by=?,reviewed_at=?,updated_at=?
                WHERE handover_id=? AND inventory_id=?
                """,
                (target, reason[:1000], actor if target != "pending" else "", now if target != "pending" else "", now, handover_id, inventory_id),
            )
            self._write_audit_connection(
                connection, f"audit_{uuid.uuid4().hex}", actor, f"handover.inventory_{action}",
                "handover_inventory", inventory_id,
                {"handover_id": handover_id, "from_status": str(row["status"]), "to_status": target},
                project_id=self.project_id, created_at=now,
            )
            updated = connection.execute(
                "SELECT * FROM handover_inventory_entries WHERE inventory_id=?", (inventory_id,),
            ).fetchone()
            return self._handover_inventory_payload(updated)

    def update_handover_item(
        self, handover_id: str, item_id: str, action: str, payload: Dict[str, Any], *, actor: str,
    ) -> Dict[str, Any]:
        now = utc_now()
        with self.database.transaction(write=True) as connection:
            case = connection.execute(
                "SELECT status,data_json FROM handover_cases WHERE project_id=? AND handover_id=?",
                (self.project_id, handover_id),
            ).fetchone()
            if not case:
                raise ValueError("交接记录不存在")
            if str(case["status"]) == "completed":
                raise ValueError("交接已关闭，不能再修改交接项")
            item = connection.execute(
                "SELECT * FROM handover_items WHERE project_id=? AND handover_id=? AND item_id=?",
                (self.project_id, handover_id, item_id),
            ).fetchone()
            if not item:
                raise ValueError("交接项不存在")
            current = str(item["status"])
            assignments = ["updated_at=?", "revision=revision+1"]
            params: List[Any] = [now]
            review_mode, proxy_reason = "", ""
            evidence_refs: List[Dict[str, str]] = []
            if action in {"accept", "reject"}:
                # Recheck persisted identities inside the write transaction to avoid stale eligibility.
                member = connection.execute(
                    "SELECT m.role FROM project_memberships m JOIN users u ON u.user_id=m.user_id "
                    "WHERE m.project_id=? AND m.user_id=? AND m.status='active' AND u.status='active'",
                    (self.project_id, actor),
                ).fetchone()
                if not member:
                    raise PermissionError("只有当前项目有效成员可以验收交接项")
                record = {**_json_load(case["data_json"], {}), "status": case["status"]}
                policy = item_review_policy(
                    record, dict(item), actor=actor,
                    can_manage=member["role"] in {"project_admin", "project_manager"},
                )
                if action not in policy["review_actions"]:
                    raise PermissionError(policy["review_block_reason"] or "当前交接项不能验收")
                review_mode = policy["review_action_mode"]
                if review_mode == "proxy":
                    proxy_reason = str(payload.get("proxy_reason") or "").strip()
                    if not proxy_reason:
                        raise ValueError("请填写代验收原因")
                    if len(proxy_reason) > 1000:
                        raise ValueError("代验收原因不能超过 1000 字")
                assignments.extend(("review_mode=?", "proxy_reason=?"))
                params.extend((review_mode, proxy_reason))
            if action == "submit":
                if current not in {"pending", "rejected"}:
                    raise ValueError("当前状态不能提交或重新提交")
                evidence = str(payload.get("evidence") or "").strip()
                raw_refs = payload.get("asset_refs")
                if not raw_refs and str(payload.get("asset_id") or "").strip():
                    raw_refs = [{
                        "asset_id": str(payload.get("asset_id") or "").strip(),
                        "version_id": str(payload.get("version_id") or "").strip(),
                        "source_kind": "legacy",
                    }]
                evidence_refs = self._normalize_handover_evidence_refs(raw_refs)
                if not evidence and not evidence_refs:
                    raise ValueError("提交交接项必须填写证据或资料说明")
                primary = evidence_refs[0] if evidence_refs else {"asset_id": "", "version_id": ""}
                assignments.extend((
                    "status='submitted'", "validity_status='unknown'", "evidence=?", "asset_id=?",
                    "version_id=?", "rejection_reason=''", "submitted_by=?", "submitted_at=?",
                    "reviewed_by=''", "reviewed_at=''", "review_mode=''", "proxy_reason=''",
                ))
                params.extend((
                    evidence[:4000], primary["asset_id"], primary["version_id"], actor, now,
                ))
            elif action == "accept":
                if current != "submitted":
                    raise ValueError("只有待验收交接项可以接收")
                assignments.extend((
                    "status='accepted'", "validity_status='valid'", "rejection_reason=''",
                    "reviewed_by=?", "reviewed_at=?",
                ))
                params.extend((actor, now))
            elif action == "reject":
                if current != "submitted":
                    raise ValueError("只有待验收交接项可以退回")
                reason = str(payload.get("reason") or "").strip()
                if not reason:
                    raise ValueError("退回交接项必须说明补充要求")
                assignments.extend((
                    "status='rejected'", "validity_status='invalid'", "rejection_reason=?",
                    "knowledge_task_id=?", "reviewed_by=?", "reviewed_at=?",
                ))
                params.extend((reason[:2000], str(payload.get("knowledge_task_id") or ""), actor, now))
            else:
                raise ValueError("不支持的交接项操作")
            params.extend((item_id, handover_id, self.project_id))
            connection.execute(
                f"UPDATE handover_items SET {','.join(assignments)} WHERE item_id=? AND handover_id=? AND project_id=?",
                tuple(params),
            )
            if action == "submit":
                self._replace_handover_item_evidence_assets_connection(
                    connection, handover_id, item_id, evidence_refs, actor=actor, now=now,
                )
                self.clear_handover_item_draft(connection, handover_id, item_id, actor)
            self._write_audit_connection(
                connection, f"audit_{uuid.uuid4().hex}", actor, f"handover.item_{action}",
                "handover_item", item_id,
                {"handover_id": handover_id, "from_status": current,
                 "has_evidence": bool(payload.get("evidence")), "evidence_asset_count": len(evidence_refs),
                 "review_mode": review_mode,
                 "proxy_reason": proxy_reason},
                project_id=self.project_id, created_at=now,
            )
            updated = connection.execute("SELECT * FROM handover_items WHERE item_id=?", (item_id,)).fetchone()
            return self._handover_item_payload(updated)

    def create_handover_risk(
        self, handover_id: str, payload: Dict[str, Any], *, actor: str,
    ) -> Dict[str, Any]:
        now = utc_now()
        risk_id = str(payload.get("risk_id") or f"hr_{uuid.uuid4().hex}")
        with self.database.transaction(write=True) as connection:
            case = connection.execute(
                "SELECT status FROM handover_cases WHERE project_id=? AND handover_id=?",
                (self.project_id, handover_id),
            ).fetchone()
            if not case:
                raise ValueError("交接记录不存在")
            if str(case["status"]) == "completed":
                raise ValueError("交接已关闭，不能新增风险")
            connection.execute(
                """
                INSERT INTO handover_risks(
                    risk_id,handover_id,project_id,title,impact,severity,owner_user_id,due_at,
                    mitigation,status,created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    risk_id, handover_id, self.project_id, str(payload.get("title") or "").strip(),
                    str(payload.get("impact") or "").strip(), str(payload.get("severity") or "medium"),
                    str(payload.get("owner_user_id") or ""), str(payload.get("due_at") or ""),
                    str(payload.get("mitigation") or "").strip(), "open", now, now,
                ),
            )
            self._write_audit_connection(
                connection, f"audit_{uuid.uuid4().hex}", actor, "handover.risk_created",
                "handover_risk", risk_id,
                {"handover_id": handover_id, "severity": str(payload.get("severity") or "medium")},
                project_id=self.project_id, created_at=now,
            )
            row = connection.execute("SELECT * FROM handover_risks WHERE risk_id=?", (risk_id,)).fetchone()
            return self._handover_risk_payload(row)

    def update_handover_risk(
        self, handover_id: str, risk_id: str, action: str, payload: Dict[str, Any], *, actor: str,
    ) -> Dict[str, Any]:
        now = utc_now()
        with self.database.transaction(write=True) as connection:
            case = connection.execute(
                "SELECT status FROM handover_cases WHERE project_id=? AND handover_id=?",
                (self.project_id, handover_id),
            ).fetchone()
            if not case:
                raise ValueError("交接记录不存在")
            if str(case["status"]) == "completed":
                raise ValueError("交接已关闭，不能再修改风险")
            risk = connection.execute(
                "SELECT * FROM handover_risks WHERE project_id=? AND handover_id=? AND risk_id=?",
                (self.project_id, handover_id, risk_id),
            ).fetchone()
            if not risk:
                raise ValueError("交接风险不存在")
            current = str(risk["status"])
            assignments = ["updated_at=?", "revision=revision+1"]
            params: List[Any] = [now]
            if action == "mitigate":
                if current not in {"open", "mitigating"}:
                    raise ValueError("当前状态不能更新缓解措施")
                mitigation = str(payload.get("mitigation") or "").strip()
                if not mitigation:
                    raise ValueError("请填写风险缓解措施")
                assignments.extend(("status='mitigating'", "mitigation=?"))
                params.append(mitigation[:4000])
            elif action == "submit_close":
                if current not in {"open", "mitigating"}:
                    raise ValueError("当前状态不能提交关闭验收")
                evidence = str(payload.get("close_evidence") or "").strip()
                if not evidence:
                    raise ValueError("提交风险关闭前必须填写关闭证据")
                mitigation = str(payload.get("mitigation") or risk["mitigation"] or "").strip()
                if not mitigation:
                    raise ValueError("提交风险关闭前必须填写缓解措施")
                assignments.extend((
                    "status='pending_close'", "mitigation=?", "close_evidence=?",
                    "submitted_by=?", "submitted_at=?",
                ))
                params.extend((mitigation[:4000], evidence[:4000], actor, now))
            elif action == "close":
                if current != "pending_close":
                    raise ValueError("只有待验收风险可以关闭")
                assignments.extend(("status='closed'", "closed_by=?", "closed_at=?"))
                params.extend((actor, now))
            elif action == "reopen":
                if current not in {"pending_close", "closed"}:
                    raise ValueError("当前状态不能重新打开风险")
                reason = str(payload.get("reason") or "").strip()
                if not reason:
                    raise ValueError("重新打开风险必须说明原因")
                assignments.extend((
                    "status='mitigating'", "close_evidence=?", "closed_by=''", "closed_at=''",
                ))
                params.append(f"原关闭证据：{risk['close_evidence']}\n重新打开原因：{reason}"[:4000])
            else:
                raise ValueError("不支持的风险操作")
            params.extend((risk_id, handover_id, self.project_id))
            connection.execute(
                f"UPDATE handover_risks SET {','.join(assignments)} WHERE risk_id=? AND handover_id=? AND project_id=?",
                tuple(params),
            )
            self._write_audit_connection(
                connection, f"audit_{uuid.uuid4().hex}", actor, f"handover.risk_{action}",
                "handover_risk", risk_id,
                {"handover_id": handover_id, "from_status": current},
                project_id=self.project_id, created_at=now,
            )
            updated = connection.execute("SELECT * FROM handover_risks WHERE risk_id=?", (risk_id,)).fetchone()
            return self._handover_risk_payload(updated)

    def get_handover_snapshot(self, handover_id: str) -> Optional[Dict[str, Any]]:
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM handover_snapshots WHERE project_id=? AND handover_id=?",
                (self.project_id, handover_id),
            ).fetchone()
            return self._handover_snapshot_payload(row, include_payload=True) if row else None

    def complete_handover(self, handover_id: str, *, actor: str) -> Dict[str, Any]:
        """Validate all gates and atomically seal an immutable versioned snapshot."""
        now = utc_now()
        with self.database.transaction(write=True) as connection:
            case = connection.execute(
                "SELECT * FROM handover_cases WHERE project_id=? AND handover_id=?",
                (self.project_id, handover_id),
            ).fetchone()
            if not case:
                raise ValueError("交接记录不存在")
            existing = connection.execute(
                "SELECT * FROM handover_snapshots WHERE project_id=? AND handover_id=?",
                (self.project_id, handover_id),
            ).fetchone()
            if existing:
                return {
                    "record": self._handover_row(connection, case),
                    "snapshot": self._handover_snapshot_payload(existing, include_payload=True),
                    "created": False,
                }
            items = connection.execute(
                "SELECT * FROM handover_items WHERE project_id=? AND handover_id=? ORDER BY position,item_id",
                (self.project_id, handover_id),
            ).fetchall()
            risks = connection.execute(
                "SELECT * FROM handover_risks WHERE project_id=? AND handover_id=? ORDER BY created_at,risk_id",
                (self.project_id, handover_id),
            ).fetchall()
            inventory = connection.execute(
                "SELECT * FROM handover_inventory_entries WHERE project_id=? AND handover_id=? ORDER BY source_type,title,inventory_id",
                (self.project_id, handover_id),
            ).fetchall()
            pending_items = [row for row in items if bool(row["required"]) and row["status"] != "accepted"]
            open_risks = [row for row in risks if row["status"] != "closed"]
            pending_inventory = [row for row in inventory if row["status"] == "pending"]
            data = _json_load(case["data_json"], {})
            if pending_items:
                raise ValueError(f"仍有 {len(pending_items)} 个必交项未验收")
            if open_risks:
                blocking = sum(row["severity"] == "blocking" for row in open_risks)
                if blocking:
                    raise ValueError(f"仍有 {blocking} 个阻断风险未关闭")
                raise ValueError(f"仍有 {len(open_risks)} 个风险未关闭")
            if pending_inventory:
                raise ValueError(f"仍有 {len(pending_inventory)} 个知识盘点项未审阅")
            if not str(case["accepted_at"] or ""):
                raise ValueError("接替人尚未确认接收")
            if str(data.get("learning_plan_status") or "") != "ready":
                raise ValueError("接替人学习计划尚未就绪，不能关闭交接")
            assets = [
                dict(row) for row in connection.execute(
                    """
                    SELECT DISTINCT asset_id,version_id,source_file,stored_file FROM documents
                    WHERE project_id=? AND handover_id=? AND status<>'deleted'
                    ORDER BY source_file,asset_id,version_id
                    """,
                    (self.project_id, handover_id),
                )
            ]
            known_asset_versions = {
                (str(item.get("asset_id") or ""), str(item.get("version_id") or "")) for item in assets
            }
            included_asset_ids = [
                str(row["reference_id"]) for row in inventory
                if row["source_type"] == "knowledge_asset" and row["status"] == "included"
            ]
            if included_asset_ids:
                placeholders = ",".join("?" for _ in included_asset_ids)
                for asset in connection.execute(
                    f"""
                    SELECT ka.asset_id,ka.current_version_id AS version_id,
                           ks.display_name AS source_file,'' AS stored_file
                    FROM knowledge_assets ka
                    JOIN knowledge_sources ks ON ks.source_id=ka.primary_source_id
                    WHERE ka.project_id=? AND ka.asset_id IN ({placeholders})
                    ORDER BY ka.title,ka.asset_id
                    """,
                    (self.project_id, *included_asset_ids),
                ).fetchall():
                    key = (str(asset["asset_id"] or ""), str(asset["version_id"] or ""))
                    if key not in known_asset_versions:
                        assets.append(dict(asset))
                        known_asset_versions.add(key)
            case_payload = {
                **{key: value for key, value in data.items() if key not in {"items", "risk_items", "snapshot"}},
                "id": handover_id, "name": str(case["name"]), "role": str(case["role"]),
                "recipient": str(case["recipient"]), "due_date": str(case["due_date"]),
                "status": "completed", "accepted_by": str(case["accepted_by"]),
                "accepted_at": str(case["accepted_at"]), "created_at": str(case["created_at"]),
            }
            snapshot_payload = {
                "schema": "handover_snapshot_v2",
                "sealed_at": now,
                "case": case_payload,
                "items": [self._handover_item_payload(row) for row in items],
                "risks": [self._handover_risk_payload(row) for row in risks],
                "inventory": [self._handover_inventory_payload(row) for row in inventory],
                "asset_versions": assets,
                "confirmation": {
                    "recipient_user_id": str(data.get("recipient_user_id") or ""),
                    "accepted_by_user_id": str(data.get("accepted_by_user_id") or ""),
                    "accepted_by": str(case["accepted_by"]), "accepted_at": str(case["accepted_at"]),
                },
                "learning_plan": {
                    "plan_id": str(data.get("learning_plan_id") or ""),
                    "status": str(data.get("learning_plan_status") or ""),
                    "synced_at": str(data.get("learning_plan_synced_at") or ""),
                },
            }
            canonical = json.dumps(snapshot_payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            checksum = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            snapshot_id = f"hs_{uuid.uuid4().hex}"
            connection.execute(
                """
                INSERT INTO handover_snapshots(
                    snapshot_id,handover_id,project_id,snapshot_version,snapshot_json,checksum,created_by,created_at
                ) VALUES(?,?,?,?,?,?,?,?)
                """,
                (snapshot_id, handover_id, self.project_id, 2, canonical, checksum, actor, now),
            )
            updated_data = {
                **data, "status": "completed", "completed_at": now,
                "completed_by_user_id": actor, "snapshot_id": snapshot_id,
                "snapshot_checksum": checksum,
            }
            connection.execute(
                "UPDATE handover_cases SET status='completed',updated_at=?,data_json=? WHERE handover_id=?",
                (now, _json_dump(updated_data), handover_id),
            )
            self._write_audit_connection(
                connection, f"audit_{uuid.uuid4().hex}", actor, "handover.completed",
                "handover", handover_id,
                {"snapshot_id": snapshot_id, "checksum": checksum, "asset_version_count": len(assets)},
                project_id=self.project_id, created_at=now,
            )
            updated_case = connection.execute("SELECT * FROM handover_cases WHERE handover_id=?", (handover_id,)).fetchone()
            snapshot = connection.execute("SELECT * FROM handover_snapshots WHERE snapshot_id=?", (snapshot_id,)).fetchone()
            return {
                "record": self._handover_row(connection, updated_case),
                "snapshot": self._handover_snapshot_payload(snapshot, include_payload=True),
                "created": True,
            }

    def save_handovers(self, records: Sequence[Dict[str, Any]]) -> None:
        with self.database.transaction(write=True) as connection:
            desired = {str(record.get("id") or record.get("handover_id") or "") for record in records}
            desired.discard("")
            existing = {
                row[0] for row in connection.execute("SELECT handover_id FROM handover_cases WHERE project_id=?", (self.project_id,))
            }
            for handover_id in existing - desired:
                connection.execute("DELETE FROM handover_cases WHERE handover_id=?", (handover_id,))
            for record in records:
                self._upsert_handover_connection(connection, record)

    def _upsert_handover_connection(self, connection: sqlite3.Connection, record: Dict[str, Any]) -> None:
        handover_id = str(record.get("id") or record.get("handover_id") or "").strip()
        if not handover_id:
            raise ValueError("交接记录缺少 ID")
        existing_case = connection.execute(
            "SELECT 1 FROM handover_cases WHERE project_id=? AND handover_id=?",
            (self.project_id, handover_id),
        ).fetchone()
        if connection.execute(
            "SELECT 1 FROM handover_snapshots WHERE handover_id=?", (handover_id,)
        ).fetchone():
            # Compatibility callers rewrite the whole handover list. A sealed
            # case remains readable but is never mutated by that legacy path.
            return
        now = utc_now()
        created_at = str(record.get("created_at") or now)
        record_data = dict(record)
        for derived_key in (
            "items", "risk_items", "snapshot", "completion", "completion_breakdown",
            "completion_gates", "risks", "file_count", "inventory", "inventory_summary",
        ):
            record_data.pop(derived_key, None)
        connection.execute(
            """
            INSERT INTO handover_cases(
                handover_id,project_id,name,role,recipient,due_date,status,accepted_by,accepted_at,
                created_at,updated_at,data_json
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(handover_id) DO UPDATE SET
                name=excluded.name,role=excluded.role,recipient=excluded.recipient,due_date=excluded.due_date,
                status=excluded.status,accepted_by=excluded.accepted_by,accepted_at=excluded.accepted_at,
                updated_at=excluded.updated_at,data_json=excluded.data_json
            """,
            (
                handover_id, self.project_id, str(record.get("name") or ""), str(record.get("role") or ""),
                str(record.get("recipient") or ""), str(record.get("due_date") or ""),
                str(record.get("status") or "pending_acceptance"), str(record.get("accepted_by") or ""),
                str(record.get("accepted_at") or ""), created_at, now, _json_dump(record_data),
            ),
        )
        self._ensure_handover_items_connection(
            connection, handover_id,
            owner_user_id=str(record.get("departing_user_id") or record.get("created_by_user_id") or ""),
            recipient_user_id=str(record.get("recipient_user_id") or ""),
            due_at=str(record.get("due_date") or ""),
            role_template_id=str(record.get("role_template_id") or "") if not existing_case else "",
        )
        connection.execute("DELETE FROM handover_files WHERE handover_id=?", (handover_id,))
        for filename in record.get("files", []) or []:
            document = connection.execute(
                """
                SELECT document_id FROM documents
                WHERE project_id=? AND handover_id=? AND (source_file=? OR stored_file=?)
                ORDER BY created_at DESC LIMIT 1
                """,
                (self.project_id, handover_id, str(filename), str(filename)),
            ).fetchone()
            connection.execute(
                "INSERT INTO handover_files(handover_id,document_id,source_file,created_at) VALUES(?,?,?,?)",
                (handover_id, document["document_id"] if document else None, str(filename), now),
            )
        departing_user_id = str(record.get("departing_user_id") or "")
        has_inventory = connection.execute(
            "SELECT 1 FROM handover_inventory_entries WHERE handover_id=? LIMIT 1", (handover_id,),
        ).fetchone()
        if departing_user_id and not has_inventory:
            self._refresh_handover_inventory_connection(
                connection, handover_id, departing_user_id,
                actor=str(record.get("created_by_user_id") or "user_system"),
            )

    @staticmethod
    def _handover_item_payload(row: sqlite3.Row) -> Dict[str, Any]:
        payload = dict(row)
        payload["required"] = bool(payload.get("required"))
        return payload

    @staticmethod
    def _handover_risk_payload(row: sqlite3.Row) -> Dict[str, Any]:
        return dict(row)

    @staticmethod
    def _handover_snapshot_payload(row: sqlite3.Row, *, include_payload: bool = False) -> Dict[str, Any]:
        payload = dict(row)
        snapshot = _json_load(payload.pop("snapshot_json", "{}"), {})
        if include_payload:
            payload["payload"] = snapshot
        return payload

    def _handover_row(self, connection: sqlite3.Connection, row: sqlite3.Row) -> Dict[str, Any]:
        data = _json_load(row["data_json"], {})
        files = [
            item["source_file"]
            for item in connection.execute(
                "SELECT source_file FROM handover_files WHERE handover_id=? ORDER BY created_at", (row["handover_id"],)
            )
        ]
        items = [
            self._handover_item_payload(item)
            for item in connection.execute(
                "SELECT * FROM handover_items WHERE handover_id=? ORDER BY position,item_id",
                (row["handover_id"],),
            )
        ]
        risk_items = [
            self._handover_risk_payload(item)
            for item in connection.execute(
                """
                SELECT * FROM handover_risks WHERE handover_id=?
                ORDER BY CASE severity WHEN 'blocking' THEN 0 WHEN 'high' THEN 1
                    WHEN 'medium' THEN 2 ELSE 3 END,created_at,risk_id
                """,
                (row["handover_id"],),
            )
        ]
        inventory = [
            self._handover_inventory_payload(item)
            for item in connection.execute(
                """
                SELECT * FROM handover_inventory_entries WHERE handover_id=?
                ORDER BY CASE source_type WHEN 'knowledge_asset' THEN 0
                    WHEN 'knowledge_task' THEN 1 ELSE 2 END,title,inventory_id
                """,
                (row["handover_id"],),
            )
        ]
        inventory_summary = {
            "total": len(inventory),
            "pending": sum(item["status"] == "pending" for item in inventory),
            "included": sum(item["status"] == "included" for item in inventory),
            "excluded": sum(item["status"] == "excluded" for item in inventory),
            "by_source": {
                source_type: sum(item["source_type"] == source_type for item in inventory)
                for source_type in ("knowledge_asset", "knowledge_task", "graph_topic")
            },
        }
        snapshot_row = connection.execute(
            "SELECT * FROM handover_snapshots WHERE handover_id=?", (row["handover_id"],)
        ).fetchone()
        return {
            **data, "id": row["handover_id"], "name": row["name"], "role": row["role"],
            "recipient": row["recipient"], "due_date": row["due_date"], "status": row["status"],
            "accepted_by": row["accepted_by"], "accepted_at": row["accepted_at"],
            "created_at": row["created_at"], "updated_at": row["updated_at"], "files": files,
            "items": items, "risk_items": risk_items,
            "inventory": inventory, "inventory_summary": inventory_summary,
            "snapshot": self._handover_snapshot_payload(snapshot_row) if snapshot_row else None,
        }

    # Feishu snapshot compatibility -------------------------------------
    def load_feishu_workspace(self) -> Dict[str, Any]:
        with self.database.transaction() as connection:
            return self._load_feishu_connection(connection)

    def _load_feishu_connection(self, connection: sqlite3.Connection) -> Dict[str, Any]:
        def payloads(table: str, order: str = "updated_at DESC") -> List[Dict[str, Any]]:
            return [
                _json_load(row["data_json"], {})
                for row in connection.execute(
                    f"SELECT data_json FROM {table} WHERE project_id=? ORDER BY {order}", (self.project_id,)
                )
            ]
        diagnostics_row = connection.execute(
            "SELECT data_json FROM feishu_diagnostics WHERE project_id=?", (self.project_id,)
        ).fetchone()
        revision_row = connection.execute(
            "SELECT value_text FROM system_state WHERE state_key=?", (f"feishu_revision:{self.project_id}",)
        ).fetchone()
        return {
            "version": 4,
            "groups": payloads("feishu_groups"),
            "messages": payloads("feishu_messages", "create_time DESC,message_id DESC"),
            "candidates": payloads("feishu_candidates"),
            "assets": payloads("feishu_assets"),
            "audit": payloads("audit_events", "created_at ASC")[-2000:],
            "diagnostics": _json_load(diagnostics_row["data_json"], {}) if diagnostics_row else {},
            "updated_at": "",
            "_revision": int(revision_row["value_text"] if revision_row else 0),
        }

    def save_feishu_workspace(self, data: Dict[str, Any]) -> int:
        with self.database.transaction(write=True) as connection:
            _, revision = self._save_feishu_workspace_connection(connection, data)
            return revision

    def _save_feishu_workspace_connection(
        self, connection: sqlite3.Connection, data: Dict[str, Any],
    ) -> tuple[Dict[str, Any], int]:
        """Persist one workspace snapshot inside the caller's transaction."""
        expected_revision = int(data.get("_revision") or 0)
        current = self._load_feishu_connection(connection)
        current_revision = int(current.get("_revision") or 0)
        snapshot = dict(data)
        if expected_revision != current_revision:
            snapshot = self._merge_feishu_snapshots(current, snapshot)
        self._replace_feishu_connection(connection, snapshot)
        next_revision = current_revision + 1
        connection.execute(
            """
            INSERT INTO system_state(state_key,value_text,updated_at) VALUES(?,?,?)
            ON CONFLICT(state_key) DO UPDATE SET value_text=excluded.value_text,updated_at=excluded.updated_at
            """,
            (f"feishu_revision:{self.project_id}", str(next_revision), utc_now()),
        )
        return snapshot, next_revision

    def revert_feishu_asset(self, data: Dict[str, Any], asset_id: str) -> Dict[str, Any]:
        """Atomically persist a Feishu revocation while retaining source evidence."""
        with self.database.transaction(write=True) as connection:
            snapshot, revision = self._save_feishu_workspace_connection(connection, data)
            asset = next(
                (item for item in snapshot.get("assets", []) if str(item.get("asset_id") or "") == asset_id),
                None,
            )
            if not asset:
                raise ValueError("知识资产不存在")
            if str(asset.get("status") or "") != "reverted":
                raise ValueError("知识资产撤销状态未生效")

            stored_file = str(asset.get("stored_file") or "").strip()
            source_file = str(asset.get("source_file") or "").strip()
            document_ref = stored_file or source_file
            rows = self._document_rows_for_delete(connection, document_ref, False) if document_ref else []
            deleted_chunks = sum(int(row["chunks"] or 0) for row in rows)
            archived_documents = 0
            for row in rows:
                connection.execute(
                    "UPDATE documents SET status='revoked',updated_at=? WHERE document_id=?",
                    (utc_now(), row["document_id"]),
                )
                archived_documents += 1

            return {
                "asset": dict(asset),
                "revision": revision,
                "deleted_documents": 0,
                "archived_documents": archived_documents,
                "deleted_chunks": deleted_chunks,
                "stored_file": stored_file,
                "source_file": source_file,
            }

    def update_feishu_asset_projection(
        self, asset_id: str, status: str, errors: Sequence[str] = (),
    ) -> Dict[str, Any]:
        """Persist derived projection health without changing authoritative revocation."""
        if status not in {"pending_refresh", "healthy", "repair_required"}:
            raise ValueError("不支持的派生状态")
        with self.database.transaction(write=True) as connection:
            current = self._load_feishu_connection(connection)
            asset = next(
                (item for item in current.get("assets", []) if str(item.get("asset_id") or "") == asset_id),
                None,
            )
            if not asset:
                raise ValueError("知识资产不存在")
            if str(asset.get("status") or "") != "reverted":
                raise ValueError("仅已撤销资产可更新撤销投影状态")
            asset["projection_status"] = status
            asset["projection_errors"] = [str(item)[:200] for item in errors if str(item)]
            asset["projection_updated_at"] = utc_now()
            snapshot, revision = self._save_feishu_workspace_connection(connection, current)
            persisted = next(
                item for item in snapshot.get("assets", []) if str(item.get("asset_id") or "") == asset_id
            )
            return {"asset": dict(persisted), "revision": revision}

    @staticmethod
    def _merge_feishu_snapshots(current: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
        keys = {
            "groups": "chat_id", "messages": "message_id", "candidates": "candidate_id",
            "assets": "asset_id", "audit": "audit_id",
        }
        merged = {**current, **incoming}
        for collection, identifier in keys.items():
            combined = {
                str(item.get(identifier) or ""): dict(item)
                for item in current.get(collection, []) if item.get(identifier)
            }
            for item in incoming.get(collection, []):
                item_id = str(item.get(identifier) or "")
                if item_id:
                    combined[item_id] = {**combined.get(item_id, {}), **dict(item)}
            merged[collection] = list(combined.values())
        merged["diagnostics"] = {**current.get("diagnostics", {}), **incoming.get("diagnostics", {})}
        return merged

    def _replace_feishu_connection(self, connection: sqlite3.Connection, data: Dict[str, Any]) -> None:
        for table in (
            "feishu_candidate_messages", "feishu_asset_messages", "feishu_assets", "feishu_candidates",
            "feishu_messages", "feishu_groups", "feishu_diagnostics",
        ):
            connection.execute(f"DELETE FROM {table} WHERE project_id=?" if table not in {"feishu_candidate_messages", "feishu_asset_messages"} else (
                f"DELETE FROM {table} WHERE "
                + ("candidate_id IN (SELECT candidate_id FROM feishu_candidates WHERE project_id=?)" if table == "feishu_candidate_messages" else "asset_id IN (SELECT asset_id FROM feishu_assets WHERE project_id=?)")
            ), (self.project_id,))
        now = utc_now()
        for group in data.get("groups", []):
            chat_id = str(group.get("chat_id") or "")
            if chat_id:
                connection.execute(
                    """
                    INSERT INTO feishu_groups(
                        chat_id,project_id,name,collection_mode,updated_at,data_json,visibility,access_policy_id
                    ) VALUES(?,?,?,?,?,?,?,?)
                    """,
                    (chat_id, self.project_id, str(group.get("name") or ""), str(group.get("collection_mode") or "review"),
                     str(group.get("updated_at") or now), _json_dump(group),
                     str(group.get("visibility") or "project"), str(group.get("access_policy_id") or "")),
                )
        for message in data.get("messages", []):
            message_id = str(message.get("message_id") or "")
            if message_id:
                provider_account_id = str(message.get("provider_account_id") or "feishu_default")
                event_id = str(message.get("event_id") or "")
                if event_id and connection.execute(
                    "SELECT 1 FROM feishu_messages WHERE project_id=? AND provider_account_id=? AND event_id=?",
                    (self.project_id, provider_account_id, event_id),
                ).fetchone():
                    continue
                connection.execute(
                    """
                    INSERT INTO feishu_messages(
                        message_id,project_id,chat_id,message_type,content_status,candidate_id,asset_id,
                        content_hash,create_time,updated_at,data_json,provider_account_id,event_id
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (message_id, self.project_id, str(message.get("chat_id") or ""),
                     str(message.get("message_type") or message.get("msg_type") or "text"),
                     str(message.get("content_status") or "archived"), str(message.get("candidate_id") or ""),
                     str(message.get("asset_id") or ""), str(message.get("content_hash") or ""),
                     str(message.get("create_time") or ""), str(message.get("synced_at") or now), _json_dump(message),
                     provider_account_id, event_id),
                )
        for candidate in data.get("candidates", []):
            candidate_id = str(candidate.get("candidate_id") or "")
            if not candidate_id:
                continue
            connection.execute(
                """
                INSERT INTO feishu_candidates(
                    candidate_id,project_id,chat_id,aggregation_key,status,category,updated_at,data_json
                ) VALUES(?,?,?,?,?,?,?,?)
                """,
                (candidate_id, self.project_id, str(candidate.get("chat_id") or ""),
                 str(candidate.get("aggregation_key") or ""), str(candidate.get("status") or "review_required"),
                 str(candidate.get("category") or ""), str(candidate.get("updated_at") or now), _json_dump(candidate)),
            )
            for message_id in candidate.get("source_message_ids", []) or []:
                if connection.execute("SELECT 1 FROM feishu_messages WHERE message_id=?", (str(message_id),)).fetchone():
                    connection.execute(
                        "INSERT OR IGNORE INTO feishu_candidate_messages(candidate_id,message_id) VALUES(?,?)",
                        (candidate_id, str(message_id)),
                    )
        for asset in data.get("assets", []):
            asset_id = str(asset.get("asset_id") or "")
            if not asset_id:
                continue
            connection.execute(
                """
                INSERT INTO feishu_assets(asset_id,project_id,candidate_id,status,stored_file,updated_at,data_json)
                VALUES(?,?,?,?,?,?,?)
                """,
                (asset_id, self.project_id, str(asset.get("candidate_id") or ""), str(asset.get("status") or "published"),
                 str(asset.get("stored_file") or ""), str(asset.get("reverted_at") or asset.get("published_at") or now),
                 _json_dump(asset)),
            )
            for message_id in asset.get("source_message_ids", []) or []:
                if connection.execute("SELECT 1 FROM feishu_messages WHERE message_id=?", (str(message_id),)).fetchone():
                    connection.execute(
                        "INSERT OR IGNORE INTO feishu_asset_messages(asset_id,message_id) VALUES(?,?)",
                        (asset_id, str(message_id)),
                    )
        for event in data.get("audit", [])[-2000:]:
            audit_id = str(event.get("audit_id") or _stable_id("audit", event.get("created_at"), event.get("action"), event.get("object_id")))
            connection.execute(
                """
                INSERT OR REPLACE INTO audit_events(
                    audit_id,project_id,actor,action,object_type,object_id,created_at,data_json
                ) VALUES(?,?,?,?,?,?,?,?)
                """,
                (audit_id, self.project_id, str(event.get("actor") or "system"), str(event.get("action") or "unknown"),
                 str(event.get("object_type") or ""), str(event.get("object_id") or ""),
                 str(event.get("created_at") or now), _json_dump({**event, "audit_id": audit_id})),
            )
        diagnostics = data.get("diagnostics") or {}
        connection.execute(
            "INSERT INTO feishu_diagnostics(project_id,data_json,updated_at) VALUES(?,?,?)",
            (self.project_id, _json_dump(diagnostics), now),
        )

    def feishu_revision(self) -> int:
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT value_text FROM system_state WHERE state_key=?", (f"feishu_revision:{self.project_id}",)
            ).fetchone()
            return int(row["value_text"] if row else 0)

    # Graph edits --------------------------------------------------------
    def save_graph_edits(self, graph: Dict[str, Any]) -> None:
        nodes = [dict(item) for item in graph.get("nodes", []) if isinstance(item, dict) and item.get("id")]
        edges = [dict(item) for item in graph.get("edges", []) if isinstance(item, dict)]
        manually_referenced_nodes = {
            str(endpoint)
            for edge in edges
            if not bool(edge.get("auto_generated"))
            and str(edge.get("method") or "manual") not in {"generated", "hybrid_auto"}
            for endpoint in (edge.get("source") or edge.get("from"), edge.get("target") or edge.get("to"))
            if endpoint
        }
        with self.database.transaction(write=True) as connection:
            connection.execute("DELETE FROM graph_edges WHERE project_id=?", (self.project_id,))
            connection.execute("DELETE FROM graph_nodes WHERE project_id=?", (self.project_id,))
            now = utc_now()
            for node in nodes:
                origin = str(node.get("origin") or "manual")
                keep = origin != "generated" or bool(node.get("manual_override")) or bool(node.get("locked"))
                keep = keep or node.get("x") is not None or node.get("y") is not None
                keep = keep or str(node.get("id")) in manually_referenced_nodes
                if keep:
                    connection.execute(
                        "INSERT INTO graph_nodes(project_id,node_id,origin,source,updated_at,data_json) VALUES(?,?,?,?,?,?)",
                        (self.project_id, str(node["id"]), origin, str(node.get("source") or ""), now, _json_dump(node)),
                    )
            node_ids = {str(node.get("id")) for node in nodes}
            for edge in edges:
                source = str(edge.get("source") or edge.get("from") or "")
                target = str(edge.get("target") or edge.get("to") or "")
                if not source or not target or source not in node_ids or target not in node_ids:
                    continue
                origin = str(edge.get("method") or ("generated" if edge.get("auto_generated") else "manual"))
                if edge.get("auto_generated") and not edge.get("protected"):
                    continue
                edge_id = str(edge.get("id") or _stable_id("edge", source, target, edge.get("label")))
                connection.execute(
                    """
                    INSERT INTO graph_edges(
                        project_id,edge_id,source_node_id,target_node_id,origin,updated_at,data_json
                    ) VALUES(?,?,?,?,?,?,?)
                    """,
                    (self.project_id, edge_id, source, target, origin, now, _json_dump({**edge, "id": edge_id})),
                )

    def load_graph_edits(self) -> Dict[str, Any]:
        with self.database.transaction() as connection:
            nodes = [_json_load(row["data_json"], {}) for row in connection.execute(
                "SELECT data_json FROM graph_nodes WHERE project_id=? ORDER BY node_id", (self.project_id,)
            )]
            edges = [_json_load(row["data_json"], {}) for row in connection.execute(
                "SELECT data_json FROM graph_edges WHERE project_id=? ORDER BY edge_id", (self.project_id,)
            )]
            return {"nodes": nodes, "edges": edges}

    def replace_graph_origin(self, origin: str, graph: Dict[str, Any]) -> None:
        """Replace one graph origin without disturbing manual edits from other origins."""
        origin = str(origin or "manual")
        nodes = [dict(item) for item in graph.get("nodes", []) if isinstance(item, dict) and item.get("id")]
        edges = [dict(item) for item in graph.get("edges", []) if isinstance(item, dict)]
        with self.database.transaction(write=True) as connection:
            connection.execute("DELETE FROM graph_edges WHERE project_id=? AND origin=?", (self.project_id, origin))
            connection.execute("DELETE FROM graph_nodes WHERE project_id=? AND origin=?", (self.project_id, origin))
            now = utc_now()
            node_ids = set()
            for item in nodes:
                node_id = str(item["id"])
                node_ids.add(node_id)
                payload = {**item, "origin": origin}
                connection.execute(
                    "INSERT OR REPLACE INTO graph_nodes(project_id,node_id,origin,source,updated_at,data_json) VALUES(?,?,?,?,?,?)",
                    (self.project_id, node_id, origin, str(item.get("source") or ""), now, _json_dump(payload)),
                )
            for item in edges:
                source = str(item.get("source") or item.get("from") or "")
                target = str(item.get("target") or item.get("to") or "")
                if not source or not target or source not in node_ids or target not in node_ids:
                    continue
                edge_id = str(item.get("id") or _stable_id("edge", source, target, item.get("label")))
                payload = {**item, "id": edge_id, "source": source, "target": target}
                connection.execute(
                    "INSERT OR REPLACE INTO graph_edges(project_id,edge_id,source_node_id,target_node_id,origin,updated_at,data_json) VALUES(?,?,?,?,?,?,?)",
                    (self.project_id, edge_id, source, target, origin, now, _json_dump(payload)),
                )

    def remove_graph_source(self, source_name: str) -> None:
        stored_suffix = f"%\\_{_like_literal(source_name)}"
        with self.database.transaction(write=True) as connection:
            node_ids = [
                row["node_id"] for row in connection.execute(
                    "SELECT node_id FROM graph_nodes WHERE project_id=? AND source <> '' "
                    "AND (source=? OR source LIKE ? ESCAPE '\\' OR ? LIKE '%' || source)",
                    (self.project_id, source_name, stored_suffix, source_name),
                )
            ]
            for node_id in node_ids:
                connection.execute(
                    "DELETE FROM graph_edges WHERE project_id=? AND (source_node_id=? OR target_node_id=?)",
                    (self.project_id, node_id, node_id),
                )
                connection.execute("DELETE FROM graph_nodes WHERE project_id=? AND node_id=?", (self.project_id, node_id))

    # Runtime observability -------------------------------------------
    def upsert_runtime_metric_buckets(self, buckets: Sequence[Dict[str, Any]]) -> int:
        """Merge bounded minute aggregates; no request content is accepted."""
        if not buckets:
            return 0
        now = utc_now()
        with self.database.transaction(write=True) as connection:
            for item in buckets:
                metric_key = str(item.get("metric_key") or "")[:80]
                route_key = str(item.get("route_key") or "")[:160]
                bucket_start = str(item.get("bucket_start") or "")[:32]
                if not metric_key or not bucket_start:
                    continue
                existing = connection.execute(
                    "SELECT histogram_json FROM runtime_metric_buckets "
                    "WHERE project_id=? AND metric_key=? AND route_key=? AND bucket_start=?",
                    (self.project_id, metric_key, route_key, bucket_start),
                ).fetchone()
                incoming_histogram = {
                    str(key): max(0, int(value))
                    for key, value in dict(item.get("histogram") or {}).items()
                }
                if existing:
                    histogram = _json_load(existing["histogram_json"], {})
                    for key, value in incoming_histogram.items():
                        histogram[key] = int(histogram.get(key, 0)) + value
                    connection.execute(
                        """
                        UPDATE runtime_metric_buckets SET
                            sample_count=sample_count+?,success_count=success_count+?,
                            error_count=error_count+?,total_duration_ms=total_duration_ms+?,
                            max_duration_ms=MAX(max_duration_ms,?),histogram_json=?,updated_at=?
                        WHERE project_id=? AND metric_key=? AND route_key=? AND bucket_start=?
                        """,
                        (
                            int(item.get("sample_count") or 0), int(item.get("success_count") or 0),
                            int(item.get("error_count") or 0), float(item.get("total_duration_ms") or 0),
                            float(item.get("max_duration_ms") or 0), _json_dump(histogram), now,
                            self.project_id, metric_key, route_key, bucket_start,
                        ),
                    )
                else:
                    connection.execute(
                        """
                        INSERT INTO runtime_metric_buckets(
                            project_id,metric_key,route_key,bucket_start,sample_count,success_count,
                            error_count,total_duration_ms,max_duration_ms,histogram_json,updated_at
                        ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            self.project_id, metric_key, route_key, bucket_start,
                            int(item.get("sample_count") or 0), int(item.get("success_count") or 0),
                            int(item.get("error_count") or 0), float(item.get("total_duration_ms") or 0),
                            float(item.get("max_duration_ms") or 0), _json_dump(incoming_histogram), now,
                        ),
                    )
        return len(buckets)

    def list_runtime_metric_buckets(self, since: str) -> List[Dict[str, Any]]:
        with self.database.transaction() as connection:
            rows = connection.execute(
                "SELECT * FROM runtime_metric_buckets WHERE project_id=? AND bucket_start>=? "
                "ORDER BY bucket_start,metric_key,route_key",
                (self.project_id, str(since)),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["histogram"] = _json_load(item.pop("histogram_json", "{}"), {})
            result.append(item)
        return result

    def create_runtime_maintenance_window(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        now = utc_now()
        item = {
            "window_id": str(payload.get("window_id") or f"mw_{uuid.uuid4().hex}"),
            "title": str(payload.get("title") or "计划维护").strip()[:120],
            "reason": str(payload.get("reason") or "").strip()[:1000],
            "starts_at": str(payload.get("starts_at") or ""),
            "ends_at": str(payload.get("ends_at") or ""),
            "created_by": str(payload.get("created_by") or "user_system"),
        }
        with self.database.transaction(write=True) as connection:
            connection.execute(
                """
                INSERT INTO runtime_maintenance_windows(
                    window_id,project_id,title,reason,starts_at,ends_at,status,
                    created_by,created_at,updated_at
                ) VALUES(?,?,?,?,?,?,'scheduled',?,?,?)
                """,
                (item["window_id"], self.project_id, item["title"], item["reason"],
                 item["starts_at"], item["ends_at"], item["created_by"], now, now),
            )
            row = connection.execute(
                "SELECT * FROM runtime_maintenance_windows WHERE window_id=?", (item["window_id"],)
            ).fetchone()
        return dict(row)

    def list_runtime_maintenance_windows(self, since: str = "", until: str = "", limit: int = 100) -> List[Dict[str, Any]]:
        where = ["project_id=?"]
        params: List[Any] = [self.project_id]
        if since:
            where.append("ends_at>=?")
            params.append(str(since))
        if until:
            where.append("starts_at<=?")
            params.append(str(until))
        params.append(max(1, min(int(limit), 500)))
        with self.database.transaction() as connection:
            rows = connection.execute(
                f"SELECT * FROM runtime_maintenance_windows WHERE {' AND '.join(where)} "
                "ORDER BY starts_at DESC LIMIT ?", tuple(params),
            ).fetchall()
        return [dict(row) for row in rows]

    def cancel_runtime_maintenance_window(self, window_id: str, actor_user_id: str) -> Dict[str, Any]:
        now = utc_now()
        with self.database.transaction(write=True) as connection:
            row = connection.execute(
                "SELECT * FROM runtime_maintenance_windows WHERE window_id=? AND project_id=?",
                (window_id, self.project_id),
            ).fetchone()
            if not row:
                raise ValueError("计划维护窗口不存在")
            if row["status"] != "scheduled":
                raise ValueError("计划维护窗口已取消")
            connection.execute(
                "UPDATE runtime_maintenance_windows SET status='cancelled',cancelled_by=?,"
                "cancelled_at=?,updated_at=? WHERE window_id=?",
                (actor_user_id, now, now, window_id),
            )
            current = connection.execute(
                "SELECT * FROM runtime_maintenance_windows WHERE window_id=?", (window_id,)
            ).fetchone()
        return dict(current)

    def start_runtime_slo_event(self, metric_key: str, subject_id: str, detail: Optional[Dict[str, Any]] = None) -> str:
        event_id, now = f"slo_{uuid.uuid4().hex}", utc_now()
        with self.database.transaction(write=True) as connection:
            connection.execute(
                """
                INSERT INTO runtime_slo_events(
                    event_id,project_id,metric_key,subject_id,status,started_at,detail_json,updated_at
                ) VALUES(?,?,?,?,'running',?,?,?)
                """,
                (event_id, self.project_id, str(metric_key)[:80], str(subject_id)[:200],
                 now, _json_dump(detail or {}), now),
            )
        return event_id

    def complete_runtime_slo_event(
        self, event_id: str, *, succeeded: bool, detail: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        now = utc_now()
        with self.database.transaction(write=True) as connection:
            row = connection.execute(
                "SELECT * FROM runtime_slo_events WHERE event_id=? AND project_id=?",
                (event_id, self.project_id),
            ).fetchone()
            if not row:
                raise ValueError("SLO 事件不存在")
            if row["status"] != "running":
                return dict(row)
            try:
                started = datetime.fromisoformat(str(row["started_at"]).replace("Z", "+00:00"))
                completed = datetime.fromisoformat(now.replace("Z", "+00:00"))
                duration_ms = max(0.0, (completed - started).total_seconds() * 1000)
            except ValueError:
                duration_ms = 0.0
            merged = _json_load(row["detail_json"], {})
            merged.update(detail or {})
            connection.execute(
                "UPDATE runtime_slo_events SET status=?,completed_at=?,duration_ms=?,detail_json=?,updated_at=? "
                "WHERE event_id=?",
                ("succeeded" if succeeded else "failed", now, duration_ms, _json_dump(merged), now, event_id),
            )
            current = connection.execute("SELECT * FROM runtime_slo_events WHERE event_id=?", (event_id,)).fetchone()
        item = dict(current)
        item["detail"] = _json_load(item.pop("detail_json", "{}"), {})
        return item

    def list_runtime_slo_events(self, metric_key: str, since: str) -> List[Dict[str, Any]]:
        with self.database.transaction() as connection:
            rows = connection.execute(
                "SELECT * FROM runtime_slo_events WHERE project_id=? AND metric_key=? AND started_at>=? "
                "ORDER BY started_at", (self.project_id, metric_key, since),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["detail"] = _json_load(item.pop("detail_json", "{}"), {})
            result.append(item)
        return result

    def processing_failure_visibility(self, since: str) -> Dict[str, Any]:
        with self.database.transaction() as connection:
            rows = connection.execute(
                """
                SELECT attempts.completed_at AS failed_at,
                       MIN(events.created_at) AS visible_at
                FROM processing_job_attempts AS attempts
                LEFT JOIN processing_job_events AS events
                  ON events.job_id=attempts.job_id
                 AND events.created_at>=attempts.completed_at
                 AND events.event_type IN ('retry_wait','failed','recovered')
                WHERE attempts.project_id=? AND attempts.completed_at>=?
                  AND attempts.status IN ('retry_wait','failed','interrupted')
                GROUP BY attempts.attempt_id
                """, (self.project_id, since),
            ).fetchall()
        delays = []
        for row in rows:
            if not row["visible_at"]:
                continue
            try:
                failed = datetime.fromisoformat(str(row["failed_at"]).replace("Z", "+00:00"))
                visible = datetime.fromisoformat(str(row["visible_at"]).replace("Z", "+00:00"))
                delays.append(max(0.0, (visible - failed).total_seconds()))
            except ValueError:
                continue
        total = len(rows)
        within = sum(1 for delay in delays if delay <= 60)
        return {
            "sample_count": total,
            "within_target_count": within,
            "visibility_percent": round(within * 100 / total, 2) if total else None,
            "max_seconds": round(max(delays), 2) if delays else None,
        }

    def save_runtime_recovery_drill(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        item = {
            "drill_id": str(payload.get("drill_id") or f"drill_{uuid.uuid4().hex}"),
            "status": "passed" if payload.get("status") == "passed" else "failed",
            "backup_name": str(payload.get("backup_name") or "")[:240],
            "backup_created_at": str(payload.get("backup_created_at") or ""),
            "rpo_hours": max(0.0, float(payload.get("rpo_hours") or 0)),
            "rto_seconds": max(0.0, float(payload.get("rto_seconds") or 0)),
            "report": dict(payload.get("report") or {}),
            "actor_user_id": str(payload.get("actor_user_id") or "user_system"),
            "created_at": str(payload.get("created_at") or utc_now()),
        }
        with self.database.transaction(write=True) as connection:
            connection.execute(
                """
                INSERT INTO runtime_recovery_drills(
                    drill_id,project_id,status,backup_name,backup_created_at,rpo_hours,
                    rto_seconds,report_json,actor_user_id,created_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?)
                """,
                (item["drill_id"], self.project_id, item["status"], item["backup_name"],
                 item["backup_created_at"], item["rpo_hours"], item["rto_seconds"],
                 _json_dump(item["report"]), item["actor_user_id"], item["created_at"]),
            )
            self._write_audit_connection(
                connection,
                f"audit_{uuid.uuid4().hex}",
                item["actor_user_id"],
                "operations.recovery_drill_completed",
                "runtime_recovery_drill",
                item["drill_id"],
                {
                    "status": item["status"],
                    "backup_name": item["backup_name"],
                    "backup_created_at": item["backup_created_at"],
                    "rpo_hours": item["rpo_hours"],
                    "rto_seconds": item["rto_seconds"],
                    "sha256_verified": bool(item["report"].get("sha256_verified")),
                    "quick_check": item["report"].get("quick_check", []),
                    "foreign_key_errors": int(item["report"].get("foreign_key_errors") or 0),
                    "schema_version": item["report"].get("schema_version"),
                    "online_database_changed": bool(item["report"].get("online_database_changed")),
                },
                project_id=self.project_id,
                created_at=item["created_at"],
            )
        return item

    def latest_runtime_recovery_drill(self) -> Optional[Dict[str, Any]]:
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM runtime_recovery_drills WHERE project_id=? ORDER BY created_at DESC,rowid DESC LIMIT 1",
                (self.project_id,),
            ).fetchone()
        if not row:
            return None
        item = dict(row)
        item["report"] = _json_load(item.pop("report_json", "{}"), {})
        return item

    def sync_runtime_alert(
        self, rule_key: str, active: bool, *, severity: str, title: str,
        summary: str = "", evidence: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        now = utc_now()
        rule_key = str(rule_key)[:100]
        with self.database.transaction(write=True) as connection:
            row = connection.execute(
                "SELECT * FROM runtime_alerts WHERE project_id=? AND rule_key=? AND status='open'",
                (self.project_id, rule_key),
            ).fetchone()
            if active:
                if row:
                    connection.execute(
                        "UPDATE runtime_alerts SET severity=?,title=?,summary=?,evidence_json=?,"
                        "occurrence_count=occurrence_count+1,last_seen_at=?,updated_at=? WHERE alert_id=?",
                        (severity, title[:200], summary[:1000], _json_dump(evidence or {}), now, now, row["alert_id"]),
                    )
                    alert_id = str(row["alert_id"])
                else:
                    alert_id = f"alert_{uuid.uuid4().hex}"
                    connection.execute(
                        """
                        INSERT INTO runtime_alerts(
                            alert_id,project_id,rule_key,severity,status,title,summary,evidence_json,
                            occurrence_count,first_seen_at,last_seen_at,resolved_at,updated_at
                        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        (alert_id, self.project_id, rule_key, severity, "open", title[:200], summary[:1000],
                         _json_dump(evidence or {}), 1, now, now, "", now),
                    )
            elif row:
                alert_id = str(row["alert_id"])
                connection.execute(
                    "UPDATE runtime_alerts SET status='resolved',resolved_at=?,updated_at=? WHERE alert_id=?",
                    (now, now, alert_id),
                )
            else:
                return None
            current = connection.execute("SELECT * FROM runtime_alerts WHERE alert_id=?", (alert_id,)).fetchone()
        item = dict(current) if current else None
        if item:
            item["evidence"] = _json_load(item.pop("evidence_json", "{}"), {})
        return item

    def list_runtime_alerts(self, status: str = "open", limit: int = 50) -> List[Dict[str, Any]]:
        params: List[Any] = [self.project_id]
        where = "project_id=?"
        if status in {"open", "resolved"}:
            where += " AND status=?"
            params.append(status)
        params.append(max(1, min(int(limit), 200)))
        with self.database.transaction() as connection:
            rows = connection.execute(
                f"SELECT * FROM runtime_alerts WHERE {where} "
                "ORDER BY CASE severity WHEN 'critical' THEN 0 WHEN 'warning' THEN 1 ELSE 2 END,"
                "last_seen_at DESC LIMIT ?",
                tuple(params),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["evidence"] = _json_load(item.pop("evidence_json", "{}"), {})
            result.append(item)
        return result

    def save_runtime_capacity_run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        item = {
            "run_id": str(payload.get("run_id") or f"cap_{uuid.uuid4().hex}"),
            "status": "passed" if payload.get("status") == "passed" else "failed",
            "message_count": max(0, int(payload.get("message_count") or 0)),
            "p50_ms": max(0.0, float(payload.get("p50_ms") or 0)),
            "p95_ms": max(0.0, float(payload.get("p95_ms") or 0)),
            "max_ms": max(0.0, float(payload.get("max_ms") or 0)),
            "report": dict(payload.get("report") or {}),
            "actor_user_id": str(payload.get("actor_user_id") or "user_system"),
            "created_at": str(payload.get("created_at") or utc_now()),
        }
        with self.database.transaction(write=True) as connection:
            connection.execute(
                """
                INSERT INTO runtime_capacity_runs(
                    run_id,project_id,status,message_count,p50_ms,p95_ms,max_ms,
                    report_json,actor_user_id,created_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?)
                """,
                (item["run_id"], self.project_id, item["status"], item["message_count"],
                 item["p50_ms"], item["p95_ms"], item["max_ms"], _json_dump(item["report"]),
                 item["actor_user_id"], item["created_at"]),
            )
        return item

    def latest_runtime_capacity_run(self) -> Optional[Dict[str, Any]]:
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM runtime_capacity_runs WHERE project_id=? ORDER BY created_at DESC,rowid DESC LIMIT 1",
                (self.project_id,),
            ).fetchone()
        if not row:
            return None
        item = dict(row)
        item["report"] = _json_load(item.pop("report_json", "{}"), {})
        return item

    def runtime_health_facts(self) -> Dict[str, int]:
        with self.database.transaction() as connection:
            failed_jobs = int(connection.execute(
                "SELECT COUNT(*) FROM processing_jobs WHERE project_id=? AND status='failed'",
                (self.project_id,),
            ).fetchone()[0])
            retry_jobs = int(connection.execute(
                "SELECT COUNT(*) FROM processing_jobs WHERE project_id=? AND status='retry_wait'",
                (self.project_id,),
            ).fetchone()[0])
        return {"failed_jobs": failed_jobs, "retry_wait_jobs": retry_jobs}

    # Migration bookkeeping --------------------------------------------
    def migration_run(self, key: str) -> Optional[Dict[str, Any]]:
        with self.database.transaction() as connection:
            row = connection.execute("SELECT * FROM migration_runs WHERE migration_key=?", (key,)).fetchone()
            return dict(row) if row else None

    def record_migration_run(
        self, key: str, source_hash: str, status: str, report_path: str = "", detail: str = "",
        started_at: str = "", completed_at: str = "",
    ) -> None:
        with self.database.transaction(write=True) as connection:
            connection.execute(
                """
                INSERT INTO migration_runs(
                    migration_key,source_hash,status,report_path,detail,started_at,completed_at
                ) VALUES(?,?,?,?,?,?,?)
                ON CONFLICT(migration_key) DO UPDATE SET
                    source_hash=excluded.source_hash,status=excluded.status,report_path=excluded.report_path,
                    detail=excluded.detail,started_at=excluded.started_at,completed_at=excluded.completed_at
                """,
                (key, source_hash, status, report_path, detail[:1000], started_at or utc_now(), completed_at),
            )

    def import_legacy_snapshot(self, snapshot: Dict[str, Any]) -> Dict[str, int]:
        """Atomically import a normalized legacy snapshot."""
        expected_counts = snapshot.get("_expected_counts") or {}
        counts = {
            "categories": 0,
            "documents": 0,
            "chunks": 0,
            "handovers": 0,
            "feishu_groups": 0,
            "feishu_messages": 0,
            "feishu_candidates": 0,
            "feishu_assets": 0,
            "audit_events": 0,
            "graph_nodes": 0,
            "graph_edges": 0,
            "upload_files": int(snapshot.get("upload_files") or 0),
        }
        with self.database.transaction(write=True) as connection:
            project_name = str(snapshot.get("project_name") or "默认项目")
            connection.execute(
                "UPDATE projects SET name=?,updated_at=? WHERE project_id=?",
                (project_name, utc_now(), self.project_id),
            )
            for index, name in enumerate(snapshot.get("categories", [])):
                self._ensure_category(connection, str(name), index)
                counts["categories"] += 1
            for item in snapshot.get("documents", []):
                self._upsert_document_connection(connection, item.get("document", {}), item.get("chunks", []))
                counts["documents"] += 1
                counts["chunks"] += len(item.get("chunks", []))
            for handover in snapshot.get("handovers", []):
                self._upsert_handover_connection(connection, handover)
                counts["handovers"] += 1
            feishu = snapshot.get("feishu") or {}
            if any(feishu.get(key) for key in ("groups", "messages", "candidates", "assets", "audit")):
                self._replace_feishu_connection(connection, feishu)
                counts["feishu_groups"] = len([item for item in feishu.get("groups", []) if item.get("chat_id")])
                counts["feishu_messages"] = len(feishu.get("messages", []))
                counts["feishu_candidates"] = len([
                    item for item in feishu.get("candidates", []) if item.get("candidate_id")
                ])
                counts["feishu_assets"] = len([
                    item for item in feishu.get("assets", []) if item.get("asset_id")
                ])
                counts["audit_events"] = len(feishu.get("audit", [])[-2000:])
                connection.execute(
                    "INSERT OR REPLACE INTO system_state(state_key,value_text,updated_at) VALUES(?,?,?)",
                    (f"feishu_revision:{self.project_id}", "1", utc_now()),
                )
            graph = snapshot.get("graph") or {}
            if graph.get("nodes") or graph.get("edges"):
                # Keep all imported/manual graph data. Generated nodes remain rebuildable.
                now = utc_now()
                for node in graph.get("nodes", []):
                    if not node.get("id"):
                        continue
                    connection.execute(
                        "INSERT OR REPLACE INTO graph_nodes(project_id,node_id,origin,source,updated_at,data_json) VALUES(?,?,?,?,?,?)",
                        (self.project_id, str(node["id"]), str(node.get("origin") or "manual"),
                         str(node.get("source") or ""), now, _json_dump(node)),
                    )
                    counts["graph_nodes"] += 1
                known_nodes = {str(item.get("id")) for item in graph.get("nodes", []) if item.get("id")}
                for edge in graph.get("edges", []):
                    source = str(edge.get("source") or edge.get("from") or "")
                    target = str(edge.get("target") or edge.get("to") or "")
                    if source not in known_nodes or target not in known_nodes:
                        continue
                    edge_id = str(edge.get("id") or _stable_id("edge", source, target, edge.get("label")))
                    connection.execute(
                        "INSERT OR REPLACE INTO graph_edges(project_id,edge_id,source_node_id,target_node_id,origin,updated_at,data_json) VALUES(?,?,?,?,?,?,?)",
                        (self.project_id, edge_id, source, target, str(edge.get("method") or edge.get("origin") or "manual"),
                         now, _json_dump({**edge, "id": edge_id})),
                    )
                    counts["graph_edges"] += 1
            mismatches = {
                key: {"expected": int(expected), "imported": int(counts.get(key, 0))}
                for key, expected in expected_counts.items()
                if int(expected) != int(counts.get(key, 0))
            }
            if mismatches:
                raise ValueError(f"迁移数量对账失败: {mismatches}")
        # Legacy rows are imported after repository construction; adopt only
        # once the import transaction has fully committed.
        self.backfill_knowledge_assets(force=True)
        return counts
