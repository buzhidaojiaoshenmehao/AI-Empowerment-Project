"""Project-scoped knowledge-governance task workflow.

AI may propose a task, but deterministic rules own assignment, transitions,
acceptance and notification side effects.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Iterable, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from backend.auth import ROLE_PERMISSIONS
from backend.auth_context import Identity
from backend.config import settings
from backend.knowledge_base.readiness import calculate_role_readiness
from backend.storage import get_repository


TASK_TYPES = {
    "knowledge_gap", "asset_review", "asset_publish", "feishu_review", "handover_gap", "qa_feedback", "manual",
}
PRIORITIES = {"low", "medium", "high", "critical"}
OPEN_STATUSES = {"unassigned", "open", "in_progress", "pending_acceptance"}
FINAL_STATUSES = {"completed", "cancelled"}
logger = logging.getLogger(__name__)


class KnowledgeTaskService:
    def __init__(self, repository_provider=get_repository) -> None:
        self.repository_provider = repository_provider

    @property
    def repository(self):
        return self.repository_provider()

    @staticmethod
    def _actor(identity: Any) -> str:
        return str(getattr(identity, "user_id", "") or "user_system")

    @staticmethod
    def _can_manage(identity: Any) -> bool:
        return bool(identity and identity.can("task.manage"))

    def create(self, payload: Dict[str, Any], identity: Any) -> Dict[str, Any]:
        title = str(payload.get("title") or "").strip()
        task_type = str(payload.get("task_type") or "manual").strip()
        priority = str(payload.get("priority") or "medium").strip()
        if not title:
            raise ValueError("任务标题不能为空")
        if task_type not in TASK_TYPES:
            raise ValueError("不支持的知识任务类型")
        if priority not in PRIORITIES:
            raise ValueError("不支持的任务优先级")
        if task_type == "asset_publish":
            self._validate_asset_publish_payload(payload, identity)
        actor = self._actor(identity)
        requested_assignee = str(payload.get("assignee_user_id") or "").strip()
        if requested_assignee and requested_assignee != actor and not self._can_manage(identity):
            raise PermissionError("只有任务管理人员可以把新任务指派给其他成员")
        assignee = requested_assignee
        if not assignee and task_type == "handover_gap":
            assignee = self._handover_gap_default_assignee(str(payload.get("handover_id") or ""))
        task = self.repository.create_or_merge_knowledge_task({
            **payload,
            "title": title,
            "task_type": task_type,
            "priority": priority,
            "reporter_user_id": actor,
            "assignee_user_id": assignee,
        })
        self.repository.write_security_audit(
            actor,
            "knowledge_task.created" if task.get("created") else "knowledge_task.merged",
            "knowledge_task",
            str(task.get("task_id") or ""),
            {
                "task_type": task_type,
                "source_type": task.get("source_type", "manual"),
                "dedupe_key": task.get("dedupe_key", ""),
                "occurrence_count": task.get("occurrence_count", 1),
            },
            project_id=str(getattr(identity, "project_id", "") or self.repository.project_id),
        )
        return task

    def _handover_gap_default_assignee(self, handover_id: str) -> str:
        """Supplementing handover material is the departing person's duty, not the recipient's."""
        if not handover_id:
            return ""
        try:
            record = self.repository.get_handover(handover_id)
        except Exception:  # noqa: BLE001 - a missing record must not block task creation
            logger.warning("handover gap default assignee lookup failed: %s", handover_id, exc_info=True)
            return ""
        return str((record or {}).get("departing_user_id") or "")

    def _validate_asset_publish_payload(self, payload: Dict[str, Any], identity: Any) -> None:
        """Require an exact, currently reviewable asset-version binding."""
        if not identity or not identity.can("knowledge.publish"):
            raise PermissionError("只有知识发布人员可以创建资产发布任务")
        asset_id = str(payload.get("linked_asset_id") or "").strip()
        version_id = str(payload.get("source_key") or "").strip()
        metadata = payload.get("metadata") or {}
        metadata_version_id = str(metadata.get("linked_version_id") or "").strip()
        try:
            asset_revision = int(metadata.get("asset_revision"))
        except (TypeError, ValueError):
            asset_revision = -1
        if (
            str(payload.get("source_type") or "") != "asset_version"
            or not asset_id or not version_id or version_id != metadata_version_id
            or asset_revision < 1
        ):
            raise ValueError("资产发布任务必须绑定准确的资产、版本和生命周期修订号")
        asset = self.repository.get_asset(asset_id, identity=identity)
        version = self.repository.get_asset_version(asset_id, version_id, identity=identity)
        if not asset or not version:
            raise ValueError("资产发布任务关联的资产或版本不存在")
        if str(asset.get("status") or "") != "pending_review":
            raise ValueError("只有待审核资产可以创建发布任务")
        if int(asset.get("lifecycle_revision") or 0) != asset_revision:
            raise ValueError("资产生命周期已变化，请刷新后重新创建发布任务")
        if str(version.get("status") or "") != "ready":
            raise ValueError("资产发布任务只能绑定 ready 版本")

    def list(self, identity: Any, **filters: Any) -> Dict[str, Any]:
        tasks = self.repository.list_knowledge_tasks(
            viewer_user_id=self._actor(identity), can_manage=self._can_manage(identity), **filters,
        )
        tasks = [self._with_allowed_actions(task, identity) for task in tasks]
        return {"tasks": tasks, "summary": self.summarize(tasks)}

    def get(self, task_id: str, identity: Any) -> Optional[Dict[str, Any]]:
        task = self.repository.get_knowledge_task(
            task_id, viewer_user_id=self._actor(identity), can_manage=self._can_manage(identity),
        )
        return self._with_allowed_actions(task, identity) if task else None

    def _with_allowed_actions(self, task: Dict[str, Any], identity: Any) -> Dict[str, Any]:
        """Expose server-authoritative actions so the UI never guesses permissions."""
        item = dict(task)
        actor = self._actor(identity)
        can_manage = self._can_manage(identity)
        status = str(item.get("status") or "")
        assignee = str(item.get("assignee_user_id") or "")
        reporter = str(item.get("reporter_user_id") or "")
        actions = []
        if status == "unassigned":
            actions.append("claim")
        if can_manage and status in OPEN_STATUSES:
            actions.append("assign")
        if status == "open" and (actor == assignee or can_manage):
            actions.append("start")
        if status == "in_progress" and (actor == assignee or can_manage):
            actions.append("submit")
        if status == "pending_acceptance" and actor != assignee and (actor == reporter or can_manage):
            actions.append("accept")
        if status == "pending_acceptance" and (actor == reporter or can_manage):
            actions.append("return")
        if status in OPEN_STATUSES and (actor == reporter or can_manage):
            actions.append("cancel")
        item["allowed_actions"] = list(dict.fromkeys(actions))
        item["primary_action_block_reason"] = self._primary_action_block_reason(
            item, actor=actor, can_manage=can_manage,
        )
        return item

    @staticmethod
    def _primary_action_block_reason(
        task: Dict[str, Any], *, actor: str, can_manage: bool,
    ) -> str:
        status = str(task.get("status") or "")
        assignee = str(task.get("assignee_user_id") or "")
        reporter = str(task.get("reporter_user_id") or "")
        assignee_name = str(task.get("assignee_name") or "任务负责人")
        if status == "open" and actor != assignee and not can_manage:
            return f"仅任务负责人“{assignee_name}”可以开始处理"
        if status == "in_progress" and actor != assignee and not can_manage:
            return f"仅任务负责人“{assignee_name}”可以提交验收"
        if status == "pending_acceptance" and actor == assignee:
            if can_manage:
                return "当前账号同时是任务负责人，不能验收自己的任务；请先转派给其他成员复核，再由你验收"
            return "任务负责人不能验收自己的任务，请由报告人或任务管理人员验收"
        if status == "pending_acceptance" and actor != reporter and not can_manage:
            return "仅报告人或任务管理人员可以验收"
        return ""

    @staticmethod
    def summarize(tasks: Iterable[Dict[str, Any]]) -> Dict[str, int]:
        items = list(tasks)
        return {
            "total": len(items),
            "unassigned": sum(item.get("status") == "unassigned" for item in items),
            "open": sum(item.get("status") == "open" for item in items),
            "in_progress": sum(item.get("status") == "in_progress" for item in items),
            "pending_acceptance": sum(item.get("status") == "pending_acceptance" for item in items),
            "overdue": sum(bool(item.get("overdue")) for item in items),
            "completed": sum(item.get("status") == "completed" for item in items),
        }

    def act(self, task_id: str, action: str, payload: Dict[str, Any], identity: Any) -> Dict[str, Any]:
        actor = self._actor(identity)
        can_manage = self._can_manage(identity)
        task = self.repository.get_knowledge_task(task_id, can_manage=True)
        if not task:
            raise ValueError("知识任务不存在")
        related = actor in {task.get("assignee_user_id"), task.get("reporter_user_id")}
        if not related and not can_manage:
            raise PermissionError("当前账号无权处理该任务")

        action = str(action or "").strip().lower()
        status = str(task.get("status") or "")
        note = str(payload.get("note") or "").strip()
        kwargs: Dict[str, Any] = {
            "actor_user_id": actor, "event_type": action, "note": note,
            "expected_status": status,
        }

        if action == "claim":
            self._expect(status, {"unassigned"}, action)
            kwargs.update(to_status="open", assignee_user_id=actor)
        elif action == "assign":
            if not can_manage:
                raise PermissionError("只有任务管理人员可以转派任务")
            self._expect(status, OPEN_STATUSES, action)
            assignee = str(payload.get("assignee_user_id") or "").strip()
            if not assignee:
                raise ValueError("请选择任务负责人")
            kwargs.update(to_status="open" if status == "unassigned" else status, assignee_user_id=assignee)
        elif action == "start":
            self._expect(status, {"open"}, action)
            if actor != task.get("assignee_user_id") and not can_manage:
                raise PermissionError("只有任务负责人可以开始处理")
            kwargs.update(to_status="in_progress")
        elif action == "submit":
            self._expect(status, {"in_progress"}, action)
            if actor != task.get("assignee_user_id") and not can_manage:
                raise PermissionError("只有任务负责人可以提交验收")
            evidence = str(payload.get("evidence") or "").strip()
            if not evidence:
                raise ValueError("提交验收前必须填写完成证据")
            kwargs.update(to_status="pending_acceptance", evidence=evidence)
        elif action == "accept":
            self._expect(status, {"pending_acceptance"}, action)
            if actor == task.get("assignee_user_id"):
                raise PermissionError("任务负责人不能验收自己的任务")
            if actor != task.get("reporter_user_id") and not can_manage:
                raise PermissionError("只有报告人或任务管理人员可以验收")
            if task.get("task_type") == "asset_publish" and (
                not identity or not identity.can("knowledge.publish")
            ):
                raise PermissionError("只有知识发布人员可以验收资产发布任务")
            kwargs.update(to_status="completed")
        elif action == "return":
            self._expect(status, {"pending_acceptance"}, action)
            if actor != task.get("reporter_user_id") and not can_manage:
                raise PermissionError("只有报告人或任务管理人员可以退回")
            if not note:
                raise ValueError("退回任务必须说明原因")
            kwargs.update(to_status="in_progress")
        elif action == "cancel":
            self._expect(status, OPEN_STATUSES, action)
            if actor != task.get("reporter_user_id") and not can_manage:
                raise PermissionError("只有报告人或任务管理人员可以取消")
            if not note:
                raise ValueError("取消任务必须说明原因")
            kwargs.update(to_status="cancelled")
        elif action == "comment":
            if not note:
                raise ValueError("备注内容不能为空")
        else:
            raise ValueError("不支持的任务操作")

        updated = self.repository.update_knowledge_task(task_id, **kwargs)
        self.repository.write_security_audit(
            actor, f"knowledge_task.{action}", "knowledge_task", task_id,
            {"from_status": status, "to_status": updated.get("status"), "has_evidence": bool(payload.get("evidence"))},
            project_id=str(getattr(identity, "project_id", "") or self.repository.project_id),
        )
        return self._with_allowed_actions(updated, identity)

    @staticmethod
    def _expect(current: str, allowed: Iterable[str], action: str) -> None:
        if current not in set(allowed):
            raise ValueError(f"当前状态不能执行 {action} 操作")

    def create_onboarding_gap_tasks(
        self, role: str, gaps: Iterable[Dict[str, Any]], identity: Any,
        *, assignee_user_id: str = "", due_days: int = 7,
    ) -> Dict[str, Any]:
        due_at = (datetime.now(timezone.utc) + timedelta(days=max(1, min(int(due_days), 90)))).isoformat(timespec="seconds")
        tasks = []
        for gap in gaps:
            category = str(gap.get("category") or "").strip()
            label = str(gap.get("label") or category).strip()
            if not category:
                continue
            tasks.append(self.create({
                "task_type": "knowledge_gap",
                "title": f"补齐{role}岗位知识：{label}",
                "description": f"新人赋能分析发现“{role}”岗位缺少“{label}”的有效知识资产。请补充、审核并发布可追溯资料。",
                "source_type": "onboarding",
                "source_key": f"{role}:{category}",
                "dedupe_key": f"onboarding_gap:{role}:{category}",
                "role_key": role,
                "priority": "high" if int(gap.get("weight") or 1) >= 2 else "medium",
                "assignee_user_id": assignee_user_id,
                "due_at": due_at,
                "metadata": {"category": category, "label": label, "weight": gap.get("weight", 1)},
            }, identity))
        return {
            "tasks": tasks,
            "created_count": sum(bool(item.get("created")) for item in tasks),
            "merged_count": sum(not bool(item.get("created")) for item in tasks),
        }

    def reconcile_onboarding_gap_tasks(
        self, *, actor: str = "user_system", trigger_asset_id: str = "",
        trigger: str = "asset_published",
    ) -> Dict[str, Any]:
        """Attach visible effective assets to onboarding gaps and request acceptance.

        Asset publication proves that matching material exists, but it does not
        replace the independent human acceptance gate. The transition therefore
        stops at ``pending_acceptance``.
        """
        tasks = self.repository.list_knowledge_tasks(
            can_manage=True, task_type="knowledge_gap", limit=500,
        )
        candidates = [
            item for item in tasks
            if item.get("source_type") == "onboarding"
            and item.get("status") in {"open", "in_progress"}
            and item.get("assignee_user_id")
        ]
        matched = []
        errors = []
        for task in candidates:
            try:
                assignee = self.repository.get_user(
                    str(task.get("assignee_user_id") or ""), project_id=self.repository.project_id,
                )
                if not assignee or assignee.get("status") != "active" or assignee.get("membership_status") != "active":
                    continue
                role = str(assignee.get("role") or "project_member")
                target = Identity(
                    user_id=str(assignee["user_id"]), email=str(assignee.get("email") or ""),
                    display_name=str(assignee.get("display_name") or ""),
                    organization_id=str(assignee.get("organization_id") or "org_default"),
                    project_id=str(assignee.get("project_id") or self.repository.project_id),
                    project_name=str(assignee.get("project_name") or self.repository.project_name()),
                    role=role, permissions=ROLE_PERMISSIONS.get(role, frozenset()),
                )
                metadata = task.get("metadata") or {}
                category = str(metadata.get("category") or "").strip()
                label = str(metadata.get("label") or category).strip()
                role_name = str(task.get("role_key") or "当前岗位")
                if not category:
                    continue
                readiness = calculate_role_readiness(
                    {"role": role_name, "topics": [{"topic_id": category, "label": label, "weight": 1}]},
                    self.repository.readiness_assets(identity=target),
                )
                evidence_items = list((readiness.get("topics") or [{}])[0].get("evidence") or [])
                if not evidence_items:
                    continue
                evidence = evidence_items[0]
                asset_id = str(evidence.get("asset_id") or "")
                version_id = str(evidence.get("version_id") or "")
                title = str(evidence.get("title") or label or "匹配资料")
                completion_evidence = f"系统已匹配生效知识《{title}》（资产 {asset_id}，版本 {version_id}）"
                updated = self.repository.update_knowledge_task(
                    str(task["task_id"]), actor_user_id=actor,
                    event_type="auto_evidence_matched", to_status="pending_acceptance",
                    evidence=completion_evidence, linked_asset_id=asset_id,
                    metadata_updates={
                        "matched_asset_id": asset_id, "matched_version_id": version_id,
                        "matched_title": title, "matched_trigger": trigger,
                    },
                    note=f"检测到负责人有权访问的“{label}”生效资料，已提交人工验收",
                    expected_status=str(task.get("status") or ""),
                )
                matched.append({
                    "task_id": str(task["task_id"]), "asset_id": asset_id,
                    "version_id": version_id, "status": updated.get("status"),
                })
            except Exception as exc:
                logger.exception("岗位知识缺口任务自动匹配失败: %s", task.get("task_id"))
                errors.append({"task_id": str(task.get("task_id") or ""), "error": str(exc)[:300]})
        return {
            "success": not errors, "checked_count": len(candidates),
            "matched_count": len(matched), "trigger_asset_id": str(trigger_asset_id or ""),
            "matched": matched, "errors": errors,
        }

    def create_asset_review_task(self, asset: Dict[str, Any], identity: Any) -> Dict[str, Any]:
        asset_id = str(asset.get("asset_id") or "")
        if not asset_id:
            raise ValueError("知识资产信息不完整")
        owner_user_id = str(asset.get("owner_user_id") or "")
        owner = self.repository.get_user(owner_user_id) if owner_user_id else None
        if not owner or owner.get("membership_status") != "active" or owner.get("status") != "active":
            owner_user_id = ""
        return self.create({
            "task_type": "asset_review",
            "title": f"复审知识资产：{asset.get('title') or asset_id}",
            "description": "该知识资产已进入待复审状态。请核对来源、有效期、责任人和内容准确性后完成复审。",
            "source_type": "knowledge_asset",
            "source_key": asset_id,
            "dedupe_key": f"asset_review:{asset_id}",
            "linked_asset_id": asset_id,
            "priority": "high",
            "assignee_user_id": owner_user_id,
            "due_at": str(asset.get("review_due_at") or ""),
        }, identity)

    def create_asset_publish_task(self, asset: Dict[str, Any], identity: Any) -> Dict[str, Any]:
        """Create the review task that may publish the latest exact ready version."""
        asset_id = str(asset.get("asset_id") or "")
        if not asset_id or str(asset.get("status") or "") != "pending_review":
            raise ValueError("只有待审核知识资产可以创建发布任务")
        versions = self.repository.list_asset_versions(asset_id, identity=identity)
        version = next((item for item in versions if item.get("status") == "ready"), None)
        if not version:
            raise ValueError("没有可审核的 ready 版本")
        version_id = str(version.get("version_id") or "")
        revision = int(asset.get("lifecycle_revision") or 0)
        owner_user_id = str(asset.get("owner_user_id") or "")
        owner = self.repository.get_user(owner_user_id) if owner_user_id else None
        if not owner or owner.get("membership_status") != "active" or owner.get("status") != "active":
            owner_user_id = ""
        return self.create({
            "task_type": "asset_publish",
            "title": f"发布知识资产：{asset.get('title') or asset_id}",
            "description": (
                f"请核对知识来源、权限范围、内容准确性及版本 v{version.get('version_no') or '?'}。"
                "验收通过后仅发布本任务绑定的版本；资产或候选版本发生变化时不会自动覆盖。"
            ),
            "source_type": "asset_version",
            "source_key": version_id,
            "dedupe_key": f"asset_publish:{asset_id}:{version_id}",
            "linked_asset_id": asset_id,
            "priority": "high",
            "assignee_user_id": owner_user_id,
            "metadata": {
                "linked_version_id": version_id,
                "asset_revision": revision,
                "version_no": int(version.get("version_no") or 0),
            },
        }, identity)

    def create_handover_gap_task(
        self, handover: Dict[str, Any], item: Dict[str, Any], reason: str, identity: Any,
    ) -> Dict[str, Any]:
        """Create the deterministic supplement task required by an item rejection."""
        handover_id = str(handover.get("id") or handover.get("handover_id") or "")
        item_id = str(item.get("item_id") or "")
        if not handover_id or not item_id:
            raise ValueError("交接补充任务缺少关联信息")
        assignee = str(item.get("owner_user_id") or handover.get("departing_user_id") or "")
        if assignee:
            owner = self.repository.get_user(assignee)
            if not owner or owner.get("status") != "active" or owner.get("membership_status") != "active":
                assignee = ""
        actor = self._actor(identity)
        task = self.repository.create_or_merge_knowledge_task({
            "task_type": "handover_gap",
            "title": f"补充交接项：{item.get('title') or item_id}",
            "description": (
                f"{handover.get('name') or '成员'} 的交接项被退回。补充要求：{str(reason).strip()}"
            ),
            "source_type": "handover_item",
            "source_key": item_id,
            "dedupe_key": f"handover_item_gap:{handover_id}:{item_id}",
            "handover_id": handover_id,
            "linked_asset_id": str(item.get("asset_id") or ""),
            "priority": "high",
            "assignee_user_id": assignee,
            "reporter_user_id": actor,
            "due_at": str(item.get("due_at") or handover.get("due_date") or ""),
            "reason": str(reason).strip(),
            "metadata": {"handover_item_id": item_id, "rejection_reason": str(reason).strip()},
        })
        self.repository.write_security_audit(
            actor,
            "knowledge_task.created" if task.get("created") else "knowledge_task.merged",
            "knowledge_task", str(task.get("task_id") or ""),
            {
                "task_type": "handover_gap", "handover_id": handover_id,
                "handover_item_id": item_id, "dedupe_key": task.get("dedupe_key", ""),
            },
            project_id=str(getattr(identity, "project_id", "") or self.repository.project_id),
        )
        return task

    def get_notification_policy(self) -> Dict[str, Any]:
        policy = self.repository.get_task_notification_policy()
        effective_target = str(policy.get("target_chat_id") or settings.FEISHU_DEFAULT_CHAT_ID or "").strip()
        return {
            **policy,
            "effective_target_chat_id": effective_target,
            "using_default_chat": bool(effective_target and not policy.get("target_chat_id")),
        }

    @staticmethod
    def _valid_clock(value: Any) -> str:
        text = str(value or "").strip()
        try:
            parsed = datetime.strptime(text, "%H:%M")
        except ValueError as exc:
            raise ValueError("静默时间必须使用 HH:MM 格式") from exc
        return parsed.strftime("%H:%M")

    def update_notification_policy(self, payload: Dict[str, Any], identity: Any) -> Dict[str, Any]:
        if not self._can_manage(identity):
            raise PermissionError("只有任务管理人员可以修改自动提醒策略")
        timezone_name = str(payload.get("timezone") or "Asia/Shanghai").strip()
        try:
            ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError as exc:
            raise ValueError("提醒策略时区无效") from exc
        normalized = {
            "enabled": bool(payload.get("enabled")),
            "target_chat_id": str(payload.get("target_chat_id") or "").strip(),
            "timezone": timezone_name,
            "quiet_start": self._valid_clock(payload.get("quiet_start") or "22:00"),
            "quiet_end": self._valid_clock(payload.get("quiet_end") or "08:00"),
        }
        bounds = {
            "remind_before_hours": (1, 168), "reminder_interval_hours": (1, 168),
            "escalation_after_hours": (1, 720), "escalation_interval_hours": (1, 720),
        }
        for key, (minimum, maximum) in bounds.items():
            try:
                value = int(payload.get(key) or 24)
            except (TypeError, ValueError) as exc:
                raise ValueError("提醒策略小时数必须为整数") from exc
            if value < minimum or value > maximum:
                raise ValueError(f"{key} 必须在 {minimum} 到 {maximum} 小时之间")
            normalized[key] = value
        effective_target = str(normalized["target_chat_id"] or settings.FEISHU_DEFAULT_CHAT_ID or "").strip()
        if normalized["enabled"] and not effective_target:
            raise ValueError("启用自动提醒前必须配置目标飞书群")
        self.repository.update_task_notification_policy(
            normalized, actor_user_id=self._actor(identity),
        )
        return self.get_notification_policy()

    @staticmethod
    def _parse_timestamp(value: Any) -> Optional[datetime]:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)

    @staticmethod
    def _next_allowed_delivery(now: datetime, policy: Dict[str, Any]) -> datetime:
        local_now = now.astimezone(ZoneInfo(str(policy.get("timezone") or "Asia/Shanghai")))
        start_hour, start_minute = map(int, str(policy.get("quiet_start") or "22:00").split(":"))
        end_hour, end_minute = map(int, str(policy.get("quiet_end") or "08:00").split(":"))
        start = local_now.replace(hour=start_hour, minute=start_minute, second=0, microsecond=0)
        end = local_now.replace(hour=end_hour, minute=end_minute, second=0, microsecond=0)
        if (start_hour, start_minute) == (end_hour, end_minute):
            return now
        if start < end:
            quiet = start <= local_now < end
            quiet_end = end
        else:
            quiet = local_now >= start or local_now < end
            quiet_end = end + timedelta(days=1) if local_now >= start else end
        return quiet_end.astimezone(timezone.utc) if quiet else now

    def schedule_due_notifications(self, *, now: Optional[datetime] = None) -> Dict[str, Any]:
        current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        policy = self.get_notification_policy()
        target_id = str(policy.get("effective_target_chat_id") or "")
        if not policy.get("enabled"):
            return {"enabled": False, "scheduled_count": 0, "job_count": 0, "reason": "disabled"}
        if not target_id:
            return {"enabled": True, "scheduled_count": 0, "job_count": 0, "reason": "target_missing"}
        available = self._next_allowed_delivery(current, policy)
        due_before = current + timedelta(hours=int(policy["remind_before_hours"]))
        candidates = self.repository.list_task_reminder_candidates(
            due_before=due_before.isoformat(timespec="seconds"), limit=500,
        )
        entries = []
        interval_hours = int(policy["reminder_interval_hours"])
        interval_seconds = interval_hours * 3600
        for task in candidates:
            due_at = self._parse_timestamp(task.get("due_at"))
            if not due_at:
                continue
            last_at = self._parse_timestamp(task.get("last_auto_notification_at"))
            if last_at and available < last_at + timedelta(hours=interval_hours):
                continue
            overdue_hours = max(0.0, (current - due_at).total_seconds() / 3600)
            if due_at > current:
                notification_kind = "due_soon"
                escalation_level = 0
                bucket = int(available.timestamp() // interval_seconds)
                suffix = f"bucket:{bucket}"
            elif overdue_hours < int(policy["escalation_after_hours"]):
                notification_kind = "overdue"
                escalation_level = 0
                suffix = "initial"
            else:
                notification_kind = "escalation"
                escalation_level = 1 + int(
                    (overdue_hours - int(policy["escalation_after_hours"]))
                    // int(policy["escalation_interval_hours"])
                )
                suffix = f"level:{escalation_level}"
            entries.append({
                "task_id": str(task["task_id"]), "notification_kind": notification_kind,
                "escalation_level": escalation_level,
                "dedupe_key": f"task-reminder:{task['task_id']}:{notification_kind}:{suffix}",
            })
        jobs = []
        for start in range(0, len(entries), 20):
            digest = self.repository.create_task_notification_digest(
                entries[start:start + 20], target_id=target_id,
                available_at=available.isoformat(timespec="seconds"),
            )
            if digest:
                jobs.append(digest)
        return {
            "enabled": True, "scheduled_count": sum(item["notification_count"] for item in jobs),
            "job_count": len(jobs), "candidate_count": len(candidates),
            "available_at": available.isoformat(timespec="seconds"),
            "deferred_by_quiet_hours": available > current,
            "jobs": [item["processing_job"] for item in jobs],
        }

    async def deliver_scheduled_notifications(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        from backend.processing_jobs import ProcessingJobError, sanitize_value

        notification_ids = [
            str(item) for item in payload.get("notification_ids") or [] if str(item)
        ]
        items = self.repository.task_notification_delivery_items(notification_ids)
        if not items:
            return {"sent_count": 0, "skipped_count": 0, "message": "没有待投递提醒"}
        policy = self.get_notification_policy()
        target = str(payload.get("target_chat_id") or "")
        if not policy.get("enabled") or target != str(policy.get("effective_target_chat_id") or ""):
            self.repository.update_task_notification_deliveries(
                notification_ids, status="skipped", error="提醒策略已关闭或目标群已变化",
                increment_attempt=False,
            )
            return {"sent_count": 0, "skipped_count": len(items), "message": "提醒策略已变化，已跳过旧投递"}
        current = datetime.now(timezone.utc)
        valid, skipped = [], []
        for item in items:
            due_at = self._parse_timestamp(item.get("due_at"))
            kind = str(item.get("notification_kind") or "")
            is_open = str(item.get("task_status") or "") in OPEN_STATUSES
            due_matches = bool(
                due_at and (
                    (kind == "due_soon" and current < due_at <= current + timedelta(hours=int(policy["remind_before_hours"])))
                    or (kind in {"overdue", "escalation"} and due_at <= current)
                )
            )
            (valid if is_open and due_matches else skipped).append(item)
        if skipped:
            self.repository.update_task_notification_deliveries(
                [str(item["notification_id"]) for item in skipped], status="skipped",
                error="任务状态或截止时间已变化", increment_attempt=False,
            )
        if not valid:
            return {"sent_count": 0, "skipped_count": len(skipped), "message": "任务状态已变化，提醒已跳过"}
        max_level = max(int(item.get("escalation_level") or 0) for item in valid)
        title = f"知识任务逾期升级 L{max_level}" if max_level else "知识任务到期提醒"
        lines = []
        for item in valid:
            marker = "已逾期" if item.get("notification_kind") in {"overdue", "escalation"} else "即将到期"
            lines.append(
                f"- **{item.get('title') or item['task_id']}** · {marker} · "
                f"{item.get('assignee_name') or '待分配'} · 截止 {item.get('due_at') or '未设置'}"
            )
        content = "\n".join(lines) + "\n\n请在知识任务中心补充证据并按流程提交验收。"
        try:
            from backend.feishu_bot import feishu_bot
            delivered = await feishu_bot.send_card_message(target, title, content)
            if not delivered:
                raise RuntimeError("飞书发送接口未成功")
        except Exception as exc:
            error = str(sanitize_value(str(exc)))[:1000]
            self.repository.update_task_notification_deliveries(
                [str(item["notification_id"]) for item in valid], status="failed", error=error,
            )
            raise ProcessingJobError(
                "知识任务提醒发送失败", code="TASK_NOTIFICATION_FAILED", retryable=True,
            ) from exc
        self.repository.update_task_notification_deliveries(
            [str(item["notification_id"]) for item in valid], status="sent",
        )
        self.repository.write_security_audit(
            "user_system", "knowledge_task.notification_digest_sent", "knowledge_task_notification",
            str(valid[0]["notification_id"]),
            {"notification_count": len(valid), "skipped_count": len(skipped), "max_escalation_level": max_level},
            project_id=self.repository.project_id,
        )
        return {
            "sent_count": len(valid), "skipped_count": len(skipped),
            "max_escalation_level": max_level, "message": f"已发送 {len(valid)} 项任务提醒",
        }

    async def notify_feishu(self, task_id: str, chat_id: str, identity: Any) -> Dict[str, Any]:
        task = self.get(task_id, identity)
        if not task:
            raise PermissionError("知识任务不存在或无权访问")
        target = str(chat_id or "").strip()
        if not target:
            result = self.repository.record_task_notification(
                task_id, channel="feishu", target_id="", status="skipped", error="未指定飞书群",
            )
            return {**result, "success": False}
        try:
            from backend.feishu_bot import feishu_bot
            content = (
                f"**{task['title']}**\n\n"
                f"状态：{task['status']}\n"
                f"优先级：{task['priority']}\n"
                f"负责人：{task.get('assignee_name') or '待分配'}\n"
                f"截止时间：{task.get('due_at') or '未设置'}"
            )
            delivered = await feishu_bot.send_card_message(target, "知识任务提醒", content)
            status, error = ("sent", "") if delivered else ("failed", "飞书发送接口未成功")
        except Exception as exc:
            status, error = "failed", str(exc)[:1000]
        result = self.repository.record_task_notification(
            task_id, channel="feishu", target_id=target, status=status, error=error,
        )
        self.repository.write_security_audit(
            self._actor(identity), "knowledge_task.notified", "knowledge_task", task_id,
            {"channel": "feishu", "target_id": target, "status": status, "error": error[:200]},
            project_id=str(getattr(identity, "project_id", "") or self.repository.project_id),
        )
        return {**result, "success": status == "sent"}


knowledge_task_service = KnowledgeTaskService()


class KnowledgeTaskReminderScheduler:
    """Small in-process clock; SQLite jobs and dedupe rows remain authoritative."""

    def __init__(
        self, service: KnowledgeTaskService, *, interval_seconds: float = 60,
        wake_callback: Optional[Callable[[], None]] = None,
    ) -> None:
        self.service = service
        self.interval_seconds = max(1.0, float(interval_seconds))
        self.wake_callback = wake_callback
        self._wake = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._stopping = False

    @property
    def running(self) -> bool:
        return bool(self._task and not self._task.done())

    def wake(self) -> None:
        self._wake.set()

    async def start(self) -> None:
        if self.running:
            return
        self._stopping = False
        self._task = asyncio.create_task(self._run(), name="knowledge-task-reminder-scheduler")
        self.wake()

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
            try:
                result = self.service.schedule_due_notifications()
                if result.get("job_count") and self.wake_callback:
                    self.wake_callback()
            except Exception:
                logger.exception("知识任务自动提醒扫描失败")
            self._wake.clear()
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=self.interval_seconds)
            except asyncio.TimeoutError:
                pass


knowledge_task_reminder_scheduler = KnowledgeTaskReminderScheduler(
    knowledge_task_service, interval_seconds=settings.TASK_REMINDER_SCAN_SECONDS,
)


__all__ = [
    "KnowledgeTaskService", "knowledge_task_service", "TASK_TYPES", "PRIORITIES",
    "OPEN_STATUSES", "FINAL_STATUSES", "KnowledgeTaskReminderScheduler",
    "knowledge_task_reminder_scheduler",
]
