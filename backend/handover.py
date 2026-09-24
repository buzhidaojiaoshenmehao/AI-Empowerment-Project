"""Project-scoped handover governance, acceptance and immutable closure."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from backend.auth import ROLE_PERMISSIONS
from backend.auth_context import Identity
from backend.handover_policy import item_review_policy
from backend.knowledge_tasks import KnowledgeTaskService, knowledge_task_service
from backend.onboarding import OnboardingService, onboarding_service
from backend.storage import get_repository


logger = logging.getLogger(__name__)


def summarize_handover(record: Dict[str, Any]) -> Dict[str, Any]:
    """Return one deterministic, explainable completion and closure view."""
    items = list(record.get("items") or [])
    required_items = [item for item in items if bool(item.get("required"))]
    accepted_items = [item for item in required_items if item.get("status") == "accepted"]
    rejected_items = [item for item in required_items if item.get("status") == "rejected"]
    risk_items = list(record.get("risk_items") or [])
    open_risks = [risk for risk in risk_items if risk.get("status") != "closed"]
    blocking_risks = [risk for risk in open_risks if risk.get("severity") == "blocking"]
    inventory = list(record.get("inventory") or [])
    pending_inventory = [item for item in inventory if item.get("status") == "pending"]
    recipient_confirmed = bool(record.get("accepted_at")) or record.get("status") in {"accepted", "completed"}
    item_ratio = (len(accepted_items) / len(required_items)) if required_items else 1.0
    risk_ratio = (
        sum(risk.get("status") == "closed" for risk in risk_items) / len(risk_items)
        if risk_items else 1.0
    )
    completion = min(100, max(0, round(item_ratio * 60 + risk_ratio * 20 + (20 if recipient_confirmed else 0))))
    learning_plan_ready = str(record.get("learning_plan_status") or "") == "ready"
    completed = str(record.get("status") or "") == "completed" and bool(record.get("snapshot"))
    gates = {
        "required_items_complete": len(accepted_items) == len(required_items),
        "risks_closed": not open_risks,
        "blocking_risks_closed": not blocking_risks,
        "inventory_reviewed": not pending_inventory,
        "recipient_confirmed": recipient_confirmed,
        "learning_plan_ready": learning_plan_ready,
    }
    gates["closable"] = all(gates.values()) and not completed

    warnings = []
    if not record.get("files") and not completed:
        warnings.append("尚未上传交接资料")
    if not record.get("recipient") and not completed:
        warnings.append("尚未指定接替人")
    pending_count = len(required_items) - len(accepted_items)
    if pending_count:
        warnings.append(f"仍有 {pending_count} 个必交项未验收")
    if rejected_items:
        warnings.append(f"{len(rejected_items)} 个交接项已退回待补充")
    if blocking_risks:
        warnings.append(f"{len(blocking_risks)} 个阻断风险未关闭")
    elif open_risks:
        warnings.append(f"{len(open_risks)} 个风险未关闭")
    if pending_inventory:
        warnings.append(f"{len(pending_inventory)} 个知识盘点项待审阅")
    if not recipient_confirmed and not completed:
        warnings.append("接替人尚未确认接收")
    elif not learning_plan_ready and not completed:
        warnings.append("接替人学习计划待生成")
    if record.get("due_date") and not completed:
        try:
            if str(record["due_date"]) < datetime.now().strftime("%Y-%m-%d"):
                warnings.append("交接已超过计划完成日期")
        except Exception:
            pass
    return {
        **record,
        "file_count": len(record.get("files") or []),
        "risks": warnings,
        "completion": completion,
        "completion_breakdown": {
            "required_items": round(item_ratio * 60),
            "risks": round(risk_ratio * 20),
            "recipient_confirmation": 20 if recipient_confirmed else 0,
            "weights": {"required_items": 60, "risks": 20, "recipient_confirmation": 20},
            "required_total": len(required_items),
            "required_accepted": len(accepted_items),
            "risk_total": len(risk_items),
            "risk_closed": len(risk_items) - len(open_risks),
        },
        "completion_gates": gates,
    }


class HandoverService:
    """Own handover participants, transitions, evidence and closure gates."""

    def __init__(
        self,
        repository_provider=get_repository,
        onboarding_provider: Optional[OnboardingService] = None,
        knowledge_task_provider: Optional[KnowledgeTaskService] = None,
    ) -> None:
        self.repository_provider = repository_provider
        self.onboarding_provider = onboarding_provider
        self.knowledge_task_provider = knowledge_task_provider

    @property
    def repository(self):
        return self.repository_provider()

    @property
    def onboarding(self) -> OnboardingService:
        return self.onboarding_provider or onboarding_service

    @property
    def knowledge_tasks(self) -> KnowledgeTaskService:
        return self.knowledge_task_provider or knowledge_task_service

    @staticmethod
    def _active_member(member: Optional[Dict[str, Any]]) -> bool:
        return bool(
            member
            and str(member.get("status") or "") == "active"
            and str(member.get("membership_status") or "") == "active"
        )

    def resolve_recipient(self, record: Dict[str, Any]) -> Dict[str, Any]:
        recipient_user_id = str(record.get("recipient_user_id") or "").strip()
        if recipient_user_id:
            member = self.repository.get_user(recipient_user_id)
            if not self._active_member(member):
                raise ValueError("交接记录绑定的接替人已停用或不属于当前项目")
            return member or {}

        recipient = str(record.get("recipient") or "").strip().lower()
        if not recipient:
            raise ValueError("请先为交接记录指定当前项目中的接替人")
        matches = [
            member
            for member in self.repository.list_project_members()
            if self._active_member(member)
            and recipient in {
                str(member.get("display_name") or "").strip().lower(),
                str(member.get("email") or "").strip().lower(),
            }
        ]
        if not matches:
            raise ValueError("接替人无法映射到当前项目成员，请重新登记并选择有效成员")
        if len(matches) > 1:
            raise ValueError("接替人姓名存在重名，请使用企业邮箱重新登记")
        return matches[0]

    def detail(self, handover_id: str, identity: Optional[Identity] = None) -> Dict[str, Any]:
        record = self.repository.get_handover(handover_id)
        if not record:
            raise ValueError("交接记录不存在")
        return self.view(record, identity)

    def view(self, record: Dict[str, Any], identity: Optional[Identity] = None) -> Dict[str, Any]:
        result = summarize_handover(record)
        items = []
        for item in result.get("items") or []:
            rendered = {
                **item,
                **item_review_policy(
                    record, item, actor=identity.user_id if identity else "",
                    can_manage=bool(identity and self._is_manager(identity)),
                ),
                "evidence_assets": self.repository.list_handover_item_evidence_assets(
                    str(record.get("id") or ""), str(item.get("item_id") or ""), identity=identity,
                ),
            }
            if (
                identity
                and str(item.get("status") or "") in {"pending", "rejected"}
                and self._can_submit_item(record, item, identity)
            ):
                draft = self.repository.get_handover_item_draft(
                    str(record.get("id") or ""), str(item.get("item_id") or ""), identity.user_id,
                )
                rendered["submission_draft"] = {
                    **draft,
                    "asset_refs": [
                        self.repository._handover_evidence_reference_payload(reference, identity=identity)
                        for reference in draft.get("asset_refs") or []
                    ],
                }
            items.append(rendered)
        result["items"] = items
        return result

    @staticmethod
    def _is_manager(identity: Identity) -> bool:
        return bool(identity.can("handover.manage"))

    def _assert_participant(self, record: Dict[str, Any], identity: Identity) -> None:
        participant_ids = {
            str(record.get("departing_user_id") or ""),
            str(record.get("created_by_user_id") or ""),
            str(record.get("recipient_user_id") or ""),
        }
        participant_ids.update(str(item.get("owner_user_id") or "") for item in record.get("items") or [])
        participant_ids.discard("")
        if not self._is_manager(identity) and identity.user_id not in participant_ids:
            raise PermissionError("只有本次交接参与人或管理角色可以处理该交接")

    def _can_review(self, record: Dict[str, Any], identity: Identity) -> bool:
        return self._is_manager(identity) or str(record.get("recipient_user_id") or "") == identity.user_id

    def _can_submit_item(self, record: Dict[str, Any], item: Dict[str, Any], identity: Identity) -> bool:
        owner_ids = {
            str(item.get("owner_user_id") or ""),
            str(record.get("departing_user_id") or ""),
            str(record.get("created_by_user_id") or ""),
        }
        owner_ids.discard("")
        return self._is_manager(identity) or identity.user_id in owner_ids

    def _assert_can_submit_item(self, record: Dict[str, Any], item: Dict[str, Any], identity: Identity) -> None:
        self._assert_participant(record, identity)
        if not self._can_submit_item(record, item, identity):
            raise PermissionError("只有交接项责任人可以提交、关联资料或上传补充文件")

    def _recipient_identity(self, record: Dict[str, Any]) -> Identity:
        member = self.resolve_recipient(record)
        role = str(member.get("role") or "")
        return Identity(
            user_id=str(member.get("user_id") or ""),
            email=str(member.get("email") or ""),
            display_name=str(member.get("display_name") or ""),
            organization_id="", project_id=self.repository.project_id,
            project_name=str(member.get("project_name") or ""), role=role,
            permissions=ROLE_PERMISSIONS.get(role, frozenset()),
        )

    def _submission_asset_options(self, record: Dict[str, Any], identity: Identity) -> list[Dict[str, Any]]:
        """Return only current assets both the submitter and recipient may read."""
        recipient_identity = self._recipient_identity(record)
        options = []
        for asset in self.repository.list_assets(identity=identity):
            asset_id = str(asset.get("asset_id") or "")
            version = dict(asset.get("current_version") or {})
            version_id = str(version.get("version_id") or asset.get("current_version_id") or "")
            if not asset_id or not version_id or not self.repository.get_asset(asset_id, identity=recipient_identity):
                continue
            source_file = str((list(version.get("documents") or []) or [{}])[0].get("source_file") or "")
            options.append({
                "asset_id": asset_id,
                "version_id": version_id,
                "title": str(asset.get("title") or source_file or "未命名资料"),
                "source_file": source_file,
                "categories": list(asset.get("categories") or []),
                "source_kind": "existing",
            })
        return options

    def _validated_submission_refs(
        self, record: Dict[str, Any], identity: Identity, value: Any,
    ) -> list[Dict[str, str]]:
        refs = self.repository._normalize_handover_evidence_refs(value)
        if len(refs) > 10:
            raise ValueError("每个必交项最多关联 10 份资料")
        allowed = {
            (entry["asset_id"], entry["version_id"]): entry
            for entry in self._submission_asset_options(record, identity)
        }
        recipient_identity = self._recipient_identity(record)

        def frozen_reference_is_still_readable(reference: Dict[str, str]) -> bool:
            """Keep a selected revision stable when a newer version is published."""
            submitter_asset = self.repository.get_asset(reference["asset_id"], identity=identity)
            recipient_asset = self.repository.get_asset(reference["asset_id"], identity=recipient_identity)
            if not submitter_asset or not recipient_asset:
                return False
            if str(submitter_asset.get("status") or "") not in {"active", "review_due"}:
                return False
            submitter_version = self.repository.get_asset_version(
                reference["asset_id"], reference["version_id"], identity=identity,
            )
            recipient_version = self.repository.get_asset_version(
                reference["asset_id"], reference["version_id"], identity=recipient_identity,
            )
            valid_statuses = {"active", "superseded"}
            return bool(
                submitter_version
                and recipient_version
                and str(submitter_version.get("status") or "") in valid_statuses
                and str(recipient_version.get("status") or "") in valid_statuses
            )

        missing = [
            reference for reference in refs
            if (reference["asset_id"], reference["version_id"]) not in allowed
            and not frozen_reference_is_still_readable(reference)
        ]
        if missing:
            raise ValueError("所选资料已失效，或提交人和接替人中至少一方无权访问")
        return [
            {
                "asset_id": reference["asset_id"],
                "version_id": reference["version_id"],
                "source_kind": reference["source_kind"],
            }
            for reference in refs
        ]

    def submission_context(self, handover_id: str, item_id: str, identity: Identity) -> Dict[str, Any]:
        record = self.repository.get_handover(handover_id)
        if not record:
            raise ValueError("交接记录不存在")
        if str(record.get("status") or "") == "completed":
            raise ValueError("交接已关闭，不能再提交资料")
        item = next((entry for entry in record.get("items") or [] if entry.get("item_id") == item_id), None)
        if not item:
            raise ValueError("交接项不存在")
        if str(item.get("status") or "") not in {"pending", "rejected"}:
            raise ValueError("当前交接项不能再编辑提交资料")
        self._assert_can_submit_item(record, item, identity)
        draft = self.repository.get_handover_item_draft(handover_id, item_id, identity.user_id)
        return {
            "success": True,
            "item_id": item_id,
            "assets": self._submission_asset_options(record, identity),
            "draft": {
                **draft,
                "asset_refs": [
                    self.repository._handover_evidence_reference_payload(reference, identity=identity)
                    for reference in draft.get("asset_refs") or []
                ],
            },
            "message": "仅展示双方均可访问的当前有效资料；已关联草稿会保留其选定版本",
        }

    def save_submission_draft(
        self, handover_id: str, item_id: str, payload: Dict[str, Any], identity: Identity,
    ) -> Dict[str, Any]:
        context = self.submission_context(handover_id, item_id, identity)
        record = self.repository.get_handover(handover_id) or {}
        refs = self._validated_submission_refs(record, identity, payload.get("asset_refs") or [])
        evidence = str(payload.get("evidence") or "").strip()
        draft = self.repository.save_handover_item_draft(
            handover_id, item_id, identity.user_id, evidence=evidence, asset_refs=refs,
        )
        self.repository.write_security_audit(
            identity.user_id, "handover.item_draft_saved", "handover_item", item_id,
            {"handover_id": handover_id, "evidence_asset_count": len(refs)},
            project_id=identity.project_id,
        )
        return {
            **context,
            "draft": {
                **draft,
                "asset_refs": [
                    self.repository._handover_evidence_reference_payload(reference, identity=identity)
                    for reference in draft.get("asset_refs") or []
                ],
            },
            "message": "提交草稿已保存",
        }

    def append_uploaded_draft_asset(
        self, handover_id: str, item_id: str, asset_id: str, version_id: str, identity: Identity,
    ) -> Dict[str, Any]:
        context = self.submission_context(handover_id, item_id, identity)
        record = self.repository.get_handover(handover_id) or {}
        draft = self.repository.get_handover_item_draft(handover_id, item_id, identity.user_id)
        refs = [*list(draft.get("asset_refs") or []), {
            "asset_id": str(asset_id or ""), "version_id": str(version_id or ""), "source_kind": "uploaded",
        }]
        resolved = self._validated_submission_refs(record, identity, refs)
        draft = self.repository.save_handover_item_draft(
            handover_id, item_id, identity.user_id,
            evidence=str(draft.get("evidence") or ""), asset_refs=resolved,
        )
        self.repository.write_security_audit(
            identity.user_id, "handover.item_attachment_uploaded", "handover_item", item_id,
            {"handover_id": handover_id, "asset_id": asset_id, "version_id": version_id},
            project_id=identity.project_id,
        )
        return {
            **context,
            "draft": {
                **draft,
                "asset_refs": [
                    self.repository._handover_evidence_reference_payload(reference, identity=identity)
                    for reference in draft.get("asset_refs") or []
                ],
            },
            "message": "补充文件已入库并关联到当前提交草稿",
        }

    def item_action(
        self, handover_id: str, item_id: str, action: str, payload: Dict[str, Any], identity: Identity,
    ) -> Dict[str, Any]:
        record = self.detail(handover_id)
        if record.get("status") == "completed":
            raise ValueError("交接已关闭，不能再修改交接项")
        self._assert_participant(record, identity)
        item = next((entry for entry in record.get("items") or [] if entry.get("item_id") == item_id), None)
        if not item:
            raise ValueError("交接项不存在")
        action = str(action or "").strip().lower()
        if action == "submit":
            self._assert_can_submit_item(record, item, identity)
            raw_refs = payload.get("asset_refs") or []
            if not raw_refs and str(payload.get("asset_id") or "").strip():
                raw_refs = [{
                    "asset_id": str(payload.get("asset_id") or "").strip(),
                    "version_id": str(payload.get("version_id") or "").strip(),
                    "source_kind": "legacy",
                }]
            payload = {**payload, "asset_refs": self._validated_submission_refs(record, identity, raw_refs)}
        elif action in {"accept", "reject"}:
            policy = item_review_policy(record, item, actor=identity.user_id, can_manage=self._is_manager(identity))
            if action not in policy["review_actions"]:
                raise PermissionError(policy["review_block_reason"] or "当前交接项不能验收")
            if policy["review_action_mode"] == "proxy":
                proxy_reason = str(payload.get("proxy_reason") or "").strip()
                if not proxy_reason:
                    raise ValueError("请填写代验收原因")
                if len(proxy_reason) > 1000:
                    raise ValueError("代验收原因不能超过 1000 字")
        else:
            raise ValueError("不支持的交接项操作")

        task = None
        if action == "reject":
            reason = str(payload.get("reason") or "").strip()
            if not reason:
                raise ValueError("退回交接项必须说明补充要求")
            if item.get("status") != "submitted":
                raise ValueError("只有待验收交接项可以退回")
            task = self.knowledge_tasks.create_handover_gap_task(record, item, reason, identity)
            payload = {**payload, "knowledge_task_id": str(task.get("task_id") or "")}
        updated = self.repository.update_handover_item(
            handover_id, item_id, action, payload, actor=identity.user_id,
        )
        return {
            "success": True, "item": updated, "record": self.detail(handover_id, identity), "task": task,
            "message": {
                "submit": "交接项已提交，等待接替人验收",
                "accept": "交接项已验收通过",
                "reject": "交接项已退回，并创建交接补充任务",
            }[action],
        }

    def refresh_inventory(self, handover_id: str, identity: Identity) -> Dict[str, Any]:
        record = self.detail(handover_id)
        if record.get("status") == "completed":
            raise ValueError("交接已关闭，不能刷新知识盘点")
        self._assert_participant(record, identity)
        if not self._can_review(record, identity):
            raise PermissionError("只有接替人或交接管理角色可以刷新知识盘点")
        departing_user_id = str(record.get("departing_user_id") or "").strip()
        if not departing_user_id:
            raise ValueError("离职人员未映射到当前项目成员，无法自动盘点")
        result = self.repository.refresh_handover_inventory(
            handover_id, departing_user_id, actor=identity.user_id,
        )
        return {
            "success": True, "inventory": result["entries"], "record": self.detail(handover_id, identity),
            "message": f"知识盘点已刷新，共识别 {result['discovered_count']} 项关联事实",
        }

    def inventory_action(
        self, handover_id: str, inventory_id: str, action: str,
        payload: Dict[str, Any], identity: Identity,
    ) -> Dict[str, Any]:
        record = self.detail(handover_id)
        if record.get("status") == "completed":
            raise ValueError("交接已关闭，不能修改知识盘点")
        self._assert_participant(record, identity)
        if not self._can_review(record, identity):
            raise PermissionError("只有接替人或交接管理角色可以审阅知识盘点")
        if not any(item.get("inventory_id") == inventory_id for item in record.get("inventory") or []):
            raise ValueError("知识盘点项不存在")
        action = str(action or "").strip().lower()
        updated = self.repository.update_handover_inventory(
            handover_id, inventory_id, action, payload, actor=identity.user_id,
        )
        return {
            "success": True, "inventory_item": updated, "record": self.detail(handover_id, identity),
            "message": {
                "include": "盘点项已纳入交接范围",
                "exclude": "盘点项已记录排除原因",
                "reset": "盘点项已恢复待审阅",
            }.get(action, "知识盘点已更新"),
        }

    def create_risk(self, handover_id: str, payload: Dict[str, Any], identity: Identity) -> Dict[str, Any]:
        record = self.detail(handover_id)
        if record.get("status") == "completed":
            raise ValueError("交接已关闭，不能新增风险")
        self._assert_participant(record, identity)
        required = {
            "title": "风险标题", "impact": "影响说明", "owner_user_id": "责任人",
            "due_at": "截止时间", "mitigation": "缓解措施",
        }
        for key, label in required.items():
            if not str(payload.get(key) or "").strip():
                raise ValueError(f"{label}不能为空")
        severity = str(payload.get("severity") or "medium")
        if severity not in {"low", "medium", "high", "blocking"}:
            raise ValueError("不支持的风险等级")
        owner = self.repository.get_user(str(payload.get("owner_user_id") or ""))
        if not self._active_member(owner):
            raise ValueError("风险责任人不是当前项目的有效成员")
        risk = self.repository.create_handover_risk(handover_id, payload, actor=identity.user_id)
        return {
            "success": True, "risk": risk, "record": self.detail(handover_id, identity),
            "message": "交接风险已登记",
        }

    def risk_action(
        self, handover_id: str, risk_id: str, action: str, payload: Dict[str, Any], identity: Identity,
    ) -> Dict[str, Any]:
        record = self.detail(handover_id)
        if record.get("status") == "completed":
            raise ValueError("交接已关闭，不能再修改风险")
        self._assert_participant(record, identity)
        risk = next((entry for entry in record.get("risk_items") or [] if entry.get("risk_id") == risk_id), None)
        if not risk:
            raise ValueError("交接风险不存在")
        action = str(action or "").strip().lower()
        if action in {"mitigate", "submit_close"}:
            if not self._is_manager(identity) and str(risk.get("owner_user_id") or "") != identity.user_id:
                raise PermissionError("只有风险责任人可以更新缓解措施或提交关闭")
        elif action in {"close", "reopen"}:
            if not self._can_review(record, identity):
                raise PermissionError("只有接替人或交接管理角色可以验收或重开风险")
        else:
            raise ValueError("不支持的风险操作")
        updated = self.repository.update_handover_risk(
            handover_id, risk_id, action, payload, actor=identity.user_id,
        )
        return {
            "success": True, "risk": updated, "record": self.detail(handover_id, identity),
            "message": {
                "mitigate": "风险缓解措施已更新",
                "submit_close": "风险已提交关闭验收",
                "close": "风险已关闭",
                "reopen": "风险已重新打开",
            }[action],
        }

    def complete(self, handover_id: str, identity: Identity) -> Dict[str, Any]:
        if not self._is_manager(identity):
            raise PermissionError("只有交接管理角色可以关闭交接并生成快照")
        result = self.repository.complete_handover(handover_id, actor=identity.user_id)
        return {
            "success": True, **result,
            "record": self.view(result["record"], identity),
            "message": "交接已关闭并生成不可变快照" if result["created"] else "交接已关闭，返回既有快照",
        }

    def cancel(self, handover_id: str, identity: Identity) -> Dict[str, Any]:
        """Allow only the initiator to withdraw a handover before acceptance."""
        record = self.detail(handover_id)
        initiator_user_id = str(
            record.get("created_by_user_id") or record.get("departing_user_id") or ""
        )
        if not initiator_user_id or identity.user_id != initiator_user_id:
            raise PermissionError("只有发起人可以取消交接")
        if str(record.get("status") or "") != "pending_acceptance":
            raise ValueError("仅待接替人确认的交接可以取消")
        self.repository.cancel_handover(handover_id, actor=identity.user_id)
        return {
            "success": True,
            "message": "交接已取消，已上传的知识文档仍保留在知识库中",
        }

    def accept(self, handover_id: str, identity: Identity) -> Dict[str, Any]:
        record = self.repository.get_handover(handover_id)
        if not record:
            raise ValueError("交接记录不存在")
        recipient = self.resolve_recipient(record)
        recipient_user_id = str(recipient.get("user_id") or "")
        if not identity.can("handover.manage") and recipient_user_id != identity.user_id:
            raise PermissionError("只能确认分配给自己的交接任务")

        if (
            str(record.get("status") or "") in {"accepted", "completed"}
            and str(record.get("learning_plan_status") or "") == "ready"
            and record.get("learning_plan_id")
        ):
            plan = self.repository.get_onboarding_plan(str(record["learning_plan_id"]))
            if plan:
                return {
                    "success": True, "partial_success": False,
                    "record": self.view(record, identity), "plan": plan, "created": False,
                    "message": "交接已确认，接替人学习计划已存在",
                }

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        updated = {
            **record,
            "status": "accepted",
            "recipient": str(recipient.get("display_name") or record.get("recipient") or ""),
            "recipient_user_id": recipient_user_id,
            "accepted_by": str(record.get("accepted_by") or identity.display_name or identity.email),
            "accepted_by_user_id": str(record.get("accepted_by_user_id") or identity.user_id),
            "accepted_at": str(record.get("accepted_at") or now),
            "learning_plan_status": "syncing",
            "learning_plan_error": "",
        }
        updated = self.repository.save_handover(
            updated, actor=identity.user_id, action="handover.accepted",
        )

        try:
            plan = self.onboarding.sync_handover_plan(updated, recipient_user_id, identity)
            unavailable = list(plan.get("unavailable_asset_ids") or [])
            plan_status = "needs_attention" if unavailable else "ready"
            plan_error = (
                f"{len(unavailable)} 份交接资料当前不在接替人的授权范围内，请调整来源权限后重试"
                if unavailable else ""
            )
            updated = self.repository.save_handover(
                {
                    **updated,
                    "learning_plan_id": str(plan.get("plan_id") or ""),
                    "learning_plan_status": plan_status,
                    "learning_plan_error": plan_error,
                    "learning_plan_synced_at": now,
                    "role_key": str(
                        self.onboarding.resolve_template(
                            str(updated.get("role_key") or updated.get("role") or ""), fallback=False,
                        ).get("role_key") or ""
                    ),
                },
                actor=identity.user_id, action="handover.learning_plan_synced",
            )
            partial = plan_status != "ready"
            return {
                "success": True, "partial_success": partial,
                "record": self.view(updated, identity), "plan": plan,
                "created": bool(plan.get("created")),
                "message": (
                    "交接已确认，但部分交接资料尚未进入接替人的学习计划"
                    if partial
                    else (
                        "交接已确认，已为接替人生成学习计划"
                        if plan.get("created")
                        else "交接已确认，本次交接资料已合并到接替人的进行中计划"
                    )
                ),
            }
        except (PermissionError, ValueError) as exc:
            updated = self.repository.save_handover(
                {
                    **updated,
                    "learning_plan_status": "needs_attention",
                    "learning_plan_error": str(exc)[:500],
                },
                actor=identity.user_id, action="handover.learning_plan_sync_failed",
            )
            return {
                "success": True, "partial_success": True,
                "record": self.view(updated, identity), "plan": None, "created": False,
                "message": f"交接已确认，但学习计划待处理：{exc}",
            }
        except Exception:
            logger.exception("交接学习计划同步失败: %s", handover_id)
            updated = self.repository.save_handover(
                {
                    **updated,
                    "learning_plan_status": "needs_attention",
                    "learning_plan_error": "学习计划生成失败，请稍后重试",
                },
                actor=identity.user_id, action="handover.learning_plan_sync_failed",
            )
            return {
                "success": True, "partial_success": True,
                "record": self.view(updated, identity), "plan": None, "created": False,
                "message": "交接已确认，但学习计划生成失败，请稍后重试",
            }


handover_service = HandoverService()


__all__ = ["HandoverService", "handover_service", "summarize_handover"]
