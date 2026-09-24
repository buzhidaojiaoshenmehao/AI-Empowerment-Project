"""Deterministic onboarding templates, readiness and personal learning plans."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional

from backend.auth import ROLE_PERMISSIONS
from backend.auth_context import Identity
from backend.knowledge_base.readiness import calculate_role_readiness
from backend.storage import get_repository


DEFAULT_ROLE_TEMPLATES: List[Dict[str, Any]] = [
    {
        "role_key": "project_manager", "name": "项目经理",
        "focus": "先掌握项目目标、里程碑、关键决策和未关闭风险，再确认责任人与推进节奏。",
        "topics": [
            ("proj_plan", "项目管理计划", 2, "根据当前计划整理未来两周的关键里程碑与责任人"),
            ("proj_decision", "决策记录", 2, "选择一项关键决策，说明背景、取舍和影响"),
            ("proj_risk", "风险日志", 2, "完成一次风险盘点并提出跟踪动作"),
            ("proj_report", "报告与状态", 1, "输出一份结构化项目状态摘要"),
            ("opa_process", "流程与规范", 1, "说明一项核心管理流程的入口与完成标准"),
            ("交接文档", "交接文档", 1, "核对交接范围、责任人与未完成事项"),
        ],
    },
    {
        "role_key": "developer", "name": "开发工程师",
        "focus": "优先理解需求边界、技术架构、接口约束、发布流程和未解决技术风险。",
        "topics": [
            ("proj_architecture", "技术架构", 2, "画出核心调用链并说明关键依赖"),
            ("proj_requirement", "需求文档", 2, "选择一项需求说明边界、异常和验收点"),
            ("proj_decision", "决策记录", 1, "复述一项技术决策及其替代方案"),
            ("opa_process", "流程与规范", 1, "在测试环境完成一次标准开发与发布流程"),
            ("proj_risk", "风险日志", 1, "识别一个技术风险并提出验证方案"),
            ("交接文档", "交接文档", 1, "核对模块负责人、运行入口和遗留事项"),
        ],
    },
    {
        "role_key": "tester", "name": "测试工程师",
        "focus": "建立需求、验收标准、风险和回归策略之间的对应关系。",
        "topics": [
            ("proj_requirement", "需求文档", 2, "为一项需求补充正常、异常和权限测试点"),
            ("proj_acceptance", "验收标准", 2, "基于验收标准执行一次端到端验收"),
            ("proj_risk", "风险日志", 1, "把一个项目风险转换为可执行测试场景"),
            ("opa_process", "流程与规范", 1, "完成缺陷提交、验证和关闭流程"),
            ("proj_report", "报告与状态", 1, "输出一份可追溯测试结果摘要"),
            ("交接文档", "交接文档", 1, "核对历史问题、回归范围和遗留风险"),
        ],
    },
    {
        "role_key": "product_manager", "name": "产品经理",
        "focus": "优先理解需求来源、业务取舍、用户反馈和关键决策依据。",
        "topics": [
            ("proj_requirement", "需求文档", 2, "梳理一项需求的用户、场景、价值和边界"),
            ("proj_decision", "决策记录", 2, "说明一项产品取舍及其证据"),
            ("eef_market", "市场环境", 1, "整理与当前产品相关的外部变化及影响"),
            ("proj_report", "报告与状态", 1, "形成一次面向干系人的进展同步"),
            ("opa_lessons", "经验教训库", 1, "把一次复盘转化为可复用产品原则"),
            ("交接文档", "交接文档", 1, "核对需求责任、决策背景和待办事项"),
        ],
    },
    {
        "role_key": "operations", "name": "运维工程师",
        "focus": "重点确认运行环境、发布回滚、监控告警和高风险依赖。",
        "topics": [
            ("eef_infrastructure", "基础设施", 2, "说明环境拓扑和关键基础设施依赖"),
            ("opa_process", "流程与规范", 2, "在测试环境完成一次发布与回滚演练"),
            ("proj_architecture", "技术架构", 1, "标注运行链路中的关键服务和故障边界"),
            ("proj_risk", "风险日志", 1, "验证一个高风险依赖的降级方案"),
            ("proj_report", "报告与状态", 1, "输出运行状态、告警和处置摘要"),
            ("交接文档", "交接文档", 1, "核对账号、配置、联系人和应急流程"),
        ],
    },
]


def _default_payload(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "role_key": item["role_key"], "name": item["name"], "focus": item["focus"],
        "topics": [
            {
                "topic_key": key, "label": label, "weight": weight,
                "practice_task": practice,
                "completion_standard": "能够结合当前项目资料说明并完成一次实际操作",
            }
            for key, label, weight, practice in item["topics"]
        ],
    }


class OnboardingService:
    """Own onboarding state without delegating permissions or completion to an LLM."""

    def __init__(self, repository_provider=get_repository) -> None:
        self.repository_provider = repository_provider

    @property
    def repository(self):
        return self.repository_provider()

    @staticmethod
    def _can_manage(identity: Identity) -> bool:
        return identity.can("onboarding.manage")

    def ensure_defaults(self) -> None:
        for item in DEFAULT_ROLE_TEMPLATES:
            if not self.repository.get_onboarding_template(role_key=item["role_key"]):
                self.repository.upsert_onboarding_template(_default_payload(item), "user_system")

    def list_templates(self, identity: Identity) -> List[Dict[str, Any]]:
        self.ensure_defaults()
        templates = self.repository.list_onboarding_templates()
        return [
            {**template, "can_manage": self._can_manage(identity)}
            for template in templates
        ]

    def save_template(self, payload: Dict[str, Any], identity: Identity) -> Dict[str, Any]:
        if not self._can_manage(identity):
            raise PermissionError("当前项目角色无权维护岗位模板")
        return self.repository.upsert_onboarding_template(payload, identity.user_id)

    def _target_identity(self, identity: Identity, user_id: str = "") -> Identity:
        target_id = str(user_id or identity.user_id).strip()
        if target_id == identity.user_id:
            return identity
        if not self._can_manage(identity):
            raise PermissionError("只能查看自己的学习计划")
        member = self.repository.get_user(target_id, project_id=identity.project_id)
        if not member or member.get("membership_status") != "active" or member.get("status") != "active":
            raise ValueError("目标成员不存在或已停用")
        role = str(member.get("role") or "project_member")
        return Identity(
            user_id=target_id, email=str(member.get("email") or ""),
            display_name=str(member.get("display_name") or ""),
            organization_id=identity.organization_id, project_id=identity.project_id,
            project_name=identity.project_name, role=role,
            permissions=ROLE_PERMISSIONS.get(role, frozenset()),
        )

    def _template(self, role: str, *, fallback: bool = True) -> Dict[str, Any]:
        self.ensure_defaults()
        text = str(role or "").strip()
        template = self.repository.get_onboarding_template(template_id=text)
        if not template:
            template = self.repository.get_onboarding_template(role_key=text)
        if not template:
            template = next(
                (item for item in self.repository.list_onboarding_templates() if item["name"] == text),
                None,
            )
        if not template and fallback:
            template = self.repository.get_onboarding_template(role_key="developer")
        if not template:
            raise ValueError("交接岗位没有对应的有效岗位模板，请先在新人赋能中维护岗位模板")
        return template

    def resolve_template(self, role: str, *, fallback: bool = False) -> Dict[str, Any]:
        """Resolve a role key/name without silently changing handover semantics."""
        return self._template(role, fallback=fallback)

    @staticmethod
    def _plan_progress(plan: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not plan:
            return None
        dimensions = []
        mapping = [
            ("reading", "必读资料", 60),
            ("practice", "实践任务", 25),
            ("manager_confirmation", "负责人确认", 15),
        ]
        overall = 0.0
        for item_type, label, formula_weight in mapping:
            items = [item for item in plan.get("items") or [] if item.get("item_type") == item_type]
            total_weight = sum(float(item.get("weight") or 1) for item in items)
            completed_weight = sum(
                float(item.get("weight") or 1) for item in items if item.get("status") == "completed"
            )
            ratio = completed_weight / total_weight if total_weight else 0.0
            contribution = ratio * formula_weight
            overall += contribution
            dimensions.append({
                "key": item_type, "label": label, "weight": formula_weight,
                "score": round(ratio * 100, 1), "contribution": round(contribution, 1),
                "completed": len([item for item in items if item.get("status") == "completed"]),
                "total": len(items),
            })
        return {
            **plan,
            "progress": round(overall, 1),
            "formula": "必读完成 60% + 实践任务 25% + 负责人确认 15%",
            "dimensions": dimensions,
            "blocked_count": len([item for item in plan.get("items") or [] if item.get("status") == "blocked"]),
        }

    def guide(self, role: str, identity: Identity, *, user_id: str = "") -> Dict[str, Any]:
        template = self._template(role)
        target = self._target_identity(identity, user_id)
        visible_assets = [dict(asset) for asset in self.repository.readiness_assets(identity=target)]
        required_topics: Dict[str, set[str]] = {}
        for topic in template["topics"]:
            for asset_id in topic.get("required_asset_ids") or []:
                required_topics.setdefault(str(asset_id), set()).add(str(topic["topic_key"]))
        for asset in visible_assets:
            required = required_topics.get(str(asset.get("asset_id") or ""), set())
            if required:
                asset["topics"] = list(dict.fromkeys([
                    *(asset.get("topics") or []), *sorted(required),
                ]))
        role_template = {
            "role": template["name"],
            "topics": [
                {
                    "topic_id": item["topic_key"], "label": item["label"],
                    "weight": item["weight"],
                }
                for item in template["topics"]
            ],
        }
        result = calculate_role_readiness(
            role_template, visible_assets,
        )
        topic_config = {item["topic_key"]: item for item in template["topics"]}
        coverage = []
        recommended = []
        seen_assets = set()
        for item in result["topics"]:
            config = topic_config.get(item["topic_id"], {})
            coverage.append({
                "category": item["topic_id"], "label": item["label"],
                "weight": item["weight"], "document_count": item["asset_count"],
                "status": "已覆盖" if item["covered"] else "待补充",
                "scores": item["scores"], "evidence": item["evidence"],
                "practice_task": config.get("practice_task", ""),
                "completion_standard": config.get("completion_standard", ""),
                "owner_user_id": config.get("owner_user_id", ""),
            })
            for evidence in item["evidence"]:
                asset_id = str(evidence.get("asset_id") or "")
                if not asset_id or asset_id in seen_assets:
                    continue
                seen_assets.add(asset_id)
                recommended.append({
                    "asset_id": asset_id, "version_id": evidence.get("version_id", ""),
                    "source_file": evidence.get("title", "未命名知识"),
                    "category": item["topic_id"], "category_label": item["label"],
                    "reason": f"覆盖岗位必需主题“{item['label']}”",
                    "quality": {
                        "authority": evidence["authority"], "freshness": evidence["freshness"],
                        "traceability": evidence["traceability"],
                    },
                })
        plans = self.repository.list_onboarding_plans(user_id=target.user_id, status="active")
        current_plan = next(
            (plan for plan in plans if plan.get("template_id") == template["template_id"]), None,
        )
        plan = self._plan_progress(current_plan)
        if plan:
            plan["template_outdated"] = int(plan.get("template_revision") or 0) != int(template["revision"])
        return {
            "success": True, "role": template["name"], "role_key": template["role_key"],
            "template": template, "target_user": {
                "user_id": target.user_id, "display_name": target.display_name, "email": target.email,
            },
            "focus": template["focus"], "readiness": result["readiness"],
            "formula": result["formula"], "dimensions": result["dimensions"],
            "coverage": coverage,
            "gaps": [
                {"category": item["topic_id"], "label": item["label"], "weight": item["weight"]}
                for item in result["gaps"]
            ],
            "recommended_documents": recommended[:12],
            "learning_path": [
                item.get("practice_task") or f"学习并说明 {item['label']}"
                for item in template["topics"]
            ],
            "document_count": result["eligible_asset_count"],
            "input_asset_count": result["input_asset_count"],
            "excluded_asset_count": result["excluded_asset_count"],
            "calculated_at": result["calculated_at"],
            "plan": plan,
            "can_manage": self._can_manage(identity),
        }

    def create_plan(self, payload: Dict[str, Any], identity: Identity) -> Dict[str, Any]:
        if not self._can_manage(identity):
            raise PermissionError("当前项目角色无权分配学习计划")
        template = self._template(str(payload.get("template_id") or payload.get("role") or ""))
        target = self._target_identity(identity, str(payload.get("user_id") or ""))
        guide = self.guide(template["role_key"], identity, user_id=target.user_id)
        coverage = {item["category"]: item for item in guide["coverage"]}
        items: List[Dict[str, Any]] = []
        for topic in template["topics"]:
            topic_key = str(topic["topic_key"])
            matched = list((coverage.get(topic_key) or {}).get("evidence") or [])
            required = set(topic.get("required_asset_ids") or [])
            if required:
                matched = [item for item in matched if item.get("asset_id") in required]
            if matched:
                for evidence in matched[:3]:
                    items.append({
                        "topic_key": topic_key, "item_type": "reading", "weight": topic["weight"],
                        "title": f"阅读：{evidence.get('title') or topic['label']}",
                        "description": f"对应岗位主题：{topic['label']}。{topic.get('completion_standard') or ''}",
                        "asset_id": evidence.get("asset_id", ""), "version_id": evidence.get("version_id", ""),
                    })
            else:
                items.append({
                    "topic_key": topic_key, "item_type": "reading", "weight": topic["weight"],
                    "title": f"待补充资料：{topic['label']}",
                    "description": "当前权限范围内没有可用资料，需先完成知识补充任务。",
                    "status": "blocked",
                })
            if topic.get("practice_task"):
                items.append({
                    "topic_key": topic_key, "item_type": "practice", "weight": topic["weight"],
                    "title": str(topic["practice_task"]),
                    "description": str(topic.get("completion_standard") or "提交可核验的实践结果"),
                })
        items.append({
            "item_type": "manager_confirmation", "title": "负责人确认已达到岗位上手标准",
            "description": "阅读和实践任务全部完成后，由负责人结合证据确认。", "weight": 1,
        })
        target_date = str(payload.get("target_date") or "").strip()
        if not target_date:
            target_date = (datetime.now(timezone.utc) + timedelta(days=30)).date().isoformat()
        plan = self.repository.create_onboarding_plan({
            "user_id": target.user_id, "template_id": template["template_id"],
            "target_date": target_date, "manager_user_id": identity.user_id,
            "source_type": str(payload.get("source_type") or "manual"),
            "source_key": str(payload.get("source_key") or ""),
        }, items, identity.user_id)
        return self._plan_progress(plan) or {}

    def _workflow_identity(self, actor: str = "user_system") -> Identity:
        return Identity(
            user_id=str(actor or "user_system"), email="", display_name="系统自动化",
            organization_id="org_default", project_id=self.repository.project_id,
            project_name=self.repository.project_name(), role="project_admin",
            permissions=frozenset({"*"}), is_system=True,
        )

    def _topic_matches(
        self, template: Dict[str, Any], target: Identity,
    ) -> Dict[str, Dict[str, Any]]:
        """Resolve current reading candidates using the target member's own ACL."""
        guide = self.guide(str(template["role_key"]), target)
        topic_config = {str(item["topic_key"]): item for item in template.get("topics") or []}
        matches: Dict[str, Dict[str, Any]] = {}
        for coverage in guide.get("coverage") or []:
            topic_key = str(coverage.get("category") or "")
            topic = topic_config.get(topic_key) or {}
            required = {
                str(asset_id) for asset_id in topic.get("required_asset_ids") or []
                if str(asset_id or "").strip()
            }
            evidence = list(coverage.get("evidence") or [])
            if required:
                evidence = [item for item in evidence if str(item.get("asset_id") or "") in required]
            label = str(topic.get("label") or coverage.get("label") or topic_key)
            matches[topic_key] = {
                "label": label,
                "candidates": [
                    {
                        "asset_id": str(item.get("asset_id") or ""),
                        "version_id": str(item.get("version_id") or ""),
                        "title": f"阅读：{item.get('title') or label}",
                        "description": (
                            f"对应岗位主题：{label}。"
                            f"{topic.get('completion_standard') or ''}"
                        ),
                        "weight": float(topic.get("weight") or coverage.get("weight") or 1),
                    }
                    for item in evidence[:3]
                    if item.get("asset_id") and item.get("version_id")
                ],
            }
        for topic_key, topic in topic_config.items():
            matches.setdefault(topic_key, {
                "label": str(topic.get("label") or topic_key), "candidates": [],
            })
        return matches

    def _refresh_plan_materials(
        self, plan: Dict[str, Any], target: Identity, *, actor: str, trigger: str,
    ) -> Dict[str, Any]:
        template = self._template(str(plan.get("template_id") or ""), fallback=False)
        refreshed = self.repository.refresh_onboarding_reading_items(
            str(plan["plan_id"]), self._topic_matches(template, target), actor,
            trigger=trigger,
        )
        return self._plan_progress(refreshed) or {}

    def refresh_plan_materials(self, plan_id: str, identity: Identity) -> Dict[str, Any]:
        plan = self.repository.get_onboarding_plan(plan_id)
        if not plan:
            raise ValueError("学习计划不存在")
        if str(plan.get("user_id") or "") != identity.user_id and not self._can_manage(identity):
            raise PermissionError("只能刷新自己的学习计划")
        target = self._target_identity(identity, str(plan.get("user_id") or ""))
        return self._refresh_plan_materials(
            plan, target, actor=identity.user_id, trigger="manual",
        )

    def refresh_active_plans(
        self, *, actor: str = "user_system", trigger_asset_id: str = "",
        trigger: str = "asset_published",
    ) -> Dict[str, Any]:
        """Best-effort reconciliation for every active plan after knowledge changes."""
        workflow = self._workflow_identity(actor)
        plans = self.repository.list_onboarding_plans(status="active")
        results: List[Dict[str, Any]] = []
        errors: List[Dict[str, str]] = []
        changed = unblocked = 0
        for plan in plans:
            try:
                target = self._target_identity(workflow, str(plan.get("user_id") or ""))
                refreshed = self._refresh_plan_materials(
                    plan, target, actor=actor, trigger=trigger,
                )
                summary = dict(refreshed.get("refresh") or {})
                changed += int(bool(summary.get("changed")))
                unblocked += int(summary.get("unblocked_count") or 0)
                results.append({
                    "plan_id": str(plan.get("plan_id") or ""),
                    "user_id": str(plan.get("user_id") or ""),
                    **summary,
                })
            except Exception as exc:
                errors.append({
                    "plan_id": str(plan.get("plan_id") or ""),
                    "error": str(exc)[:300],
                })
        return {
            "success": not errors, "plan_count": len(plans),
            "changed_plan_count": changed, "unblocked_count": unblocked,
            "trigger_asset_id": str(trigger_asset_id or ""),
            "results": results, "errors": errors,
        }

    def sync_handover_plan(
        self, record: Dict[str, Any], recipient_user_id: str, identity: Identity,
    ) -> Dict[str, Any]:
        """Create/reuse a plan and idempotently attach this handover's visible assets."""
        template = self._template(
            str(record.get("role_key") or record.get("role") or ""), fallback=False,
        )
        workflow_identity = Identity(
            user_id=identity.user_id, email=identity.email, display_name=identity.display_name,
            organization_id=identity.organization_id, project_id=identity.project_id,
            project_name=identity.project_name, role=identity.role,
            permissions=frozenset({"*"}), session_id=identity.session_id, is_system=True,
        )
        target = self._target_identity(workflow_identity, recipient_user_id)
        guide = self.guide(template["role_key"], workflow_identity, user_id=target.user_id)
        preferred_asset_ids = {
            str(value)
            for value in (
                list(record.get("asset_ids") or [])
                + self.repository.list_handover_asset_ids(str(record.get("id") or ""))
            )
            if str(value or "").strip()
        }
        visible_scope: Dict[str, Dict[str, Any]] = {}
        for coverage in guide.get("coverage") or []:
            for evidence in coverage.get("evidence") or []:
                asset_id = str(evidence.get("asset_id") or "")
                if asset_id and asset_id in preferred_asset_ids:
                    visible_scope[asset_id] = {
                        "topic_key": str(coverage.get("category") or "交接文档"),
                        "title": f"交接必读：{evidence.get('title') or coverage.get('label') or '交接资料'}",
                        "description": (
                            f"本次交接范围资料，对应岗位主题：{coverage.get('label') or '交接文档'}。"
                            "阅读后核对职责、风险和未完成事项。"
                        ),
                        "asset_id": asset_id,
                        "version_id": str(evidence.get("version_id") or ""),
                        "weight": float(coverage.get("weight") or 1),
                    }
        plan = self.create_plan({
            "user_id": target.user_id, "template_id": template["template_id"],
            "target_date": str(record.get("due_date") or ""),
            "source_type": "handover", "source_key": str(record.get("id") or ""),
        }, workflow_identity)
        was_created = bool(plan.get("created"))
        linked = self.repository.add_onboarding_reading_items(
            str(plan["plan_id"]), list(visible_scope.values()), identity.user_id,
        )
        result = self._plan_progress(linked) or {}
        result.update({
            "created": was_created,
            "scope_asset_count": len(preferred_asset_ids),
            "linked_scope_asset_count": len(visible_scope),
            "unavailable_asset_ids": sorted(preferred_asset_ids - set(visible_scope)),
            "added_scope_asset_count": int(linked.get("added_count") or 0),
        })
        return result

    def list_plans(self, identity: Identity, *, user_id: str = "") -> List[Dict[str, Any]]:
        target = self._target_identity(identity, user_id)
        return [self._plan_progress(plan) for plan in self.repository.list_onboarding_plans(user_id=target.user_id)]

    def list_supervised_plans(self, identity: Identity) -> List[Dict[str, Any]]:
        """Return compact active-plan summaries for the current plan manager.

        The supervisor worklist deliberately excludes task descriptions and
        evidence. Those are loaded only after the supervisor opens one member's
        plan, while the summary exposes the minimum information needed to decide
        who needs follow-up or is ready for confirmation.
        """
        if not self._can_manage(identity):
            raise PermissionError("当前项目角色无权查看主管学习计划")
        summaries: List[Dict[str, Any]] = []
        for raw_plan in self.repository.list_onboarding_plans(
            manager_user_id=identity.user_id, status="active",
        ):
            plan = self._plan_progress(raw_plan) or {}
            confirmation = next(
                (item for item in plan.get("items") or [] if item.get("item_type") == "manager_confirmation"),
                None,
            )
            pending_items = [
                item for item in plan.get("items") or []
                if item.get("item_type") != "manager_confirmation" and item.get("status") != "completed"
            ]
            blocked_items = [item for item in pending_items if item.get("status") == "blocked"]
            can_confirm = bool(
                confirmation
                and confirmation.get("status") != "completed"
                and not pending_items
            )
            if can_confirm:
                confirmation_hint = "阅读与实践任务均已完成，可由负责人确认"
            elif blocked_items:
                confirmation_hint = f"仍有 {len(blocked_items)} 项资料待补充"
            else:
                confirmation_hint = f"仍有 {len(pending_items)} 项阅读或实践任务待完成"
            summaries.append({
                "plan_id": plan.get("plan_id", ""),
                "user_id": plan.get("user_id", ""),
                "user_name": plan.get("user_name", ""),
                "user_email": plan.get("user_email", ""),
                "role_key": plan.get("role_key", ""),
                "role_name": plan.get("role_name", ""),
                "status": plan.get("status", "active"),
                "target_date": plan.get("target_date", ""),
                "updated_at": plan.get("updated_at", ""),
                "progress": plan.get("progress", 0),
                "blocked_count": plan.get("blocked_count", 0),
                "dimensions": plan.get("dimensions", []),
                "manager_confirmation": {
                    "item_id": confirmation.get("item_id", "") if confirmation else "",
                    "status": confirmation.get("status", "pending") if confirmation else "pending",
                    "can_confirm": can_confirm,
                    "pending_count": len(pending_items),
                    "hint": confirmation_hint,
                },
            })
        return sorted(
            summaries,
            key=lambda item: (
                0 if item["manager_confirmation"]["can_confirm"] else 1,
                0 if not item.get("blocked_count") else 1,
                str(item.get("target_date") or "9999-12-31"),
                str(item.get("updated_at") or ""),
                str(item.get("plan_id") or ""),
            ),
        )

    def update_item(
        self, plan_id: str, item_id: str, payload: Dict[str, Any], identity: Identity,
    ) -> Dict[str, Any]:
        plan = self.repository.get_onboarding_plan(plan_id)
        if not plan:
            raise ValueError("学习计划不存在")
        can_manage = self._can_manage(identity)
        if plan["user_id"] != identity.user_id and not can_manage:
            raise PermissionError("只能更新自己的学习任务")
        item = next((item for item in plan.get("items") or [] if item["item_id"] == item_id), None)
        if not item:
            raise ValueError("学习任务不存在")
        status = str(payload.get("status") or "completed")
        evidence = str(payload.get("evidence") or "").strip()
        if item["item_type"] == "manager_confirmation":
            if not can_manage:
                raise PermissionError("负责人确认只能由项目管理角色完成")
            if status == "completed" and any(
                candidate["item_type"] != "manager_confirmation" and candidate["status"] != "completed"
                for candidate in plan.get("items") or []
            ):
                raise ValueError("阅读与实践任务尚未全部完成，不能负责人确认")
        elif item["status"] == "blocked" and status == "completed" and not item.get("asset_id"):
            raise ValueError("该主题缺少可用资料，请先补齐知识并重新生成计划")
        if item["item_type"] == "practice" and status == "completed" and len(evidence) < 4:
            raise ValueError("实践任务完成时需要填写可核验的完成证据")
        prospective = [
            {**candidate, "status": status if candidate["item_id"] == item_id else candidate["status"]}
            for candidate in plan.get("items") or []
        ]
        complete_plan = bool(prospective) and all(candidate["status"] == "completed" for candidate in prospective)
        updated = self.repository.update_onboarding_item(
            plan_id, item_id, status=status, evidence=evidence,
            actor=identity.user_id, complete_plan=complete_plan,
        )
        return self._plan_progress(updated) or {}


onboarding_service = OnboardingService()


__all__ = ["DEFAULT_ROLE_TEMPLATES", "OnboardingService", "onboarding_service"]
