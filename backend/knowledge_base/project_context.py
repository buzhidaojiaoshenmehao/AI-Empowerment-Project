"""
项目知识语境管理模块

将项目知识分为三大类，实现结构化管理和上下文感知检索：

1. 事业环境因素 (EEF)  — 组织文化、行业标准、法规要求、市场环境
2. 组织过程资产 (OPA)  — 流程规范、模板库、历史经验、培训材料
3. 项目自身资产          — 管理计划、裁剪指南、决策记录、风险日志、验收报告

每个知识块关联项目阶段、知识类别、标签，实现精准的上下文检索。
"""
import re
import logging
from functools import wraps
from threading import RLock
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


def _synchronized(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapped


class KnowledgeCategory(str, Enum):
    """知识类别（PMI 标准分类扩展）"""
    # 事业环境因素
    EEF_CULTURE = "eef_culture"            # 组织文化与结构
    EEF_STANDARD = "eef_standard"           # 行业标准与法规
    EEF_MARKET = "eef_market"               # 市场环境
    EEF_INFRASTRUCTURE = "eef_infrastructure"  # 基础设施

    # 组织过程资产
    OPA_PROCESS = "opa_process"             # 流程与规范
    OPA_TEMPLATE = "opa_template"           # 模板与工具
    OPA_LESSONS = "opa_lessons"             # 经验教训库
    OPA_TRAINING = "opa_training"           # 培训材料
    OPA_HISTORICAL = "opa_historical"       # 历史数据

    # 项目自身资产
    PROJ_PLAN = "proj_plan"                 # 项目管理计划
    PROJ_TAILORING = "proj_tailoring"       # 裁剪指南
    PROJ_DECISION = "proj_decision"         # 决策记录
    PROJ_RISK = "proj_risk"                 # 风险日志
    PROJ_REPORT = "proj_report"             # 报告与状态
    PROJ_REQUIREMENT = "proj_requirement"   # 需求文档
    PROJ_ARCHITECTURE = "proj_architecture" # 技术架构
    PROJ_COMMUNICATION = "proj_communication"  # 沟通记录
    PROJ_ACCEPTANCE = "proj_acceptance"     # 验收标准

    @classmethod
    def get_group(cls, category: str) -> str:
        if category.startswith("eef_"):
            return "事业环境因素 (EEF)"
        elif category.startswith("opa_"):
            return "组织过程资产 (OPA)"
        elif category.startswith("proj_"):
            return "项目自身资产"
        return "未分类"


KNOWLEDGE_CATEGORY_LABELS = {
    "eef_culture": "组织文化与结构",
    "eef_standard": "行业标准与法规",
    "eef_market": "市场环境",
    "eef_infrastructure": "基础设施",
    "opa_process": "流程与规范",
    "opa_template": "模板与工具",
    "opa_lessons": "经验教训库",
    "opa_training": "培训材料",
    "opa_historical": "历史数据",
    "proj_plan": "项目管理计划",
    "proj_tailoring": "裁剪指南",
    "proj_decision": "决策记录",
    "proj_risk": "风险日志",
    "proj_report": "报告与状态",
    "proj_requirement": "需求文档",
    "proj_architecture": "技术架构",
    "proj_communication": "沟通记录",
    "proj_acceptance": "验收标准",
}


def knowledge_category_label(category: Any) -> str:
    """Return a user-facing category name without exposing taxonomy keys."""
    value = str(category or "").strip()
    return KNOWLEDGE_CATEGORY_LABELS.get(value, value or "未分类")


class ProjectContext:
    """项目语境管理"""

    def __init__(self, project_name: str = "默认项目"):
        self.project_name = project_name
        self._metadata: Dict[str, Dict] = {}
        self._categories: Dict[str, List[str]] = {c.value: [] for c in KnowledgeCategory}
        self._loaded = False
        self._loaded_scope = ""
        self._lock = RLock()

    def _rebuild_category_index(self):
        self._categories = {c.value: [] for c in KnowledgeCategory}
        for doc_id, metadata in self._metadata.items():
            for category in metadata.get("categories", []):
                self._categories.setdefault(category, []).append(doc_id)

    @staticmethod
    def _parse_time(value: Any) -> Optional[datetime]:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            return None

    @classmethod
    def _is_current_document(cls, item: Dict[str, Any], now: Optional[datetime] = None) -> bool:
        """Keep the context cache aligned with the authoritative active version."""
        if item.get("is_current_version") is False:
            return False
        allowed = {"active", "published", "effective", "review_due"}
        for key in ("status", "asset_status", "version_status"):
            value = str(item.get(key) or "").strip().lower()
            if value and value not in allowed:
                return False
        current = now or datetime.now(timezone.utc)
        valid_from = cls._parse_time(item.get("valid_from"))
        valid_until = cls._parse_time(item.get("valid_until"))
        if valid_from and valid_from > current:
            return False
        if valid_until and valid_until <= current:
            return False
        return True

    def _ensure_loaded(self, force: bool = False):
        """Refresh the authorized current projection before every read.

        Lifecycle and ACL changes are security events, so an identity-scoped
        in-process cache is not sufficient: a revoked version or a tightened
        access policy must disappear without waiting for a process restart.
        """
        try:
            from backend.auth_context import get_current_identity

            identity = get_current_identity()
            scope = identity.cache_key if identity else "system"
        except Exception:
            scope = "system"
        try:
            from backend.storage import get_repository

            repository = get_repository()
            self.project_name = repository.project_name() or self.project_name
            self._metadata = {
                str(item.get("stored_file") or item.get("name") or item.get("source_file")): dict(item)
                for item in repository.list_documents()
                if (item.get("stored_file") or item.get("name") or item.get("source_file"))
                and self._is_current_document(item)
            }
            self._rebuild_category_index()
            self._loaded = True
            self._loaded_scope = scope
        except Exception as exc:
            logger.warning("读取 SQLite 项目语境失败: %s", exc)
            self._metadata = {}
            self._rebuild_category_index()
            self._loaded = False
            self._loaded_scope = ""
            raise RuntimeError("项目知识数据暂时不可用") from exc

    @_synchronized
    def reload(self):
        self._ensure_loaded(force=True)

    def classify_document(self, filename: str, content: str) -> List[str]:
        """
        自动判断文档的知识类别（基于关键词匹配 + 规则）
        返回匹配的知识类别列表
        """
        content_lower = content.lower()
        matched_categories = []

        # ── 事业环境因素 ──
        eef_keywords = {
            KnowledgeCategory.EEF_CULTURE: ["组织文化", "组织架构", "汇报关系", "部门职责",
                                             "culture", "organization", "org chart"],
            KnowledgeCategory.EEF_STANDARD: ["标准", "规范", "法规", "合规", "iso", "标准规范",
                                              "standard", "regulation", "compliance"],
            KnowledgeCategory.EEF_MARKET: ["市场", "竞品", "行业趋势", "客户需求",
                                            "market", "competitive", "industry"],
            KnowledgeCategory.EEF_INFRASTRUCTURE: ["基础设施", "系统环境", "办公环境",
                                                    "infrastructure", "tools", "environment"],
        }

        # ── 组织过程资产 ──
        opa_keywords = {
            KnowledgeCategory.OPA_PROCESS: ["流程", "规程", "sop", "操作手册", "checklist",
                                            "process", "procedure", "workflow"],
            KnowledgeCategory.OPA_TEMPLATE: ["模板", "表单", "报告模板", "计划模板",
                                              "template", "form", "report format"],
            KnowledgeCategory.OPA_LESSONS: ["经验教训", "复盘", "总结", "回顾", "lessons learned",
                                            "retrospective", "postmortem"],
            KnowledgeCategory.OPA_TRAINING: ["培训", "教程", "指南", "学习", "training",
                                              "tutorial", "guide", "learning"],
            KnowledgeCategory.OPA_HISTORICAL: ["历史", "历史数据", "往期", "previous",
                                                "historical", "archive"],
        }

        # ── 项目自身资产 ──
        proj_keywords = {
            KnowledgeCategory.PROJ_PLAN: ["项目管理计划", "项目计划", "进度计划", "项目章程",
                                          "project plan", "project charter", "schedule"],
            KnowledgeCategory.PROJ_TAILORING: ["裁剪", "定制", "适配", "tailoring", "customize",
                                               "adaptation"],
            KnowledgeCategory.PROJ_DECISION: ["决策", "决议", "变更控制", "变更请求",
                                              "decision", "change request", "approval"],
            KnowledgeCategory.PROJ_RISK: ["风险", "问题", "issue", "risk", "mitigation",
                                          "应急", "应对措施"],
            KnowledgeCategory.PROJ_REPORT: ["报告", "周报", "月报", "状态", "milestone",
                                            "report", "weekly", "status", "进展"],
            KnowledgeCategory.PROJ_REQUIREMENT: ["需求", "需求规格", "功能需求", "非功能需求",
                                                  "requirement", "user story", "spec"],
            KnowledgeCategory.PROJ_ARCHITECTURE: ["架构", "设计", "技术方案", "系统设计",
                                                   "architecture", "design", "technical"],
            KnowledgeCategory.PROJ_COMMUNICATION: ["会议纪要", "沟通", "会议记录", "minutes",
                                                    "meeting notes", "communication"],
            KnowledgeCategory.PROJ_ACCEPTANCE: ["验收", "测试", "质量标准", "交付",
                                                 "acceptance", "qa", "test", "delivery"],
        }

        for cat_map in [eef_keywords, opa_keywords, proj_keywords]:
            for category, keywords in cat_map.items():
                for kw in keywords:
                    if kw.lower() in content_lower:
                        matched_categories.append(category.value)
                        break

        return matched_categories if matched_categories else ["未分类"]

    def extract_project_metadata(self, content: str) -> dict:
        """从文档内容中提取项目元数据"""
        metadata = {
            "project_name": self.project_name,
            "phase": "unknown",
            "stakeholders": [],
            "key_dates": [],
            "decisions": [],
        }

        # 提取项目阶段
        phase_patterns = [
            (r"启动阶段|启动", "initiation"),
            (r"规划阶段|规划", "planning"),
            (r"执行阶段|执行", "execution"),
            (r"监控阶段|监控", "monitoring"),
            (r"收尾阶段|收尾|验收", "closure"),
        ]
        for pattern, phase in phase_patterns:
            if re.search(pattern, content):
                metadata["phase"] = phase
                break

        # 提取干系人
        stakeholder_pattern = r"(?:干系人|相关方|stakeholder|参与人)[：:]\s*([^\n。]+)"
        for match in re.finditer(stakeholder_pattern, content, re.IGNORECASE):
            names = re.split(r"[、,，/]", match.group(1))
            metadata["stakeholders"].extend([n.strip() for n in names if n.strip()])

        # 提取关键日期
        date_pattern = r"(\d{4})[年/.-](\d{1,2})[月/.-](\d{1,2})[日]?"
        for match in re.finditer(date_pattern, content):
            metadata["key_dates"].append(f"{match.group(1)}-{match.group(2)}-{match.group(3)}")

        return metadata

    @_synchronized
    def register_document(self, doc_id: str, metadata: dict):
        """注册文档元数据"""
        self._ensure_loaded()
        previous = self._metadata.get(doc_id, {})
        self._metadata[doc_id] = {
            **previous,
            **metadata,
            "registered_at": previous.get("registered_at") or datetime.now().isoformat(),
        }
        from backend.storage import get_repository

        get_repository().upsert_document_metadata(doc_id, self._metadata[doc_id])
        self._rebuild_category_index()

    @_synchronized
    def sync_from_vector_metadata(self, metadata_items: List[dict]) -> int:
        """从持久化向量元数据恢复文档语境，兼容历史数据。"""
        self._ensure_loaded()
        grouped: Dict[str, dict] = {}
        for metadata in metadata_items or []:
            if not isinstance(metadata, dict):
                continue
            doc_id = str(metadata.get("stored_file") or metadata.get("source_file") or "").strip()
            if not doc_id:
                continue
            record = grouped.setdefault(doc_id, {"chunks": 0, "metadata": metadata})
            record["chunks"] += 1

        changed = 0
        for doc_id, item in grouped.items():
            metadata = item["metadata"]
            existing = self._metadata.get(doc_id, {})
            categories = metadata.get("categories") or ([metadata.get("category")] if metadata.get("category") else [])
            merged = {
                **existing,
                "source_file": metadata.get("source_file") or existing.get("source_file") or doc_id,
                "stored_file": metadata.get("stored_file") or existing.get("stored_file") or doc_id,
                "categories": categories or existing.get("categories", []),
                "category": metadata.get("category") or existing.get("category") or (categories[0] if categories else "未分类"),
                "uploader": metadata.get("uploader") or existing.get("uploader"),
                "role": metadata.get("role") or existing.get("role"),
                "handover_id": metadata.get("handover_id") or existing.get("handover_id"),
                "upload_date": metadata.get("upload_date") or existing.get("upload_date"),
                "organization_id": metadata.get("organization_id") or existing.get("organization_id"),
                "project_id": metadata.get("project_id") or existing.get("project_id"),
                "source_id": metadata.get("source_id") or existing.get("source_id"),
                "asset_id": metadata.get("asset_id") or existing.get("asset_id"),
                "version_id": metadata.get("version_id") or existing.get("version_id"),
                "is_current_version": metadata.get("is_current_version", existing.get("is_current_version", True)),
                "status": metadata.get("status") or existing.get("status"),
                "asset_status": metadata.get("asset_status") or existing.get("asset_status"),
                "version_status": metadata.get("version_status") or existing.get("version_status"),
                "valid_from": metadata.get("valid_from") or existing.get("valid_from"),
                "valid_until": metadata.get("valid_until") or existing.get("valid_until"),
                "review_due_at": metadata.get("review_due_at") or existing.get("review_due_at"),
                "authority_level": metadata.get("authority_level") or existing.get("authority_level"),
                "authority_score": metadata.get("authority_score", existing.get("authority_score")),
                "source_refs": metadata.get("source_refs") or existing.get("source_refs"),
                "applicable_roles": metadata.get("applicable_roles") or existing.get("applicable_roles"),
                "topics": metadata.get("topics") or existing.get("topics"),
                "chunks": item["chunks"],
                "registered_at": existing.get("registered_at") or datetime.now().isoformat(),
            }
            if existing != merged:
                self._metadata[doc_id] = merged
                from backend.storage import get_repository

                get_repository().upsert_document_metadata(doc_id, merged)
                changed += 1

        if changed:
            self._rebuild_category_index()
        return changed

    @_synchronized
    def unregister_document(self, filename: str) -> bool:
        """移除一个明确版本；来源有多个版本时拒绝模糊删除。"""
        self._ensure_loaded()
        from backend.storage import get_repository

        database_deleted = get_repository().delete_document(filename)
        self.reload()
        return bool(database_deleted)

    @_synchronized
    def list_documents(self) -> List[Dict[str, Any]]:
        """返回所有已注册的文档元数据列表"""
        self._ensure_loaded()
        docs = []
        for doc_id, meta in self._metadata.items():
            doc = {
                "name": doc_id,
                "stored_file": meta.get("stored_file", doc_id),
                "document_id": meta.get("document_id", ""),
                "source_id": meta.get("source_id", ""),
                "asset_id": meta.get("asset_id", ""),
                "version_id": meta.get("version_id", ""),
                "source_file": meta.get("source_file", doc_id),
                "categories": meta.get("categories", []),
                "category": meta.get("category") or (meta.get("categories", [])[:1] or ["未分类"])[0],
                "uploader": meta.get("uploader"),
                "role": meta.get("role"),
                "handover_id": meta.get("handover_id"),
                "status": meta.get("status", "active"),
                "asset_status": meta.get("asset_status", ""),
                "version_status": meta.get("version_status", ""),
                "is_current_version": meta.get("is_current_version", True),
                "valid_from": meta.get("valid_from", ""),
                "valid_until": meta.get("valid_until", ""),
                "review_due_at": meta.get("review_due_at", ""),
                "authority_level": meta.get("authority_level", ""),
                "authority_score": meta.get("authority_score"),
                "source_refs": meta.get("source_refs") or [],
                "applicable_roles": meta.get("applicable_roles") or [],
                "topics": meta.get("topics") or [],
                "upload_date": meta.get("registered_at", ""),
                "chunks": meta.get("chunks", 0),
            }
            docs.append(doc)
        return docs

    @_synchronized
    def get_full_context(self) -> Dict[str, Any]:
        """返回可用于接口与仪表盘展示的真实项目语境。"""
        self._ensure_loaded()
        category_counts = {}
        for category, doc_ids in self._categories.items():
            if doc_ids:
                category_counts[category] = len(doc_ids)
        return {
            "project_name": self.project_name,
            "document_count": len(self._metadata),
            "category_counts": category_counts,
            "documents": self.list_documents(),
        }

    @_synchronized
    def get_context_prompt(self, query: str, categories: List[str] = None) -> str:
        """
        生成项目语境提示，增强 LLM 对项目上下文的理解
        用于注入到 RAG 提示词中
        """
        self._ensure_loaded()
        parts = [f"📋 当前项目: {self.project_name}"]

        if categories:
            cat_names = [KnowledgeCategory.get_group(c) for c in categories]
            parts.append(f"📂 知识范围: {' → '.join(set(cat_names))}")

        # 添加各类文档统计
        total_docs = len(self._metadata)
        parts.append(f"📊 知识库规模: {total_docs} 篇文档")

        # 按类别统计
        cat_counts = {}
        for cat, doc_ids in self._categories.items():
            if doc_ids:
                group = KnowledgeCategory.get_group(cat)
                cat_counts[group] = cat_counts.get(group, 0) + len(doc_ids)

        for group, count in cat_counts.items():
            parts.append(f"  · {group}: {count} 个知识块")

        parts.append(f"\n📌 用户当前关注: {query}")
        parts.append("请基于以上项目语境，结合检索到的知识内容进行回答。")

        return "\n".join(parts)


# 全局单例
project_context = ProjectContext()
