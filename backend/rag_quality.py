"""Deterministic RAG evidence, citation, evaluation and feedback contracts.

The model may phrase an answer, but SQLite lifecycle/ACL state decides whether
evidence and citations are usable. Offline evaluation deliberately measures the
retrieval evidence before generation so strategy comparisons are reproducible
and do not depend on a live model supplier.
"""
from __future__ import annotations

import re
import unicodedata
import uuid
from typing import Any, Dict, Iterable, List, Optional

from backend.config import settings
from backend.knowledge_base.retriever import retriever
from backend.storage import get_repository


NO_ANSWER_TEXT = (
    "当前知识不足，暂时无法基于你有权访问的项目资料可靠回答。"
    "你可以提交“无帮助”反馈，系统会创建知识补充任务。"
)
SCENARIOS = {"fact", "process", "decision", "risk", "handover", "no_answer"}
FEEDBACK_REASONS = {
    "citation_error", "knowledge_outdated", "incomplete", "permission_blocked", "other",
}
REASON_LABELS = {
    "citation_error": "引用错误",
    "knowledge_outdated": "知识过期",
    "incomplete": "答案不完整",
    "permission_blocked": "无权限误拦截",
    "other": "其他问题",
}


def _strings(value: Any, *, limit: int = 50) -> List[str]:
    if not isinstance(value, list):
        return []
    result: List[str] = []
    for item in value[:limit]:
        text = str(item or "").strip()
        if text and text not in result:
            result.append(text[:500])
    return result


def _tokenize(text: str) -> set[str]:
    text = unicodedata.normalize("NFKC", str(text or ""))
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9_\-]{2,}|[\u4e00-\u9fff]{2,}", text)
        if token.strip()
    }


class RAGQualityService:
    def __init__(self, repository_provider=get_repository, retriever_instance=retriever) -> None:
        self.repository_provider = repository_provider
        self.retriever = retriever_instance

    @property
    def repository(self):
        return self.repository_provider()

    @staticmethod
    def actor(identity: Any) -> str:
        return str(getattr(identity, "user_id", "") or "user_system")

    @staticmethod
    def answer_id() -> str:
        return f"raga_{uuid.uuid4().hex}"

    def citations_for_documents(self, documents: Iterable[Any], identity: Any) -> List[Dict[str, Any]]:
        citations = []
        for index, document in enumerate(documents, 1):
            metadata = dict(getattr(document, "metadata", {}) or {})
            citations.append({
                "citation_id": f"C{index}",
                "source": str(metadata.get("source_file") or "").strip(),
                "content": str(getattr(document, "page_content", "") or "")[:240],
                "asset_id": str(metadata.get("asset_id") or "").strip(),
                "version_id": str(metadata.get("version_id") or "").strip(),
                "document_id": str(metadata.get("document_id") or "").strip(),
                "chunk_id": str(metadata.get("chunk_id") or "").strip(),
                "page": metadata.get("page"),
                "section": str(metadata.get("section") or "").strip(),
            })
        return self.validate_citations(citations, identity)

    def validate_citations(self, citations: Iterable[Dict[str, Any]], identity: Any) -> List[Dict[str, Any]]:
        items = [dict(item or {}) for item in citations]
        chunk_ids = [str(item.get("chunk_id") or "") for item in items if item.get("chunk_id")]
        active_chunks = self.repository.active_chunk_ids(chunk_ids)
        allowed_documents = self.repository.accessible_document_ids(identity)
        documents = {
            str(item.get("document_id") or ""): item
            for item in self.repository.list_documents()
            if item.get("document_id")
        }
        validated = []
        for item in items:
            document_id = str(item.get("document_id") or "")
            chunk_id = str(item.get("chunk_id") or "")
            source = str(item.get("source") or "").strip()
            document = documents.get(document_id)
            reasons = []
            if not source:
                reasons.append("missing_source")
            if not document_id or document_id not in allowed_documents or not document:
                reasons.append("inaccessible_document")
            if not chunk_id or chunk_id not in active_chunks:
                reasons.append("inactive_chunk")
            if document:
                asset_id = str(item.get("asset_id") or "")
                version_id = str(item.get("version_id") or "")
                if asset_id and asset_id != str(document.get("asset_id") or ""):
                    reasons.append("asset_mismatch")
                if version_id and version_id != str(document.get("version_id") or ""):
                    reasons.append("version_mismatch")
            item["valid"] = not reasons
            item["invalid_reasons"] = reasons
            validated.append(item)
        return validated

    def retrieve_evidence(self, question: str, identity: Any, *, k: int = 4) -> Dict[str, Any]:
        documents = self.retriever.retrieve(question, k=k)
        citations = self.citations_for_documents(documents, identity)
        usable = [item for item in citations if item.get("valid")]
        sufficient = bool(documents and usable and len(usable) == len(citations))
        return {
            "documents": documents if sufficient else [],
            "citations": citations,
            "sufficient": sufficient,
            "no_answer": not sufficient,
            "citation_validity_rate": round(len(usable) / len(citations), 4) if citations else 0.0,
        }

    def create_evaluation_set(self, payload: Dict[str, Any], identity: Any) -> Dict[str, Any]:
        name = str(payload.get("name") or "").strip()
        if not name:
            raise ValueError("评测集名称不能为空")
        version = int(payload.get("version") or 1)
        if version < 1:
            raise ValueError("评测集版本必须大于 0")
        status = str(payload.get("status") or "draft")
        if status not in {"draft", "active", "archived"}:
            raise ValueError("不支持的评测集状态")
        return self.repository.create_rag_evaluation_set({
            **payload,
            "name": name,
            "version": version,
            "status": status,
            "min_case_target": max(1, min(int(payload.get("min_case_target") or 30), 500)),
            "created_by": self.actor(identity),
        })

    def list_evaluation_sets(self) -> Dict[str, Any]:
        sets = self.repository.list_rag_evaluation_sets()
        return {
            "evaluation_sets": sets,
            "summary": {
                "total": len(sets),
                "ready": sum(bool(item.get("ready")) for item in sets),
                "cases": sum(int(item.get("case_count") or 0) for item in sets),
                "runs": sum(int(item.get("run_count") or 0) for item in sets),
            },
        }

    def get_evaluation_set(self, evaluation_set_id: str) -> Optional[Dict[str, Any]]:
        return self.repository.get_rag_evaluation_set(evaluation_set_id)

    def add_evaluation_case(
        self, evaluation_set_id: str, payload: Dict[str, Any], identity: Any,
    ) -> Dict[str, Any]:
        question = str(payload.get("question") or "").strip()
        if not question:
            raise ValueError("评测问题不能为空")
        scenario = str(payload.get("scenario") or "fact")
        if scenario not in SCENARIOS:
            raise ValueError("不支持的评测场景")
        expect_no_answer = bool(payload.get("expect_no_answer")) or scenario == "no_answer"
        expected_points = _strings(payload.get("expected_points"))
        if not expect_no_answer and not expected_points:
            raise ValueError("非无答案案例至少需要一个期望答案要点")
        return self.repository.add_rag_evaluation_case(evaluation_set_id, {
            **payload,
            "question": question,
            "scenario": "no_answer" if expect_no_answer else scenario,
            "expect_no_answer": expect_no_answer,
            "expected_points": expected_points,
            "allowed_asset_ids": _strings(payload.get("allowed_asset_ids")),
            "allowed_source_files": _strings(payload.get("allowed_source_files")),
            "created_by": self.actor(identity),
        })

    @staticmethod
    def _point_matches(point: str, evidence_text: str) -> bool:
        point_text = unicodedata.normalize("NFKC", str(point or "")).strip().lower()
        evidence_text = unicodedata.normalize("NFKC", str(evidence_text or "")).lower()
        if not point_text:
            return False
        if point_text in evidence_text:
            return True
        tokens = _tokenize(point_text)
        if not tokens:
            return False
        evidence_tokens = _tokenize(evidence_text)
        return len(tokens & evidence_tokens) / len(tokens) >= 0.6

    def run_evaluation(self, evaluation_set_id: str, identity: Any) -> Dict[str, Any]:
        evaluation_set = self.get_evaluation_set(evaluation_set_id)
        if not evaluation_set:
            raise ValueError("评测集不存在")
        cases = list(evaluation_set.get("cases") or [])
        if not cases:
            raise ValueError("评测集还没有问题")
        results = []
        for case in cases:
            evidence = self.retrieve_evidence(str(case.get("question") or ""), identity)
            citations = list(evidence.get("citations") or [])
            no_answer = bool(evidence.get("no_answer"))
            expected_no_answer = bool(case.get("expect_no_answer"))
            allowed_assets = set(_strings(case.get("allowed_asset_ids")))
            allowed_sources = set(_strings(case.get("allowed_source_files")))
            if not citations:
                allowed_precision = 1.0 if expected_no_answer else 0.0
            elif not allowed_assets and not allowed_sources:
                allowed_precision = 1.0
            else:
                allowed_count = sum(
                    (str(item.get("asset_id") or "") in allowed_assets)
                    or (str(item.get("source") or "") in allowed_sources)
                    for item in citations
                )
                allowed_precision = allowed_count / len(citations)
            evidence_text = "\n".join(
                str(getattr(document, "page_content", "") or "").lower()
                for document in evidence.get("documents") or []
            )
            points = _strings(case.get("expected_points"))
            matched_points = [point for point in points if self._point_matches(point, evidence_text)]
            coverage = len(matched_points) / len(points) if points else (1.0 if expected_no_answer else 0.0)
            validity = float(evidence.get("citation_validity_rate") or 0)
            if expected_no_answer:
                passed = no_answer
            else:
                passed = not no_answer and validity >= 0.95 and allowed_precision >= 0.95 and coverage >= 0.6
            results.append({
                "evaluation_case_id": case.get("evaluation_case_id"),
                "no_answer": no_answer,
                "citation_validity_rate": round(validity, 4),
                "allowed_source_precision": round(allowed_precision, 4),
                "expected_evidence_coverage": round(coverage, 4),
                "passed": passed,
                "citations": citations,
                "matched_points": matched_points,
                "detail": "无答案判断符合预期" if expected_no_answer and passed else (
                    "检索证据达到门槛" if passed else "检索证据未达到评测门槛"
                ),
            })
        total = len(results)
        citations = [item for result in results for item in result.get("citations") or []]
        valid_citations = sum(bool(item.get("valid")) for item in citations)
        permission_leaks = sum(
            "inaccessible_document" in set(item.get("invalid_reasons") or [])
            for item in citations
        )
        metrics = {
            "case_count": total,
            "passed_count": sum(bool(item.get("passed")) for item in results),
            "pass_rate": round(sum(bool(item.get("passed")) for item in results) / total, 4),
            "citation_count": len(citations),
            "citation_validity_rate": round(valid_citations / len(citations), 4) if citations else 1.0,
            "permission_leak_count": permission_leaks,
            "no_answer_accuracy": round(sum(
                bool(item.get("no_answer")) == bool(case.get("expect_no_answer"))
                for item, case in zip(results, cases)
            ) / total, 4),
            "ready_case_target_met": bool(evaluation_set.get("ready")),
        }
        strategy = {
            "evaluation_contract": "retrieval_evidence_v1",
            "text_normalization": "NFKC",
            "retriever_k": 4,
            "score_threshold": settings.RETRIEVER_SCORE_THRESHOLD,
            "embedding_model": settings.LOCAL_EMBEDDING_MODEL,
        }
        return self.repository.save_rag_evaluation_run(
            evaluation_set_id, strategy=strategy, metrics=metrics, results=results,
            created_by=self.actor(identity),
        )

    def record_feedback(self, payload: Dict[str, Any], identity: Any, knowledge_tasks: Any) -> Dict[str, Any]:
        answer_id = str(payload.get("answer_id") or "").strip()
        if not answer_id:
            raise ValueError("回答标识不能为空")
        helpful = payload.get("helpful")
        if not isinstance(helpful, bool):
            raise ValueError("请选择回答是否有帮助")
        reason = str(payload.get("reason") or "").strip()
        if not helpful and reason not in FEEDBACK_REASONS:
            raise ValueError("无帮助反馈需要选择原因")
        if helpful:
            reason = ""
        feedback = self.repository.record_rag_answer_feedback({
            "answer_id": answer_id,
            "user_id": self.actor(identity),
            "question": str(payload.get("question") or "").strip(),
            "helpful": helpful,
            "reason": reason,
            "note": str(payload.get("note") or "").strip(),
            "citations": list(payload.get("citations") or [])[:20],
        })
        task = None
        if not helpful:
            linked_asset_id = next((
                str(item.get("asset_id") or "")
                for item in feedback.get("citations") or [] if item.get("asset_id")
            ), "")
            task = knowledge_tasks.create({
                "task_type": "qa_feedback",
                "title": f"改进问答：{REASON_LABELS.get(reason, '回答质量问题')}",
                "description": (
                    f"用户反馈回答“{answer_id}”存在{REASON_LABELS.get(reason, '质量问题')}。"
                    "请核对引用、知识时效、答案覆盖范围和来源权限。"
                ),
                "source_type": "rag_feedback",
                "source_key": str(feedback.get("feedback_id") or ""),
                "dedupe_key": f"rag_feedback:{feedback.get('feedback_id')}",
                "linked_asset_id": linked_asset_id,
                "priority": "high" if reason in {"citation_error", "knowledge_outdated", "permission_blocked"} else "medium",
                "metadata": {
                    "answer_id": answer_id,
                    "feedback_reason": reason,
                    "citation_ids": [item.get("citation_id") for item in feedback.get("citations") or []],
                },
            }, identity)
            self.repository.link_rag_feedback_task(
                str(feedback.get("feedback_id") or ""), str(task.get("task_id") or ""),
            )
            feedback["knowledge_task_id"] = str(task.get("task_id") or "")
        return {"feedback": feedback, "knowledge_task": task}


rag_quality_service = RAGQualityService()
