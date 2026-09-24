"""Deterministic role knowledge-readiness calculation.

The caller supplies assets that have already passed project and source access
checks. This module rechecks lifecycle eligibility and calculates explainable
scores without asking an LLM to make authorization or quality decisions.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional


DIMENSION_WEIGHTS = {
    "topic_coverage": 0.50,
    "authority": 0.20,
    "freshness": 0.20,
    "traceability": 0.10,
}

ACTIVE_STATUSES = {"active", "published", "effective", "review_due"}
AUTHORITY_LEVELS = {
    "official": 1.0,
    "authoritative": 1.0,
    "high": 0.9,
    "reviewed": 0.8,
    "medium": 0.65,
    "standard": 0.6,
    "internal": 0.6,
    "low": 0.35,
    "unknown": 0.4,
    "官方": 1.0,
    "权威": 1.0,
    "高": 0.9,
    "已审核": 0.8,
    "中": 0.65,
    "普通": 0.6,
    "低": 0.35,
    "未知": 0.4,
}


def _score(value: Any, fallback: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    if number > 1:
        number /= 100.0
    return max(0.0, min(1.0, number))


def _time(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _values(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in re.split(r"[,，;；|]", value) if item.strip()]
    if isinstance(value, Iterable) and not isinstance(value, (bytes, Mapping)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _metadata(asset: Mapping[str, Any]) -> Mapping[str, Any]:
    value = asset.get("metadata")
    return value if isinstance(value, Mapping) else {}


def _field(asset: Mapping[str, Any], *names: str) -> Any:
    """Read a normalized field from the asset row or its metadata payload."""
    for source in (asset, _metadata(asset)):
        for name in names:
            value = source.get(name)
            if value is not None and value != "":
                return value
    return None


def _role_values(value: Any) -> List[str]:
    """Accept both legacy role strings and SQLite role assignment objects."""
    if isinstance(value, Mapping):
        role = value.get("role") or value.get("role_key") or value.get("name") or value.get("id")
        return _values(role)
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        result: List[str] = []
        for item in value:
            result.extend(_role_values(item))
        return result
    return _values(value)


def _key(value: Any) -> str:
    return re.sub(r"[^a-z0-9_\u4e00-\u9fff]", "", str(value or "").strip().lower())


def _asset_is_eligible(asset: Mapping[str, Any], now: datetime) -> bool:
    if _field(asset, "is_current_version") is False:
        return False
    current_version_id = str(_field(asset, "current_version_id") or "").strip()
    supplied_version_id = str(_field(asset, "version_id") or "").strip()
    if current_version_id and supplied_version_id and current_version_id != supplied_version_id:
        return False
    for field in ("status", "asset_status", "version_status"):
        status = str(_field(asset, field) or "").strip().lower()
        if status and status not in ACTIVE_STATUSES:
            return False
    projection_status = str(_field(asset, "projection_status") or "").strip().lower()
    if projection_status in {"failed", "repair_required", "pending_refresh"}:
        return False
    valid_from = _time(_field(asset, "valid_from"))
    valid_until = _time(_field(asset, "valid_until"))
    return not ((valid_from and valid_from > now) or (valid_until and valid_until <= now))


def _role_applies(asset: Mapping[str, Any], role: str) -> bool:
    roles = {_key(item) for item in _role_values(_field(asset, "applicable_roles"))}
    return not roles or _key(role) in roles


def _asset_topic_keys(asset: Mapping[str, Any]) -> set[str]:
    values: List[str] = []
    for field in ("topics", "categories", "tags"):
        values.extend(_values(_field(asset, field)))
    for field in ("topic", "category", "primary_category"):
        values.extend(_values(_field(asset, field)))
    return {_key(item) for item in values if _key(item)}


def _authority(asset: Mapping[str, Any]) -> float:
    authority_score = _field(asset, "authority_score")
    if authority_score is not None:
        return _score(authority_score, 0.0)
    raw_level = _field(asset, "authority_level")
    if isinstance(raw_level, (int, float)) or str(raw_level or "").strip().replace(".", "", 1).isdigit():
        return _score(raw_level, 0.0)
    level = str(raw_level or "unknown").strip().lower()
    return AUTHORITY_LEVELS.get(level, AUTHORITY_LEVELS["unknown"])


def _freshness(asset: Mapping[str, Any], now: datetime) -> float:
    freshness_score = _field(asset, "freshness_score")
    if freshness_score is not None:
        return _score(freshness_score, 0.0)
    review_due = _time(_field(asset, "review_due_at"))
    status = str(_field(asset, "asset_status", "status") or "").strip().lower()
    if status == "review_due" or (review_due and review_due <= now):
        return 0.4
    if review_due:
        return 1.0
    # Missing review metadata is usable, but cannot receive full freshness credit.
    return 0.6


def _traceability(asset: Mapping[str, Any]) -> float:
    traceability_score = _field(asset, "traceability_score")
    if traceability_score is not None:
        return _score(traceability_score, 0.0)
    references = _values(_field(asset, "source_refs"))
    references += _values(_field(asset, "feishu_message_ids"))
    has_source_id = bool(str(_field(asset, "source_id", "primary_source_id") or "").strip())
    has_location = bool(
        references
        or str(_field(asset, "source_file", "external_key") or "").strip()
        or str(_field(asset, "page", "section") or "").strip()
    )
    if has_source_id and has_location:
        return 1.0
    if has_source_id or has_location:
        return 0.7
    return 0.0


def _template_topics(role_template: Mapping[str, Any]) -> List[Dict[str, Any]]:
    raw_topics = role_template.get("topics") or role_template.get("categories") or []
    topics = []
    for index, item in enumerate(raw_topics):
        if isinstance(item, Mapping):
            topic_id = str(item.get("topic_id") or item.get("id") or item.get("category") or item.get("name") or "").strip()
            label = str(item.get("label") or item.get("name") or topic_id).strip()
            weight = max(0.0, float(item.get("weight") or 1.0))
        else:
            topic_id = str(item).strip()
            label = topic_id
            weight = 1.0
        if topic_id:
            topics.append({"topic_id": topic_id, "label": label, "weight": weight, "index": index})
    return topics


def calculate_role_readiness(
    role_template: Mapping[str, Any],
    assets: Iterable[Mapping[str, Any]],
    *,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Calculate role readiness from authorized current assets with evidence."""
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    role = str(role_template.get("role") or role_template.get("name") or "当前岗位")
    topics = _template_topics(role_template)
    supplied_assets = [dict(asset) for asset in assets if isinstance(asset, Mapping)]
    eligible_assets = [
        asset for asset in supplied_assets
        if _asset_is_eligible(asset, current)
        and _role_applies(asset, role)
    ]
    total_weight = sum(item["weight"] for item in topics)
    if not topics or total_weight <= 0:
        return {
            "role": role,
            "readiness": 0.0,
            "formula": "主题覆盖 50% + 权威性 20% + 新鲜度 20% + 可追溯性 10%",
            "dimensions": {
                key: {"score": 0.0, "weight": round(weight * 100), "contribution": 0.0}
                for key, weight in DIMENSION_WEIGHTS.items()
            },
            "topics": [],
            "gaps": [],
            "input_asset_count": len(supplied_assets),
            "eligible_asset_count": len(eligible_assets),
            "excluded_asset_count": len(supplied_assets) - len(eligible_assets),
            "calculated_at": current.isoformat(),
        }

    topic_results = []
    weighted = {key: 0.0 for key in DIMENSION_WEIGHTS}
    for topic in topics:
        topic_key = _key(topic["topic_id"])
        matched = [asset for asset in eligible_assets if topic_key in _asset_topic_keys(asset)]
        scores = {
            "topic_coverage": 1.0 if matched else 0.0,
            "authority": max((_authority(asset) for asset in matched), default=0.0),
            "freshness": max((_freshness(asset, current) for asset in matched), default=0.0),
            "traceability": max((_traceability(asset) for asset in matched), default=0.0),
        }
        for dimension, score in scores.items():
            weighted[dimension] += topic["weight"] * score
        evidence = [
            {
                "asset_id": str(asset.get("asset_id") or ""),
                "version_id": str(_field(asset, "version_id", "current_version_id") or ""),
                "source_id": str(_field(asset, "source_id", "primary_source_id") or ""),
                "title": str(_field(asset, "title", "source_file") or "未命名知识"),
                "authority": round(_authority(asset) * 100, 1),
                "freshness": round(_freshness(asset, current) * 100, 1),
                "traceability": round(_traceability(asset) * 100, 1),
            }
            for asset in matched
        ]
        topic_results.append({
            "topic_id": topic["topic_id"],
            "label": topic["label"],
            "weight": topic["weight"],
            "covered": bool(matched),
            "asset_count": len(matched),
            "scores": {key: round(value * 100, 1) for key, value in scores.items()},
            "evidence": evidence,
        })

    dimensions = {}
    readiness = 0.0
    for dimension, formula_weight in DIMENSION_WEIGHTS.items():
        score = weighted[dimension] / total_weight
        contribution = score * formula_weight
        readiness += contribution
        dimensions[dimension] = {
            "score": round(score * 100, 1),
            "weight": round(formula_weight * 100),
            "contribution": round(contribution * 100, 1),
        }

    return {
        "role": role,
        "readiness": round(readiness * 100, 1),
        "formula": "主题覆盖 50% + 权威性 20% + 新鲜度 20% + 可追溯性 10%",
        "dimensions": dimensions,
        "topics": topic_results,
        "gaps": [
            {"topic_id": item["topic_id"], "label": item["label"], "weight": item["weight"]}
            for item in topic_results if not item["covered"]
        ],
        "input_asset_count": len(supplied_assets),
        "eligible_asset_count": len(eligible_assets),
        "excluded_asset_count": len(supplied_assets) - len(eligible_assets),
        "calculated_at": current.isoformat(),
    }
