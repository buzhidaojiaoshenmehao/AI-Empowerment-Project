"""Shared, deterministic eligibility for reviewing a handover checklist item."""
from typing import Any, Dict


def item_review_policy(
    record: Dict[str, Any], item: Dict[str, Any], *, actor: str, can_manage: bool,
) -> Dict[str, Any]:
    recipient = str(record.get("recipient_user_id") or "")
    waiting = f"等待{record.get('recipient') or '接替人'}验收"
    result = {"review_actions": [], "review_action_mode": "", "review_block_reason": ""}
    if record.get("status") == "completed" or item.get("status") != "submitted":
        return result
    if not actor:
        return {**result, "review_block_reason": waiting}
    excluded = {
        str(record.get("departing_user_id") or ""),
        str(item.get("submitted_by") or ""),
    }
    if actor in excluded:
        return {**result, "review_block_reason": f"{waiting}。提交人、离职人员不能验收自己交出的资料"}
    if actor == recipient:
        return {**result, "review_actions": ["accept", "reject"], "review_action_mode": "recipient"}
    if can_manage:
        return {**result, "review_actions": ["accept", "reject"], "review_action_mode": "proxy"}
    return {**result, "review_block_reason": waiting}
