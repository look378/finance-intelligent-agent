"""
Structured audit logging for financial customer service.

Records tool executions, guardrail events, and risk-assessment results
with a consistent schema so every sensitive operation is traceable
(who, when, what, outcome).  Written via stdlib logging under the
``audit`` logger with a JSON-friendly extra payload.
"""
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger("audit")

_audit_enabled = True


def set_audit_enabled(enabled: bool) -> None:
    """Globally enable/disable audit logging (driven by settings)."""
    global _audit_enabled
    _audit_enabled = enabled


def log_audit(
    event: str,
    actor: Optional[Any] = None,
    action: Optional[str] = None,
    target: Optional[str] = None,
    outcome: str = "success",
    metadata: Optional[Dict[str, Any]] = None,
    session_id: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Emit one audit record and return it for state persistence.

    Args:
        event: Event category, e.g. "tool_execution", "guardrail", "risk_assessment"
        actor: Who triggered it (user_id / session identity)
        action: Specific action, e.g. "run_risk_assessment", "block_return_promise"
        target: What it acted on, e.g. tool intent / pii type
        outcome: success / failure / blocked / redacted
        metadata: Additional structured data
        session_id: Chat session id for correlation
    """
    if not _audit_enabled:
        return {}

    record: Dict[str, Any] = {
        "event": event,
        "action": action or "",
        "actor": actor,
        "target": target,
        "outcome": outcome,
        "session_id": session_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
    }

    logger.info(
        "AUDIT event=%s action=%s actor=%s target=%s outcome=%s session=%s",
        record["event"],
        record["action"],
        record["actor"],
        record["target"],
        record["outcome"],
        record["session_id"],
        extra={"audit": record},
    )

    return record


def audit_payload(record: Dict[str, Any]) -> str:
    """Serialize an audit record for storage / display."""
    try:
        return json.dumps(record, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(record)
