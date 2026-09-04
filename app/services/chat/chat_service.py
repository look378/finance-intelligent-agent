"""
Chat orchestration service.

Delegates dialogue management to a compiled LangGraph, while retaining
optional backward-compatible services (LLM, memory, guardrails) for
history management and fallback use.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncGenerator, List, Optional

from app.services.guardrails.output_guard import COMPLIANCE_BLOCK_MESSAGE as COMPLIANCE_FALLBACK


@dataclass
class ChatResponse:
    """
    Chat response data.

    Attributes:
        content: Response text content
        session_id: Session identifier
        intent: Detected intent
        sources: Optional list of source document IDs
        metadata: Optional response metadata
    """

    content: str
    session_id: int
    intent: str
    sources: Optional[List[str]] = None
    metadata: Optional[dict] = None


@dataclass
class ChatMessage:
    """
    Chat message data.

    Attributes:
        role: Message role (user/assistant/system)
        content: Message content
        timestamp: Optional timestamp
    """

    role: str
    content: str
    timestamp: Optional[str] = None


class ChatService:
    """
    Chat orchestration service backed by a LangGraph dialogue graph.

    The graph handles intent detection, slot filling, retrieval routing,
    guardrails, and LLM generation internally. This service is a thin
    adapter that invokes the graph and translates results into
    ``ChatResponse`` / ``ChatMessage`` dataclasses.
    """

    def __init__(
        self,
        graph,  # Compiled LangGraph
        llm_service=None,  # backward compat
        memory_strategy=None,  # backward compat
        guardrail_service=None,  # backward compat
    ) -> None:
        self.graph = graph
        self.llm_service = llm_service
        self.memory_strategy = memory_strategy
        self.guardrail_service = guardrail_service

    async def process_message(
        self,
        session_id: int,
        message: str,
        user_id: int,
        max_tokens: Optional[int] = None,
    ) -> ChatResponse:
        """
        Process a user message through the LangGraph dialogue graph.

        Args:
            session_id: Session identifier
            message: User message
            user_id: User identifier
            max_tokens: Optional max tokens (reserved, not yet forwarded)

        Returns:
            ChatResponse with graph output
        """
        config = {"configurable": {"thread_id": str(session_id)}}
        # 每轮重置瞬态字段，避免上一轮的 tool_result / retrieved_docs / route
        # 等状态残留污染本轮回答（跨轮状态污染修复）。
        input_state = {
            "message": message,
            "session_id": session_id,
            "user_id": user_id,
            # 瞬态键重置为空
            "tool_result": None,
            "retrieved_docs": [],
            "sources": [],
            "slot_prompt": "",
            "response": "",
            "confidence": 0.0,
            "route": "",
            "blocked": False,
            "blocked_reason": "",
        }
        result = await self.graph.ainvoke(input_state, config)
        response_text = result.get("response", "")
        audit_events = list(result.get("audit_events") or [])

        # Apply output guardrail (financial compliance: return-promise detection,
        # risk disclaimers, PII redaction) to every generated response.
        if self.guardrail_service is not None:
            output_result = self.guardrail_service.check_output(response_text)
            if output_result.was_blocked:
                response_text = output_result.sanitized_content or COMPLIANCE_FALLBACK
                audit_record = self._record_guardrail_audit(
                    event="guardrail",
                    action="block_output",
                    target=", ".join(output_result.violations),
                    outcome="blocked",
                    session_id=session_id,
                )
                if audit_record:
                    audit_events.append(audit_record)
            else:
                response_text = output_result.safe_content
                for violation in output_result.violations:
                    if violation.startswith("pii_") or violation == "risk_disclaimer_appended":
                        audit_record = self._record_guardrail_audit(
                            event="guardrail",
                            action=violation,
                            target=violation,
                            outcome="redacted" if violation.startswith("pii_") else "appended",
                            session_id=session_id,
                        )
                        if audit_record:
                            audit_events.append(audit_record)

        return ChatResponse(
            content=response_text,
            session_id=session_id,
            intent=result.get("intent", "unknown"),
            sources=result.get("sources"),
            metadata={
                "confidence": result.get("confidence"),
                "pending_slots": result.get("pending_slots", []),
                "filled_slots": result.get("filled_slots", {}),
                "audit_events": audit_events,
            },
        )

    @staticmethod
    def _record_guardrail_audit(
        event: str,
        action: str,
        target: str,
        outcome: str,
        session_id: Optional[int],
    ) -> Optional[dict]:
        """Emit an audit record for guardrail events via the audit logger."""
        try:
            from app.services.observability.audit_log import log_audit

            return log_audit(
                event=event,
                action=action,
                target=target,
                outcome=outcome,
                session_id=session_id,
            )
        except Exception:
            return None

    async def process_message_stream(
        self,
        session_id: int,
        message: str,
        user_id: int,
    ) -> AsyncGenerator[str, None]:
        """
        Process a user message with streaming via LangGraph.

        Yields response chunks from the ``generate_response`` or
        ``direct_response`` graph nodes.

        Args:
            session_id: Session identifier
            message: User message
            user_id: User identifier

        Yields:
            str: Response text chunks
        """
        config = {"configurable": {"thread_id": str(session_id)}}
        async for event in self.graph.astream(
            {"message": message, "session_id": session_id, "user_id": user_id},
            config,
        ):
            for node_name, node_state in event.items():
                if node_name == "generate_response" and "response" in node_state:
                    yield node_state["response"]
                elif node_name == "direct_response" and "response" in node_state:
                    yield node_state["response"]

    async def get_chat_history(
        self,
        session_id: int,
        limit: int = 50,
    ) -> List[ChatMessage]:
        """
        Get chat history for a session.

        Args:
            session_id: Session identifier
            limit: Maximum number of messages

        Returns:
            List[ChatMessage]: List of chat messages
        """
        if not self.memory_strategy:
            return []
        context = await self.memory_strategy.get_context(session_id=session_id)
        return [
            ChatMessage(role=msg.role, content=msg.content)
            for msg in context[:limit]
        ]

    async def clear_chat_history(self, session_id: int) -> None:
        """
        Clear chat history for a session.

        Args:
            session_id: Session identifier
        """
        if self.memory_strategy:
            await self.memory_strategy.clear_session(session_id=session_id)
