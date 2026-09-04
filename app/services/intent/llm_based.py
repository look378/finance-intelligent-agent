"""
LLM-based intent detector for business intents.

Uses language model understanding to classify customer service queries
into business intent categories.
"""
import json
import re
from typing import Optional

from app.services.intent.base import IntentDetector, IntentResult
from app.models.enums.intent import Intent
from app.services.llm.base import LLMServiceBase, LLMMessage
from app.core.exceptions import ExternalServiceError


class LLMIntentDetector(IntentDetector):
    """
    LLM-based intent detector for wealth-management business intents.

    Classifies queries into task-oriented (wealth_query, fund_query, insurance_query,
    portfolio_query, risk_assessment, yield_calc), knowledge (faq, policy),
    dialogue (chitchat, greeting), and meta intents.
    """

    def __init__(
        self,
        llm_service: LLMServiceBase,
        prompt_template: Optional[str] = None,
    ) -> None:
        self.llm_service = llm_service
        self.prompt_template = prompt_template
        self.intents = [i.value for i in Intent]

    async def detect(
        self,
        query: str,
        context: Optional[dict] = None,
    ) -> Intent:
        result = await self.detect_with_confidence(query, context)
        return result.intent

    async def detect_with_confidence(
        self,
        query: str,
        context: Optional[dict] = None,
    ) -> IntentResult:
        if not query or not query.strip():
            return IntentResult(intent=Intent.UNKNOWN, confidence=0.0)

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(query, context)

        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]

        try:
            response = await self.llm_service.generate(
                messages=messages,
                max_tokens=50,
                temperature=0.0,
            )
            return self._parse_response(response.content, query)

        except ExternalServiceError:
            raise
        except Exception as e:
            raise ExternalServiceError(
                service="LLM Intent Detection",
                message=f"Failed to detect intent: {str(e)}",
            ) from e

    def _build_system_prompt(self) -> str:
        if self.prompt_template:
            return self.prompt_template

        return """You are an intent classifier for a wealth-management customer service chatbot.
Classify user queries into one of these business intents:

Task-oriented (read-only consultation / query):
- wealth_query: User asks about wealth-management products (recommendation, comparison, product info)
- fund_query: User asks about fund details (NAV, fees, size, subscription/redemption)
- insurance_query: User asks about insurance (coverage, premium, policy types)
- portfolio_query: User asks about their holdings / positions / returns
- risk_assessment: User wants to take a risk tolerance assessment
- yield_calc: User wants to calculate expected returns for an amount and term

Knowledge:
- faq: General question about products, services, or processes
- policy: Question about product terms, regulatory policies, fee rules

Dialogue:
- chitchat: Casual conversation, not service-related
- greeting: Greetings (hello, hi, good morning)

Meta:
- confirm: User confirms something (yes, correct, continue)
- deny: User denies something (no, not right)
- cancel: User wants to cancel current operation
- unknown: Cannot determine intent

Rules:
- Respond with ONLY the intent name (lowercase)
- If user mentions a fund code or product code with a service request, classify by the service type
- Default to "faq" for general questions about how things work
- Default to "unknown" if truly uncertain"""

    def _build_user_prompt(self, query: str, context: Optional[dict]) -> str:
        prompt = f"Classify this query: {query}\n\nIntent:"

        if context:
            if "previous_messages" in context:
                messages = context["previous_messages"]
                if messages:
                    recent = messages[-2:] if len(messages) > 2 else messages
                    conversation = "\n".join(
                        f"{m.get('role', 'user')}: {m.get('content', '')}"
                        for m in recent
                    )
                    prompt = f"""Conversation context:
{conversation}

New query: {query}

Intent:"""
            if "current_intent" in context:
                prompt = f"""Current active task: {context['current_intent']}
Filled slots: {context.get('filled_slots', {})}

New query: {query}

Intent:"""

        return prompt

    def _parse_response(self, response: str, query: str) -> IntentResult:
        cleaned = response.strip().lower()

        if cleaned.startswith("{"):
            try:
                data = json.loads(cleaned)
                intent_str = data.get("intent", cleaned)
                confidence = float(data.get("confidence", 0.9))

                for intent in Intent:
                    if intent.value == intent_str:
                        return IntentResult(
                            intent=intent,
                            confidence=min(max(confidence, 0.0), 1.0),
                        )
            except json.JSONDecodeError:
                pass

        intent_str = re.sub(
            r"^(intent:|classification:|category:)\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()

        # 精确匹配（修复旧子串匹配误判：如 "I cannot confirm" 误判 confirm）
        if re.fullmatch(r"[a-z_]+", intent_str):
            for intent in Intent:
                if intent.value == intent_str:
                    return IntentResult(
                        intent=intent,
                        confidence=0.9,
                        metadata={"method": "llm", "raw_response": response},
                    )

        return IntentResult(
            intent=Intent.UNKNOWN,
            confidence=0.0,
            metadata={"method": "llm", "raw_response": response},
        )
