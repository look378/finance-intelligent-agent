"""
Demo chat service for open API testing.

Provides simple responses without requiring full RAG pipeline setup.
"""
from dataclasses import dataclass, field
from typing import Optional, List, AsyncGenerator
import random


@dataclass
class DemoChatResponse:
    """Demo chat response data."""
    content: str
    session_id: int
    intent: str
    sources: Optional[List[str]] = None
    metadata: Optional[dict] = None


@dataclass
class DemoChatMessage:
    """Demo chat message data."""
    role: str
    content: str
    timestamp: Optional[str] = None


class DemoChatService:
    """
    Demo chat service for open API testing.

    Returns simple responses without requiring LLM, embeddings, or vector DB.
    """

    def __init__(self):
        """Initialize demo chat service."""
        self.demo_responses = {
            "hello": "您好！我是金融财富管理客服助手，可以为您提供理财、基金、保险等产品咨询。",
            "hi": "您好！请问有什么可以帮您？例如：查询理财产品、基金净值、持仓收益，或进行风险测评。",
            "help": "我可以为您提供理财咨询、基金查询、保险咨询、持仓查询、风险测评和收益测算等服务。这是演示模式，回复内容为固定文案。",
            "what": "这是很好的问题！在正式环境中，我会检索金融知识库中的产品条款、费率与政策信息来回答您。",
            "how": "让我为您说明：正式环境中，系统通过意图识别、记忆和知识检索（向量+知识图谱）来提供准确的金融咨询回答。",
            "default": "感谢您的咨询！这是演示回复。正式环境中我将通过金融知识库与工具链为您提供理财产品、基金与保险等咨询服务。",
        }

    async def process_message(
        self,
        session_id: int,
        message: str,
        user_id: int,
        max_tokens: Optional[int] = None,
    ) -> DemoChatResponse:
        """
        Process a user message and generate demo response.

        Args:
            session_id: Session identifier
            message: User message
            user_id: User identifier
            max_tokens: Optional max tokens for response

        Returns:
            DemoChatResponse: Generated response
        """
        # Simple intent detection based on keywords
        message_lower = message.lower()

        if any(word in message_lower for word in ["hello", "hey"]):
            intent = "greeting"
            response = self.demo_responses["hello"]
        elif any(word in message_lower for word in ["help", "assist"]):
            intent = "faq"
            response = self.demo_responses["help"]
        elif "what" in message_lower or "什么" in message:
            intent = "faq"
            response = self.demo_responses["what"]
        elif "how" in message_lower or "怎么" in message:
            intent = "faq"
            response = self.demo_responses["how"]
        else:
            intent = "chitchat"
            response = self.demo_responses["default"]

        return DemoChatResponse(
            content=response,
            session_id=session_id,
            intent=intent,
            sources=None,
            metadata={
                "demo": True,
                "tokens_used": len(response.split()),
            },
        )

    async def process_message_stream(
        self,
        session_id: int,
        message: str,
        user_id: int,
    ) -> AsyncGenerator[str, None]:
        """
        Process message with streaming demo response.

        Args:
            session_id: Session identifier
            message: User message
            user_id: User identifier

        Yields:
            str: Response chunks
        """
        response = await self.process_message(session_id, message, user_id)
        words = response.content.split()

        for word in words:
            yield word + " "

    async def get_chat_history(
        self,
        session_id: int,
        limit: int = 50,
    ) -> List[DemoChatMessage]:
        """
        Get demo chat history.

        Args:
            session_id: Session identifier
            limit: Maximum number of messages

        Returns:
            List[DemoChatMessage]: Empty list (demo mode doesn't persist)
        """
        return []

    async def clear_chat_history(self, session_id: int) -> None:
        """
        Clear demo chat history (no-op in demo mode).

        Args:
            session_id: Session identifier
        """
        pass
