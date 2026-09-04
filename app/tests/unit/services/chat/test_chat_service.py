"""Tests for chat service"""
import pytest
from unittest.mock import Mock, AsyncMock

from app.services.chat.chat_service import ChatService, ChatResponse, ChatMessage


def _make_graph(return_value: dict):
    """Create a mock compiled LangGraph that returns the given value on ainvoke."""
    graph = Mock()
    graph.ainvoke = AsyncMock(return_value=return_value)
    return graph


class TestChatService:
    """Test chat orchestration service"""

    def test_chat_service_initialization(self):
        mock_graph = Mock()
        mock_llm = Mock()
        mock_memory = Mock()
        service = ChatService(
            graph=mock_graph,
            llm_service=mock_llm,
            memory_strategy=mock_memory,
        )
        assert service.graph is mock_graph
        assert service.llm_service is mock_llm
        assert service.memory_strategy is mock_memory

    @pytest.mark.asyncio
    async def test_process_message_simple(self):
        mock_graph = _make_graph({
            "response": "您好！有什么可以帮您的？",
            "intent": "greeting",
            "confidence": 0.95,
        })
        service = ChatService(graph=mock_graph)
        response = await service.process_message(
            session_id=1, message="你好", user_id=1,
        )
        assert response.content == "您好！有什么可以帮您的？"
        assert response.session_id == 1
        assert response.intent == "greeting"
        mock_graph.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_message_with_retrieval(self):
        mock_graph = _make_graph({
            "response": "理财非存款，产品有风险，投资须谨慎。",
            "intent": "policy",
            "sources": ["doc1"],
            "confidence": 0.9,
        })
        service = ChatService(graph=mock_graph)
        response = await service.process_message(
            session_id=1, message="理财有风险吗", user_id=1,
        )
        assert "风险" in response.content
        assert response.sources is not None
        assert len(response.sources) > 0

    @pytest.mark.asyncio
    async def test_process_message_with_memory(self):
        mock_graph = _make_graph({"response": "好的", "intent": "chitchat"})
        mock_memory = Mock()
        msg1 = Mock(role="user", content="My name is Alice")
        msg2 = Mock(role="assistant", content="Hello Alice!")
        mock_memory.get_context = AsyncMock(return_value=[msg1, msg2])
        service = ChatService(graph=mock_graph, memory_strategy=mock_memory)
        history = await service.get_chat_history(session_id=1)
        assert len(history) == 2
        assert history[0].role == "user"

    @pytest.mark.asyncio
    async def test_process_message_with_streaming(self):
        async def mock_astream(*args, **kwargs):
            yield {"generate_response": {"response": "您好"}}
            yield {"generate_response": {"response": "有什么可以帮您"}}

        mock_graph = Mock()
        mock_graph.astream = mock_astream
        service = ChatService(graph=mock_graph)
        chunks = []
        async for chunk in service.process_message_stream(
            session_id=1, message="你好", user_id=1,
        ):
            chunks.append(chunk)
        assert len(chunks) == 2

    @pytest.mark.asyncio
    async def test_get_chat_history(self):
        mock_graph = Mock()
        mock_memory = Mock()
        mock_memory.get_context = AsyncMock(
            return_value=[
                Mock(role="user", content="Hello"),
                Mock(role="assistant", content="Hi there!"),
                Mock(role="user", content="How are you?"),
            ]
        )
        service = ChatService(graph=mock_graph, memory_strategy=mock_memory)
        history = await service.get_chat_history(session_id=1)
        assert len(history) == 3
        assert history[0].content == "Hello"

    @pytest.mark.asyncio
    async def test_clear_chat_history(self):
        mock_graph = Mock()
        mock_memory = Mock()
        mock_memory.clear_session = AsyncMock()
        service = ChatService(graph=mock_graph, memory_strategy=mock_memory)
        await service.clear_chat_history(session_id=1)
        mock_memory.clear_session.assert_called_once_with(session_id=1)

    @pytest.mark.asyncio
    async def test_get_chat_history_no_memory(self):
        service = ChatService(graph=Mock())
        history = await service.get_chat_history(session_id=1)
        assert history == []

    @pytest.mark.asyncio
    async def test_process_message_returns_metadata(self):
        mock_graph = _make_graph({
            "response": "请提供基金代码",
            "intent": "fund_query",
            "confidence": 0.9,
            "pending_slots": ["fund_code"],
            "filled_slots": {},
        })
        service = ChatService(graph=mock_graph)
        response = await service.process_message(
            session_id=1, message="查一下基金", user_id=1,
        )
        assert response.metadata["pending_slots"] == ["fund_code"]


class TestChatResponse:
    """Test ChatResponse dataclass"""

    def test_chat_response_creation(self):
        response = ChatResponse(
            content="Hello!",
            session_id=1,
            intent="greeting",
            sources=["doc1", "doc2"],
            metadata={"tokens": 50},
        )
        assert response.content == "Hello!"
        assert response.session_id == 1
        assert response.intent == "greeting"
        assert len(response.sources) == 2

    def test_chat_response_without_sources(self):
        response = ChatResponse(
            content="Hi!",
            session_id=1,
            intent="greeting",
        )
        assert response.sources is None
