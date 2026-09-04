"""Integration tests for chat pipeline (LangGraph-backed ChatService)."""
import pytest
from unittest.mock import Mock, AsyncMock


class TestChatPipelineIntegration:
    """Test complete chat pipeline integration"""

    @pytest.mark.asyncio
    async def test_full_chat_pipeline_with_retrieval(self):
        """Test complete pipeline: graph returns RAG-style response"""
        # Arrange
        from app.services.chat.chat_service import ChatService

        mock_graph = Mock()
        mock_graph.ainvoke = AsyncMock(
            return_value={
                "response": "理财非存款，产品有风险，投资须谨慎。",
                "intent": "policy",
                "sources": ["doc1"],
                "confidence": 0.9,
            }
        )

        service = ChatService(graph=mock_graph)

        # Act
        response = await service.process_message(
            session_id=1,
            message="理财有风险吗？",
            user_id=1,
        )

        # Assert - Complete pipeline executed
        assert response.content is not None
        assert "风险" in response.content
        assert response.intent == "policy"
        assert response.sources is not None
        assert len(response.sources) == 1
        assert response.sources[0] == "doc1"

        # Verify graph was invoked with the right payload
        mock_graph.ainvoke.assert_called_once()
        call_kwargs = mock_graph.ainvoke.call_args
        assert call_kwargs[0][0]["message"] == "理财有风险吗？"

    @pytest.mark.asyncio
    async def test_chat_pipeline_with_conversation_memory(self):
        """Test pipeline returning conversation context through graph"""
        # Arrange
        from app.services.chat.chat_service import ChatService

        mock_graph = Mock()
        mock_graph.ainvoke = AsyncMock(
            return_value={
                "response": "您之前提到想了解基金，请问具体想了解哪只基金？",
                "intent": "fund_query",
                "confidence": 0.9,
            }
        )

        mock_memory = Mock()
        mock_memory.get_context = AsyncMock(return_value=[])

        service = ChatService(graph=mock_graph, memory_strategy=mock_memory)

        # Act
        response = await service.process_message(
            session_id=1,
            message="介绍一下基金",
            user_id=1,
        )

        # Assert
        assert "基金" in response.content
        assert response.intent == "fund_query"

    @pytest.mark.asyncio
    async def test_chat_pipeline_streaming(self):
        """Test complete pipeline with streaming response"""
        # Arrange
        from app.services.chat.chat_service import ChatService

        async def mock_astream(*args, **kwargs):
            yield {"generate_response": {"response": "您好"}}
            yield {"generate_response": {"response": "有什么可以帮您"}}

        mock_graph = Mock()
        mock_graph.astream = mock_astream

        service = ChatService(graph=mock_graph)

        # Act
        chunks = []
        async for chunk in service.process_message_stream(
            session_id=1,
            message="你好",
            user_id=1,
        ):
            chunks.append(chunk)

        # Assert
        assert chunks == ["您好", "有什么可以帮您"]

    @pytest.mark.asyncio
    async def test_chat_pipeline_error_recovery(self):
        """Test pipeline handles errors gracefully"""
        # Arrange
        from app.services.chat.chat_service import ChatService

        mock_graph = Mock()
        mock_graph.ainvoke = AsyncMock(side_effect=RuntimeError("Graph failed"))

        service = ChatService(graph=mock_graph)

        # Act & Assert - Exception propagates to caller for HTTP layer handling
        with pytest.raises(RuntimeError):
            await service.process_message(
                session_id=1,
                message="Test message",
                user_id=1,
            )

    @pytest.mark.asyncio
    async def test_chat_pipeline_with_factory(self):
        """Test creating complete pipeline with factory"""
        # Arrange
        from app.services.chat.factory import ChatServiceFactory

        mock_llm = Mock()
        mock_message_repo = Mock()
        mock_session_repo = Mock()

        # Mock methods
        mock_llm.generate = AsyncMock(
            return_value=Mock(content="Response", model="gpt-4", usage={})
        )
        mock_message_repo.get_recent_messages = AsyncMock(return_value=[])
        mock_message_repo.create = AsyncMock()
        mock_message_repo.count_messages = AsyncMock(return_value=0)

        # Create service with factory (graph built internally, fallback to None on failure)
        service = ChatServiceFactory.create_with_defaults(
            llm_service=mock_llm,
            message_repo=mock_message_repo,
            session_repo=mock_session_repo,
            memory_type="sliding_window",
            intent_type="rule_based",
        )

        # Assert
        assert service is not None


class TestChatEndToEndScenarios:
    """Test end-to-end chat scenarios"""

    @pytest.mark.asyncio
    async def test_question_answering_scenario(self):
        """Test QA scenario with retrieval metadata"""
        # Arrange
        from app.services.chat.chat_service import ChatService

        mock_graph = Mock()
        mock_graph.ainvoke = AsyncMock(
            return_value={
                "response": "安心货币A是一款货币基金，风险等级R1，7日年化约1.85%。",
                "intent": "faq",
                "sources": ["doc1"],
                "confidence": 0.95,
            }
        )

        service = ChatService(graph=mock_graph)

        # Act
        response = await service.process_message(
            session_id=1,
            message="安心货币A是什么？",
            user_id=1,
        )

        # Assert
        assert "安心货币A" in response.content
        assert response.sources is not None
        assert len(response.sources) == 1

    @pytest.mark.asyncio
    async def test_greeting_scenario(self):
        """Test greeting scenario without retrieval"""
        # Arrange
        from app.services.chat.chat_service import ChatService

        mock_graph = Mock()
        mock_graph.ainvoke = AsyncMock(
            return_value={
                "response": "您好！我是金融财富管理客服助手，可以为您提供理财、基金、保险等咨询服务。",
                "intent": "greeting",
                "confidence": 0.95,
            }
        )

        service = ChatService(graph=mock_graph)

        # Act
        response = await service.process_message(
            session_id=1,
            message="你好",
            user_id=1,
        )

        # Assert
        assert "您好" in response.content or "help" in response.content
        assert response.sources is None  # No retrieval for greetings
