"""Tests for prompt templates"""
import pytest


class TestPromptTemplates:
    """Test prompt template management"""

    def test_chat_system_prompt(self):
        """Test chat system prompt template (financial wealth management)"""
        # Arrange & Act
        from app.services.llm.prompt_templates import PromptTemplates

        prompt = PromptTemplates.get_chat_system_prompt()

        # Assert
        assert "assistant" in prompt.lower()
        assert "wealth" in prompt.lower() or "financial" in prompt.lower()
        assert len(prompt) > 50

    def test_rag_system_prompt(self):
        """Test RAG system prompt template"""
        # Arrange & Act
        from app.services.llm.prompt_templates import PromptTemplates

        prompt = PromptTemplates.get_rag_system_prompt()

        # Assert
        assert "retrieval" in prompt.lower() or "context" in prompt.lower()
        assert len(prompt) > 50

    def test_format_chat_prompt(self):
        """Test formatting chat prompt with context"""
        # Arrange
        from app.services.llm.prompt_templates import PromptTemplates

        context = {
            "user_name": "John",
            "conversation_topic": "AI"
        }

        # Act
        prompt = PromptTemplates.format_chat_prompt(
            system_prompt="You are talking to {user_name} about {conversation_topic}.",
            **context
        )

        # Assert
        assert "John" in prompt
        assert "AI" in prompt

    def test_format_rag_prompt(self):
        """Test formatting RAG prompt with retrieved context"""
        # Arrange
        from app.services.llm.prompt_templates import PromptTemplates

        query = "What is RAG?"
        retrieved_docs = [
            {"content": "RAG stands for Retrieval-Augmented Generation.", "source": "doc1.pdf"},
            {"content": "It combines retrieval systems with LLMs.", "source": "doc2.pdf"},
        ]

        # Act
        prompt = PromptTemplates.format_rag_prompt(query, retrieved_docs)

        # Assert
        assert "What is RAG?" in prompt
        assert "Retrieval-Augmented Generation" in prompt
        assert "doc1.pdf" in prompt or "doc2.pdf" in prompt

    def test_format_rag_prompt_no_docs(self):
        """Test formatting RAG prompt with no retrieved documents"""
        # Arrange
        from app.services.llm.prompt_templates import PromptTemplates

        query = "Tell me a joke"
        retrieved_docs = []

        # Act
        prompt = PromptTemplates.format_rag_prompt(query, retrieved_docs)

        # Assert
        assert query in prompt
        # Should handle empty docs gracefully

    def test_summarization_prompt(self):
        """Test summarization prompt template"""
        # Arrange
        from app.services.llm.prompt_templates import PromptTemplates

        text = "This is a long text that needs to be summarized. " * 10

        # Act
        prompt = PromptTemplates.get_summarization_prompt(text)

        # Assert
        assert "summarize" in prompt.lower()
        assert text in prompt

    def test_intent_detection_prompt(self):
        """Test intent detection prompt template"""
        # Arrange
        from app.services.llm.prompt_templates import PromptTemplates

        query = "How do I implement a binary search tree?"
        intents = ["question", "how_to", "code_help", "chitchat"]

        # Act
        prompt = PromptTemplates.get_intent_detection_prompt(query, intents)

        # Assert
        assert query in prompt
        for intent in intents:
            assert intent in prompt

    def test_custom_template(self):
        """Test custom prompt template"""
        # Arrange
        from app.services.llm.prompt_templates import PromptTemplates

        template = "You are a {role} specializing in {topic}."
        vars = {"role": "Python expert", "topic": "FastAPI"}

        # Act
        prompt = PromptTemplates.format_template(template, **vars)

        # Assert
        assert "Python expert" in prompt
        assert "FastAPI" in prompt

    def test_memory_aware_prompt(self):
        """Test memory-aware prompt with conversation history"""
        # Arrange
        from app.services.llm.prompt_templates import PromptTemplates

        conversation_summary = "User asked about Python basics."
        current_message = "How do I use list comprehensions?"

        # Act
        prompt = PromptTemplates.get_memory_aware_prompt(
            conversation_summary,
            current_message
        )

        # Assert
        assert conversation_summary in prompt
        assert current_message in prompt

    def test_code_generation_prompt(self):
        """Test code generation prompt template"""
        # Arrange
        from app.services.llm.prompt_templates import PromptTemplates

        requirements = "Create a function to validate email addresses"
        language = "Python"

        # Act
        prompt = PromptTemplates.get_code_generation_prompt(requirements, language)

        # Assert
        assert requirements in prompt
        assert language in prompt
        assert "function" in prompt.lower()

    def test_few_shot_prompt(self):
        """Test few-shot learning prompt template"""
        # Arrange
        from app.services.llm.prompt_templates import PromptTemplates

        examples = [
            {"input": "Hello", "output": "Hi there!"},
            {"input": "How are you?", "output": "I'm doing well!"},
        ]
        test_input = "Good morning!"

        # Act
        prompt = PromptTemplates.get_few_shot_prompt(examples, test_input)

        # Assert
        assert "Hello" in prompt
        assert "Hi there!" in prompt
        assert "Good morning!" in prompt
