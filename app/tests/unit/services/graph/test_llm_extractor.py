"""Unit tests for LLM entity extractor."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.services.graph.extraction.llm_extractor import LLMEntityExtractor


class MockLLMResponse:
    def __init__(self, content: str):
        self.content = content


class TestLLMEntityExtractor:
    def setup_method(self):
        self.mock_llm = AsyncMock()
        self.extractor = LLMEntityExtractor(llm_service=self.mock_llm)

    @pytest.mark.asyncio
    async def test_extract_valid_json(self):
        self.mock_llm.generate = AsyncMock(
            return_value=MagicMock(
                content='{"entities": [{"name": "安心货币A", "type": "FinancialProduct", "description": "货币基金", "properties": {"risk_level": "R1"}}], "relations": [{"source": "安心货币A", "target": "R1", "type": "BELONGS_TO", "properties": {}}]}'
            )
        )

        result = await self.extractor.extract("安心货币A是一款货币基金，风险等级R1")
        assert len(result.entities) == 1
        assert result.entities[0]["name"] == "安心货币A"
        assert result.entities[0]["type"] == "FinancialProduct"
        assert len(result.relations) == 1
        assert result.relations[0]["type"] == "BELONGS_TO"

    @pytest.mark.asyncio
    async def test_extract_invalid_json(self):
        self.mock_llm.generate = AsyncMock(
            return_value=MagicMock(content="not json at all")
        )

        result = await self.extractor.extract("some text")
        assert result.entities == []
        assert result.relations == []

    @pytest.mark.asyncio
    async def test_extract_with_markdown_fences(self):
        self.mock_llm.generate = AsyncMock(
            return_value=MagicMock(
                content='```json\n{"entities": [{"name": "重疾险", "type": "Insurance", "description": "", "properties": {}}], "relations": []}\n```'
            )
        )

        result = await self.extractor.extract("test text")
        assert len(result.entities) == 1
        assert result.entities[0]["name"] == "重疾险"

    @pytest.mark.asyncio
    async def test_extract_filters_unknown_entity_type(self):
        self.mock_llm.generate = AsyncMock(
            return_value=MagicMock(
                content='{"entities": [{"name": "test", "type": "UnknownType", "properties": {}}], "relations": []}'
            )
        )

        result = await self.extractor.extract("test")
        assert len(result.entities) == 0

    @pytest.mark.asyncio
    async def test_extract_handles_llm_error(self):
        self.mock_llm.generate = AsyncMock(side_effect=Exception("LLM error"))

        result = await self.extractor.extract("test text")
        assert result.entities == []
        assert result.raw_text == "test text"

    @pytest.mark.asyncio
    async def test_extract_batch(self):
        self.mock_llm.generate = AsyncMock(
            return_value=MagicMock(
                content='{"entities": [{"name": "a", "type": "Fund", "properties": {}}], "relations": []}'
            )
        )

        results = await self.extractor.extract_batch(["text1", "text2"])
        assert len(results) == 2
