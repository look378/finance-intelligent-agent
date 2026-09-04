"""
LLM-based entity and relationship extraction from unstructured text.

Sends text chunks to the LLM with a structured prompt that includes the
domain schema, requesting JSON output of entities and relations.
"""
import json
import logging
from typing import Optional, List, Dict, Any

from app.services.graph.extraction.base import EntityExtractor, ExtractionResult
from app.services.graph.schema import ENTITY_TYPES, RELATION_TYPES
from app.services.llm.base import LLMServiceBase, LLMMessage

logger = logging.getLogger(__name__)

_EXTRACTION_PROMPT = """You are an expert knowledge graph builder for a financial wealth-management customer service system.

Given the text below, extract all entities and relationships. Follow this schema strictly:

Entity Types:
{entity_schema}

Relationship Types:
{relation_schema}

Instructions:
1. Extract every entity mentioned in the text.
2. For each entity, provide: "name" (canonical), "type" (from schema), "description" (brief), and "properties" (key facts from text).
3. Extract relationships between entities. For each: "source" (entity name), "target" (entity name), "type" (from schema), "properties" (any relevant data).
4. Normalize entity names (e.g., "稳健理财" and "稳健型理财" both map to one canonical name like "稳健型理财").
5. Only extract relationships that are explicitly stated or can be directly inferred.
6. Financial facts (rates, fees, risk levels) must be preserved as properties verbatim.

Return ONLY a JSON object with this exact structure:
{{
  "entities": [
    {{"name": "...", "type": "...", "description": "...", "properties": {{...}}}}
  ],
  "relations": [
    {{"source": "...", "target": "...", "type": "...", "properties": {{...}}}}
  ]
}}

Text:
{text}"""


class LLMEntityExtractor(EntityExtractor):
    """Extract entities and relations using an LLM."""

    def __init__(
        self,
        llm_service: LLMServiceBase,
        entity_types: Optional[Dict[str, Dict[str, Any]]] = None,
        relation_types: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        self._llm = llm_service
        self._entity_types = entity_types or ENTITY_TYPES
        self._relation_types = relation_types or RELATION_TYPES
        self._prompt = self._build_prompt_template()

    def _build_prompt_template(self) -> str:
        entity_schema_parts: List[str] = []
        for name, info in self._entity_types.items():
            props = ", ".join(info.get("properties", []))  # type: ignore[union-attr]
            entity_schema_parts.append(
                f"  {name}: {info.get('description', '')}. Properties: [{props}]"
            )

        relation_schema_parts: List[str] = []
        for name, info in self._relation_types.items():
            relation_schema_parts.append(
                f"  ({info.get('source', '')})-[{name}]->({info.get('target', '')}): "
                f"{info.get('description', '')}"
            )

        prompt = _EXTRACTION_PROMPT.replace(
            "{entity_schema}", "\n".join(entity_schema_parts)
        ).replace(
            "{relation_schema}", "\n".join(relation_schema_parts)
        )
        return prompt

    async def extract(
        self, text: str, context: Optional[Dict[str, Any]] = None
    ) -> ExtractionResult:
        prompt = self._prompt.format(text=text)
        messages = [
            LLMMessage(role="system", content="You extract structured knowledge about financial products, funds, insurance, risk levels, and policies from customer service text. Output only valid JSON."),
            LLMMessage(role="user", content=prompt),
        ]

        try:
            response = await self._llm.generate(messages=messages, max_tokens=2048)
            return self._parse_response(response.content, text)
        except Exception as e:
            logger.warning("Entity extraction failed: %s", e)
            return ExtractionResult(raw_text=text)

    async def extract_batch(
        self, texts: List[str], context: Optional[Dict[str, Any]] = None
    ) -> List[ExtractionResult]:
        results: List[ExtractionResult] = []
        for text in texts:
            result = await self.extract(text, context)
            results.append(result)
        return results

    def _parse_response(self, content: str, raw_text: str) -> ExtractionResult:
        text = content.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:])
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse extraction JSON: %s", e)
            return ExtractionResult(raw_text=raw_text)

        entities = data.get("entities", [])
        relations = data.get("relations", [])

        # Validate entity types against schema
        valid_entities: List[Dict[str, Any]] = []
        for ent in entities:
            if not isinstance(ent, dict) or "name" not in ent or "type" not in ent:
                continue
            if ent["type"] in self._entity_types:
                valid_entities.append(ent)
            else:
                logger.debug("Skipping entity with unknown type: %s", ent.get("type"))

        # Validate relation types against schema
        valid_relations: List[Dict[str, Any]] = []
        for rel in relations:
            if not isinstance(rel, dict) or "source" not in rel or "target" not in rel:
                continue
            if "type" in rel and rel["type"] in self._relation_types:
                valid_relations.append(rel)
            else:
                logger.debug("Skipping relation with unknown type: %s", rel.get("type"))

        return ExtractionResult(
            entities=valid_entities,
            relations=valid_relations,
            raw_text=raw_text,
        )
