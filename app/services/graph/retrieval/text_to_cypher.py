"""
Text-to-Cypher service.

Generates Cypher queries from natural language using the LLM, executes
them against the graph database, and returns structured results.
"""
import json
import logging
import time
from typing import Optional, List, Dict, Any

from app.services.graph.base import (
    GraphClient,
    GraphEntity,
    GraphRelation,
    GraphSearchResult,
    GraphClientError,
)
from app.services.graph.schema import get_schema_prompt_text
from app.services.llm.base import LLMServiceBase, LLMMessage

logger = logging.getLogger(__name__)

_CYPHER_PROMPT = """You are a Neo4j Cypher query expert for a wealth-management customer service knowledge graph. Generate a read-only Cypher query for the user's question.

Graph Schema:
{schema}

Example queries:
1. "稳健型投资者能买哪些理财产品？" -> MATCH (p:Entity:FinancialProduct)-[:BELONGS_TO]->(r:Entity:RiskLevel {{level: 'R2'}}) RETURN p.name, p.benchmark, p.min_amount
2. "易方达优质精选混合基金的费率和净值？" -> MATCH (f:Entity:Fund {{fund_name: '易方达优质精选混合'}}) RETURN f.nav, f.subscription_fee, f.management_fee, f.custody_fee, f.year_return
3. "哪些产品适合养老规划人群？" -> MATCH (p:Entity:FinancialProduct)-[:SUITABLE_FOR]->(c:Entity:CustomerProfile {{profile_name: '养老规划'}}) RETURN p.name, p.category
4. "重疾险的保费大概多少？" -> MATCH (i:Entity:Insurance {{insurance_type: '重疾险'}}) RETURN i.plan_name, i.premium_example, i.coverage
5. "货币基金属于什么风险等级？" -> MATCH (f:Entity:Fund)-[:BELONGS_TO_LEVEL]->(r:Entity:RiskLevel) WHERE f.fund_name CONTAINS '货币' RETURN f.fund_name, r.level, r.level_name

Rules:
- Generate ONLY read-only MATCH/WHERE/RETURN queries. No CREATE, MERGE, SET, DELETE.
- Always add LIMIT (max {max_results}).
- Use parameterized values via $params where appropriate.
- Return results as: {{"query": "...", "params": {{...}}}}
- If the question cannot be answered from the schema, return {{"query": null, "params": {{}}}}

Question: {question}"""


class TextToCypherService:
    """Convert natural language to Cypher and execute against the graph."""

    def __init__(
        self,
        llm_service: LLMServiceBase,
        graph_client: GraphClient,
        query_timeout: float = 10.0,
        max_results: int = 20,
    ) -> None:
        self._llm = llm_service
        self._graph = graph_client
        self._timeout = query_timeout
        self._max_results = max_results
        self._schema_cache: Optional[str] = None
        self._schema_cache_time: float = 0
        self._cache_ttl: float = 300.0  # 5 minutes

    async def query(
        self, natural_language_query: str, max_results: int = 20, entity_hints: Optional[list] = None
    ) -> List[GraphSearchResult]:
        """Generate Cypher from NL, execute, and return results."""
        schema = await self._get_schema()
        cypher = await self._generate_cypher(natural_language_query, schema, max_results, entity_hints)

        if not cypher.get("query"):
            return []

        try:
            results = await self._graph.execute_cypher(
                cypher["query"], cypher.get("params", {})
            )
        except GraphClientError as e:
            logger.warning("Cypher execution failed: %s", e)
            return []

        return self._to_search_results(results, natural_language_query)

    async def _get_schema(self) -> str:
        now = time.time()
        if self._schema_cache and (now - self._schema_cache_time) < self._cache_ttl:
            return self._schema_cache

        dynamic_schema = await self._graph.get_schema()
        static_schema = get_schema_prompt_text()
        self._schema_cache = f"{static_schema}\n\nCurrent Graph State:\n{dynamic_schema}"
        self._schema_cache_time = now
        return self._schema_cache

    async def _generate_cypher(
        self, question: str, schema: str, max_results: int, entity_hints: Optional[list] = None
    ) -> Dict[str, Any]:
        prompt = _CYPHER_PROMPT.format(
            schema=schema,
            question=question,
            max_results=max_results,
        )
        if entity_hints:
            hints_lines = [f"  - {h['type']}: {h['name']}" for h in entity_hints]
            prompt += "\n\nKnown entities in the query:\n" + "\n".join(hints_lines)
        messages = [
            LLMMessage(role="system", content="Output only valid JSON."),
            LLMMessage(role="user", content=prompt),
        ]

        try:
            response = await self._llm.generate(messages=messages, max_tokens=1024)
            return self._parse_cypher_response(response.content)
        except Exception as e:
            logger.warning("Cypher generation failed: %s", e)
            return {}

    def _parse_cypher_response(self, content: str) -> Dict[str, Any]:
        text = content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:])
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        try:
            data = json.loads(text)
            query = data.get("query")
            if query and not query.strip().upper().startswith("MATCH"):
                logger.warning("Generated query is not a MATCH query: %s", query[:100])
                return {}
            return data
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse Cypher JSON: %s", e)
            return {}

    def _to_search_results(
        self, records: List[Dict[str, Any]], query: str
    ) -> List[GraphSearchResult]:
        results: List[GraphSearchResult] = []
        for record in records:
            parts: List[str] = []
            entities: List[GraphEntity] = []
            relations: List[GraphRelation] = []

            for key, value in record.items():
                if isinstance(value, dict):
                    if "name" in value:
                        entities.append(
                            GraphEntity(
                                id=value.get("id", ""),
                                name=value["name"],
                                type=value.get("type", ""),
                                description=value.get("description"),
                            )
                        )
                    parts.append(f"{key}: {value}")
                else:
                    parts.append(f"{key}: {value}")

            content = "; ".join(parts)
            results.append(
                GraphSearchResult(
                    content=content,
                    entities=entities,
                    relations=relations,
                    score=1.0,
                    source_type="text_to_cypher",
                    metadata={"query": query},
                )
            )
        return results
