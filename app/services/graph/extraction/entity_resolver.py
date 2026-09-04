"""
Entity resolution and deduplication.

Normalizes entity names across chunks and merges properties from multiple
extractions to ensure consistency in the knowledge graph.
"""
import logging
import re
from typing import Dict, List, Optional, Tuple, Any

from app.services.graph.extraction.base import ExtractionResult

logger = logging.getLogger(__name__)


class EntityResolver:
    """Resolve and deduplicate extracted entities across chunks."""

    def __init__(self) -> None:
        self._entity_registry: Dict[str, Dict[str, Any]] = {}

    def resolve(
        self, extraction_results: List[ExtractionResult]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Process extraction results and return deduplicated entities/relations.

        Returns:
            Tuple of (resolved_entities, resolved_relations)
        """
        resolved_entities: Dict[str, Dict[str, Any]] = {}
        resolved_relations: List[Dict[str, Any]] = []

        for result in extraction_results:
            for entity in result.entities:
                key = self._make_key(entity["name"], entity["type"])
                if key in resolved_entities:
                    self._merge_entity(resolved_entities[key], entity)
                else:
                    resolved_entities[key] = {
                        "name": self._normalize_name(entity["name"]),
                        "type": entity["type"],
                        "description": entity.get("description"),
                        "properties": dict(entity.get("properties", {})),
                    }

            for relation in result.relations:
                source_key = self._resolve_name(relation["source"], resolved_entities)
                target_key = self._resolve_name(relation["target"], resolved_entities)
                if source_key and target_key:
                    resolved_relations.append({
                        "source": resolved_entities[source_key]["name"],
                        "target": resolved_entities[target_key]["name"],
                        "type": relation["type"],
                        "properties": dict(relation.get("properties", {})),
                    })

        return list(resolved_entities.values()), resolved_relations

    def _make_key(self, name: str, entity_type: str) -> str:
        normalized = self._normalize_name(name)
        return f"{entity_type}:{normalized}"

    @staticmethod
    def _normalize_name(name: str) -> str:
        name = name.strip()
        name = re.sub(r"\s+", " ", name)
        return name

    @staticmethod
    def _merge_entity(existing: Dict[str, Any], new: Dict[str, Any]) -> None:
        if not existing.get("description") and new.get("description"):
            existing["description"] = new["description"]
        existing["properties"].update(new.get("properties", {}))

    def _resolve_name(
        self, name: str, entities: Dict[str, Dict[str, Any]]
    ) -> Optional[str]:
        normalized = self._normalize_name(name)
        for key, entity in entities.items():
            if self._normalize_name(entity["name"]) == normalized:
                return key
            if normalized in self._normalize_name(entity["name"]):
                return key
        return None
