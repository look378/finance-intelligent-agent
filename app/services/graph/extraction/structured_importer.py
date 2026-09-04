"""
Structured data importer for CSV/Excel sources.

Maps columns to entity/relation types via a configurable schema mapping.
"""
import csv
import io
import logging
from typing import Optional, List, Dict, Any

from app.services.graph.base import GraphEntity, GraphRelation

logger = logging.getLogger(__name__)


class ColumnMapping:
    """Defines how a CSV column maps to graph entity/relation properties."""

    def __init__(
        self,
        entity_type: str,
        name_column: str,
        property_columns: Optional[Dict[str, str]] = None,
        description_column: Optional[str] = None,
    ) -> None:
        self.entity_type = entity_type
        self.name_column = name_column
        self.property_columns = property_columns or {}
        self.description_column = description_column


class RelationMapping:
    """Defines how two entity mappings connect via a relation."""

    def __init__(
        self,
        source_entity_type: str,
        source_name_column: str,
        target_entity_type: str,
        target_name_column: str,
        relation_type: str,
        relation_properties: Optional[Dict[str, str]] = None,
    ) -> None:
        self.source_entity_type = source_entity_type
        self.source_name_column = source_name_column
        self.target_entity_type = target_entity_type
        self.target_name_column = target_name_column
        self.relation_type = relation_type
        self.relation_properties = relation_properties or {}


class StructuredDataImporter:
    """Import structured data (CSV) into graph entities and relations."""

    def import_csv(
        self,
        csv_content: str,
        entity_mappings: List[ColumnMapping],
        relation_mappings: Optional[List[RelationMapping]] = None,
        delimiter: str = ",",
        encoding: str = "utf-8",
    ) -> tuple[List[GraphEntity], List[GraphRelation]]:
        """Parse CSV content and produce graph entities and relations.

        Args:
            csv_content: Raw CSV text
            entity_mappings: How columns map to entity types
            relation_mappings: How columns map to relations between entities
            delimiter: CSV delimiter

        Returns:
            Tuple of (entities, relations)
        """
        reader = csv.DictReader(io.StringIO(csv_content), delimiter=delimiter)
        entities: List[GraphEntity] = []
        relations: List[GraphRelation] = []

        for row_idx, row in enumerate(reader):
            row_entities: Dict[str, GraphEntity] = {}

            for mapping in entity_mappings:
                name = row.get(mapping.name_column, "").strip()
                if not name:
                    continue

                entity_id = f"{mapping.entity_type}_{name}_{row_idx}"
                props: Dict[str, Any] = {}
                for prop_name, col_name in mapping.property_columns.items():
                    val = row.get(col_name, "").strip()
                    if val:
                        props[prop_name] = self._parse_value(val)

                description = None
                if mapping.description_column:
                    description = row.get(mapping.description_column, "").strip() or None

                entity = GraphEntity(
                    id=entity_id,
                    name=name,
                    type=mapping.entity_type,
                    properties=props,
                    description=description,
                )
                row_entities[mapping.entity_type] = entity
                entities.append(entity)

            if relation_mappings:
                for rmap in relation_mappings:
                    source = row_entities.get(rmap.source_entity_type)
                    target = row_entities.get(rmap.target_entity_type)
                    if not source or not target:
                        continue

                    rel_props: Dict[str, Any] = {}
                    for prop_name, col_name in rmap.relation_properties.items():
                        val = row.get(col_name, "").strip()
                        if val:
                            rel_props[prop_name] = self._parse_value(val)

                    relations.append(
                        GraphRelation(
                            id=f"rel_{source.id}_{rmap.relation_type}_{target.id}",
                            source_entity_id=source.id,
                            target_entity_id=target.id,
                            relation_type=rmap.relation_type,
                            properties=rel_props,
                        )
                    )

        return entities, relations

    @staticmethod
    def _parse_value(value: str) -> Any:
        """Try to parse a string value as a number, fall back to string."""
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        return value
