"""
Domain schema definition for the knowledge graph (wealth management).

Defines entity types, relationship types, and their properties for the
financial wealth-management customer service domain.
"""

ENTITY_TYPES: dict[str, dict[str, str | list[str]]] = {
    "FinancialProduct": {
        "description": "A wealth-management product (理财/基金/保险/存款)",
        "properties": [
            "product_name", "product_code", "category", "risk_level",
            "benchmark", "min_amount", "term", "issuer",
        ],
    },
    "Fund": {
        "description": "A mutual fund (公募基金)",
        "properties": [
            "fund_name", "fund_code", "fund_type", "nav", "year_return",
            "subscription_fee", "management_fee", "custody_fee", "size",
        ],
    },
    "Insurance": {
        "description": "An insurance plan (重疾/医疗/意外/寿险/年金)",
        "properties": [
            "plan_name", "insurance_type", "coverage", "premium_example", "features",
        ],
    },
    "RiskLevel": {
        "description": "A risk rating level R1-R5 (投资者风险承受等级/产品风险等级)",
        "properties": ["level", "level_name", "description"],
    },
    "FeeRule": {
        "description": "A fee / rate rule (费率、手续费、管理费、托管费)",
        "properties": ["fee_type", "rate", "description"],
    },
    "Policy": {
        "description": "A regulatory policy or product term (监管政策/产品条款/适当性规定)",
        "properties": ["title", "category", "effective_date", "summary"],
    },
    "Institution": {
        "description": "A financial institution (银行/基金公司/保险公司)",
        "properties": ["institution_name", "institution_type"],
    },
    "FAQ": {
        "description": "A frequently asked question and its answer",
        "properties": ["question", "answer", "category"],
    },
    "CustomerProfile": {
        "description": "A customer profile segment (客群画像，如 保守型投资者/养老规划人群)",
        "properties": ["profile_name", "description"],
    },
}

RELATION_TYPES: dict[str, dict[str, str]] = {
    "BELONGS_TO": {
        "source": "FinancialProduct",
        "target": "RiskLevel",
        "description": "Product belongs to this risk level",
    },
    "HAS_FEE_RULE": {
        "source": "FinancialProduct",
        "target": "FeeRule",
        "description": "Product has this fee/rate rule",
    },
    "REGULATES": {
        "source": "Policy",
        "target": "FinancialProduct",
        "description": "Policy regulates this product",
    },
    "ISSUED_BY": {
        "source": "FinancialProduct",
        "target": "Institution",
        "description": "Product is issued by this institution",
    },
    "SUITABLE_FOR": {
        "source": "FinancialProduct",
        "target": "CustomerProfile",
        "description": "Product is suitable for this customer profile (适当性匹配)",
    },
    "INVESTS_IN": {
        "source": "Fund",
        "target": "FinancialProduct",
        "description": "Fund invests in this underlying product/asset",
    },
    "ANSWERS": {
        "source": "FAQ",
        "target": "FinancialProduct",
        "description": "FAQ answers question about this product",
    },
    "RELATED_TO": {
        "source": "FinancialProduct",
        "target": "FinancialProduct",
        "description": "Products are related or comparable",
    },
    "BELONGS_TO_LEVEL": {
        "source": "Fund",
        "target": "RiskLevel",
        "description": "Fund belongs to this risk level",
    },
    "BELONGS_TO_TYPE": {
        "source": "Insurance",
        "target": "RiskLevel",
        "description": "Insurance plan belongs to this risk level",
    },
}


def get_schema_prompt_text() -> str:
    """Format the schema as text suitable for LLM prompts."""
    lines = ["Entity Types:"]

    for name, info in ENTITY_TYPES.items():
        props = ", ".join(info["properties"])  # type: ignore[union-attr]
        lines.append(f"  {name}: {info['description']}. Properties: [{props}]")

    lines.append("\nRelationship Types:")
    for name, info in RELATION_TYPES.items():
        lines.append(
            f"  ({info['source']})-[{name}]->({info['target']}): "
            f"{info['description']}"
        )

    return "\n".join(lines)
