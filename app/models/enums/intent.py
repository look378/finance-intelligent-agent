"""Business intent enums for wealth-management customer service dialogue."""
from enum import Enum


class Intent(str, Enum):
    """Business intent categories aligned with wealth-management operations.

    Task-oriented intents are read-only consultation/query operations
    (no fund-transfer / capital-movement execution in this system).
    """

    # Task-oriented intents (→ Function Calling, all read-only consultation)
    WEALTH_QUERY = "wealth_query"          # 理财咨询：产品查询/推荐/对比
    FUND_QUERY = "fund_query"              # 基金查询：净值/费率/规模
    INSURANCE_QUERY = "insurance_query"    # 保险咨询：险种/保障/保费估算
    PORTFOLIO_QUERY = "portfolio_query"    # 持仓/收益查询
    RISK_ASSESSMENT = "risk_assessment"    # 风险测评（问卷→风险等级）
    YIELD_CALC = "yield_calc"              # 收益测算（金额×期限→预期区间）

    # Knowledge intents (→ RAG)
    FAQ = "faq"
    POLICY = "policy"                      # 产品条款/监管政策/费率规则

    # Dialogue intents (→ Direct response)
    CHITCHAT = "chitchat"
    GREETING = "greeting"

    # Meta intents (→ State action)
    CONFIRM = "confirm"
    DENY = "deny"
    CANCEL = "cancel"
    UNKNOWN = "unknown"

    # Graph-related (→ GraphRAG, legacy support)
    RELATIONSHIP_QUERY = "relationship_query"
    GLOBAL_SUMMARY = "global_summary"
    ENTITY_LOOKUP = "entity_lookup"


# Routing groups
TASK_INTENTS = {
    "wealth_query",
    "fund_query",
    "insurance_query",
    "portfolio_query",
    "risk_assessment",
    "yield_calc",
}
RAG_INTENTS = {"faq", "policy"}
DIRECT_INTENTS = {"chitchat", "greeting"}
META_INTENTS = {"confirm", "deny", "cancel"}
GRAPH_INTENTS = {"relationship_query", "global_summary", "entity_lookup"}

# Display names for Chinese UI
INTENT_DISPLAY_NAMES = {
    "wealth_query": "理财咨询",
    "fund_query": "基金查询",
    "insurance_query": "保险咨询",
    "portfolio_query": "持仓查询",
    "risk_assessment": "风险测评",
    "yield_calc": "收益测算",
    "faq": "常见问题",
    "policy": "政策条款",
}
