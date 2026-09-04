"""
Tool definitions and registry for Function Calling (wealth management).

Provides read-only mock tool handlers for financial consultation and
query operations (product lookup, fund detail, insurance plans, portfolio
inquiry, risk assessment, yield calculation).  NO capital-movement
operations (transfer / purchase / redemption execution) are provided in
this system — all tools are information & consultation services.
"""
from dataclasses import dataclass
from typing import Callable, Optional

from app.models.enums.intent import TASK_INTENTS

@dataclass
class ToolDefinition:
    """Definition of a callable tool."""
    name: str
    intent: str
    description: str
    required_slots: list[str]
    handler: Callable[[dict], dict]


@dataclass
class ToolResult:
    """Result from tool execution."""
    success: bool
    data: dict
    message: str = ""


class ToolRegistry:
    """Registry of available tools for Function Calling."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        self._tools[tool.intent] = tool

    def get_tool_for_intent(self, intent: str) -> Optional[ToolDefinition]:
        return self._tools.get(intent)

    async def execute(self, intent: str, args: dict) -> ToolResult:
        tool = self._tools.get(intent)
        if not tool:
            return ToolResult(success=False, data={}, message=f"No tool for intent: {intent}")
        try:
            result = tool.handler(args)
            return ToolResult(success=True, data=result, message="OK")
        except Exception as e:
            return ToolResult(success=False, data={}, message=str(e))


# ── Mock product universe (realistic rules) ──────────────────────────────

# Risk levels: R1 (conservative) → R5 (aggressive)
# Suitability: a client rated Rk may buy products rated R1..Rk.
RISK_LEVEL_NAMES = {
    "保守型": "R1",
    "稳健型": "R2",
    "平衡型": "R3",
    "进取型": "R4",
    "激进型": "R5",
}

WEALTH_PRODUCTS = [
    {
        "product_code": "WM001",
        "name": "安心货币A",
        "category": "货币基金",
        "risk_level": "R1",
        "benchmark": "7日年化 1.85%",
        "term": "无固定期限",
        "min_amount": 100,
        "rules": "支持T+1快速赎回，赎回上限1万元/日",
    },
    {
        "product_code": "WM002",
        "name": "稳盈短债C",
        "category": "短债基金",
        "risk_level": "R2",
        "benchmark": "近1年年化 3.10%",
        "term": "30天最短持有",
        "min_amount": 1000,
        "rules": "持有满30天后可随时赎回，T+1到账",
    },
    {
        "product_code": "WM003",
        "name": "固收+精选1号",
        "category": "固收+理财",
        "risk_level": "R2",
        "benchmark": "业绩比较基准 3.5%~4.2%",
        "term": "6个月封闭期",
        "min_amount": 10000,
        "rules": "封闭期内不可赎回，到期自动滚存可修改",
    },
    {
        "product_code": "WM004",
        "name": "平衡混合先锋",
        "category": "混合基金",
        "risk_level": "R3",
        "benchmark": "近1年收益 8.5%",
        "term": "无固定期限",
        "min_amount": 1000,
        "rules": "随时可赎回，T+1到账；波动中等",
    },
    {
        "product_code": "WM005",
        "name": "成长股票精选",
        "category": "股票基金",
        "risk_level": "R4",
        "benchmark": "近1年收益 15.2%",
        "term": "无固定期限",
        "min_amount": 1000,
        "rules": "随时可赎回，T+1到账；波动较大，适合长期持有",
    },
]

FUND_UNIVERSE = {
    "110011": {
        "fund_name": "易方达优质精选混合",
        "type": "混合型",
        "nav": 2.3150,
        "daily_change": "+0.85%",
        "year_return": "12.4%",
        "subscription_fee": "1.5%（申购费，通常打1折）",
        "management_fee": "1.50%/年",
        "custody_fee": "0.25%/年",
        "size": "186.5亿",
    },
    "000001": {
        "fund_name": "华夏成长混合",
        "type": "混合型",
        "nav": 1.2840,
        "daily_change": "+0.32%",
        "year_return": "6.8%",
        "subscription_fee": "1.5%",
        "management_fee": "1.50%/年",
        "custody_fee": "0.25%/年",
        "size": "92.3亿",
    },
    "003003": {
        "fund_name": "华夏现金增利货币A",
        "type": "货币型",
        "nav": 1.0000,
        "daily_change": "+0.01%",
        "year_return": "1.9%",
        "subscription_fee": "0%",
        "management_fee": "0.33%/年",
        "custody_fee": "0.10%/年",
        "size": "420.8亿",
    },
}

INSURANCE_PLANS = {
    "重疾险": {
        "coverage": "覆盖120种重大疾病，确诊即赔，保额可选30万/50万/100万",
        "premium_example": "30岁男性50万保额，保至70岁，年缴约4800元",
        "features": "含轻症/中症赔付，豁免保费",
    },
    "医疗险": {
        "coverage": "住院医疗费用报销，年度限额200万，含社保外用药",
        "premium_example": "30岁人群年缴约300元，0免赔版本约800元",
        "features": "百万医疗，支持直付，续保至99岁",
    },
    "意外险": {
        "coverage": "意外身故/伤残最高100万，意外医疗5万，含住院津贴",
        "premium_example": "30岁人群年缴约200元",
        "features": "交通意外双倍赔付",
    },
    "寿险": {
        "coverage": "身故/全残赔付保额，定期寿险可选10/20/30年",
        "premium_example": "30岁男性100万保额30年期，年缴约2500元",
        "features": "家庭支柱必备，杠杆高",
    },
    "年金险": {
        "coverage": "按约定年龄领取生存金，可附加万能账户二次增值",
        "premium_example": "年缴1万元，缴10年，60岁起每年领取",
        "features": "锁定长期利率，适合养老规划",
    },
}


# ── Mock handlers (read-only consultation) ───────────────────────────────

def mock_query_wealth_products(args: dict) -> dict:
    """Recommend products filtered by risk level (suitability first)."""
    risk_level = args.get("risk_level", "")
    level = RISK_LEVEL_NAMES.get(risk_level)
    if level is None:
        normalized = risk_level.strip().upper()
        level = normalized if normalized in {"R1", "R2", "R3", "R4", "R5"} else None

    if level not in {"R1", "R2", "R3", "R4", "R5"}:
        return {
            "status": "need_risk_level",
            "message": "根据投资者适当性管理要求，请先完成风险测评或提供风险承受等级（保守型/稳健型/平衡型/进取型/激进型）。",
        }

    level_num = int(level[1])
    eligible = [p for p in WEALTH_PRODUCTS if int(p["risk_level"][1]) <= level_num]

    return {
        "status": "success",
        "client_risk_level": level,
        "eligible_products": eligible,
        "disclaimer": "以上产品信息仅供参考，不构成投资建议。理财非存款，产品有风险，投资须谨慎。",
    }


def mock_query_fund_detail(args: dict) -> dict:
    """Return fund NAV / fees / size by fund code."""
    fund_code = (args.get("fund_code") or "").strip()
    fund = FUND_UNIVERSE.get(fund_code)

    if not fund:
        # Loose lookup by name keyword
        for code, info in FUND_UNIVERSE.items():
            if fund_code and fund_code in info["fund_name"]:
                fund = info
                fund_code = code
                break

    if not fund:
        return {
            "status": "not_found",
            "message": f"未查询到基金代码/名称「{fund_code}」的信息，请确认后重试。",
        }

    return {
        "status": "success",
        "fund_code": fund_code,
        **fund,
        "disclaimer": "基金净值数据仅供参考，过往业绩不代表未来表现。",
    }


def mock_query_insurance_plans(args: dict) -> dict:
    """Return insurance coverage / premium by insurance type."""
    insurance_type = (args.get("insurance_type") or "").strip()
    plan = INSURANCE_PLANS.get(insurance_type)

    if not plan:
        return {
            "status": "need_type",
            "message": "请提供险种名称（重疾险/医疗险/意外险/寿险/年金险）。",
        }

    return {
        "status": "success",
        "insurance_type": insurance_type,
        **plan,
        "disclaimer": "保费为示例测算，实际以投保时核保结果为准。",
    }


def mock_query_portfolio(args: dict) -> dict:
    """Return mock holdings for the current user."""
    return {
        "status": "success",
        "holdings": [
            {
                "product_name": "安心货币A",
                "product_code": "WM001",
                "shares": "50,000份",
                "cost": "50,000元",
                "market_value": "50,092元",
                "profit": "+92元 (+0.18%)",
            },
            {
                "product_name": "稳盈短债C",
                "product_code": "WM002",
                "shares": "20,000份",
                "cost": "20,000元",
                "market_value": "20,560元",
                "profit": "+560元 (+2.80%)",
            },
            {
                "product_name": "平衡混合先锋",
                "product_code": "WM004",
                "shares": "10,000份",
                "cost": "10,000元",
                "market_value": "10,850元",
                "profit": "+850元 (+8.50%)",
            },
        ],
        "total_cost": "80,000元",
        "total_market_value": "81,502元",
        "total_profit": "+1,502元 (+1.88%)",
        "disclaimer": "持仓数据为演示数据，仅供参考。",
    }


def _score_risk_assessment(args: dict) -> tuple[str, int]:
    """Score the 5-question questionnaire → risk level R1..R5.

    评分规则（符合投资者适当性方向）：
    - q1 年龄：**反向计分**（年龄越大风险承受越低）：A=4, B=3, C=2, D=1
    - q2-q5（经验/期限/亏损承受/收入来源）：A=1, B=2, C=3, D=4
    总分映射 R1（保守）.. R5（激进）。
    """
    q1_map = {"a": 4, "b": 3, "c": 2, "d": 1, "A": 4, "B": 3, "C": 2, "D": 1}
    q_map = {
        "a": 1, "b": 2, "c": 3, "d": 4,
        "A": 1, "B": 2, "C": 3, "D": 4,
    }
    total = 0
    for key in ("q1", "q2", "q3", "q4", "q5"):
        answer = str(args.get(key, "")).strip()
        if key == "q1":
            total += q1_map.get(answer, 0)
        else:
            total += q_map.get(answer, 0)

    if total <= 7:
        return "R1", total
    if total <= 11:
        return "R2", total
    if total <= 14:
        return "R3", total
    if total <= 17:
        return "R4", total
    return "R5", total


def mock_run_risk_assessment(args: dict) -> dict:
    """Score the questionnaire and return suitable product level."""
    level, total = _score_risk_assessment(args)
    level_num = int(level[1])
    eligible = [p for p in WEALTH_PRODUCTS if int(p["risk_level"][1]) <= level_num]

    return {
        "status": "success",
        "risk_level": level,
        "risk_level_name": next((name for name, lv in RISK_LEVEL_NAMES.items() if lv == level), level),
        "score": total,
        "eligible_products": [p["name"] for p in eligible],
        "message": f"您的风险承受等级为{level}（{next((n for n, l in RISK_LEVEL_NAMES.items() if l == level), '')}）。"
                   "根据适当性管理要求，可购买风险等级不高于您承受等级的产品。",
        "disclaimer": "测评结果仅供投资参考，不构成投资建议。",
    }


def mock_calc_expected_return(args: dict) -> dict:
    """Estimate expected return for amount × term (illustrative only)."""
    amount_raw = args.get("amount", "")
    try:
        amount = float(str(amount_raw).replace(",", "").replace("万", "").replace("元", "").strip())
        if "亿" in str(amount_raw):
            amount *= 100_000_000
        elif "万" in str(amount_raw):
            amount *= 10000
    except (TypeError, ValueError):
        amount = 0.0

    term_text = args.get("term") or "1年"
    # Parse term to years (default 1 year)
    import re
    term_match = re.search(r"(\d+(?:\.\d+)?)\s*(年|个月|月|天)", term_text)
    if term_match:
        value = float(term_match.group(1))
        unit = term_match.group(2)
        if unit in ("年",):
            years = value
        elif unit in ("个月", "月"):
            years = value / 12
        else:
            years = value / 365
    else:
        years = 1.0

    rate_text = args.get("rate") or "3.5%"
    rate_match = re.search(r"(\d+(?:\.\d+)?)\s*%", rate_text)
    rate = float(rate_match.group(1)) / 100 if rate_match else 0.035

    if amount <= 0:
        return {
            "status": "need_amount",
            "message": "请提供投资金额，例如：10万。",
        }

    # 参数合理性校验（合规）：利率 0~20%、期限 ≤30 年、金额 ≤1 亿
    if amount > 100_000_000:
        return {
            "status": "invalid_amount",
            "message": "投资金额超出测算上限（1亿元），请确认金额后重试。",
        }
    if rate <= 0 or rate > 0.20:
        return {
            "status": "invalid_rate",
            "message": "年化收益率超出合理范围（0~20%），请以官方产品资料为准。",
        }
    if years <= 0 or years > 30:
        return {
            "status": "invalid_term",
            "message": "投资期限超出合理范围（30年以内），请确认期限后重试。",
        }

    expected = amount * rate * years
    return {
        "status": "success",
        "amount": f"{amount:,.0f}元",
        "term": term_text,
        "annual_rate": f"{rate * 100:.2f}%",
        "expected_return": f"{expected:,.2f}元",
        "message": f"按年化{rate * 100:.2f}%测算，投资{amount:,.0f}元{term_text}的预期收益约为{expected:,.2f}元。",
        "disclaimer": "以上为模拟测算，不构成收益承诺。理财产品过往业绩不代表未来表现，市场有风险，投资须谨慎。",
    }


def create_default_tool_registry() -> ToolRegistry:
    """Create and populate registry with all financial consultation tools."""
    registry = ToolRegistry()
    for tool in DEFAULT_TOOLS:
        registry.register(tool)
    return registry


DEFAULT_TOOLS: list[ToolDefinition] = [
    ToolDefinition(
        name="query_wealth_products",
        intent="wealth_query",
        description="根据风险承受等级查询可购买的理财产品",
        required_slots=["risk_level"],
        handler=mock_query_wealth_products,
    ),
    ToolDefinition(
        name="query_fund_detail",
        intent="fund_query",
        description="查询基金净值、费率、规模等信息",
        required_slots=["fund_code"],
        handler=mock_query_fund_detail,
    ),
    ToolDefinition(
        name="query_insurance_plans",
        intent="insurance_query",
        description="查询保险险种保障范围与保费示例",
        required_slots=["insurance_type"],
        handler=mock_query_insurance_plans,
    ),
    ToolDefinition(
        name="query_portfolio",
        intent="portfolio_query",
        description="查询当前用户的持仓与收益情况",
        required_slots=[],
        handler=mock_query_portfolio,
    ),
    ToolDefinition(
        name="run_risk_assessment",
        intent="risk_assessment",
        description="根据5道问卷题目评分，输出风险承受等级与可购产品建议",
        required_slots=["q1", "q2", "q3", "q4", "q5"],
        handler=mock_run_risk_assessment,
    ),
    ToolDefinition(
        name="calc_expected_return",
        intent="yield_calc",
        description="按金额、期限与年化收益率测算预期收益",
        required_slots=["amount"],
        handler=mock_calc_expected_return,
    ),
]
