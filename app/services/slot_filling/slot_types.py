"""
Slot schemas per business intent (wealth management).

Defines required/optional slots for each task-oriented intent,
with extraction prompts for missing slot values.
"""
from typing import Dict, Any

# Per-intent slot definitions for task-oriented flows
INTENT_SLOT_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "wealth_query": {
        "required": ["risk_level"],
        "optional": ["amount", "term"],
        "slots": {
            "risk_level": {
                "type": "string",
                "prompt": "根据投资者适当性管理要求，请问您的风险承受能力等级是？（保守型/稳健型/平衡型/进取型/激进型）",
                "patterns": [
                    r"(保守型|稳健型|平衡型|进取型|激进型)",
                    r"风险(?:承受)?(?:能力|等级)?[是为：:\s]*(保守|稳健|平衡|进取|激进)",
                    r"(R[1-5])",
                ],
            },
            "amount": {
                "type": "number",
                "prompt": "请问您计划投资的金额是多少？",
                "patterns": [
                    r"(\d+(?:\.\d+)?)\s*(万|万元|元)",
                    r"金额[是为：:\s]*(\d+(?:\.\d+)?)",
                ],
            },
            "term": {
                "type": "string",
                "prompt": "请问您的投资期限是多久？（如：3个月/1年/3年）",
                "patterns": [
                    r"(\d+(?:\.\d+)?)\s*(天|个月|月|年)",
                    r"期限[是为：:\s]*(.{1,10})",
                ],
            },
        },
    },
    "fund_query": {
        "required": ["fund_code"],
        "optional": [],
        "slots": {
            "fund_code": {
                "type": "string",
                "prompt": "请提供基金代码或基金名称",
                "patterns": [
                    r"(\d{6})",
                    r"(?:基金代码|代码)[：:]?\s*([A-Za-z0-9]+)",
                    r"(基金|名称)[为是：:\s]*([\u4e00-\u9fa5A-Za-z0-9]{2,20})",
                ],
            },
        },
    },
    "insurance_query": {
        "required": ["insurance_type"],
        "optional": ["age", "budget"],
        "slots": {
            "insurance_type": {
                "type": "string",
                "prompt": "请问您想了解哪类保险？（重疾险/医疗险/意外险/寿险/年金险）",
                "patterns": [
                    r"(重疾险|医疗险|意外险|寿险|年金险|重疾|医疗|意外|寿险|年金)",
                    r"保险[为是：:\s]*([\u4e00-\u9fa5]{2,6})",
                ],
            },
            "age": {
                "type": "number",
                "prompt": "请问被保人的年龄是多少？",
                "patterns": [
                    r"(\d{1,3})\s*岁",
                    r"年龄[为是：:\s]*(\d{1,3})",
                ],
            },
            "budget": {
                "type": "number",
                "prompt": "请问您的年预算保费大概是多少？",
                "patterns": [
                    r"(\d+(?:\.\d+)?)\s*(万|万元|元)",
                    r"预算[为是：:\s]*(\d+(?:\.\d+)?)",
                ],
            },
        },
    },
    "portfolio_query": {
        "required": [],
        "optional": ["product_name"],
        "slots": {
            "product_name": {
                "type": "string",
                "prompt": "请问您想查询哪只产品的持仓？",
            },
        },
    },
    "risk_assessment": {
        "required": ["q1", "q2", "q3", "q4", "q5"],
        "optional": [],
        "slots": {
            "q1": {
                "type": "string",
                "prompt": "风险测评第1题：请问您的年龄段是？（A.18-30岁 B.31-50岁 C.51-65岁 D.65岁以上）",
                "patterns": [
                    r"([A-Da-d])",
                    r"(18[-—]?30|31[-—]?50|51[-—]?65|65以上)",
                ],
            },
            "q2": {
                "type": "string",
                "prompt": "风险测评第2题：您的投资经验是？（A.无经验 B.不足1年 C.1-3年 D.3年以上）",
                "patterns": [
                    r"([A-Da-d])",
                    r"(无经验|不足1年|1-3年|3年以上|一年|三年)",
                ],
            },
            "q3": {
                "type": "string",
                "prompt": "风险测评第3题：您的投资期限偏好是？（A.1年以内 B.1-3年 C.3-5年 D.5年以上）",
                "patterns": [
                    r"([A-Da-d])",
                    r"(1年以内|1-3年|3-5年|5年以上)",
                ],
            },
            "q4": {
                "type": "string",
                "prompt": "风险测评第4题：如果投资出现10%的亏损，您会？（A.立即赎回 B.部分赎回 C.继续持有 D.加仓）",
                "patterns": [
                    r"([A-Da-d])",
                    r"(立即赎回|部分赎回|继续持有|加仓)",
                ],
            },
            "q5": {
                "type": "string",
                "prompt": "风险测评第5题：您的家庭主要收入来源是？（A.工资收入 B.经营收入 C.投资收益 D.其他）",
                "patterns": [
                    r"([A-Da-d])",
                    r"(工资|经营|投资|其他)",
                ],
            },
        },
    },
    "yield_calc": {
        "required": ["amount"],
        "optional": ["term", "rate"],
        "slots": {
            "amount": {
                "type": "number",
                "prompt": "请问您计划投资的金额是多少？",
                "patterns": [
                    r"(\d+(?:\.\d+)?)\s*(万|万元|元)",
                    r"金额[为是：:\s]*(\d+(?:\.\d+)?)",
                ],
            },
            "term": {
                "type": "string",
                "prompt": "请问投资期限是多久？（如：3个月/1年/3年）",
                "patterns": [
                    r"(\d+(?:\.\d+)?)\s*(天|个月|月|年)",
                ],
            },
            "rate": {
                "type": "string",
                "prompt": "请问预期年化收益率大概是多少？（如：3.5%）",
                "patterns": [
                    r"(\d+(?:\.\d+)?)\s*%",
                ],
            },
        },
    },
}


def get_missing_slots(intent: str, filled: Dict[str, Any]) -> list[str]:
    """Return list of required slot names not yet filled."""
    schema = INTENT_SLOT_SCHEMAS.get(intent, {})
    required = schema.get("required", [])
    return [s for s in required if s not in filled]


def get_next_prompt(intent: str, filled: Dict[str, Any]) -> str | None:
    """Get the prompt for the next missing required slot."""
    schema = INTENT_SLOT_SCHEMAS.get(intent, {})
    missing = get_missing_slots(intent, filled)
    if not missing:
        return None
    slots = schema.get("slots", {})
    return slots.get(missing[0], {}).get("prompt", f"请提供{missing[0]}")


def extract_slots_from_message(intent: str, message: str, existing: Dict[str, Any]) -> Dict[str, Any]:
    """Extract slot values from user message using regex patterns."""
    import re

    schema = INTENT_SLOT_SCHEMAS.get(intent, {})
    slots = schema.get("slots", {})
    extracted = dict(existing)

    # 问卷类意图（risk_assessment）：每个槽位对应一道题，按顺序逐题作答。
    # 只允许"回答当前缺失的第一题"（取消息中第一个字母选项），
    # 避免一条消息同时填满 q1..q5 导致问卷被静默跳过。
    if intent == "risk_assessment":
        missing = [s for s in schema.get("required", []) if s not in extracted]
        if not missing:
            return extracted
        current_slot = missing[0]
        slot_def = slots.get(current_slot, {})
        patterns = slot_def.get("patterns", [])
        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                try:
                    extracted[current_slot] = match.group(1)
                except IndexError:
                    extracted[current_slot] = match.group(0)
                break
        return extracted

    for slot_name, slot_def in slots.items():
        if slot_name in extracted:
            continue
        patterns = slot_def.get("patterns", [])
        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                try:
                    extracted[slot_name] = match.group(1)
                except IndexError:
                    extracted[slot_name] = match.group(0)
                break

    return extracted
