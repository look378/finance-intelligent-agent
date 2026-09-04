"""
Rule-based intent detector using keyword matching for wealth-management intents.

Fast, lightweight intent detection for financial customer service scenarios.
"""
import re
from typing import Optional, Dict, List

from app.services.intent.base import IntentDetector, IntentResult
from app.models.enums.intent import Intent


class RuleBasedIntentDetector(IntentDetector):
    """
    Rule-based intent detector using keyword matching for business intents.

    Maps wealth-management customer service keywords to business intent
    categories: task-oriented (wealth/fund/insurance/portfolio/risk/yield),
    knowledge (faq, policy), dialogue (chitchat, greeting),
    and meta intents (confirm, deny, cancel).
    """

    def __init__(self) -> None:
        self.rules = self._build_rules()

    def _build_rules(self) -> Dict[Intent, List[dict]]:
        return {
            # Task-oriented intents (read-only consultation & query)
            Intent.WEALTH_QUERY: [
                {"keywords": [
                    "理财", "理财产品", "买什么理财", "推荐理财", "财富管理",
                    "wealth", "固收", "结构性存款", "大额存单", "稳健理财",
                ], "weight": 1.5},
                {"patterns": [
                    r"(理财|财富).{0,6}(推荐|选择|买|产品|哪个)",
                    r"(理财|基金).{0,4}(收益|怎么样|好不好|靠谱)",
                ], "weight": 1.3},
            ],
            Intent.FUND_QUERY: [
                {"keywords": [
                    "基金", "基金净值", "申购", "赎回", "基金代码", "fund",
                    "定投", "指数基金", "货币基金", "ETF",
                ], "weight": 1.5},
                {"patterns": [
                    r"(基金|场内).{0,4}(净值|费率|规模|涨|跌)",
                    r"(申购|赎回).{0,4}(费|规则|时间)",
                ], "weight": 1.3},
            ],
            Intent.INSURANCE_QUERY: [
                {"keywords": [
                    "保险", "重疾险", "医疗险", "意外险", "寿险", "年金险",
                    "insurance", "保费", "保障", "保额",
                ], "weight": 1.5},
                {"patterns": [
                    r"(保险|重疾|医疗|意外|寿险|年金).{0,4}(买|保|赔|保障|多少钱)",
                ], "weight": 1.3},
            ],
            Intent.PORTFOLIO_QUERY: [
                {"keywords": [
                    "持仓", "我的理财", "我的基金", "收益", "盈亏", "市值",
                    "portfolio", "持有", "份额", "浮盈", "浮亏",
                ], "weight": 1.5},
                {"patterns": [
                    r"(持仓|持有|收益|盈亏).{0,4}(多少|查询|怎么样)",
                    r"我.{0,6}(买了|持有).{0,6}(理财|基金)",
                ], "weight": 1.3},
            ],
            Intent.RISK_ASSESSMENT: [
                {"keywords": [
                    "风险测评", "风险承受", "风险评估", "测评", "风险等级",
                    "risk assessment", "风险偏好",
                ], "weight": 1.5},
                {"patterns": [
                    r"(风险|承受能力|风险偏好).{0,4}(测评|评估|测试|等级)",
                ], "weight": 1.3},
            ],
            Intent.YIELD_CALC: [
                {"keywords": [
                    "收益测算", "收益计算", "算一下收益", "收益率", "年化",
                    "yield", "测算", "预期收益",
                ], "weight": 1.5},
                {"patterns": [
                    r"(投|买|放).{0,6}(\d+万?|金额).{0,6}(收益|利息|赚)",
                    r"(\d+万?).{0,4}(一年|两年|三年).{0,4}(收益|利息|赚)",
                ], "weight": 1.3},
            ],
            # Knowledge intents
            Intent.FAQ: [
                {"keywords": ["怎么", "如何", "为什么", "能不能", "可以", "是否"], "weight": 0.5},
                {"patterns": [
                    r"(怎么|如何).{1,10}(买|开户|赎回|定投|查)",
                    r"(费率|手续费|管理费|托管费|销售服务费).{0,4}(多少|怎么算)",
                ], "weight": 1.2},
            ],
            Intent.POLICY: [
                {"keywords": ["政策", "规定", "规则", "条款", "监管", "协议", "合规"], "weight": 1.0},
                {"patterns": [
                    r"(赎回|申购|定投|购买).{0,6}(规则|规定|政策|条件)",
                    r"(政策|规定|规则|条件).{0,4}(是什么|有哪些|怎么样)",
                    r"(适当性|投资者保护|风险揭示).{0,6}(要求|规定)",
                ], "weight": 2.0},
            ],
            # Dialogue intents
            Intent.CHITCHAT: [
                {"keywords": [
                    "哈哈", "呵呵", "搞笑", "笑话", "无聊", "天气",
                    "周末", "吃什么", "推荐电影", "玩游戏",
                ], "weight": 0.8},
            ],
            Intent.GREETING: [
                {"keywords": [
                    "你好", "在吗", "hello", "hi", "hey",
                    "您好", "早上好", "下午好", "晚上好",
                    "good morning", "good afternoon",
                ], "weight": 1.5},
                {"patterns": [r"^(你好|您好|hi|hello|hey)[!！。]*$"], "weight": 1.5},
            ],
            # Meta intents
            Intent.CONFIRM: [
                {"keywords": ["是的", "对", "确认", "好的", "没错", "正确", "可以", "继续", "yes", "确定", "要的"], "weight": 1.5},
                {"patterns": [r"^(是的|对|好的|确认|没错|可以|继续|要的)[！!。]*$"], "weight": 1.3},
            ],
            Intent.DENY: [
                {"keywords": ["不是", "不对", "不要", "不行", "不可以", "no", "没有", "不是的"], "weight": 1.5},
                {"patterns": [r"^(不是|不对|不要|不行|没有)[！!。]*$"], "weight": 1.3},
            ],
            Intent.CANCEL: [
                {"keywords": ["取消", "算了", "不要了", "cancel", "不办了", "算了不做了"], "weight": 1.5},
                {"patterns": [r"^(取消|算了|不要了|不办了)[！!。]*$"], "weight": 1.3},
            ],
        }

    def detect(
        self,
        query: str,
        context: Optional[dict] = None,
    ) -> Intent:
        result = self.detect_with_confidence(query, context)
        return result.intent

    def detect_with_confidence(
        self,
        query: str,
        context: Optional[dict] = None,
    ) -> IntentResult:
        if not query or not query.strip():
            return IntentResult(intent=Intent.UNKNOWN, confidence=0.0)

        normalized_query = self._normalize_query(query)
        scores: Dict[Intent, float] = {intent: 0.0 for intent in Intent}
        matched_rules: list = []

        for intent, rules in self.rules.items():
            for rule in rules:
                if "keywords" in rule:
                    matched = self._contains_any(normalized_query, rule["keywords"])
                    if matched:
                        scores[intent] += rule["weight"]
                        matched_rules.append({
                            "intent": intent.value,
                            "rule_type": "keyword",
                            "weight": rule["weight"],
                        })

                if "patterns" in rule:
                    for pattern in rule["patterns"]:
                        if re.search(pattern, normalized_query, re.IGNORECASE):
                            scores[intent] += rule["weight"]
                            matched_rules.append({
                                "intent": intent.value,
                                "rule_type": "pattern",
                                "pattern": pattern,
                                "weight": rule["weight"],
                            })

        max_intent = Intent.UNKNOWN
        max_score = 0.0

        for intent, score in scores.items():
            if score > max_score:
                max_score = score
                max_intent = intent

        confidence = min(max_score / 3.0, 1.0) if max_score > 0 else 0.0

        return IntentResult(
            intent=max_intent,
            confidence=confidence,
            metadata={
                "matched_rules": matched_rules,
                "raw_score": max_score,
            } if matched_rules else None,
        )

    @staticmethod
    def _normalize_query(query: str) -> str:
        return query.lower().strip()

    @staticmethod
    def _contains_any(text: str, keywords: list) -> bool:
        return any(kw.lower() in text for kw in keywords)
