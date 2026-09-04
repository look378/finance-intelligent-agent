#!/usr/bin/env python
"""
Initialize Qdrant knowledge base with financial wealth-management demo documents.

Run from the project root:
    python scripts/init_kb.py

Uses the application's embedding service (BGE-M3 local by default) and the
application's async Qdrant client so the demo knowledge base matches the
runtime vector configuration.
"""
import asyncio
import sys
import os
import uuid
from pathlib import Path

# Allow running as a standalone script from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.embeddings import EmbeddingFactory  # noqa: E402
from app.services.retrieval.qdrant_client import QdrantClient  # noqa: E402
from app.services.retrieval.vector_base import Document  # noqa: E402

COLLECTION_NAME = "documents"

# ── Financial demo knowledge base ─────────────────────────────────────────
FINANCIAL_DOCS = [
    {
        "title": "安心货币A产品说明",
        "category": "产品说明",
        "content": (
            "安心货币A是一款货币市场基金，风险等级R1（低风险）。"
            "业绩比较基准为7日年化收益率约1.85%。"
            "起购金额100元，无固定期限。"
            "支持T+1快速赎回，单日快速赎回上限1万元。"
            "申购费率0%，管理费率0.33%/年，托管费率0.10%/年。"
        ),
    },
    {
        "title": "稳盈短债C产品说明",
        "category": "产品说明",
        "content": (
            "稳盈短债C是一款短期债券基金，风险等级R2（中低风险）。"
            "近1年年化收益率约3.10%。"
            "起购金额1000元，最短持有期30天。"
            "持有满30天后可随时赎回，赎回资金T+1到账。"
        ),
    },
    {
        "title": "固收+精选1号产品说明",
        "category": "产品说明",
        "content": (
            "固收+精选1号是一款固收+理财产品，风险等级R2（中低风险）。"
            "业绩比较基准为年化3.5%至4.2%。"
            "起购金额1万元，封闭期6个月。"
            "封闭期内不可赎回，到期可选择自动滚存或修改。"
        ),
    },
    {
        "title": "平衡混合先锋基金说明",
        "category": "产品说明",
        "content": (
            "平衡混合先锋是一款混合型基金，风险等级R3（中风险）。"
            "近1年收益约8.5%。"
            "起购金额1000元，无固定期限，随时可赎回，赎回T+1到账。"
            "股票和债券资产灵活配置，波动中等。"
        ),
    },
    {
        "title": "成长股票精选基金说明",
        "category": "产品说明",
        "content": (
            "成长股票精选是一款股票型基金，风险等级R4（中高风险）。"
            "近1年收益约15.2%。"
            "起购金额1000元，无固定期限。"
            "主要投资于成长型股票，波动较大，适合长期持有。"
        ),
    },
    {
        "title": "重疾险保障方案",
        "category": "保险",
        "content": (
            "重疾险覆盖120种重大疾病，确诊即赔，保额可选30万、50万、100万。"
            "示例保费：30岁男性50万保额保至70岁，年缴约4800元。"
            "包含轻症、中症赔付责任，被保险人确诊重疾后豁免后续保费。"
        ),
    },
    {
        "title": "医疗险保障方案",
        "category": "保险",
        "content": (
            "医疗险提供住院医疗费用报销，年度限额200万元，包含社保外用药。"
            "示例保费：30岁人群年缴约300元，0免赔版本约800元。"
            "支持直付服务，可续保至99岁。"
        ),
    },
    {
        "title": "投资者适当性管理规则",
        "category": "监管政策",
        "content": (
            "根据投资者适当性管理要求，向投资者销售金融产品前，"
            "应当进行风险承受能力评估。"
            "投资者风险承受等级分为保守型（R1）、稳健型（R2）、平衡型（R3）、"
            "进取型（R4）、激进型（R5）五档。"
            "投资者只能购买风险等级不高于其自身承受等级的产品。"
            "销售过程不得承诺收益，不得使用保本、稳赚等违规表述。"
        ),
    },
    {
        "title": "理财非存款风险提示",
        "category": "监管政策",
        "content": (
            "理财非存款，产品有风险，投资须谨慎。"
            "理财产品过往业绩不代表未来表现，业绩比较基准不构成收益承诺。"
            "投资者应仔细阅读产品说明书和风险揭示书，根据自身风险承受能力"
            "审慎决策。基金净值可能波动，投资可能产生亏损。"
        ),
    },
    {
        "title": "基金赎回规则",
        "category": "业务规则",
        "content": (
            "货币基金支持T+1快速赎回，单日快速赎回上限1万元，超过部分T+1普通赎回。"
            "债券基金、混合基金、股票基金赎回资金T+1到账。"
            "封闭期内产品不可赎回，到期后可选择滚存或退出。"
            "赎回到账时间以产品合同约定为准。"
        ),
    },
    {
        "title": "基金申购费率规则",
        "category": "业务规则",
        "content": (
            "混合型基金申购费率通常为1.5%，多数销售渠道可享1折优惠。"
            "货币基金申购费率为0%。"
            "基金管理费按日计提，混合基金约1.50%/年，货币基金约0.33%/年。"
            "托管费由托管银行收取，混合基金约0.25%/年。"
        ),
    },
    {
        "title": "年金险养老规划",
        "category": "保险",
        "content": (
            "年金险按约定年龄领取生存金，可附加万能账户实现二次增值。"
            "示例方案：年缴1万元，缴费10年，60岁起每年领取生存金。"
            "年金险适合有养老规划需求的客户，可锁定长期利率。"
        ),
    },
]


async def main() -> None:
    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    embedding_service = EmbeddingFactory.create_from_settings(use_cache=False)

    # 确保 collection 存在（Qdrant 不会自动创建）
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.models import Distance, VectorParams

    qdrant = AsyncQdrantClient(url=qdrant_url)
    collections = await qdrant.get_collections()
    existing = {c.name for c in collections.collections}
    if COLLECTION_NAME not in existing:
        await qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=embedding_service.dimensions,
                distance=Distance.COSINE,
            ),
        )
        print(f"Created collection '{COLLECTION_NAME}' (dim={embedding_service.dimensions})")
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists")
    await qdrant.close()

    client = QdrantClient(
        url=qdrant_url,
        collection_name=COLLECTION_NAME,
        embedding_service=embedding_service,
    )

    print(f"Loading {len(FINANCIAL_DOCS)} financial documents...")

    documents = []
    for doc in FINANCIAL_DOCS:
        documents.append(
            Document(
                id=str(uuid.uuid4()),
                content=doc["content"],
                metadata={
                    "title": doc["title"],
                    "category": doc["category"],
                    "source": "financial_demo_kb",
                },
            )
        )

    print("Upserting documents into Qdrant...")
    doc_ids = await client.add_documents(documents)
    print(f"Done! Inserted {len(doc_ids)} documents into '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    asyncio.run(main())
