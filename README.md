<div align="center">

# Finance Intelligent Agent｜金融智能客服

**一款由 LangGraph 驱动的生产级金融财富管理智能客服 —— 9节点对话图、函数调用、混合 RAG 和 GraphRAG 知识检索。**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-FF6B6B)](https://github.com/langchain-ai/langgraph)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

<!-- 🎬 录制说明:用 lic_ecap/kap 录 30 秒对话演示,放到 docs/assets/demo.gif -->
<!--     录制内容:输入"我想买理财" → 多轮槽位收集(风险等级) → 工具推荐产品 → 风险提示 -->
<img src="docs/assets/demo.gif" alt="Finance Chatbot Demo" width="80%">

*🎬 Replace this with a 30s GIF of the dialogue flow — see [Recording Guide](#-demo-recording-guide) below*

</div>

---

## 📌 Table of Contents

- [Why This Project](#-why-this-project)
- [Key Highlights](#-key-highlights)
- [How It Works](#-how-it-works)
- [Quick Start](#-quick-start)
- [Live Demo](#-live-demo)
- [Architecture](#-architecture)
- [Configuration](#-configuration)
- [Testing & Quality](#-testing--quality)
- [Roadmap](#-roadmap)
- [中文说明](#-中文说明)

---

## 💡 Why This Project

Most "RAG chatbot" tutorials stop at a single vector search call. **Real financial customer service is much harder**:

- ❌ Users interrupt midway ("wait, what about the risk level?") then expect to resume
- ❌ Consultation tasks need multi-turn slot collection (risk tolerance, amount, term...)
- ❌ You need function calling to actually query products, holdings, and calculate returns — not just chat
- ❌ Knowledge questions (product terms, policies) and task questions need different handling paths
- ❌ PII leakage, prompt injection, and non-compliant return promises (保本/稳赚) are serious compliance risks

This project solves all of them with a **LangGraph StateGraph** — the same architecture used by enterprises running mission-critical dialogue systems. It's not a demo; it's a reference implementation you can learn from and extend.

> 💬 **What you'll learn**: how to structure a multi-node dialogue graph, design financial intent routing with task resume, integrate hybrid retrieval (vector + GraphRAG), and enforce financial compliance (PII masking, return-promise detection, risk disclaimers) in production.

---

## ✨ Key Highlights

<div align="center">

| 🎯 Dialogue Engine | 🔧 Function Calling | 📚 RAG |
|:---:|:---:|:---:|
| **9-node** StateGraph | **6** financial tools | Hybrid + GraphRAG |
| Intent switch & resume | ToolRegistry pattern | Vector + BM25 + RRF |
| Slot filling (risk/amount/term) | Read-only consultation | Cross-Encoder reranking |

| 🛡️ Compliance & Safety | 📊 Observability | 🚀 Production |
|:---:|:---:|:---:|
| Input/Output guardrails | OpenTelemetry tracing | Docker Compose |
| Financial PII masking (Luhn) | Prometheus + Grafana | K8s manifests |
| Return-promise detection | Audit logging | Health checks |

| 📈 Stats | | |
|:---:|:---:|:---:|
| **104+** test cases | **3** LLM providers | **80%+** coverage |
| **9** dialogue nodes | **4** data stores | **4** memory strategies |

</div>

### 🧠 What makes it different

1. **LangGraph StateGraph, not a chain** — 9 nodes with conditional edges, checkpointed for resume
2. **Intent switch with state stack** — push/pop pattern to save & restore in-flight tasks
3. **Tri-route dispatch** — task → tool execution / RAG → retrieval / direct → LLM
4. **Financial compliance built-in** — Luhn-validated bank-card masking, return-promise detection, mandatory risk disclaimers
5. **Hybrid retrieval** — vector (Qdrant) + BM25 keyword → RRF fusion → Cross-Encoder rerank
6. **GraphRAG (optional)** — Neo4j financial knowledge graph + Text-to-Cypher + community detection
7. **Audit trail** — structured audit logging for every tool execution and guardrail event

---

## 🔄 How It Works

A user says *"我想买理财"*. Here's what happens:

```
1. guardrail        → scan for prompt injection, redact financial PII (Luhn-validated)
2. detect_intent    → classify: wealth_query (task) | policy (rag) | chitchat (direct)
3. handle_switch    → if intent changed, push current state to stack
4. route_intent     → dispatch to the right branch
   ├── task → collect_slots → execute_tool → generate_response
   ├── rag   → rag_lookup   → generate_response
   └── direct → direct_response
5. checkpoint       → save state by session_id (resume on next turn)
```

### Intent Switch & Resume — the killer feature

```
User: "我想买理财"
→ wealth_query, filled={}, pending=[risk_level]
→ "根据投资者适当性管理要求，请问您的风险承受能力等级是？（保守型/稳健型/平衡型/进取型/激进型）"

User: "我是稳健型"
→ wealth_query, filled={risk_level:"稳健型"}, pending=[]
→ execute_tool: query_wealth_products → 推荐R1/R2级产品列表（含费率/期限/申赎规则）
   + 风险提示: "理财非存款，产品有风险，投资须谨慎"

User: "那基金怎么查净值？"            ← 🔄 intent switched!
→ push wealth_query state to stack
→ fund_query → RAG/工具查询
→ "请提供基金代码" ...

User: "继续理财，10万买一年呢？"
→ pop stack → resume wealth_query + filled={risk_level:"稳健型"}
→ collect_slots: amount=10万 → execute_tool: calc_expected_return
→ "按年化3.5%测算，预期收益约3,500元" + 免责声明
```

### Financial Compliance Built-in

```
Input:  "帮我转账到6222021234567890123"
→ input_guardrail: Luhn 校验卡号 → 脱敏（full/partial mask）

Output: "这款产品保本稳赚！"
→ output_guardrail: 检测到承诺收益 → 拦截并返回合规提示

Output: "该理财产品预期年化3.5%..."
→ output_guardrail: 投资相关 → 自动附加风险提示与免责声明
```

### 🧠 What makes it different — Financial Wealth Management Domain

The dialogue engine is configured for **financial wealth-management customer service**:

- **Task intents (read-only consultation)**: `wealth_query` 理财咨询 / `fund_query` 基金查询 / `insurance_query` 保险咨询 / `portfolio_query` 持仓查询 / `risk_assessment` 风险测评 / `yield_calc` 收益测算
- **Knowledge intents**: `policy` 产品条款与监管政策 / `faq` 常见问题
- **No capital-movement operations** — all tools are information & consultation services (product lookup, NAV, insurance plans, holdings, risk scoring, return calculation)
- **Suitability first** — wealth product recommendations require the user's risk level (适当性管理), enforced by slot filling
- **Compliance** — return promises are blocked, risk disclaimers are auto-appended

```python
# app/services/dialogue/tools.py — 6 financial tools
DEFAULT_TOOLS = [
    ToolDefinition(name="query_wealth_products", intent="wealth_query", ...),
    ToolDefinition(name="query_fund_detail", intent="fund_query", ...),
    ToolDefinition(name="query_insurance_plans", intent="insurance_query", ...),
    ToolDefinition(name="query_portfolio", intent="portfolio_query", ...),
    ToolDefinition(name="run_risk_assessment", intent="risk_assessment", ...),
    ToolDefinition(name="calc_expected_return", intent="yield_calc", ...),
]
```

---

## 🚀 Quick Start

### Option 1: Docker Compose (recommended, ~2 min)

```bash
git clone https://github.com/look378/finance-intelligent-agent.git
cd finance-intelligent-agent

# Configure environment
cp .env.example .env
# Edit .env: set DEEPSEEK_API_KEY (or OPENAI_API_KEY / ANTHROPIC_API_KEY)

# Start core services (API + PostgreSQL + Redis + Qdrant)
docker-compose up -d

# Or with monitoring stack (Prometheus + Grafana)
docker-compose --profile monitoring up -d

# Verify
curl http://localhost:8000/health
# → {"status":"healthy","database":"connected","redis":"connected",...}
```

### Option 2: Try the Live Demo first

Don't want to deploy? 在本地启动后访问 http://localhost:8000 即可体验（无需注册）。

> 🔑 **Note on API keys**: DeepSeek is recommended for Chinese workloads (best cost/quality ratio). The system auto-falls back: DeepSeek → OpenAI → Anthropic. Embeddings default to local BGE-M3 (free, no API key needed).

### Option 3: Local development

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Start PostgreSQL + Redis + Qdrant via docker-compose
docker-compose up -d postgres redis qdrant

# Run migrations & dev server
alembic upgrade head
uvicorn app.main:create_app --factory --reload --host 0.0.0.0 --port 8000
```

📖 **API docs** (when running): http://localhost:8000/docs

---

## 🌐 Live Demo

本项目提供开箱即用的本地使用页面（中文聊天界面）：

> 🎯 本地启动后访问 **http://localhost:8000/** 即可使用

Try these scenarios:
- 🔄 **Intent switch**: Ask "我想买理财", answer risk tolerance, then ask "基金怎么查净值"
- 📚 **RAG retrieval**: Ask about product terms, regulatory policies, fee rules
- 🛠️ **Function Calling**: Complete a wealth consultation / risk assessment / return calc workflow

---

## 🏗️ Architecture

### Tech Stack

| Layer | Choice | Why |
|-------|--------|-----|
| **Dialogue orchestration** | LangGraph StateGraph | Conditional routing + checkpointing |
| **API framework** | FastAPI + async | High concurrency, OpenAPI built-in |
| **Vector DB** | Qdrant | Fast hybrid search, open-source |
| **Graph DB** (optional) | Neo4j | Financial knowledge graph for GraphRAG |
| **Relational DB** | PostgreSQL | Sessions, users, feedback |
| **Cache** | Redis | Rate limiting, embeddings cache |
| **LLM** | DeepSeek / OpenAI / Anthropic | Auto-fallback by API key availability |
| **Embeddings** | BGE-M3 (local) | Free, multilingual, no API key |
| **Observability** | OpenTelemetry + Prometheus + Grafana | Production-grade tracing & metrics |
| **Compliance** | Luhn PII masking + return-promise detection | Financial marketing compliance |

### LangGraph Dialogue Graph

```
START
  │
  ▼
guardrail (输入安全检查)
  │
  ▼
detect_intent (意图识别)
  │   cancel 优先 → 槽位值保持任务意图 → 元意图保持
  ▼
handle_switch (意图切换)
  │   推栈保存 / 弹栈恢复
  ▼
route_intent (路由决策)
  │
  ├── task  → collect_slots (槽位收集)
  │             │
  │             ├── complete → execute_tool → generate_response → END
  │             └── missing  → generate_response (追问) → END
  │
  ├── rag   → rag_lookup → generate_response → END
  │
  ├── direct → direct_response → END
  │
  └── meta  → generate_response → END
```

### Project Structure

```
finance_intelligent_agent/
├── app/
│   ├── api/v1/              # REST endpoints (chat, auth, feedback, docs)
│   ├── services/
│   │   ├── dialogue/        # 🎯 LangGraph engine
│   │   │   ├── graph.py     #   StateGraph construction + compile
│   │   │   ├── nodes.py     #   9 dialogue nodes + conditional edges
│   │   │   ├── state.py     #   DialogueState TypedDict
│   │   │   └── tools.py     #   ToolRegistry + 6 financial tools
│   │   ├── retrieval/       # Hybrid search + rerank + Qdrant
│   │   ├── graph/           # GraphRAG (Neo4j, financial schema, community)
│   │   ├── intent/          # Rule + LLM + hybrid intent detection
│   │   ├── slot_filling/    # Per-intent slot schemas (risk/amount/term...)
│   │   ├── guardrails/      # Input/output safety + financial compliance
│   │   ├── observability/   # Audit logging + LLM tracing
│   │   ├── memory/          # 4 memory strategies
│   │   └── llm/             # Multi-provider LLM clients
│   ├── models/              # ORM + Pydantic schemas
│   ├── repositories/        # Async data access layer
│   └── tests/               # Unit + integration + e2e
├── deploy/k8s/              # Kubernetes manifests
└── docker-compose*.yml
```

---

## ⚙️ Configuration

### Core Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `DATABASE_URL` | PostgreSQL connection string | — | ✅ |
| `REDIS_URL` | Redis connection string | — | ✅ |
| `QDRANT_URL` | Qdrant vector DB URL | — | ✅ |
| `SECRET_KEY` | JWT secret key | — | ✅ |
| `DEEPSEEK_API_KEY` | DeepSeek API key | — | (any LLM) |
| `OPENAI_API_KEY` | OpenAI API key | — | (any LLM) |
| `ANTHROPIC_API_KEY` | Anthropic API key | — | (any LLM) |
| `EMBEDDING_PROVIDER` | Embedding source | `local` | ❌ |
| `GRAPH_RAG_ENABLED` | Enable GraphRAG (needs Neo4j) | `false` | ❌ |

### Pluggable Strategies

The system is designed for swappable components:

```bash
# Intent detection: hybrid | rule_based | llm_based
INTENT_TYPE=hybrid
CONFIDENCE_THRESHOLD=0.7

# Memory: optimized | sliding_window | summarization | hybrid
MEMORY_TYPE=optimized
MEMORY_MAX_RECENT=3
MEMORY_TOKEN_BUDGET=4096

# Retrieval weights
VECTOR_WEIGHT=0.7
USE_RERANKING=true
GRAPH_RAG_FUSION_WEIGHT=0.3   # only if GraphRAG enabled

# Guardrails
GUARDRAILS_ENABLED=true
GUARDRAILS_HARDENING_ENABLED=true
```

See [.env.example](.env.example) for the full list.

---

## 🧪 Testing & Quality

```bash
# Full test suite with coverage (80% minimum enforced)
pytest

# Targeted runs
pytest app/tests/unit/services/dialogue/ -v    # LangGraph tests
pytest app/tests/unit/services/intent/ -v      # Intent detection
pytest app/tests/integration/ -v               # API integration
```

### Code Quality Gates

```bash
ruff check app/      # Lint
ruff format app/     # Format
mypy app/            # Type check
```

| Metric | Value |
|--------|-------|
| Test cases | **104+** |
| Test coverage | **80%+** (CI-enforced) |
| Python files | ~200 |
| Test files | ~58 |

---

## 🗺️ Roadmap

- [x] LangGraph StateGraph with 9 nodes + conditional routing
- [x] Intent switch & resume via state stack
- [x] Financial wealth-management intent/slot/tool chain (理财/基金/保险/持仓/风险测评/收益测算)
- [x] Financial compliance: Luhn PII masking, return-promise detection, risk disclaimers
- [x] Hybrid retrieval (vector + BM25 + rerank)
- [x] GraphRAG with Neo4j + financial knowledge-graph schema
- [x] Audit logging for tool executions and guardrail events
- [x] 104+ test cases, 80%+ coverage
- [ ] PostgresSaver checkpointing (production-grade persistence)
- [ ] Fine-tuned intent classifier (replace LLM-based with small specialized model)
- [ ] A/B testing framework for prompt variants
- [ ] Multi-tenant knowledge bases

---

## 🤝 Contributing

PRs welcome! Especially:
- 🐛 Bug fixes — please include a failing test
- ✨ New dialogue node types or slot strategies
- 📚 More GraphRAG use cases (currently: customer service KB)
- 🌍 i18n improvements

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup.

---

## 📜 License

[MIT](LICENSE) — free for personal and commercial use.

If this project helped you, please ⭐ star the repo — it helps others discover it.

---

## 📬 Contact

- 💬 **Issues**: [GitHub Issues](https://github.com/look378/finance-intelligent-agent/issues)

---

## 🇨🇳 中文说明

**金融智能客服（Finance Intelligent Agent）** — 基于 LangGraph 构建的金融财富管理智能客服对话系统。

### 核心亮点

- **LangGraph StateGraph 对话图**:9 节点编排(安全检查 → 意图识别 → 意图切换 → 路由 → 槽位/工具/RAG → 生成回复)
- **金融业务意图分类**:6 种任务型(理财咨询/基金查询/保险咨询/持仓查询/风险测评/收益测算)+ 知识型 + 对话型 + 元意图
- **多轮槽位收集**:风险承受等级、投资金额、期限、基金代码等，缺槽追问，适当性前置
- **Function Calling**:ToolRegistry + 6 个金融咨询工具（全部只读，不含资金变动操作）
- **意图切换与恢复**:State Stack 推栈保存/弹栈恢复
- **金融合规**:银行卡号 Luhn 校验脱敏、承诺收益表述检测拦截、投资回答自动附加风险提示与免责声明
- **审计日志**:工具执行与护栏事件的结构化审计记录（audit log）
- **GraphRAG**:Neo4j 金融知识图谱（产品/风险等级/费率/政策/机构/客群）+ Text-to-Cypher + 社区发现
- **混合检索**:向量 + BM25 → RRF 融合 → Cross-Encoder 重排序
- **全链路可观测**:OpenTelemetry + Prometheus + Grafana

### 快速开始

```bash
git clone https://github.com/look378/finance-intelligent-agent.git
cd finance-intelligent-agent
cp .env.example .env  # 填入 DEEPSEEK_API_KEY
docker-compose up -d
```

访问 http://localhost:8000/docs 查看 API 文档。

---

<details>
<summary>🎬 Demo Recording Guide (for maintainers)</summary>

### How to record the hero GIF

1. **Tool**: [licecap](https://www.cockos.com/licecap/) (Mac/Win, free) or [kap](https://getkap.co/) (Mac, OSS)
2. **Content** (~30s):
   - 0-5s: Type "我想买理财", show bot asking for risk tolerance (适当性管理)
   - 5-15s: Provide "稳健型", watch slot filling → product recommendation with risk disclaimer
   - 15-20s: Ask "基金怎么查净值" mid-flow (show intent switch)
   - 20-30s: Resume wealth consultation, provide amount, watch return calculation
3. **Save to**: `docs/assets/demo.gif` (keep under 5MB)
4. **Update**: Replace the placeholder `<img>` in the hero section

### Architecture diagram

Use [excalidraw](https://excalidraw.com/) (free) or [mermaid](https://mermaid.live/) to export a clean PNG of the dialogue graph, save to `docs/assets/architecture.png`.

</details>

<!--
RECORDING_TODO:
1. Record demo.gif → docs/assets/demo.gif
2. Draw architecture.png → docs/assets/architecture.png
3. Replace placeholder img tags in hero section
4. Update Live Demo URL (your real domain)
-->
