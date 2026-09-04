# 工程日志 ENGINEERING LOG

> 项目：Finance Intelligent Agent｜金融智能客服
> 定位：基于 LangGraph 的金融财富管理智能客服系统
> 本文件记录项目的开发目标、关键决策、功能模块与验证结果，作为工程可追溯性（audit trail）的一部分。

---

## 1. 项目目标

| # | 目标 | 状态 |
|---|------|------|
| 1 | 金融业务意图/槽位/工具链（财富管理，只读咨询） | ✅ 已完成 |
| 2 | 金融合规加固：PII 分级脱敏、合规表述检测、风险提示注入 | ✅ 已完成 |
| 3 | 审计日志（结构化 audit log + 对话状态透出） | ✅ 已完成 |
| 4 | 金融知识图谱 GraphRAG（schema/提取/Text-to-Cypher/语料） | ✅ 已完成 |
| 5 | 测试、README、.env.example、部署配置 | ✅ 已完成 |
| 6 | 回归验证（pytest 全量）与全链路冒烟 | ✅ 已完成 |

## 2. 关键设计决策

1. **金融子领域**：财富管理（理财/基金/保险咨询）
2. **业务规则仿真**：风险等级 R1-R5、适当性匹配、费率/期限/申赎规则贴近真实
3. **不做资金变动操作**：全部工具为信息查询与业务咨询（无转账/购买执行）
4. **合规文案**：理财非存款，产品有风险，投资须谨慎
5. **工程日志**：本文件 + 代码级结构化审计日志（`audit_log.py`）

## 3. 功能模块

### 3.1 对话引擎（LangGraph 9 节点状态机）

- `app/services/dialogue/graph.py`：StateGraph 构建与编译
- `app/services/dialogue/nodes.py`：9 个对话节点 + 条件边（安全检查 → 意图识别 → 意图切换 → 路由 → 槽位/工具/RAG → 生成）
- `app/services/dialogue/tools.py`：ToolRegistry + 6 个金融咨询工具（只读）
- `app/services/dialogue/state.py`：DialogueState 定义（含审计事件）

### 3.2 意图识别与槽位填充

- `app/services/intent/`：规则 + LLM 混合意图检测（6 任务意图 + 知识/对话/元意图）
- `app/services/slot_filling/slot_types.py`：金融槽位 schema（风险等级/金额/期限/基金代码/险种 + 风险测评问卷）

### 3.3 金融合规护栏

- `app/services/guardrails/input_guard.py`：金融 PII 识别（银行卡 Luhn 校验）、分级脱敏
- `app/services/guardrails/output_guard.py`：承诺收益表述拦截、风险提示注入
- `app/services/observability/audit_log.py`：结构化审计日志

### 3.4 检索与知识库

- `app/services/retrieval/`：混合检索（向量 + BM25 → RRF → Cross-Encoder 重排）
- `app/services/documents/`：文档预处理、分块、入库
- `app/services/graph/`：金融知识图谱 GraphRAG（Neo4j + Text-to-Cypher + 社区发现）
- `scripts/init_kb.py` / `scripts/import_kb.py`：知识库初始化与企业批量导入

### 3.5 LLM 与嵌入

- `app/services/llm/`：DeepSeek（默认）/ OpenAI / Anthropic 多提供商
- `app/services/embeddings/`：本地 BGE-M3（离线加载）

### 3.6 部署

- `docker-compose.yml` / `docker-compose.prod.yml`：开发与生产编排
- `Dockerfile`、`scripts/entrypoint.sh`、`scripts/deploy.sh`：镜像构建与一键部署

## 4. 合规文案（本工程采用）

- **风险提示**：理财非存款，产品有风险，投资须谨慎。以上内容仅供参考，不构成投资建议，实际收益以产品净值和合同条款为准。
- **拦截提示**：抱歉，根据金融营销宣传合规要求，我不能承诺或暗示保证收益。理财非存款，产品有风险，投资须谨慎。
- **收益测算免责**：以上为模拟测算，不构成收益承诺。理财产品过往业绩不代表未来表现，市场有风险，投资须谨慎。

## 5. 测试验证

- 单元测试：intent / dialogue / slot_filling / chat / graph / guardrails / llm / retrieval 等模块
- 全链路冒烟：理财咨询 → 适当性问询 → 产品推荐；风险测评逐题收集；合规拦截；RAG 检索带来源
- 全包导入扫描：132 模块全部通过

## 6. 代码审计（已修复）

- 鉴权与越权：会话归属校验、图谱端点鉴权、PII 脱敏
- 对话状态机：会话解锁、跨轮状态重置、问卷逐题收集
- 安全：Cypher 只读白名单、限流真实 IP、提示注入隔离
- 检索：文档删除修复、分块死循环防御、入库返回值
- 清理：移除未接线的遗留服务与死代码，仅保留金融业务相关

## 7. 已知遗留问题

- 环境级测试失败（memory/retrieval/auth/config/logging 等部分）源于 Python 3.13 与新版依赖兼容性
- ruff/mypy 存量风格与类型错误（约 1150-1260 项）可后续用 `ruff --fix` 与补注解处理
- GraphRAG 依赖 Neo4j 环境（`GRAPH_RAG_ENABLED=true` 时启用），默认关闭不影响运行
- 金融语料为演示数据，正式使用需替换为真实产品条款/监管文件并人工核验
