# 全面代码审计报告 — 金融智能客服

> 审计日期：2026-08-27
> 范围：app/ 全部源码（约 200 个 Python 文件）逐行审核（6 组并行）
> 结果：**202 项发现** — 漏洞 28 / 缺陷 128 / 无关残留 34 / 合规 12；高 18 / 中 91 / 低 93

---

## 1. 高危问题（18 项，必须优先修复）

### 1.1 鉴权与越权（5 项）
| # | 位置 | 问题 |
|---|------|------|
| H1 | `app/api/deps/__init__.py` | `get_current_user` 匿名放行返回 None，除 documents 上传/删除外**所有业务端点可匿名访问**；`get_current_active_user` 未处理 None 会抛 500 |
| H2 | `app/api/v1/chat.py` | chat/history/clear **未校验 session 归属**（session_id 客户端自填），枚举即越权读写他人会话（IDOR） |
| H3 | `app/api/v1/graph.py` | 图谱全部端点**无鉴权**，/import/structured 可任意改图库 |
| H4 | `app/api/v1/documents.py` | 上传写入 `uploaded_by_email` 明文入向量库，搜索**未按用户过滤**，跨用户泄露文档与邮箱（合规+越权） |
| H5 | `app/api/rate_limit.py` | 限流**信任 X-Forwarded-For**（取首 IP），伪造头即可绕过；速率硬编码 10/min 与配置 60 不符 |

### 1.2 对话状态机（3 项）
| # | 位置 | 问题 |
|---|------|------|
| H6 | `nodes.py:59-60,342` | **会话被永久锁定**：guardrail 拦截后 blocked 持久化，之后所有合法消息都返回"被拦截"文案 |
| H7 | `nodes.py:357-365` | **跨轮状态污染**：tool_result/retrieved_docs 等瞬态键跨轮残留，上轮工具结果会污染本轮 RAG 回答 |
| H8 | `slot_types.py:104-140` | **风险测评失效**：q1-q5 正则都是单字符 `[A-D]`，一条消息填满全部题目，问卷被跳过、风险等级结论错误 |

### 1.3 检索/文档（3 项）
| # | 位置 | 问题 |
|---|------|------|
| H9 | `documents/ingestion.py:226` | `ingest_text` **缺 return**，入库结果全部丢失 |
| H10 | `documents/ingestion.py:197` + `qdrant_client.py:264` | 图实体 ID 不一致（悬挂关系）+ `delete_by_filter` 过滤语法不匹配（**删除功能整体失效**） |
| H11 | `documents/chunking.py:44` | FixedSizeChunking `overlap>=size` 时 **while 死循环**挂死 worker |

### 1.4 GraphRAG 安全（3 项）
| # | 位置 | 问题 |
|---|------|------|
| H12 | `neo4j_client.py:19,137` | **Cypher 只读校验是黑名单且可绕过**（`SET\n`、写过程不在名单），LLM 生成可执行任意写操作 |
| H13 | `text_to_cypher.py:74` | LLM 生成仅查 MATCH 前缀 → **提示注入→任意 Cypher 执行**链路成立 |
| H14 | `graph/community/*` | 社区检测管线**多处静默失败断链**（communityId 写回永不生效、投影泄漏、global_search 恒空） |

### 1.5 其它（1 项，来自 LLM 组）
| # | 位置 | 问题 |
|---|------|------|
| H15 | `services/chat/chat_service.py` + `services/memory/*` | 记忆层 dict/ORM 接口混用 + **会话归属越权**（记忆按 session_id 无用户校验，复用 H2） |

## 2. 中危问题（91 项，代表性 20 项）

| 位置 | 问题 |
|------|------|
| `api/v1/auth.py` | register/login 无限流防暴力破解；register 错误把 str(e) 返回客户端泄露内部信息 |
| `api/v1/sessions.py` | 匿名时 current_user.id 抛错被捕获后以 detail 返回内部错误；5 端点全无日志 |
| `api/v1/chat.py` SSE | 生成器内异常未捕获，中断无 [DONE]；chunk 换行未转义 |
| `api/v1/documents.py` | 上传无大小限制（内存 DoS）；每请求新建 embedding 服务/Qdrant 客户端不复用 |
| `api/database.py` | `conn.execute("SELECT 1")` 未用 text() 必然报错；engine 从不 dispose（连接泄露） |
| `main.py` | RateLimiterMiddleware/ErrorHandler 从未注册（全局限流失效）；/metrics 未鉴权 |
| `api/v1/feedback.py` | 声明 current_user 未使用，任意用户可给他人消息评分（IDOR） |
| `api/v1/graph.py` | set_graph_client 从未被调用 → **GraphRAG 端点全部恒 503，功能实际不可用** |
| `intent/base.py` | 基类同步抽象方法 vs 实现 async，await 同步方法会 TypeError（rule_based 路径必崩） |
| `intent/llm_based.py` | 子串匹配 `intent.value in intent_str` 误判（"cannot confirm"→confirm） |
| `intent/rule_based.py` | confirm/deny 关键词过宽（"可以/对/好的"误判 meta） |
| `slot_types.py:52` | fund_code 正则双捕获组取错 → 基金代码提取为"基金"二字 |
| `slot_types.py:103` | **风险测评 q1 年龄越大分越高**，65+ 最易评 R5 激进型，方向与适当性相反 |
| `nodes.py:68` | 被拦截消息仍流入槽位/工具管线（拦截可绕过） |
| `nodes.py:177` | state_stack 无上限，反复切换任务栈无限增长 |
| `nodes.py:499` | 直答路径（chitchat/fallback）提示词无金融合规约束 |
| `retrieval/hybrid_search.py` | 合法空结果被误判"Both failed"抛错；两分支裸 except 无日志 |
| `embeddings/factory.py` | 声明 openai provider 但模块不存在（死分支） |
| `embeddings/cached_embeddings.py` | Redis 故障直接拖垮 embed（应降级直连） |
| `documents/preprocessing.py` | 引号归一化写反（左引号变右引号，无实质归一） |
| `graph/factory.py` | 硬编码默认凭据 password="password" |
| `graph/extraction/llm_extractor.py` | 文本直接 format 进 prompt 无隔离（提示注入）；抽取 PII 入图谱未脱敏 |
| `graph/extraction/entity_resolver.py` | 子串包含做实体归并，"稳健理财"与"稳健型理财"错误合并 |

## 3. 无关残留（34 项，可删除）

### 3.1 死代码/未接线（应删）
| 位置 | 内容 |
|------|------|
| `app/middleware/error_handler.py` | ErrorHandlerMiddleware 从未注册（与 main.py 内联处理器重复） |
| `app/middleware/rate_limiter.py` | RateLimiterMiddleware 从未注册（与 api/rate_limit.py 双套冲突） |
| `app/api/v1/openapi.py` | setup_openapi 从未被调用，整文件死代码 |
| `app/services/slot_filling/*`（base/rule_based/hybrid/llm_based/factory + SLOT_DEFINITIONS） | 整套服务从未被图流程调用（legacy，与 INTENT_SLOT_SCHEMAS 双套并存） |
| `app/services/retrieval/document_metadata.py` | 内存态演示实现混入生产（self.session 未用） |
| `app/middleware/request_id.py` get_request_id | 仅测试用，应用内无调用 |
| `app/api/v1/graph.py` set_graph_client | 从未被调用 |
| `nodes.py:367-369` | generate_response case5 与 case2 重复（不可达死代码） |

### 3.2 旧领域残留（应改/删）
| 位置 | 残留 |
|------|------|
| `app/config/constants.py` | 旧 Intent 类（QUESTION/HOW_TO/CODE_HELP…）与 INTENT_CATEGORIES，通用问答残留 |
| `app/services/intent/llm_based.py` docstring | 与当前意图集不符的过时描述 |
| `app/services/slot_filling/rule_based.py` docstring | 与金融场景不符的领域描述 |
| `app/services/slot_filling/llm_based.py` prompt | Normalize 示例与金融领域不符 |
| `app/middleware/metrics.py` / `tracing.py` | service_name 旧项目名残留 |
| `app/services/dialogue/tools.py` | `import random` 未使用 |
| `app/services/documents/ingestion.py` | `Optional/Dict` 未使用 import；循环内重复 import uuid |
| `app/services/documents/preprocessing.py` | `text_lower` 未使用 |
| `app/services/slot_filling/rule_based.py` | quarter/year_range 死分支 |

## 4. 合规问题（12 项，金融特有）

| 位置 | 问题 |
|------|------|
| `api/v1/documents.py` | 邮箱 PII 明文入向量库（高危，见 H4） |
| `slot_types.py:103` | 风险测评年龄评分方向反（见中危） |
| `nodes.py:499` | 直答路径无合规约束 |
| `tools.py:341` | 收益测算无利率/期限/金额上限，可算天价"预期收益" |
| `documents/ingestion.py:285` | 绝对 file_path 入向量库泄露服务器路径 |
| `graph/extraction/llm_extractor.py` | 抽取 PII 入图谱未脱敏 |
| `retrieval/reranking.py` | 检索结果不校验来源可信度，不可信来源同权参与排序 |
| `services/llm/*` | 三个客户端缺超时/重试；`temperature=0` 被 `or` 短路覆盖 |
| `services/memory/summarization.py` | 摘要持久化未脱敏 |

---

## 5. 修复优先级建议

**P0（上线前必修，约 12 项）**：H1-H5 鉴权越权、H6-H8 状态机、H12-H13 Cypher 注入、ingest_text return、delete_document 失效、FixedSizeChunking 死循环。

**P1（重要，约 25 项）**：风险测评年龄方向、intent 基类 async、直答合规约束、限流统一与 IP 可信、上传大小限制、资源复用/关闭、graph 端点注册、提示注入隔离（RAG/重排/抽取）。

**P2（清理，约 30 项）**：死代码删除（middleware/openapi/slot_filling 栈）、旧领域 docstring、未用 import。

**P3（增强，其余）**：MemorySaver→持久化、token 估算中文修正、缓存降级、批次 upsert、hash 稳定化等。

---

*原始 202 项明细见 `docs/code-audit-report-raw.txt`*
