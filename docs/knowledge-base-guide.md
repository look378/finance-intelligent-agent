# 企业知识库整理与上传指南

> 适用对象：开发者 / 企业知识库管理员
> 目标：把企业真实文档（产品条款、监管政策、业务规则、FAQ）整理成可被 RAG 检索的高质量知识库

---

## 1. 整体流程（四步）

```
① 整理源文档 → ② 元数据规范 → ③ 导入入库 → ④ 检索质量验证
   （目录/格式）    （分类/版本/来源）   （脚本或 API）    （问答测试/调优）
```

---

## 2. 第一步：整理源文档

### 2.1 推荐目录结构

按业务分类建目录，子目录名即分类（category），导入时自动带上：

```
kb/
├── 产品条款/                 # 理财产品说明书、基金招募书
│   ├── 安心货币A产品说明书.md
│   └── 稳盈短债C产品说明书.md
├── 监管政策/                 # 适当性管理、风险提示规定
│   └── 投资者适当性管理办法.md
├── 业务规则/                 # 申购赎回规则、费率规则
│   └── 基金赎回规则.md
├── 保险/                     # 险种条款
│   └── 重疾险条款.md
└── FAQ/
    └── 常见问题.md
```

### 2.2 支持格式

| 格式 | 说明 |
|------|------|
| `.md` / `.txt` | 直接读取（自动兼容 UTF-8 / GB18030 编码） |
| `.pdf` | 文本提取（`pypdf`，扫描件需先 OCR） |

> ⚠️ 金融 PDF 常见问题：扫描件/图片型 PDF 提取不到文字，需先 OCR（如 PaddleOCR）转为文本再入库。

### 2.3 源文档质量要求（决定检索质量的关键）

- **每篇文档主题单一**：一篇讲一个产品/一项政策，不要混杂
- **保留结构化标题**：Markdown 标题（#/##）能提升分块效果
- **避免大段表格**：表格提取后易乱，建议转为"字段: 值"文本行
- **数字/日期准确**：费率、期限等关键数字必须与官方文件核对

---

## 3. 第二步：元数据规范（sidecar 文件）

为每篇文档配一个**同名 `.json` 元数据文件**，检索时可精确过滤：

```
产品条款/安心货币A产品说明书.md
产品条款/安心货币A产品说明书.json   ← sidecar 元数据
```

```json
{
  "title": "安心货币A产品说明书",
  "category": "产品条款",
  "tags": ["货币基金", "R1", "低风险"],
  "version": "2024-06-01",
  "effective_date": "2024-06-01",
  "source": "产品管理部",
  "owner": "资金运营中心"
}
```

**推荐字段**：

| 字段 | 用途 |
|------|------|
| `title` | 文档标题（检索结果展示） |
| `category` | 分类（可用 API 过滤） |
| `tags` | 标签，提高召回精度 |
| `version` / `effective_date` | 版本管理（政策更新后可定位旧版） |
| `source` | 来源（审计/追溯） |

---

## 4. 第三步：导入入库

### 方式 A：批量脚本（推荐，企业场景）

```bash
# 基本导入（子目录名自动作为分类）
python scripts/import_kb.py --dir ./kb --recursive

# 指定分类 + 去重 + 调分块参数
python scripts/import_kb.py --dir ./kb --recursive \
    --category 产品条款 \
    --dedupe \
    --strategy recursive --chunk-size 600 --overlap 80
```

| 参数 | 说明 |
|------|------|
| `--dir` | 文档目录（必填） |
| `--recursive` | 递归扫描子目录 |
| `--strategy` | 分块策略：`semantic`（默认）/ `recursive` / `fixed` |
| `--chunk-size` | 分块大小（默认 512 字符） |
| `--overlap` | 分块重叠（默认 50） |
| `--dedupe` | 按内容哈希去重 |
| `--category` | 默认分类（覆盖子目录推断） |

### 方式 B：REST API（单篇/程序化）

```bash
# 上传文本
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -H "Content-Type: application/json" \
  -d '{"title": "产品条款X", "content": "文档正文...", "metadata": {"category": "产品条款"}}'

# 上传文件（需登录 token）
curl -X POST http://localhost:8000/api/v1/documents/upload/file \
  -H "Authorization: Bearer <token>" \
  -F "file=@产品说明书.pdf" -F "title=产品说明书X"
```

---

## 5. 第四步：检索质量验证与调优

### 5.1 验证检索命中

```bash
# 用文档搜索接口直接验证
curl -X POST http://localhost:8000/api/v1/documents/search \
  -H "Content-Type: application/json" \
  -d '{"query": "货币基金赎回到账时间", "top_k": 3}'
```

检查：命中是否相关、得分是否合理（>0.5 通常可用）。

### 5.2 分块参数调优经验

| 场景 | 建议策略 | chunk-size |
|------|---------|-----------|
| 产品说明书（条款式） | `semantic` | 400-600 |
| 监管政策（长段落） | `recursive` | 500-800 |
| FAQ（问答对） | `fixed` | 200-400（小分块更精准） |

### 5.3 知识更新策略（重要）

| 场景 | 做法 |
|------|------|
| 政策更新 | 更新文档 + sidecar 的 `version`/`effective_date`，**重新导入覆盖** |
| 产品下架 | 删除对应文档（API `DELETE /documents/{document_id}`） |
| 定期巡检 | 用检索接口抽查高频问题命中情况，发现失效内容及时清理 |

> ⚠️ **合规红线**：金融知识库务必以官方文件为准，导入前人工核验；
> 回答"资料中没有相关内容"时，系统会如实告知（已在 RAG 提示词中内置）。

---

## 6. 常见问题（FAQ）

**Q1：导入后聊天还是答不上来？**
检查：① 检索接口能否命中该问题；② 问法是否与文档表述差异太大（向量检索对同义改写敏感）。

**Q2：如何让回答更严谨？**
知识类问题系统只基于检索结果作答（不凭 LLM 记忆），并附来源文档 ID；若命中分数低，回答会倾向"暂未查询到"。

**Q3：文档更新后旧内容还在？**
导入新文档前先删除旧文档（按 document_id），或用 `--dedupe` 防重复。

**Q4：演示语料怎么清掉？**
当前 `documents` collection 里有 12 篇演示文档，可重建 collection：
```bash
# 删除并重建（需 Docker 权限）
docker compose down qdrant && docker volume rm finance_intelligent_agent_qdrant_data
docker compose up -d qdrant
# 然后用 --dir 导入你的真实文档
python scripts/import_kb.py --dir ./kb --recursive
```

---

## 7. 检索链路架构（导入后）

```
用户问题
   ↓
意图识别（policy/faq → RAG）
   ↓
Qdrant 向量检索（BGE-M3 本地 embedding，1024 维）
   ↓
Top-K 文档（带 metadata：title/category/version/source）
   ↓
DeepSeek 基于检索结果生成回答（附来源）
```

- 检索与生成分离：回答内容严格来自检索到的文档，来源可追溯
- embedding 本地运行（models/bge-m3），不依赖外部 API
