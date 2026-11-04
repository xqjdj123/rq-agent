# 燃气 AI 客服智能体系 — ReAct Agent 架构

<p align="center"><img src="web/avatar/main.webp" width="160" alt="燃气 AI 客服形象"></p>

基于 **ReAct Agent 模式** 的智能燃气客服系统，覆盖报装、缴费、安检、抢修、安全用气、政策咨询等 17 个业务域、106 个二级场景。

> **说明**：本仓库为脱敏展示版。企业名称、电话、营业网点和业务数据均已替换为示例值；完整知识库、原始业务资料与评估明细未公开。项目为团队合作成果，已获授权用于技术展示。

## 核心技术点

- **ReAct Agent 决策循环**：LLM 自主决定工具调用时机与顺序，最多 5 轮「推理 → 行动 → 观察 → 再推理」
- **7 个 Function Calling 工具**：知识库检索、意图分类、法规追溯、服务信息、关联问题、安全判断、人工转接
- **混合检索**：FAISS 语义检索 + BM25 关键词检索 + RRF 融合 + 意图域加权
- **双层安全防线**：17 个安全关键词硬预检（绕过 LLM，毫秒级）+ Agent 内语义级 `check_safety`
- **知识图谱**：业务域 → 意图 → 法规条款的实体关系网络，支持按法规编码追溯
- **可观测前端**：零框架单页应用，展示完整 Agent 决策轨迹（每轮工具、参数、结果）

---

## 什么是 Agent？

Agent（智能体）以 LLM 为决策中枢，自主决定工具调用的时机与顺序。与传统的固定流程不同，Agent 通过 **ReAct 循环**（推理 → 行动 → 观察 → 再推理）灵活应对各类用户问题，而非沿预设路径执行。

### Agent 工作流程

```
用户: "某市燃气开户需要什么材料？"
  │
  ▼
[安全预检] — 不命中，进入 Agent
  │
  ▼
Agent 思考: "用户想了解开户材料，我需要检索知识库"
  │
  ▼
Agent 调用: search_knowledge_base("燃气开户 报装材料")
  │
  ▼
检索结果: [{标准答: "居民开户需提供..."}]
  │
  ▼
Agent 思考: "信息充足，可以回答。结果中有法规编码 FG-15，查一下法规全称"
  │
  ▼
Agent 调用: lookup_regulation("FG-15")
  │
  ▼
Agent 生成回答: "居民办理燃气开户需要以下材料：1.房产证或购房合同..."

依据：《城镇燃气管理条例》第十五条
```

---

## 系统架构

```
用户问句
  │
  ▼
[安全关键词检测] — 17个安全关键词 + 否定语境识别，命中走强制安全模板
  │
  ├─ 命中 → 安全模板 + 转抢修（不经过 Agent）
  │
  └─ 未命中 → Agent 循环 ─┐
                          │
              ┌───────────┘
              ▼
         Agent 推理（DeepSeek Function Calling）
              │
              ├─ 需要检索   → search_knowledge_base (FAISS + BM25 + RRF)
              ├─ 需要分类   → classify_user_intent (关键词 + LLM few-shot)
              ├─ 需要法规   → lookup_regulation (知识图谱查询)
              ├─ 需要推荐   → get_related_questions
              ├─ 需要信息   → get_service_info (电话/地址/营业时间)
              ├─ 安全检查   → check_safety (语义安全判断)
              ├─ 无法处理   → request_human_transfer
              │
              ▼
         综合信息 → 生成最终回答
```

---

## Agent 拥有的工具（7 个）

| 工具 | 功能 | 触发场景 |
|------|------|----------|
| `search_knowledge_base` | FAISS+BM25双路混合检索 + RRF融合 + 意图域加权 | 需要获取业务知识时 |
| `classify_user_intent` | 关键词匹配 + DeepSeek few-shot 两级意图分类 | 用户意图不明确或需要消歧时 |
| `lookup_regulation` | 通过知识图谱查询法规条款详情 | 需要引用/追溯法规依据时 |
| `get_service_info` | 返回服务电话、地址、营业时间等静态信息 | 查询客服热线/网点/时间时 |
| `get_related_questions` | 从知识图谱获取关联常见问题 | 回答后推荐后续问题 |
| `check_safety` | 语义级安全判断（Agent内部第二道防线） | 查询有潜在安全风险暗示时 |
| `request_human_transfer` | 转接人工客服（含原因和紧急程度） | 无法回答或用户明确要求时 |

---

## 项目结构

```
RQAI-Agent/
├── start.py                 # 一键启动（自动加载 .env、检查数据、启动服务）
├── .env.example             # 环境变量模板（API Key、模型名）
├── requirements.txt         # Python 依赖
├── README.md
│
├── engine/                  # Agent 引擎层
│   ├── __init__.py
│   ├── agent.py             # ReAct Agent 核心决策循环（max 5轮迭代）
│   ├── tools.py             # 7个工具定义（OpenAI Function Calling Schema）+ 执行器
│   ├── retriever.py         # 混合检索（FAISS + BM25 + RRF + 意图域加权）
│   ├── safety_gateway.py    # 安全网关（17关键词硬预检 + 否定语境识别 + 4类安全模板）
│   └── session.py           # 会话管理（内存 + 磁盘持久化）
│
├── api/
│   └── main.py              # FastAPI 后端（15个端点，含对话/诊断/图谱/企业信息）
│
├── web/                     # 前端
│   ├── index.html           # 单页应用（零框架，Markdown渲染 + Agent决策追溯）
│   └── avatar/              # 头像与欢迎背景（6种情绪，WebP 压缩）
│
├── data/                    # 数据层
│   ├── knowledge_base.json  # 脱敏样例知识库（12条，字段结构与真实库一致）
│   ├── preprocess.py        # 数据预处理（Excel → JSON + jieba关键词提取）
│   └── merge_new.py         # 增量数据合并（去重 + 字段映射 + 安全自动标记）
│
├── kg/                      # 知识图谱层
│   ├── build_graph.py       # 图谱构建（6种实体 × 4种关系 → NetworkX → JSON）
│   ├── query.py             # 图谱查询（法规追溯 / 意图链 / 关联问题 / 概览统计）
│   └── visualize.py         # 图谱可视化（pyvis 交互式 HTML）
│
├── eval/                    # 评估系统
│   └── evaluate.py          # 完整评估脚本（5项指标 + 混淆矩阵 + 分类明细）
│
└── BAAI/bge-base-zh-v1.5/  # Embedding 模型（需自行下载，不入库）
```

> `data/indexes/`、`data/logs/`、`kg/graph_export.json` 等运行期产物已在 `.gitignore` 中，首次启动时自动重建。

---

## 快速启动

### 环境要求

- Python 3.9+
- DeepSeek API Key
- BGE-base-zh-v1.5 模型（约 400MB）

### 安装

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY=sk-xxx

# 3. 下载 BGE 模型到项目根目录（二选一）
# HuggingFace:
huggingface-cli download BAAI/bge-base-zh-v1.5 --local-dir BAAI/bge-base-zh-v1.5
# ModelScope（国内网络）:
modelscope download --model BAAI/bge-base-zh-v1.5 --local_dir BAAI/bge-base-zh-v1.5
```

> 仓库内置 12 条脱敏样例知识库，可直接跑通检索与 Agent 链路；接入真实业务时替换 `data/knowledge_base.json` 并重建索引。

### 启动

```bash
python start.py
```

`start.py` 会自动完成：
1. 检查 `.env` 中的 API Key
2. 如缺少则自动构建知识库和知识图谱
3. 扫描可用端口（8000-8009）并启动 FastAPI 服务
4. 预热所有引擎组件（Agent + 检索器 + 图谱）

### 访问

| 地址 | 说明 |
|------|------|
| `http://localhost:8000` | 客服对话界面 |
| `http://localhost:8000/docs` | API 文档（Swagger UI） |
| `http://localhost:8000/api/health` | 健康检查 |
| `http://localhost:8000/api/debug` | 系统诊断（Agent + 知识库 + DeepSeek API） |
| `http://localhost:8000/api/stats` | 系统统计（活跃会话 + 知识库规模） |

---

## API 端点一览

### 对话与会话

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/chat` | 主对话接口（安全网关 → Agent → 回答 + 工具调用明细） |
| `GET` | `/api/sessions` | 历史会话列表（最近50条） |
| `GET` | `/api/history/{id}` | 会话详情（含完整消息历史） |
| `DELETE` | `/api/sessions/{id}` | 删除会话 |

### 标签与知识图谱

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/tags` | 标签体系树（17域 × 106标签） |
| `GET` | `/api/graph/summary` | 知识图谱概览（节点/边/类型统计） |
| `GET` | `/api/graph/regulation/{code}` | 法规详情查询（按 FG 编码） |

### 诊断与调试

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/health` | 组件健康检查 |
| `GET` | `/api/debug` | 全组件诊断（Agent状态 + 知识库规模 + 会话数 + API连通性） |
| `GET` | `/api/stats` | 实时统计 |
| `POST` | `/api/intent` | 独立意图检测（调试用，输入 query 返回意图分类结果） |

### 企业信息（示例仓库不包含资料文件）

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/company/profile` | 公司概述（docx → HTML） |
| `GET` | `/api/company/culture` | 企业文化（docx → HTML） |
| `GET` | `/api/company/qrcode` | 企业二维码图片 |
| `GET` | `/api/company/background` | 背景图 |

> 以上接口依赖本地 `data/` 下的企业资料文件；脱敏版未提供，接口会返回缺省提示。

### `/api/chat` 响应示例

```json
{
  "answer": "居民办理燃气开户需要以下材料：1. 房产证或购房合同...",
  "session_id": "sess_1783952014225",
  "tool_calls_count": 3,
  "tool_calls_detail": [
    {"iteration": 1, "tool": "search_knowledge_base", "arguments": {"query": "燃气开户材料"}},
    {"iteration": 2, "tool": "lookup_regulation", "arguments": {"fg_code": "FG-15"}},
    {"iteration": 3, "tool": "get_related_questions", "arguments": {"intent_name": "报装开户>居民新装报装"}}
  ],
  "agent_iterations": 3,
  "source": "agent",
  "intent": {"intent_name": "报装开户", "sub_intent": "居民新装报装", "confidence": 0.9},
  "regulation_ref": "FG-15",
  "response_time_ms": 4520
}
```

---

## 安全红线机制

系统采用**双层纵深防御**：

### 第一道防线：Agent 外部硬预检
- **17 个安全关键词**（漏气、煤气味、着火、爆炸、中毒等），O(n) 子串匹配
- **否定语境识别**：区分"抢修电话是多少"（信息咨询）和"家里管道抢修"（紧急），避免误触发
- 命中后**直接返回强制安全模板**，完全绕过 Agent 和 LLM，响应延迟 < 1ms

### 第二道防线：Agent 内部 check_safety 工具
- 未命中硬预检但存在潜在安全风险时，Agent 在 ReAct 循环中主动调用
- 语义级安全判断，作为硬预检的补充

### 安全模板（4 类）

| 模板类型 | 触发关键词 | 内容要点 |
|---------|-----------|---------|
| 疑似漏气处置 | 漏气、煤气味、臭鸡蛋味 | 关阀→通风→禁火→室外拨打 400-000-0000 |
| 报警器报警 | 报警器响了、报警器鸣叫 | 勿忽视→通风→检查→联系抢修 |
| 燃气中毒 | 中毒、头晕恶心、一氧化碳 | 通风→撤离→120→抢修电话 |
| 火灾爆炸 | 着火、起火、爆炸 | 撤离→119→抢修电话 |

---

## 评估系统

```bash
python eval/evaluate.py
```

生成 303 条独立测试集，覆盖 5 项指标：

| 指标 | 实测值 | 目标 | 说明 |
|------|--------|------|------|
| 意图分类准确率 | 63.4% (184/290) | ≥85% | 17 个一级域分类，含混淆矩阵 |
| 安全检测召回率 | 100% (23/23) | ≥95% | 安全关键词 100% 命中 |
| 检索 Top-1 命中率 | 73.1% (207/283) | ≥60% | FAISS+BM25+RRF 混合检索 |
| 检索 Top-3 召回率 | 92.6% (262/283) | ≥80% | Top-3 覆盖绝大部分正确答案 |
| 自动应答率 | 84.2% (250/297) | ≥85% | 排除边缘/噪声的有效应答比例 |

评估脚本会在 `eval/` 下生成本地报告（JSON + HTML 仪表盘），属于运行期产物，不随仓库分发。

---

## 知识图谱可视化

```bash
python kg/visualize.py
```

生成交互式 HTML 图谱（使用 pyvis）：

| 文件 | 说明 |
|------|------|
| `kg/graph_export.json` | 图谱导出，运行 `python kg/build_graph.py` 本地生成 |

> 全量图与分域子图由 `kg/visualize.py` 在本地根据知识库重建，仓库不包含真实业务数据。

---

## 知识库规模（原始项目）

| 维度 | 数据 |
|------|------|
| 知识条目 | 1003 条（每条 12 字段；仓库内为 12 条脱敏样例） |
| 一级域 | 17 个 |
| 二级标签 | 106 个 |
| 法规编码 | 31 个（FL/XZ/BM/DF/HY/CZ/FT/QY 八层级） |
| 知识图谱 | 1479 节点 / 2168 边（需本地重建） |
| 向量索引 | FAISS IndexFlatIP（768 维） |
| 关键词索引 | BM25 Okapi（jieba 分词） |
| 数据来源 | 国家法律法规库 / 国务院公报 / 住建部 / 省市政府 / 企业官网 / 国家标准 |

---

## 设计原则

1. **安全硬网关**：安全关键词检测在 Agent 之前，命中直接走强制模板，严禁 LLM 介入安全响应
2. **LLM 做决策，不做事实源**：Agent 自主决策工具调用，但所有业务事实来自知识库检索结果
3. **ReAct 模式**：Reasoning + Acting 循环，LLM 推理 → 行动 → 观察 → 再推理
4. **工具即能力**：Agent 的能力边界由工具集定义，新增能力只需添加工具定义
5. **混合检索**：FAISS 语义向量 + BM25 精确关键词 + RRF 融合 + 意图域加权，兼顾语义和术语

---

