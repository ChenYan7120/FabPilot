# FabPilot 架构

FabPilot 是一个面向工业时序 forecasting 和故障诊断的 **hypothesis 驱动的多 agent 系统**，
针对半导体 fab 设备（etch 腔室）。它是一个分层 LangGraph 系统：一个 Supervisor 通过
共享的、checkpointed 状态编排三个专业 agent。

## 设计原则

1. **动态路由，而非固定 pipeline。** Supervisor 运行自己的 ReAct
   循环，并在*运行时*决定委托给哪个专业 agent、按什么顺序。没有任何硬编码。
2. **真正的 Hypothesis-Evidence Memory。** Diagnosis Agent 的 memory 是一个
   确定性的状态机（`candidate → active → supported / refuted /
   inconclusive`），而非仅靠 prompt 的声明。统计相关性被记录为
   *evidence*，而绝不会被记录为已确认的 causation。
3. **Tools 连接真实内部实现。** 每个 tool 都会校验类型化参数，并针对真实的
   database、forecast registry 或 knowledge base 运行。执行路径中
   没有任何 `pass` 占位符。
4. **单一的 forecast-model 边界。** Forecasting 是 *纯回归*，不包含
   任何 anomaly 解读；anomaly detection、统计和诊断都位于 agents 之中。

## 组件拓扑

```mermaid
flowchart TB
    subgraph Supervisor["Supervisor (hierarchical ReAct)"]
        SLP["ReAct loop（遍历 delegation tools）"]
        DTS["delegate_to_timeseries"]
        DDX["delegate_to_diagnosis"]
        DKN["delegate_to_knowledge"]
        FIN["finish(report)"]
    end

    subgraph TS["Time-Series Agent"]
        TS_LOOP["ReAct loop"]
        TS_TOOLS["sensor_data · forecast · anomaly_detection · correlation · lag_analysis · statistical_analysis · chart"]
    end

    subgraph DX["Diagnosis Agent"]
        DX_LOOP["ReAct loop"]
        MEM["Hypothesis-Evidence Memory"]
        DX_TOOLS["sensor_data · alarm_log · maintenance_record · fault_case · correlation · lag_analysis · memory ops"]
    end

    subgraph KN["Knowledge Agent"]
        KN_LOOP["ReAct loop"]
        KN_TOOLS["document_retrieval · similar_case_retrieval"]
    end

    SLP --> DTS --> TS_LOOP --> TS_TOOLS
    SLP --> DDX --> DX_LOOP --> DX_TOOLS
    DX_LOOP <--> MEM
    SLP --> DKN --> KN_LOOP --> KN_TOOLS
    SLP --> FIN

    subgraph Infra["共享基础设施"]
        STATE["共享 LangGraph state（append-only logs）"]
        REG["Forecast Model Registry"]
        KB["Knowledge Base (hybrid RAG)"]
        DB["MES/FDC database (Postgres/SQLite)"]
    end

    TS_TOOLS --> REG
    TS_TOOLS --> DB
    DX_TOOLS --> DB
    KN_TOOLS --> KB
    DX_TOOLS --> KB
    SLP -.writes.-> STATE
    STATE -.checkpointed.-> INFRA
```

## Agent 职责与边界

| Agent | 负责 | 明确**不**负责 |
|---|---|---|
| **Supervisor** | 任务分解、委托顺序、最终报告组装 | 数据分析、诊断、检索 —— 它只负责路由 |
| **Time-Series** | 数值分析：query、forecast、anomaly *detection*、correlation、lag、描述性统计、图表 | 根因推断、因果性结论 |
| **Diagnosis** | 因果推理：hypothesis 生成、证据收集、hypothesis 重新评估、根因排序 | 原始检索、模型训练 |
| **Knowledge** | 带引用的文档 / 相似案例检索 | 解读、诊断 |

这种划分落实了规范中 *forecasting ≠ anomaly analysis* 以及 *correlation ≠ causation* 的要求：
Time-Series Agent 只返回「数字 + anomalies」，而 Diagnosis Agent 是唯一被允许把
证据转化为 supported/refuted hypothesis 的组件。

## 共享状态

LangGraph 的 `AgentState` 是唯一事实来源。仅追加的日志字段
（`messages`、`tool_history`、`delegation_records`、`trace`、`errors`、
`subtasks`、`forecast_results`、`anomaly_results`、`retrieved_docs`）使用
`operator.add` reducer，因此每个节点的 delta 只携带*新增*的条目。
可替换的 memory 字段（`hypotheses`、`evidence`、`pending_checks`）由
Diagnosis Agent 整体覆盖。

LangGraph checkpointer（`MemorySaver` 或 `SqliteSaver`）使运行可恢复，
并可通过 `thread_id` 检查。

## ReAct 循环与审计轨迹

每个 agent（包括 Supervisor）都运行一个共享的 `run_react_loop`，它会：

1. 把系统 prompt + 消息历史发送给 LLM。
2. 将响应解析为结构化的 tool 调用或停止信号。
3. 通过 `ToolRegistry` 执行每个 tool（经过校验、异常安全）。
4. 将**决策摘要**（而非隐藏的 chain-of-thought）以及 tool 结果
   追加到 trace 中。

预算（`supervisor_max_iterations`、`agent_max_iterations`、`tool_budget`）以及
一个 loop 检测窗口可以阻止失控循环。

## Forecasting 模型 registry

单一的 `ForecastModel` 协议（`request → ForecastResult`）由以下实现：

- `StatisticalBaselineModel` —— **真实实现**：在 lookback 窗口上拟合线性趋势，
  并以约 95% 的预测区间进行外推。
- `PatchTSTAdapter`、`iTransformerAdapter` —— **仅接口的骨架**，在加载权重之前会
  抛出 `ForecastModelNotAvailable`。

这就是诚实的边界：pipeline 是真实端到端的，工业模型通过同一个 registry
接入，而无需改动 agents。

## RAG pipeline

混合检索（设计借鉴自 [ragent](https://github.com/infiniflow/ragent) 的
dense + 关键词并行召回、RRF 融合、rerank —— 用 Python 重新实现）：

1. `load_documents` 按段落对 markdown 进行 chunking，并带重叠；父目录即
   `doc_type`（`sop` / `alarm` / `fault_cases`）。
2. `HashEmbedder`（确定性的、离线）或 OpenAI 兼容的 embedder
   生成 dense 向量。
3. `HybridRetriever` 通过 RRF 融合 dense 余弦相似度与 BM25，然后进行 rerank。
4. 每个命中结果都带有一个来源引用（`sop/mfc_flow_drift.md`，…）。

## 持久化

- 业务数据：基于 PostgreSQL（生产）或 SQLite（开发/测试）的 SQLAlchemy 2.0。
- 会话：JSON 文件的 `SessionStore`，用于 API 历史记录 + 最终报告。
- LangGraph checkpoints：`MemorySaver` / `SqliteSaver`。
