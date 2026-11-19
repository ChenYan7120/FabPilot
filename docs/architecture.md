# FabPilot Architecture

FabPilot is a **hypothesis-driven multi-agent system** for industrial time-series
forecasting and fault diagnosis, targeting semiconductor fab equipment (etch
chambers). It is a hierarchical LangGraph system: one Supervisor orchestrates
three specialist agents through a shared, checkpointed state.

## Design principles

1. **Dynamic routing, not a fixed pipeline.** The Supervisor runs its own ReAct
   loop and decides *at runtime* which specialist to delegate to and in what
   order. Nothing is hard-wired.
2. **Real Hypothesis-Evidence Memory.** The Diagnosis Agent's memory is a
   deterministic state machine (`candidate → active → supported / refuted /
   inconclusive`), not a prompt-only claim. Statistical correlations are recorded
   as *evidence*, never as confirmed causation.
3. **Tools connect to real internals.** Every tool validates typed arguments and
   runs against the real database, forecast registry, or knowledge base. There
   are no `pass` placeholders in the execution path.
4. **A single forecast-model boundary.** Forecasting is a *pure regression* with
   no anomaly interpretation; anomaly detection, statistics and diagnosis live in
   the agents.

## Component topology

```mermaid
flowchart TB
    subgraph Supervisor["Supervisor (hierarchical ReAct)"]
        SLP["ReAct loop over delegation tools"]
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

    subgraph Infra["Shared infrastructure"]
        STATE["Shared LangGraph state (append-only logs)"]
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

## Agent responsibilities and boundaries

| Agent | Owns | Explicitly does **not** own |
|---|---|---|
| **Supervisor** | Task decomposition, delegation order, final report assembly | Data analysis, diagnosis, retrieval — it only routes |
| **Time-Series** | Numerical analysis: query, forecast, anomaly *detection*, correlation, lag, descriptive statistics, charts | Root-cause inference, causal claims |
| **Diagnosis** | Causal reasoning: hypothesis generation, evidence collection, hypothesis re-evaluation, root-cause ranking | Raw retrieval, model training |
| **Knowledge** | Document / similar-case retrieval with citations | Interpretation, diagnosis |

This split enforces the spec's requirement that *forecasting ≠ anomaly
analysis* and *correlation ≠ causation*: the Time-Series Agent returns
"numbers + anomalies", and the Diagnosis Agent is the only component allowed to
turn evidence into a supported/refuted hypothesis.

## Shared state

The LangGraph `AgentState` is the single source of truth. Append-only log fields
(`messages`, `tool_history`, `delegation_records`, `trace`, `errors`,
`subtasks`, `forecast_results`, `anomaly_results`, `retrieved_docs`) use
`operator.add` reducers so every node's delta only carries *added* items. The
replaceable memory fields (`hypotheses`, `evidence`, `pending_checks`) are
overwritten in full by the Diagnosis Agent.

A LangGraph checkpointer (`MemorySaver` or `SqliteSaver`) makes runs resumable and
inspectable by `thread_id`.

## ReAct loop and audit trail

Each agent (including the Supervisor) runs a shared `run_react_loop` that:

1. Sends the system prompt + message history to the LLM.
2. Parses the response into structured tool calls or a stop.
3. Executes each tool through the `ToolRegistry` (validated, exception-safe).
4. Appends a **decision summary** (not the hidden chain-of-thought) plus the
   tool results to the trace.

Budgets (`supervisor_max_iterations`, `agent_max_iterations`, `tool_budget`) and
a loop-detection window stop runaway loops.

## Forecasting model registry

A single `ForecastModel` protocol (`request → ForecastResult`) is implemented by:

- `StatisticalBaselineModel` — **real**: fits a linear trend on the lookback
  window and extrapolates with a ~95% prediction interval.
- `PatchTSTAdapter`, `iTransformerAdapter` — **interface-only skeletons** that
  raise `ForecastModelNotAvailable` until weights are loaded.

This is the honest boundary: the pipeline is real end-to-end, and industrial
models plug in through the same registry without touching the agents.

## RAG pipeline

Hybrid retrieval (design borrowed from [ragent](https://github.com/infiniflow/ragent)'s
dense + keyword parallel recall, RRF fusion, rerank — reimplemented in Python):

1. `load_documents` chunks markdown by paragraph with overlap; the parent
   directory is the `doc_type` (`sop` / `alarm` / `fault_cases`).
2. `HashEmbedder` (deterministic, offline) or an OpenAI-compatible embedder
   produces dense vectors.
3. `HybridRetriever` fuses dense cosine similarity and BM25 via RRF, then reranks.
4. Every hit carries a source citation (`sop/mfc_flow_drift.md`, …).

## Persistence

- Business data: SQLAlchemy 2.0 over PostgreSQL (production) or SQLite (dev/test).
- Sessions: JSON-file `SessionStore` for API history + final reports.
- LangGraph checkpoints: `MemorySaver` / `SqliteSaver`.
