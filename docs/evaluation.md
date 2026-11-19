# FabPilot Evaluation

The evaluation harness measures the *agent system* — routing, hypothesis-evidence
reasoning, and RAG — on top of the *real* tools. It does **not** fabricate run
results: every metric is computed from actual tool outputs over a seeded,
deterministic database and knowledge base.

## How it works

```bash
fabpilot eval          # runs all configs × scenarios, writes data/reports/evaluation.md + .json
```

The harness (`fabpilot.eval.runner`):

1. Seeds a **temporary SQLite database** with the synthetic fab simulation
   (`seed=42`, `n_equipment`, `days`).
2. Builds the real forecast registry and knowledge base.
3. Builds a fresh runtime per (config, scenario) with a **`ScriptedLLM`** — a
   deterministic test double that emits the tool calls the scenario requires.
   This isolates agent behaviour from model nondeterminism.
4. Invokes the graph and captures the final shared state + wall time.
5. Computes metrics from the state (tool history, hypotheses, retrieved docs,
   held-out forecast residuals, injected anomaly labels).

## Comparison configs

| Config | Supervisor | Diagnosis agent | RAG |
|---|---|---|---|
| `full` | ✅ | ✅ | ✅ |
| `no_rag` | ✅ | ✅ | ❌ |
| `baseline` | ❌ (flat tool loop) | ❌ | ❌ |

The `baseline` config runs a flat single-node graph over the domain tools with no
hierarchical routing, no hypothesis memory and no retrieval — an honest lower
bound for "what does the orchestration actually add?"

## Scenarios and ground truth

| Scenario | Task type | Ground truth |
|---|---|---|
| A | Forecast | Held-out series (split at `forecast_cut_fraction`) |
| B | Anomaly detection | Injected anomaly-label windows from the simulator |
| C | Diagnosis | Root cause string + expected agent routing |
| D | Knowledge | Expected source citations |

## Metric categories

| Category | Metric(s) | Computed from |
|---|---|---|
| Task success | `task_success` | final `status` + non-empty report |
| Tool accuracy | `tool_accuracy`, `tool_error_rate` | tool history `ok` flags |
| Routing | `routing_accuracy` | delegated agents vs expected |
| Diagnosis | `diagnosis_top_k` | hypothesis text vs ground-truth root cause |
| Evidence consistency | `evidence_consistency` | evidence linked to real hypothesis ids |
| RAG | `rag_recall_at_k` | retrieved sources vs expected sources |
| Time-series | `ts_forecast_mae`, `ts_forecast_mse`, `ts_anomaly_f1` | held-out forecast residuals / label windows |
| Efficiency | `efficiency_tool_calls`, `efficiency_agent_steps`, `efficiency_wall_time_s` | tool history / trace / wall clock |
| Robustness | `robustness` | clean completion with no error entries |

Metrics that don't apply to a config (e.g. `routing_accuracy` when
`use_supervisor=False`, or `rag_recall_at_k` when `use_rag=False`) are `None` and
excluded from per-config averages.

## Report output

`write_report` produces both:

- `data/reports/evaluation.md` — human-readable comparison tables.
- `data/reports/evaluation.json` — machine-readable aggregates + per-run detail.

## Honest boundaries

- `ts_forecast_mae` / `ts_forecast_mse` come from the **real** statistical
  baseline model over a held-out split — genuine numbers, not scripted strings.
- `ts_anomaly_f1` compares flagged points against the simulator's injected
  ground-truth windows.
- The `ScriptedLLM` is a test double only for the *LLM*; every tool it calls is
  the real implementation.
