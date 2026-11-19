# FabPilot API

FastAPI REST API exposing two surfaces:

- **MES/FDC data + analytics** — real tools, no LLM required.
- **Agent tasks** — run the full hierarchical agent system and fetch results.

## Run

```bash
# start Postgres (optional; SQLite also works)
docker compose up -d postgres

# create tables + seed synthetic data
fabpilot init-db
fabpilot seed

# start the server
fabpilot api --host 0.0.0.0 --port 8000
```

Interactive docs: http://localhost:8000/docs

## Endpoints

### Health

```
GET /api/v1/health
```

### Equipment / FDC

| Method | Path | Description |
|---|---|---|
| GET | `/api/v1/equipment` | List all equipment |
| GET | `/api/v1/equipment/{equipment_id}` | Equipment metadata / status |
| GET | `/api/v1/equipment/{equipment_id}/sensor-data` | Sensor readings (optional `channel`, `start`, `end`, `limit`) |
| GET | `/api/v1/equipment/{equipment_id}/alarms` | Alarm log |
| GET | `/api/v1/equipment/{equipment_id}/maintenance` | Maintenance records |
| GET | `/api/v1/fault-cases` | Historical fault cases (`equipment_id`, `limit`) |

### Analytics

| Method | Path | Description |
|---|---|---|
| POST | `/api/v1/forecast` | Forecast a channel (see `ForecastRequest`) |
| POST | `/api/v1/anomaly-detection` | Detect anomalies (see `AnomalyRequest`) |
| POST | `/api/v1/search` | Hybrid knowledge retrieval (see `SearchRequest`) |

### Agent tasks

| Method | Path | Description |
|---|---|---|
| POST | `/api/v1/tasks` | Run a natural-language task |
| GET | `/api/v1/tasks` | List persisted task records |
| GET | `/api/v1/tasks/{thread_id}` | Fetch a task result |

## Request / response examples

### Forecast

```bash
curl -X POST http://localhost:8000/api/v1/forecast \
  -H 'Content-Type: application/json' \
  -d '{"equipment_id": "ETCH-01", "target": "gas_flow", "model": "statistical_baseline"}'
```

```json
{
  "equipment_id": "ETCH-01",
  "target": "gas_flow",
  "model_name": "statistical_baseline",
  "horizon": 16,
  "forecast": [
    {"ts": "2026-01-08T00:00:00", "value": 42.1, "lower": 41.0, "upper": 43.2}
  ],
  "in_sample_metrics": {"method": "linear_trend", "in_sample_mae": 0.3}
}
```

### Anomaly detection

```bash
curl -X POST http://localhost:8000/api/v1/anomaly-detection \
  -H 'Content-Type: application/json' \
  -d '{"equipment_id": "ETCH-01", "channel": "gas_flow", "method": "rolling_zscore", "window": 24, "threshold": 3.0}'
```

### Agent task

```bash
curl -X POST http://localhost:8000/api/v1/tasks \
  -H 'Content-Type: application/json' \
  -d '{"user_task": "ETCH-01 的 gas_flow 出现漂移，请诊断根因并给出处置建议。"}'
```

```json
{
  "thread_id": "sess_...",
  "session_id": "sess_...",
  "status": "completed",
  "final_report": "## 根因诊断 ...",
  "delegated_agents": ["timeseries", "diagnosis", "knowledge"],
  "n_tool_calls": 12,
  "hypotheses": [{"id": "hyp_...", "description": "...", "status": "supported"}],
  "errors": []
}
```

## Architecture notes

- `FabPilotService` builds non-LLM dependencies eagerly (database session scope,
  forecast registry, knowledge base, tool registry) so the data/analytics
  endpoints work without an API key.
- The full `Runtime` (LLM + LangGraph) is built **lazily** on the first task
  call, so starting the API never requires an LLM API key.
- Task results are persisted through `SessionStore`
  (`data/sessions.json`) and retrievable by `thread_id`.
