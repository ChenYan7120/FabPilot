# FabPilot API

暴露两个界面的 FastAPI REST API：

- **MES/FDC 数据 + 分析** —— 真实工具，无需 LLM。
- **Agent 任务** —— 运行完整的分层 agent 系统并获取结果。

## 运行

```bash
# 启动 Postgres（可选；SQLite 也可用）
docker compose up -d postgres

# 建表 + 填充合成数据
fabpilot init-db
fabpilot seed

# 启动服务
fabpilot api --host 0.0.0.0 --port 8000
```

交互式文档：http://localhost:8000/docs

## 端点

### 健康检查

```
GET /api/v1/health
```

### 设备 / FDC

| Method | Path | 描述 |
|---|---|---|
| GET | `/api/v1/equipment` | 列出所有设备 |
| GET | `/api/v1/equipment/{equipment_id}` | 设备元数据 / 状态 |
| GET | `/api/v1/equipment/{equipment_id}/sensor-data` | 传感器读数（可选 `channel`、`start`、`end`、`limit`） |
| GET | `/api/v1/equipment/{equipment_id}/alarms` | 报警日志 |
| GET | `/api/v1/equipment/{equipment_id}/maintenance` | 维护记录 |
| GET | `/api/v1/fault-cases` | 历史故障案例（`equipment_id`、`limit`） |

### 分析

| Method | Path | 描述 |
|---|---|---|
| POST | `/api/v1/forecast` | 对某个通道做预测（参见 `ForecastRequest`） |
| POST | `/api/v1/anomaly-detection` | 检测异常（参见 `AnomalyRequest`） |
| POST | `/api/v1/search` | 混合知识检索（参见 `SearchRequest`） |

### Agent 任务

| Method | Path | 描述 |
|---|---|---|
| POST | `/api/v1/tasks` | 运行一个自然语言任务 |
| GET | `/api/v1/tasks` | 列出已持久化的任务记录 |
| GET | `/api/v1/tasks/{thread_id}` | 获取某个任务的结果 |

## 请求 / 响应示例

### 预测

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

### 异常检测

```bash
curl -X POST http://localhost:8000/api/v1/anomaly-detection \
  -H 'Content-Type: application/json' \
  -d '{"equipment_id": "ETCH-01", "channel": "gas_flow", "method": "rolling_zscore", "window": 24, "threshold": 3.0}'
```

### Agent 任务

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

## 架构说明

- `FabPilotService` 会预先构建非 LLM 依赖（数据库会话作用域、
  预测模型注册表、知识库、工具注册表），因此数据/分析
  端点无需 API key 即可工作。
- 完整的 `Runtime`（LLM + LangGraph）在首次任务调用时才被**懒加载**构建，
  因此启动 API 从不需要 LLM API key。
- 任务结果通过 `SessionStore`（`data/sessions.json`）持久化，
  并可通过 `thread_id` 检索。
