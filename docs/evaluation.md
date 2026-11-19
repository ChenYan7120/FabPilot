# FabPilot 评估

evaluation 框架在 *真实* tools 之上衡量 *agent 系统* —— 包括路由、hypothesis-evidence
推理和 RAG。它**不会**伪造运行结果：每个指标都基于已填充种子数据的、
确定性的 database 和 knowledge base 上的实际 tool 输出计算。

## 工作原理

```bash
fabpilot eval          # runs all configs × scenarios, writes data/reports/evaluation.md + .json
```

该框架（`fabpilot.eval.runner`）：

1. 用合成的 fab 仿真数据填充一个**临时 SQLite 数据库**
   （`seed=42`、`n_equipment`、`days`）。
2. 构建真实的 forecast registry 和 knowledge base。
3. 为每个（config, scenario）组合构建一个全新的 runtime，使用 **`ScriptedLLM`** —— 一个
   确定性的测试替身，会发出 scenario 所需的 tool 调用。
   这能把 agent 行为与模型的不确定性隔离开来。
4. 调用 graph，并捕获最终的共享状态 + 墙钟时间。
5. 从状态中计算指标（tool 历史、hypotheses、检索到的文档、
   留出的 forecast 残差、注入的 anomaly 标签）。

## 对比配置

| 配置 | Supervisor | Diagnosis agent | RAG |
|---|---|---|---|
| `full` | ✅ | ✅ | ✅ |
| `no_rag` | ✅ | ✅ | ❌ |
| `baseline` | ❌（扁平 tool 循环） | ❌ | ❌ |

`baseline` 配置在领域 tools 之上运行一个扁平的、单节点的 graph，没有分层路由、
没有 hypothesis memory、也没有检索 —— 这是对「编排到底带来了什么」的一个诚实下限。

## 场景与 ground truth

| 场景 | 任务类型 | Ground truth |
|---|---|---|
| A | Forecast | 留出序列（在 `forecast_cut_fraction` 处切分） |
| B | Anomaly detection | 仿真器注入的 anomaly 标签窗口 |
| C | Diagnosis | 根因字符串 + 预期的 agent 路由 |
| D | Knowledge | 预期的来源引用 |

## 指标类别

| 类别 | 指标 | 计算依据 |
|---|---|---|
| 任务成功 | `task_success` | 最终 `status` + 非空报告 |
| Tool 准确率 | `tool_accuracy`、`tool_error_rate` | tool 历史中的 `ok` 标志 |
| 路由 | `routing_accuracy` | 委托的 agents 与预期对比 |
| 诊断 | `diagnosis_top_k` | hypothesis 文本与 ground-truth 根因对比 |
| 证据一致性 | `evidence_consistency` | 关联到真实 hypothesis ids 的证据 |
| RAG | `rag_recall_at_k` | 检索到的来源与预期来源对比 |
| 时间序列 | `ts_forecast_mae`、`ts_forecast_mse`、`ts_anomaly_f1` | 留出的 forecast 残差 / 标签窗口 |
| 效率 | `efficiency_tool_calls`、`efficiency_agent_steps`、`efficiency_wall_time_s` | tool 历史 / trace / 墙钟时间 |
| 鲁棒性 | `robustness` | 无错误条目的干净完成 |

不适用的指标（例如当 `use_supervisor=False` 时的 `routing_accuracy`，
或当 `use_rag=False` 时的 `rag_recall_at_k`）为 `None`，并从各配置的平均值中排除。

## 报告输出

`write_report` 会同时生成：

- `data/reports/evaluation.md` —— 人类可读的对比表格。
- `data/reports/evaluation.json` —— 机器可读的聚合数据 + 每次运行的明细。

## 诚实的边界

- `ts_forecast_mae` / `ts_forecast_mse` 来自**真实**的 statistical
  baseline 模型在留出切分上的结果 —— 是真实数字，而非脚本化字符串。
- `ts_anomaly_f1` 将标记出的点与仿真器注入的 ground-truth 窗口进行对比。
- `ScriptedLLM` 只是 *LLM* 的测试替身；它调用的每个 tool 都是真实实现。
