# 报警解读参考

FabPilot 的合成 fab 环境会在某个物理 channel 的 rolling z-score 超过阈值时触发报警。
本参考文档说明如何解读这些报警，以及每个 channel 的报警通常意味着什么。

## Channel → 典型含义

| Channel | 典型报警含义 |
|---------|-----------------------|
| `gas_flow` | MFC drift / 设定值偏差，或上游气体供应问题 |
| `pressure` | 节流阀、泵或气体流量耦合问题 |
| `temperature` | 腔室热漂移、加热器/ESC 故障 |
| `rf_power` | Match-network / 发生器不稳定、打弧 |
| `bias_voltage` | Plasma-sheath / 晶圆 chuck 耦合问题 |
| `vacuum_pressure` | 泄漏、泵退化、放气 |
| `helium_flow` | 背面氦气泄漏（晶圆 chuck） |

## 正确解读报警

- 单个 channel 报警通常指向 channel 特定的故障
  （传感器或其测量的子系统）。
- 多个 channel 一起报警（例如 `rf_power` + `bias_voltage`，或
  `gas_flow` + `pressure`）通常指向一个共同的物理根因。
- 在判定根因之前，务必先调取该设备的维护与故障案例历史 —— correlation 不等于 causation。

## 优先级
- 高：`vacuum_pressure`、`rf_power`（存在晶圆报废 / 设备损坏风险）。
- 中：`gas_flow`、`pressure`（工艺漂移）。
- 低：`temperature`、`helium_flow`（逐渐退化）。
