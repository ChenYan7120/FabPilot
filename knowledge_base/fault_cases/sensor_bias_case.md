# 故障案例：pressure 传感器偏差（ETCH-02）

## 摘要
ETCH-02 上的 `pressure` channel 连续两天相对其 baseline 读出恒定的 +0.9 Pa 偏移，
且没有物理工艺变化。

## 证据链
1. `pressure` 残差（观测值 − baseline）恒定 ≈ +0.9 Pa。
2. `vacuum_pressure`（冗余）没有出现该偏移，并如预期保持与
   `gas_flow` 耦合。
3. 其他任何 channel 均无报警。

## 根因
`pressure` 传感器的零点偏移（接线/接地环路）。

## 处置
重新调零传感器消除了偏移。检查了电缆屏蔽层并重新做了端接。

## 经验教训
当物理上冗余的 channel 一致时，恒定的单 channel 偏移是传感器偏差的指纹，
而非工艺漂移。真正的工艺漂移会让耦合的 channel 一起变化。
