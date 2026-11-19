# 研究：Samsung Semiconductor Defect Agent

这篇笔记记录了设计灵感以及 FabPilot 与之分歧之处。它基于三星 semiconductor defect-classification 工作的
**公开描述**；FabPilot 不包含任何三星源代码，也不复刻任何专有模型。

## 我们从公开资料中学到了什么

三星的 semiconductor defect agent（据称与 high-bandwidth memory / 芯片制造相关）
被描述为结合了：

1. 缺陷显微图的**逐图像、逐晶圆 segmentation + defect classification**，
   用于确定哪些晶圆需要优先复查。
2. **Hypothesis 驱动的工作流** —— 操作员提出一个 hypothesis，系统收集证据来
   证实或反驳它，而不是盲目地对每个缺陷进行排序。
3. **一个反馈回路**，把发现的缺陷回灌到 classification / review pipeline 中，
   使系统随着已标注样本的积累而不断改进。

反复出现的主题是 *hypothesis 优先的分诊*：先提出一个 hypothesis 来缩小搜索空间，
再让证据来决定。

## FabPilot 借鉴了什么（概念层面）

FabPilot 把 **hypothesis 驱动的推理模式**（而非视觉模型）移植到
工业时序诊断中：

- 用户任务描述一个症状（例如「ETCH-01 的 gas_flow drift」）。
- **Diagnosis Agent** 生成显式的 hypotheses，并通过 tools（lag 分析、
  correlation、维护历史、相似案例）收集证据。
- 确定性的 **Hypothesis-Evidence Memory** 让每个 hypothesis 依次经历
  `candidate → active → supported / refuted / inconclusive`。
- 证据被*关联*到 hypotheses 上并*重新评估* —— 同样的
  证实/反驳循环，只不过作用于遥测数据而非显微图。

## FabPilot 有意分歧之处

| Samsung（公开描述） | FabPilot |
|---|---|
| 视觉：缺陷显微图分类 | 时序：传感器遥测、alarms、维护 |
| 由操作员提出 hypothesis 输入 | Supervisor + Diagnosis Agent 共同生成 hypotheses |
| 面向晶圆的缺陷优先级分诊 | 面向 hypotheses 的根因排序 |
| 专有视觉模型 | 真实 statistical baseline + 仅接口的 PatchTST/iTransformer adapters |
| （公开资料未详述） | 在共享状态中的完全开放、可审计的 ReAct 轨迹 |

## 为什么「hypothesis 优先」对诊断很重要

一个把每个统计量都倾泻到症状上的朴素 agent，会被相关 channel 淹没。
hypothesis 优先的分诊要求*首先*提出一个候选的因果叙事，然后用有针对性的证据来验证。
这就是为什么 FabPilot 把 correlation 当作 hypothesis 的**证据**，而绝不会单独当作 causation ——
正是这种自律让三星所报道的系统避免过度信任分类器的置信度。

## 参考资料

- 关于三星 semiconductor AI agent（HBM 缺陷分类与优先级复查）的公开报道 —— 参见对
  Samsung/ASML/Applied Materials fab 自动化公告的科技媒体报道。
- [ragent](https://github.com/infiniflow/ragent) —— 混合
  dense+sparse RRF 检索设计模式的来源（已在 `fabpilot/rag/` 中用 Python 重新实现）。
