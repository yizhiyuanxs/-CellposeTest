# 阶段 D 说明

阶段 D 的目标是把“能训练”升级成“能做对比实验并自动出表”。

当前已完成：

- 常规分割指标：
  - Dice
  - IoU
  - Precision
  - Recall
- 边界专项指标：
  - Boundary Precision
  - Boundary Recall
  - Boundary F1
- 四组实验自动汇总：
  - baseline
  - se
  - cbam
  - se_cbam
- 自动输出：
  - JSON
  - CSV
  - Markdown 表格

推荐命令：

```powershell
python -m research.engine.summarize_ablation --config configs/ablation_toy.yaml
```

说明：

- 这一步主要解决“评估口径统一”和“消融结果自动汇总”。
- 当前结果仍基于 toy dataset，只能作为流程验证，不代表正式课题结论。
