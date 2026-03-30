# 阶段 G 说明

阶段 G 的目标是把前面已经产出的代码结果自动汇总成一份 Markdown 报告草稿。

当前已支持整合：

- 消融实验结果
- 边界指标结果
- 速度测试结果
- 可视化样本目录
- 当前已完成项与待验证项

推荐命令：

```powershell
python -m research.engine.generate_report --ablation reports\ablation_toy\ablation_summary.json --benchmark reports\benchmarks\benchmark_1200px.json --visuals reports\visualizations\baseline_toy --output reports\研究报告草稿.md
```

说明：

- 当前生成的是“研究报告草稿”，不是最终论文定稿。
- 正式课程报告仍需加入数学原理推导、数据集背景、实验设置细节和最终结论论证。
