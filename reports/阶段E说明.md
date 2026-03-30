# 阶段 E 说明

阶段 E 的目标是把实验输出从“数字”扩展到“图像证据”。

当前已支持生成：

- 原图
- GT 掩膜图
- 预测掩膜图
- 误差图
- Grad-CAM 风格热力图
- 边界对比图
- 六联图总览面板
- 多模型横向对比面板

推荐命令：

```powershell
python -m research.engine.visualize --config configs/baseline_toy.yaml --checkpoint runs\stage_b\<run>\best.pt --split test --limit 4 --output reports\visualizations\baseline_toy
```

说明：

- 当前热力图是基于分割输出目标对深层特征做梯度回传生成的，可用于观察模型关注区域。
- 当前可视化主要用于流程验证与报告插图准备。
- 后续若切换到真实数据集，可直接复用同一套脚本。

多模型对比命令：

```powershell
python -m research.engine.compare_visualizations --baseline reports\visualizations\baseline_toy --se reports\visualizations\se_toy --cbam reports\visualizations\cbam_toy --se-cbam reports\visualizations\se_cbam_toy --output reports\visualizations\ablation_compare
```
