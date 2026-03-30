# 阶段 C 说明

阶段 C 的目标是把注意力模块真正接入分割基线，而不是只写独立模块文件。

当前已完成：

- `SEBlock`
- `CBAMBlock`
- `SE + CBAM` 串联版本
- UNet 主干的统一注意力接入
- 四组配置切换：
  - `baseline`
  - `se`
  - `cbam`
  - `se_cbam`

当前验证方式：

1. 四组配置都已完成前向形状验证。
2. `se`、`cbam`、`se_cbam` 三组都已完成最小训练并产出 checkpoint。

说明：

- 目前这一步的重点是“结构接入正确、训练链路可跑通”。
- 当前 toy dataset 很小、训练轮数只有 2，注意力模型尚未调优，因此不能据此判断注意力优于基线。
- 后续阶段需要通过正式消融实验统一训练策略、统一轮数和统一评估口径。

推荐命令：

```powershell
python -m research.engine.train --config configs/se_toy.yaml
python -m research.engine.train --config configs/cbam_toy.yaml
python -m research.engine.train --config configs/se_cbam_toy.yaml
```
