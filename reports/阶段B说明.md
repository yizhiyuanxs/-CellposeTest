# 阶段 B 说明

阶段 B 的目标是先建立一个最小可训练的分割基线闭环。

当前已规划并落地：

- Toy dataset 生成脚本
- 标准分割数据集读取类
- 小型 UNet 基线
- 训练脚本
- 评估脚本
- 推理脚本

推荐命令：

```powershell
python -m research.datasets.generate_toy_dataset --output data/toy_cells
python -m research.engine.train --config configs/baseline_toy.yaml
python -m research.engine.evaluate --config configs/baseline_toy.yaml --checkpoint runs/stage_b/<run>/best.pt --split val
python -m research.engine.infer --config configs/baseline_toy.yaml --checkpoint runs/stage_b/<run>/best.pt --input data/toy_cells/test/images --output runs/infer_demo
```

说明：

- 这套流程用于验证“训练/验证/推理闭环”已经可运行。
- 真正的课程实验数据集接入后，只需要替换配置中的 `dataset.root` 即可继续使用。
