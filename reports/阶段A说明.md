# 阶段 A 说明

阶段 A 的目标不是训练模型，而是把研究型代码骨架搭起来。

当前已覆盖：

- 统一 YAML 配置加载
- 标准数据集目录约定
- 图像与标注一一对应检查
- 最小启动命令与结构化运行摘要输出

推荐验证命令：

```powershell
python -m research.engine.minimal_run --config configs/stage_a_minimal.yaml
```

如果后续补充了真实数据集，可把 `configs/stage_a_minimal.yaml` 中的 `dataset.root` 改为真实数据路径，再重新执行。
