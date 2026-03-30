# 阶段 F 说明

阶段 F 的目标是把速度要求转成真实可复现的 benchmark 输出。

当前已支持：

- 指定 checkpoint 做推理测速
- 指定输入目录批量测速
- 自动把输入统一 resize 到目标尺寸，例如 1200 像素
- 输出均值、Median、P95
- 自动判断是否满足 `<= 5s`
- 多模型 benchmark 汇总表

推荐命令：

```powershell
python -m research.engine.benchmark --config configs/baseline_toy.yaml --checkpoint runs\stage_b\<run>\best.pt --input data\toy_cells\test\images --resize 1200 --output reports\benchmarks
```

说明：

- 当前 benchmark 是基于现有模型和当前设备真实执行得到。
- 若换成真实课程模型或 GPU 设备，结论会变化，必须重新跑。

多模型汇总命令：

```powershell
python -m research.engine.summarize_benchmarks --config configs/benchmark_toy.yaml
```
