"""Cellpose 推理与研究流程一体化入口。

默认行为保持原有的 Cellpose 推理流程。
传入 ``--study`` 后，会一键运行研究流水线，包括训练 toy 模型、
生成可视化、进行 1200px 推理测速、汇总消融结果并生成报告草稿。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _has_option(args: list[str], option: str) -> bool:
    return any(arg == option or arg.startswith(option + "=") for arg in args)


def _remove_option(args: list[str], option: str) -> list[str]:
    result: list[str] = []
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg == option:
            skip_next = True
            continue
        if arg.startswith(option + "="):
            continue
        result.append(arg)
    return result


def _project_dir() -> Path:
    return Path(__file__).resolve().parent


def _run_command(cmd: list[str], cwd: Path, dry_run: bool = False) -> None:
    printable = " ".join(f'"{part}"' if " " in part else part for part in cmd)
    print(f"> {printable}", flush=True)
    if dry_run:
        return
    result = subprocess.run(cmd, cwd=str(cwd), check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def _find_latest_checkpoint(output_root: Path, experiment_name: str) -> Path | None:
    candidates = []
    prefix_length = len("YYYYMMDD_HHMMSS_")
    for candidate in output_root.glob("*/best.pt"):
        run_dir_name = candidate.parent.name
        if len(run_dir_name) <= prefix_length:
            continue
        if run_dir_name[:8].isdigit() and run_dir_name[9:15].isdigit() and run_dir_name[8] == "_":
            if run_dir_name[prefix_length:] == experiment_name:
                candidates.append(candidate)
    candidates.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _parse_study_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="通过单条命令端到端运行研究流水线。",
    )
    parser.add_argument("--study", action="store_true", help="运行研究流水线。")
    parser.add_argument(
        "--generate-toy-data",
        action="store_true",
        help="训练前重新生成 toy 数据集。",
    )
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="即使已有最佳 checkpoint，也重新训练全部模型。",
    )
    parser.add_argument(
        "--visual-limit",
        type=int,
        default=4,
        help="每个模型生成可视化的测试样本数量。",
    )
    parser.add_argument(
        "--benchmark-resize",
        type=int,
        default=1200,
        help="测速前将输入缩放到该正方形尺寸。",
    )
    parser.add_argument(
        "--benchmark-warmup",
        type=int,
        default=1,
        help="测速预热次数。",
    )
    parser.add_argument(
        "--benchmark-input",
        default=None,
        help="可选的测速输入路径，默认使用 toy 测试图像目录。",
    )
    parser.add_argument(
        "--report-output",
        default="reports/研究报告草稿.md",
        help="生成报告草稿的输出路径。",
    )
    parser.add_argument(
        "--demo-input",
        default=None,
        help="可选的研究模型推理演示输入图像或文件夹。",
    )
    parser.add_argument(
        "--demo-output",
        default="runs/infer_demo",
        help="可选研究模型推理演示的输出目录。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印计划执行的命令，不真正运行。",
    )
    return parser.parse_args(argv)


def _run_study_pipeline(argv: list[str]) -> None:
    args = _parse_study_args(argv)
    project_dir = _project_dir()
    python_exe = sys.executable

    from research.utils.config import load_config

    studies = [
        {
            "label": "baseline",
            "config": "configs/baseline_toy.yaml",
            "visual_dir": "reports/visualizations/baseline_toy",
            "benchmark_dir": "reports/benchmarks",
        },
        {
            "label": "se",
            "config": "configs/se_toy.yaml",
            "visual_dir": "reports/visualizations/se_toy",
            "benchmark_dir": "reports/benchmarks/se_toy",
        },
        {
            "label": "cbam",
            "config": "configs/cbam_toy.yaml",
            "visual_dir": "reports/visualizations/cbam_toy",
            "benchmark_dir": "reports/benchmarks/cbam_toy",
        },
        {
            "label": "se_cbam",
            "config": "configs/se_cbam_toy.yaml",
            "visual_dir": "reports/visualizations/se_cbam_toy",
            "benchmark_dir": "reports/benchmarks/se_cbam_toy",
        },
    ]

    baseline_config = load_config(studies[0]["config"])
    toy_dataset_root = Path(baseline_config["dataset"]["root"])
    benchmark_input = (
        Path(args.benchmark_input).expanduser().resolve()
        if args.benchmark_input
        else toy_dataset_root / "test" / baseline_config["dataset"].get("image_dir_name", "images")
    )

    if args.generate_toy_data or not toy_dataset_root.exists():
        print("正在准备 toy 数据集...", flush=True)
        _run_command(
            [
                python_exe,
                "-m",
                "research.datasets.generate_toy_dataset",
                "--output",
                str(toy_dataset_root),
            ],
            cwd=project_dir,
            dry_run=args.dry_run,
        )

    checkpoints: dict[str, Path] = {}
    for study in studies:
        config = load_config(study["config"])
        output_root = Path(config["runtime"]["output_root"])
        experiment_name = str(config["experiment"]["name"])
        checkpoint = None if args.force_train else _find_latest_checkpoint(output_root, experiment_name)
        if checkpoint is None:
            print(f"正在训练 {study['label']} 模型...", flush=True)
            _run_command(
                [python_exe, "-m", "research.engine.train", "--config", study["config"]],
                cwd=project_dir,
                dry_run=args.dry_run,
            )
            checkpoint = _find_latest_checkpoint(output_root, experiment_name)
        if checkpoint is None and not args.dry_run:
            raise SystemExit(f"未能为以下配置找到 checkpoint：{study['config']}")

        if checkpoint is not None:
            checkpoints[study["label"]] = checkpoint
            print(f"{study['label']} 使用 checkpoint：{checkpoint}", flush=True)

    if args.dry_run:
        fake_checkpoint = project_dir / "runs" / "<latest>" / "best.pt"
        for study in studies:
            checkpoints.setdefault(study["label"], fake_checkpoint)

    for study in studies:
        checkpoint = checkpoints[study["label"]]
        print(f"正在生成 {study['label']} 的可视化结果...", flush=True)
        _run_command(
            [
                python_exe,
                "-m",
                "research.engine.visualize",
                "--config",
                study["config"],
                "--checkpoint",
                str(checkpoint),
                "--split",
                "test",
                "--limit",
                str(args.visual_limit),
                "--output",
                study["visual_dir"],
            ],
            cwd=project_dir,
            dry_run=args.dry_run,
        )

        print(f"正在对 {study['label']} 执行 {args.benchmark_resize}px 推理测速...", flush=True)
        _run_command(
            [
                python_exe,
                "-m",
                "research.engine.benchmark",
                "--config",
                study["config"],
                "--checkpoint",
                str(checkpoint),
                "--input",
                str(benchmark_input),
                "--output",
                study["benchmark_dir"],
                "--resize",
                str(args.benchmark_resize),
                "--warmup",
                str(args.benchmark_warmup),
            ],
            cwd=project_dir,
            dry_run=args.dry_run,
        )

    if args.demo_input:
        print("正在运行可选的研究模型推理演示...", flush=True)
        _run_command(
            [
                python_exe,
                "-m",
                "research.engine.infer",
                "--config",
                studies[0]["config"],
                "--checkpoint",
                str(checkpoints["baseline"]),
                "--input",
                str(Path(args.demo_input).expanduser().resolve()),
                "--output",
                args.demo_output,
            ],
            cwd=project_dir,
            dry_run=args.dry_run,
        )

    print("正在汇总消融实验结果...", flush=True)
    _run_command(
        [python_exe, "-m", "research.engine.summarize_ablation", "--config", "configs/ablation_toy.yaml"],
        cwd=project_dir,
        dry_run=args.dry_run,
    )

    print("正在汇总测速结果...", flush=True)
    _run_command(
        [python_exe, "-m", "research.engine.summarize_benchmarks", "--config", "configs/benchmark_toy.yaml"],
        cwd=project_dir,
        dry_run=args.dry_run,
    )

    print("正在生成跨模型对比拼图...", flush=True)
    _run_command(
        [python_exe, "-m", "research.engine.compare_visualizations"],
        cwd=project_dir,
        dry_run=args.dry_run,
    )

    print("正在生成报告草稿...", flush=True)
    _run_command(
        [
            python_exe,
            "-m",
            "research.engine.generate_report",
            "--ablation",
            "reports/ablation_toy/ablation_summary.json",
            "--benchmark",
            "reports/benchmarks/benchmark_1200px.json",
            "--visuals",
            "reports/visualizations/baseline_toy",
            "--output",
            args.report_output,
        ],
        cwd=project_dir,
        dry_run=args.dry_run,
    )

    print("研究流水线执行完成。", flush=True)
    if not args.dry_run:
        print(f"报告：{(project_dir / args.report_output).resolve()}", flush=True)
        print(f"消融汇总：{(project_dir / 'reports/ablation_toy/ablation_summary.md').resolve()}", flush=True)
        print(
            f"测速汇总：{(project_dir / 'reports/benchmarks/summary/benchmark_summary.md').resolve()}",
            flush=True,
        )


def _run_cellpose_inference() -> None:
    project_dir = Path(__file__).resolve().parent
    main_py = project_dir / "main.py"
    fixed_output_dir = project_dir / "outputs"

    user_args = sys.argv[1:]
    args = _remove_option(list(user_args), "--output")
    if not _has_option(args, "--input"):
        args.extend(["--input", str(project_dir / "images")])
    args.extend(["--output", str(fixed_output_dir)])

    cmd = [sys.executable, str(main_py), *args]
    raise SystemExit(subprocess.run(cmd, check=False).returncode)


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--study" in argv:
        _run_study_pipeline(argv)
    else:
        _run_cellpose_inference()
