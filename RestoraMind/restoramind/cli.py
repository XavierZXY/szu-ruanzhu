# -*- coding: utf-8 -*-
"""
命令行入口：restoramind <子命令>

    make-data   从干净图像目录合成复合退化数据集（Parquet）
    serve-tools 启动复原工具 HTTP 服务
    train       GRPO 训练（轻量表格化策略；大模型训练见 scripts/train_verl.sh）
    restore     单图复原
    evaluate    数据集评估
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from PIL import Image

from . import __software_name__, __version__


def _build_policy(args):
    from .agent.policy import HeuristicPolicy, OpenAICompatiblePolicy
    if args.policy == "openai":
        return OpenAICompatiblePolicy(base_url=args.api_url, api_key=args.api_key, model=args.model)
    if args.policy == "softmax":
        from .train.softmax_policy import SoftmaxToolPolicy
        if not args.policy_file:
            raise SystemExit("--policy softmax 需要同时指定 --policy-file（train 子命令产出的 policy.json）")
        policy = SoftmaxToolPolicy(seed=0)
        policy.load(args.policy_file)
        return policy
    return HeuristicPolicy(mode=args.mode)


def _build_toolbox(args):
    from .tools.toolbox import build_toolbox
    return build_toolbox(args.toolbox)


def cmd_make_data(args) -> int:
    """子命令 make-data：从干净图像目录合成复合退化数据集。"""
    from .agent.prompts import build_system_prompt, build_first_user_prompt
    from .data.dataset import build_parquet_dataset, list_images
    images = list_images(args.clean_dir)
    if not images:
        print(f"未在 {args.clean_dir} 找到图像", file=sys.stderr)
        return 1
    hint_template = "<image>\nImage analysis indicates these degradation types (order unknown): {degradations}\nSelect the FIRST tool to apply."
    n = build_parquet_dataset(images, args.output, build_system_prompt(args.mode), hint_template, seed=args.seed,
                              clean_prob=args.clean_prob, max_size=args.max_size,
                              samples_per_image=args.samples_per_image)
    print(f"写入 {n} 个样本 -> {args.output}")
    return 0


def cmd_serve_tools(args) -> int:
    """子命令 serve-tools：启动复原工具 HTTP 服务。"""
    from .tool_server import create_app, load_backend
    app = create_app(backend=load_backend(args.backend), family=args.family)
    app.run(host=args.host, port=args.port)
    return 0


def cmd_train(args) -> int:
    """子命令 train：运行轻量策略的 GRPO 训练。"""
    from .agent.env import EnvConfig, RestorationEnv
    from .data.dataset import load_parquet_samples
    from .reward.composite import RewardConfig
    from .train.grpo import GRPOConfig, GRPOTrainer
    from .train.softmax_policy import SoftmaxToolPolicy
    toolbox = _build_toolbox(args)
    samples = load_parquet_samples(args.data, args.num_samples)
    policy = SoftmaxToolPolicy(lr=args.lr, seed=args.seed)
    trainer = GRPOTrainer(lambda: RestorationEnv(toolbox, EnvConfig(mode="single_tool_iterative", max_turns=args.max_turns)),
                          policy, GRPOConfig(group_size=args.group_size, epochs=args.epochs, reward=RewardConfig()))
    trainer.fit(samples, batch_size=args.batch_size,
                on_step=lambda s: print(f"step {s.step}: reward={s.mean_reward:.4f} quality={s.mean_quality:.4f} "
                                        f"violations={s.format_violation_rate:.2f} tools={s.mean_tool_calls:.2f}"))
    policy.save(args.output)
    print(f"策略已保存 -> {args.output}")
    return 0


def cmd_restore(args) -> int:
    """子命令 restore：对单张图像执行复原。"""
    from .agent.env import EnvConfig
    from .inference.restore import restore_image
    policy = _build_policy(args)    # 先校验策略参数，再读取图像
    image = Image.open(args.input).convert("RGB")
    hint: Optional[List[str]] = [x.strip() for x in args.degradations.split(",") if x.strip()] if args.degradations else None
    result = restore_image(image, policy, _build_toolbox(args),
                           EnvConfig(mode=args.mode, max_turns=args.max_turns), hint)
    result.image.save(args.output)
    print(f"工具序列: {' -> '.join(result.tool_sequence) or '(无)'}")
    print(f"复原结果 -> {args.output}")
    return 0


def cmd_evaluate(args) -> int:
    """子命令 evaluate：对数据集执行评估并输出结果。"""
    from .agent.env import EnvConfig
    from .data.dataset import load_parquet_samples
    from .inference.evaluate import evaluate_samples
    samples = load_parquet_samples(args.data, args.num_samples)
    summary = evaluate_samples(samples, _build_policy(args), _build_toolbox(args),
                               EnvConfig(mode=args.mode, max_turns=args.max_turns,
                                         provide_degradation_hint=not args.no_hint), output_dir=args.output)
    print(summary.to_text())
    return 0


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--mode", default="single_tool_iterative", choices=["single_tool_iterative", "multi_tool_planning"])
    p.add_argument("--toolbox", default="local", choices=["local", "remote"], help="local=经典算子；remote=远程专家模型服务")
    p.add_argument("--policy", default="heuristic", choices=["heuristic", "softmax", "openai"],
                   help="heuristic=规则策略；softmax=train 产出的轻量策略（需 --policy-file）；openai=vLLM 部署的多模态智能体")
    p.add_argument("--policy-file", default=None, help="softmax 策略参数文件（policy.json）")
    p.add_argument("--api-url", default="http://localhost:8000/v1")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--model", default=None)
    p.add_argument("--max-turns", type=int, default=6)


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(prog="restoramind", description=f"{__software_name__} V{__version__}")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("make-data", help="合成复合退化数据集")
    p.add_argument("--clean-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--mode", default="single_tool_iterative", choices=["single_tool_iterative", "multi_tool_planning"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--clean-prob", type=float, default=0.0)
    p.add_argument("--max-size", type=int, default=512)
    p.add_argument("--samples-per-image", type=int, default=1)
    p.set_defaults(func=cmd_make_data)

    p = sub.add_parser("serve-tools", help="启动复原工具服务")
    p.add_argument("--family", default=None, help="仅服务某个模型家族，如 SCUNet")
    p.add_argument("--backend", default="classical", help="classical 或 module:function")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=5000)
    p.set_defaults(func=cmd_serve_tools)

    p = sub.add_parser("train", help="GRPO 训练（轻量策略）")
    _common(p)
    p.add_argument("--data", required=True)
    p.add_argument("--output", default="policy.json")
    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("restore", help="单图复原")
    _common(p)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--degradations", default=None, help="可选的退化类型提示，逗号分隔")
    p.set_defaults(func=cmd_restore)

    p = sub.add_parser("evaluate", help="数据集评估")
    _common(p)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument("--no-hint", action="store_true",
                   help="不向智能体提供真实退化类型提示（检验自行诊断能力；规则策略依赖提示，此时会直接作答）")
    p.set_defaults(func=cmd_evaluate)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """命令行入口，返回进程退出码。"""
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
