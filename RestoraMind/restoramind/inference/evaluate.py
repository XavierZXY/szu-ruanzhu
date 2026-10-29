# -*- coding: utf-8 -*-
"""
数据集评估：对每个样本运行复原流程，计算退化图/复原图相对真值的 PSNR、SSIM（可选 LPIPS）
与无参考质量分，统计工具调用次数与耗时，输出 CSV、汇总文本与图像。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import pandas as pd

from ..agent.env import EnvConfig, RestorationEnv
from ..agent.policy import Policy
from ..agent.runner import run_episode
from ..data.dataset import Sample
from ..reward.composite import RewardConfig, compute_episode_reward
from ..reward.quality_metrics import lpips_distance, no_reference_score, psnr, ssim
from ..tools.toolbox import Toolbox


@dataclass
class EvalSummary:
    """数据集评估汇总：平均指标与逐样本结果表。"""
    num_samples: int
    mean_psnr_degraded: float
    mean_psnr_restored: float
    mean_ssim_degraded: float
    mean_ssim_restored: float
    mean_no_ref_restored: float
    mean_tool_calls: float
    mean_seconds: float
    mean_reward: float
    table: pd.DataFrame = field(repr=False)

    def to_text(self) -> str:
        """生成汇总文本。"""
        return "\n".join([
            f"samples: {self.num_samples}",
            f"PSNR  degraded -> restored: {self.mean_psnr_degraded:.2f} -> {self.mean_psnr_restored:.2f}",
            f"SSIM  degraded -> restored: {self.mean_ssim_degraded:.4f} -> {self.mean_ssim_restored:.4f}",
            f"no-reference score (restored): {self.mean_no_ref_restored:.4f}",
            f"mean tool calls: {self.mean_tool_calls:.2f}",
            f"mean seconds per image: {self.mean_seconds:.2f}",
            f"mean episode reward: {self.mean_reward:.4f}",
        ])


def evaluate_samples(samples: Sequence[Sample], policy: Policy, toolbox: Toolbox,
                     env_config: Optional[EnvConfig] = None, reward_config: Optional[RewardConfig] = None,
                     output_dir: Optional[str | Path] = None, use_lpips: bool = False,
                     progress: Optional[Callable[[int, int], None]] = None) -> EvalSummary:
    """对样本集合运行完整复原流程并计算指标，可输出 CSV、汇总与图像。"""
    env_config = env_config or EnvConfig()
    reward_config = reward_config or RewardConfig()
    rows: List[Dict] = []
    out = Path(output_dir) if output_dir else None
    if out:
        (out / "images").mkdir(parents=True, exist_ok=True)
        (out / "details").mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(samples):
        env = RestorationEnv(toolbox, env_config)
        start = time.time()
        traj = run_episode(env, policy, sample)
        elapsed = time.time() - start
        breakdown = compute_episode_reward(traj, reward_config)
        row: Dict = {
            "sample_id": sample.index, "gt_degradations": ", ".join(sample.degradation_types),
            "tool_sequence": " -> ".join(traj.executed_tools), "restoration_log": ", ".join(traj.restoration_log),
            "num_tool_calls": traj.num_tool_calls, "num_turns": len(traj.steps), "finished": traj.finished,
            "seconds": elapsed, "reward": breakdown.score, "format_score": breakdown.format_score,
            "no_ref_restored": no_reference_score(traj.final_image),
        }
        if sample.clean is not None:
            row.update({
                "psnr_degraded": psnr(sample.degraded, sample.clean), "psnr_restored": psnr(traj.final_image, sample.clean),
                "ssim_degraded": ssim(sample.degraded, sample.clean), "ssim_restored": ssim(traj.final_image, sample.clean),
            })
            if use_lpips:
                row["lpips_degraded"] = lpips_distance(sample.degraded, sample.clean)
                row["lpips_restored"] = lpips_distance(traj.final_image, sample.clean)
        rows.append(row)
        if out:
            d = out / "images" / f"sample_{sample.index:04d}"
            d.mkdir(exist_ok=True)
            sample.degraded.save(d / "degraded.png")
            traj.final_image.save(d / "restored.png")
            if sample.clean is not None:
                sample.clean.save(d / "gt.png")
            (out / "details" / f"sample_{sample.index:04d}.json").write_text(
                json.dumps({"responses": traj.responses, "row": row}, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8")
        if progress:
            progress(i + 1, len(samples))

    table = pd.DataFrame(rows)
    has_ref = "psnr_restored" in table.columns and len(table) > 0
    summary = EvalSummary(
        num_samples=len(table),
        mean_psnr_degraded=float(table["psnr_degraded"].mean()) if has_ref else float("nan"),
        mean_psnr_restored=float(table["psnr_restored"].mean()) if has_ref else float("nan"),
        mean_ssim_degraded=float(table["ssim_degraded"].mean()) if has_ref else float("nan"),
        mean_ssim_restored=float(table["ssim_restored"].mean()) if has_ref else float("nan"),
        mean_no_ref_restored=float(table["no_ref_restored"].mean()) if len(table) else float("nan"),
        mean_tool_calls=float(table["num_tool_calls"].mean()) if len(table) else 0.0,
        mean_seconds=float(table["seconds"].mean()) if len(table) else 0.0,
        mean_reward=float(table["reward"].mean()) if len(table) else 0.0,
        table=table,
    )
    if out:
        table.to_csv(out / "evaluation_results.csv", index=False)
        (out / "summary.txt").write_text(summary.to_text(), encoding="utf-8")
    return summary
