# -*- coding: utf-8 -*-
"""运行离线评估与训练，保存说明书引用的真实数值与训练曲线图。"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent
font_manager.fontManager.addfont("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
plt.rcParams["font.family"] = "WenQuanYi Zen Hei"

from restoramind.agent.env import EnvConfig, RestorationEnv  # noqa: E402
from restoramind.agent.policy import HeuristicPolicy  # noqa: E402
from restoramind.data.dataset import Sample  # noqa: E402
from restoramind.data.synthesize import synthesize_sample  # noqa: E402
from restoramind.inference.evaluate import evaluate_samples  # noqa: E402
from restoramind.reward.composite import RewardConfig  # noqa: E402
from restoramind.tools.toolbox import build_toolbox  # noqa: E402
from restoramind.train.grpo import GRPOConfig, GRPOTrainer  # noqa: E402
from restoramind.train.softmax_policy import SoftmaxToolPolicy  # noqa: E402


def make_images(n, size=128):
    imgs = []
    for i in range(n):
        yy, xx = np.mgrid[0:size, 0:size] / float(size)
        f = 4 + (i % 7)
        img = np.stack([255 * (0.5 + 0.5 * np.sin(f * np.pi * xx)), 255 * yy ** (1 + 0.1 * (i % 5)),
                        255 * (0.5 + 0.5 * np.cos((3 + i % 4) * np.pi * yy * xx))], -1)
        a = 20 + (i * 7) % 40
        img[a:a + 40, a:a + 40] = 200 + (i * 11) % 50
        imgs.append(np.clip(img, 0, 255).astype(np.uint8))
    return imgs


def make_samples(n, seed):
    rng = np.random.default_rng(seed)
    out = []
    for i, clean in enumerate(make_images(n)):
        syn = synthesize_sample(clean, rng, num_degradations=(2, 3))
        out.append(Sample(Image.fromarray(syn.degraded), Image.fromarray(clean), syn.degradations, i))
    return out


if __name__ == "__main__":
    toolbox = build_toolbox("local")
    samples = make_samples(40, seed=123)
    summary = evaluate_samples(samples, HeuristicPolicy(), toolbox, EnvConfig(max_turns=6))
    t = summary.table
    n_degs = np.array([len(s.degradations) for s in samples])
    result = {
        "num_samples": summary.num_samples,
        "psnr_before": summary.mean_psnr_degraded, "psnr_after": summary.mean_psnr_restored,
        "ssim_before": summary.mean_ssim_degraded, "ssim_after": summary.mean_ssim_restored,
        "mean_tool_calls": summary.mean_tool_calls, "mean_degradations": float(n_degs.mean()),
        "mean_seconds": summary.mean_seconds, "mean_reward": summary.mean_reward,
        "format_violations": int((t["format_score"] <= 0).sum()),
        "psnr_improved_ratio": float((t["psnr_restored"] > t["psnr_degraded"]).mean()),
        "ssim_improved_ratio": float((t["ssim_restored"] > t["ssim_degraded"]).mean()),
    }

    # 线性关系验证：工具调用次数 vs 退化数量
    by_n = {int(k): float(v) for k, v in t.assign(n=n_degs).groupby("n")["num_tool_calls"].mean().items()}
    result["tool_calls_by_num_degradations"] = by_n

    # GRPO 训练曲线（轻量表格化策略）
    train_samples = make_samples(12, seed=7)
    policy = SoftmaxToolPolicy(lr=0.8, seed=0)
    trainer = GRPOTrainer(lambda: RestorationEnv(toolbox, EnvConfig(max_turns=6)), policy,
                          GRPOConfig(group_size=4, epochs=6, reward=RewardConfig()))
    hist = trainer.fit(train_samples, batch_size=4)
    result["train_steps"] = [{"step": h.step, "mean_reward": h.mean_reward, "mean_quality": h.mean_quality,
                              "mean_tool_calls": h.mean_tool_calls} for h in hist]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot([h.step for h in hist], [h.mean_reward for h in hist], "o-", label="平均轨迹奖励")
    ax.plot([h.step for h in hist], [h.mean_quality for h in hist], "s--", label="平均质量得分")
    ax.set_xlabel("训练步"); ax.set_ylabel("得分"); ax.set_title("GRPO 训练曲线（轻量策略，离线工具）")
    ax.grid(alpha=0.3); ax.legend()
    fig.savefig(OUT / "figures" / "fig6_train_curve.png", dpi=160, bbox_inches="tight"); plt.close(fig)

    (OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
