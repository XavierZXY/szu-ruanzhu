# -*- coding: utf-8 -*-
"""生成软件说明书中的示意图（matplotlib，纯代码绘制，可重复生成）。"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib import font_manager
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

for f in ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",):
    if Path(f).exists():
        font_manager.fontManager.addfont(f)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
plt.rcParams["axes.unicode_minus"] = False

BLUE, GREEN, ORANGE, GRAY, RED = "#dbe8f7", "#dcefd9", "#fbe6c8", "#eeeeee", "#f6d5d5"


def box(ax, x, y, w, h, text, color=BLUE, fs=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec="#333333", lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, p, q, text=None, rad=0.0, fs=8):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, lw=1.1, color="#333333",
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 0.12, text, ha="center", fontsize=fs)


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def fig_architecture():
    fig, ax = canvas(10, 6.4)
    box(ax, 0.3, 4.6, 2.0, 0.9, "数据模块\n退化合成 / Parquet", GREEN)
    box(ax, 0.3, 3.3, 2.0, 0.9, "统计先验模块\n亮度·边缘·噪声·雾", GREEN)
    box(ax, 3.0, 4.6, 2.2, 0.9, "智能体环境\n状态 / 动作 / 转移", BLUE)
    box(ax, 3.0, 3.3, 2.2, 0.9, "策略\n多模态大模型 / 规则", BLUE)
    box(ax, 5.9, 4.6, 2.0, 0.9, "工具箱\n28 个复原工具", ORANGE)
    box(ax, 5.9, 3.3, 2.0, 0.9, "工具服务\nHTTP /process", ORANGE)
    box(ax, 8.4, 4.6, 1.4, 0.9, "奖励模块", RED)
    box(ax, 3.0, 1.7, 2.2, 0.9, "推理模块\n单图复原 / 评估", GRAY)
    box(ax, 0.3, 1.7, 2.0, 0.9, "命令行 / 脚本", GRAY)
    box(ax, 5.9, 1.7, 2.3, 0.9, "专家模型\nSwinIR·Restormer·SCUNet…", GRAY, fs=8)
    box(ax, 8.4, 1.7, 1.4, 0.9, "GRPO 训练", RED)
    arrow(ax, (2.3, 5.05), (3.0, 5.05)); arrow(ax, (2.3, 3.75), (3.0, 4.7), rad=-0.1)
    arrow(ax, (4.1, 4.6), (4.1, 4.2)); arrow(ax, (5.2, 5.05), (5.9, 5.05)); arrow(ax, (6.9, 4.6), (6.9, 4.2))
    arrow(ax, (7.9, 5.05), (8.4, 5.05)); arrow(ax, (9.1, 4.6), (9.1, 2.6)); arrow(ax, (8.4, 2.15), (5.2, 3.5), rad=-0.15); ax.text(8.0, 3.0, "更新策略参数", fontsize=8)
    arrow(ax, (6.9, 3.3), (6.9, 2.6)); arrow(ax, (4.1, 3.3), (4.1, 2.6)); arrow(ax, (2.3, 2.15), (3.0, 2.15))
    ax.text(5, 0.7, "训练：环境采样轨迹 → 奖励 → 组相对优势 → 更新策略      推理：策略逐步决策 → 工具执行 → 输出复原图", ha="center", fontsize=9)
    fig.savefig(OUT / "fig1_architecture.png", dpi=170, bbox_inches="tight"); plt.close(fig)


def fig_flow():
    fig, ax = canvas(10.3, 6.6)
    box(ax, 0.3, 4.9, 2.0, 0.8, "输入退化图像 $I_0$", GRAY)
    box(ax, 3.0, 4.9, 2.4, 0.8, "提取统计先验 $F_t$", GREEN)
    box(ax, 6.1, 4.9, 3.5, 0.8, "构造状态 $S_t=(I_t, F_t,$ 工具史, 退化史$)$", BLUE)
    box(ax, 0.3, 3.3, 3.0, 0.8, "退化推理：剩余退化 $\\hat{D}_t$", BLUE)
    box(ax, 3.9, 3.3, 3.0, 0.8, "工具选择：同类候选 + 先验", ORANGE)
    box(ax, 7.5, 3.3, 2.1, 0.8, "执行工具 $T_t$", ORANGE)
    box(ax, 7.5, 1.8, 2.1, 0.8, "得到 $I_{t+1}$，更新状态", GREEN)
    box(ax, 3.9, 1.8, 3.0, 0.8, "终止判断\n轮数/剩余退化/收益", RED)
    box(ax, 0.3, 1.8, 3.0, 0.8, "输出最终复原图 + 日志", GRAY)
    arrow(ax, (2.3, 5.3), (3.0, 5.3)); arrow(ax, (5.4, 5.3), (6.1, 5.3)); arrow(ax, (7.8, 4.9), (2.0, 4.1), rad=0.0)
    arrow(ax, (3.3, 3.7), (3.9, 3.7)); arrow(ax, (6.9, 3.7), (7.5, 3.7)); arrow(ax, (8.55, 3.3), (8.55, 2.6))
    arrow(ax, (7.5, 2.2), (6.9, 2.2)); arrow(ax, (3.9, 2.2), (3.3, 2.2), "结束")
    ax.plot([9.6, 9.95, 9.95], [2.2, 2.2, 5.3], color="#333333", lw=1.1)
    arrow(ax, (9.95, 5.3), (9.6, 5.3))
    ax.text(9.8, 3.9, "继续", rotation=90, ha="center", va="center", fontsize=9)
    ax.text(5, 0.9, "每一步仅调用一个被选工具，工具调用次数随退化类型数近似线性增长；回退重试受固定轮数和成本惩罚约束。", ha="center", fontsize=9)
    fig.savefig(OUT / "fig2_flow.png", dpi=170, bbox_inches="tight"); plt.close(fig)


def fig_reward():
    fig, ax = canvas(10, 4.6)
    box(ax, 0.3, 2.6, 2.2, 0.9, "图像质量增益\n$\\alpha\\Delta Q_t$", GREEN)
    box(ax, 2.8, 2.6, 2.2, 0.9, "退化推理奖励\n$\\beta R_{deg}$", BLUE)
    box(ax, 5.3, 2.6, 2.2, 0.9, "工具成本惩罚\n$-\\gamma Cost(a_t)$", RED)
    box(ax, 7.8, 2.6, 1.9, 0.9, "步骤惩罚\n$-\\delta$", RED)
    box(ax, 2.6, 0.9, 4.8, 0.9, "$r_t=\\alpha\\Delta Q_t+\\beta R_{deg}-\\gamma Cost-\\delta$    终止：$\\kappa R_{quality}(I_T)$", ORANGE)
    for x in (1.4, 3.9, 6.4, 8.75):
        arrow(ax, (x, 2.6), (min(max(x, 3.2), 6.8), 1.8))
    fig.savefig(OUT / "fig3_reward.png", dpi=170, bbox_inches="tight"); plt.close(fig)


def fig_selection():
    fig, ax = canvas(10, 4.8)
    box(ax, 0.3, 2.0, 2.3, 1.2, "当前图像\n退化类型：低分辨率\n边缘密度 / 纹理强度", GRAY)
    box(ax, 3.4, 2.0, 2.3, 1.2, "智能体策略\n$\\pi(a|S)$", BLUE)
    box(ax, 6.6, 3.2, 3.1, 0.8, "边缘密度低 → 稳健型工具", GREEN)
    box(ax, 6.6, 1.0, 3.1, 0.8, "边缘密度高 → 细节恢复型工具", ORANGE)
    arrow(ax, (2.6, 2.6), (3.4, 2.6)); arrow(ax, (5.7, 2.8), (6.6, 3.6)); arrow(ax, (5.7, 2.4), (6.6, 1.4))
    fig.savefig(OUT / "fig4_selection.png", dpi=170, bbox_inches="tight"); plt.close(fig)


def fig_demo():
    """真实运行一次离线流程，把退化图 / 复原图 / 原图并排保存。"""
    from restoramind.agent.env import EnvConfig, RestorationEnv
    from restoramind.agent.policy import HeuristicPolicy
    from restoramind.agent.runner import run_episode
    from restoramind.data.dataset import Sample
    from restoramind.data.synthesize import synthesize_sample
    from restoramind.reward.quality_metrics import psnr, ssim
    from restoramind.tools.toolbox import build_toolbox

    size = 160
    yy, xx = np.mgrid[0:size, 0:size] / float(size)
    img = np.stack([255 * (0.5 + 0.5 * np.sin(7 * np.pi * xx)), 255 * yy, 255 * (0.5 + 0.5 * np.cos(5 * np.pi * yy * xx))], -1)
    img[40:90, 40:90] = 235
    clean = np.clip(img, 0, 255).astype(np.uint8)
    syn = synthesize_sample(clean, np.random.default_rng(4), num_degradations=(2,))
    sample = Sample(Image.fromarray(syn.degraded), Image.fromarray(clean), syn.degradations, 0)
    traj = run_episode(RestorationEnv(build_toolbox("local"), EnvConfig(max_turns=6)), HeuristicPolicy(), sample)
    panels = [("原始清晰图像", clean), (f"退化图像\n{' + '.join(syn.degradation_types)}", np.array(sample.degraded)),
              (f"复原图像\n{' → '.join(traj.executed_tools)}", np.array(traj.final_image))]
    fig, axes = plt.subplots(1, 3, figsize=(10, 4.2))
    for a, (t, im) in zip(axes, panels):
        a.imshow(im); a.set_title(t, fontsize=8.5, pad=8); a.axis("off")
    d, r = np.array(sample.degraded), np.array(traj.final_image)
    fig.suptitle("离线演示：原始图像、退化图像与复原图像", fontsize=11, y=0.97)
    fig.subplots_adjust(top=0.76)
    fig.savefig(OUT / "fig5_demo.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    return {"psnr_before": psnr(d, clean), "psnr_after": psnr(r, clean), "ssim_before": ssim(d, clean),
            "ssim_after": ssim(r, clean), "degradations": syn.degradation_types, "tools": traj.executed_tools}


if __name__ == "__main__":
    fig_architecture(); fig_flow(); fig_reward(); fig_selection()
    print(fig_demo())
