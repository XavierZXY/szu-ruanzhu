# -*- coding: utf-8 -*-
"""
复合奖励。

步级奖励（用于 PPO 等逐步信用分配）：
    r_t = α·ΔQ_t + β·R_deg(D^_t, D) - γ·Cost(a_t) - δ
    终止奖励 r_final = κ·R_quality(I_T)，未在最大轮数内完成则给失败惩罚
轨迹级奖励（用于 GRPO 等序列级优化，与 VeRL 训练路径一致）：
    R = w_fmt·format + w_q·quality [+ w_deg·type_match] [+ w_mid·intermediate] [+ diversity_bonus]
    格式违规时仅保留负的格式项，不计算其他奖励。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..agent.trajectory import Trajectory
from .degradation_reward import degradation_set_reward, degradation_type_match, is_clean_answer
from .format_reward import FormatConfig, check_format_enhanced, check_format_progressive
from .quality_metrics import discretize, no_reference_score, quality_score


@dataclass
class RewardConfig:
    # 步级奖励系数
    """奖励配置：步级系数、轨迹级权重、质量指标与格式模式。"""
    alpha: float = 1.0
    beta: float = 0.5
    gamma: float = 0.05
    delta: float = 0.02
    kappa: float = 1.0
    failure_penalty: float = -0.5
    # 轨迹级奖励权重
    format_weight: float = 0.3
    quality_weight: float = 0.7
    enable_degradation_type_reward: bool = False
    degradation_type_weight: float = 1.0
    enable_intermediate_reward: bool = False
    intermediate_weight: float = 0.5
    tool_diversity_bonus: float = 0.1
    # 质量指标
    use_no_reference: bool = False
    use_lpips: bool = False
    discretize_levels: int = 0
    # 格式
    format_mode: str = "enhanced"          # enhanced | progressive
    format: FormatConfig = field(default_factory=FormatConfig)


@dataclass
class RewardBreakdown:
    """轨迹奖励的分项明细，便于训练监控。"""
    score: float
    format_score: float
    format_reason: str
    quality_score: float
    degradation_type_score: float = 0.0
    intermediate_quality_score: float = 0.0
    tool_diversity_bonus: float = 0.0
    clean_accuracy: Optional[float] = None
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """转换为扁平字典。"""
        d = {
            "score": self.score, "format_score": self.format_score, "format_reason": self.format_reason,
            "quality_score": self.quality_score, "degradation_type_score": self.degradation_type_score,
            "intermediate_quality_score": self.intermediate_quality_score,
            "tool_diversity_bonus": self.tool_diversity_bonus,
        }
        if self.clean_accuracy is not None:
            d["clean_accuracy"] = self.clean_accuracy
        d.update({f"metric_{k}": v for k, v in self.metrics.items()})
        return d


def _quality(image, reference, cfg: RewardConfig) -> Dict[str, float]:
    if cfg.use_no_reference or reference is None:
        return {"score": no_reference_score(image)}
    return quality_score(image, reference, use_lpips=cfg.use_lpips)


def compute_step_rewards(traj: Trajectory, cfg: Optional[RewardConfig] = None) -> List[float]:
    """按步计算 r_t；长度与 traj.steps 相同。"""
    cfg = cfg or RewardConfig()
    truth = traj.sample.degradation_types
    ref = traj.sample.clean
    q_cache: Dict[int, float] = {}

    def q(idx: int) -> float:
        if idx not in q_cache:
            q_cache[idx] = _quality(traj.image_history[idx], ref, cfg)["score"]
        return q_cache[idx]

    rewards: List[float] = []
    for step in traj.steps:
        if step.is_answer:
            r = cfg.kappa * q(step.image_index_after) if traj.finished else cfg.failure_penalty
            rewards.append(r)
            continue
        delta_q = q(step.image_index_after) - q(step.image_index_before)
        r_deg = degradation_set_reward(step.predicted_degradations, truth) if truth else 0.0
        r = cfg.alpha * delta_q + cfg.beta * r_deg - cfg.gamma * step.tool_cost - cfg.delta
        if step.format_errors or step.error:
            r -= 0.1
        rewards.append(r)
    if traj.truncated and not traj.finished and rewards:
        rewards[-1] += cfg.failure_penalty
    return rewards


def compute_episode_reward(traj: Trajectory, cfg: Optional[RewardConfig] = None) -> RewardBreakdown:
    """轨迹级复合奖励（GRPO 序列级）。"""
    cfg = cfg or RewardConfig()
    sample = traj.sample
    truth = sample.degradation_types
    is_clean = sample.is_clean
    fmt_cfg = FormatConfig(mode=traj.mode, max_tools_per_turn=cfg.format.max_tools_per_turn,
                           enable_total_tools_upper_limit=cfg.format.enable_total_tools_upper_limit,
                           allowed_tools=cfg.format.allowed_tools)
    checker = check_format_progressive if cfg.format_mode == "progressive" else check_format_enhanced
    fmt, reason = checker(traj.concatenated_response, len(truth), is_clean, fmt_cfg)

    if is_clean:
        acc = 1.0 if is_clean_answer(traj.restoration_log, traj.num_tool_calls) else 0.0
        score = cfg.format_weight * fmt + (cfg.quality_weight * acc if fmt > 0 else 0.0)
        return RewardBreakdown(score, fmt, reason, acc, clean_accuracy=acc)

    if fmt <= 0:
        return RewardBreakdown(cfg.format_weight * fmt, fmt, reason, 0.0)

    executed_any = len(traj.image_history) > 1
    metrics = _quality(traj.final_image, sample.clean, cfg) if executed_any else {"score": 0.0}
    q_final = discretize(metrics["score"], cfg.discretize_levels)
    score = cfg.format_weight * fmt + cfg.quality_weight * q_final

    deg_score = 0.0
    if cfg.enable_degradation_type_reward:
        predicted = traj.restoration_log or [d for s in traj.steps for d in s.predicted_degradations]
        deg_score = degradation_type_match(predicted, truth)
        score += cfg.degradation_type_weight * deg_score

    mid_score = 0.0
    if cfg.enable_intermediate_reward and len(traj.image_history) > 2:
        mids = traj.image_history[1:-1]
        mid_score = sum(no_reference_score(m) for m in mids) / len(mids)
        score += cfg.intermediate_weight * mid_score

    bonus = 0.0
    tools = traj.executed_tools
    if tools and len(set(tools)) == len(tools) and 1 <= len(tools) <= len(truth) + 1:
        bonus = cfg.tool_diversity_bonus
        score += bonus

    return RewardBreakdown(score, fmt, reason, q_final, deg_score, mid_score, bonus,
                           metrics={k: v for k, v in metrics.items() if k != "score"})
