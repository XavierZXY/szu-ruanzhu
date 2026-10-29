# -*- coding: utf-8 -*-
"""
组相对策略优化（GRPO）训练循环。

对每个训练样本采样 G 条轨迹，以组内奖励均值/标准差归一化得到优势
    A_i = (r_i - mean(r)) / (std(r) + ε)
再交由可训练策略按优势更新参数。该循环与具体策略实现解耦：
    - SoftmaxToolPolicy   : 轻量表格化策略，可在 CPU 上完整运行（测试 / 原型）
    - 多模态大模型策略    : 由 VeRL 框架负责参数更新，本模块通过 verl_plugin 提供奖励函数
"""
from __future__ import annotations

import statistics
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence

from ..agent.env import RestorationEnv
from ..agent.policy import Policy
from ..agent.runner import run_episode
from ..agent.trajectory import Trajectory
from ..data.dataset import Sample
from ..reward.composite import RewardConfig, compute_episode_reward


class TrainablePolicy(Policy):
    """可训练策略接口：在轨迹采样前后记录动作，并按优势更新。"""
    @abstractmethod
    def begin_episode(self) -> None:
        """开始记录一条轨迹的动作。"""

    @abstractmethod
    def end_episode(self) -> List[object]:
        """返回本轨迹中记录的动作（供 update 使用）。"""
    @abstractmethod
    def update(self, episodes: Sequence[List[object]], advantages: Sequence[float]) -> Dict[str, float]:
        """按组相对优势更新策略参数并返回统计。"""


def group_relative_advantages(rewards: Sequence[float], eps: float = 1e-6) -> List[float]:
    """计算组内标准化优势 (r-mean)/(std+eps)。"""
    if len(rewards) == 0:
        return []
    if len(rewards) == 1:
        return [0.0]
    mean = statistics.fmean(rewards)
    std = statistics.pstdev(rewards)
    return [(r - mean) / (std + eps) for r in rewards]


@dataclass
class GRPOConfig:
    """GRPO 配置：组大小、训练轮数与奖励配置。"""
    group_size: int = 4
    epochs: int = 1
    reward: RewardConfig = field(default_factory=RewardConfig)


@dataclass
class TrainStats:
    """单个训练步的统计量。"""
    step: int
    mean_reward: float
    mean_quality: float
    format_violation_rate: float
    mean_tool_calls: float
    policy_stats: Dict[str, float] = field(default_factory=dict)


class GRPOTrainer:
    """组相对策略优化训练循环，与具体策略实现解耦。"""
    def __init__(self, env_factory: Callable[[], RestorationEnv], policy: TrainablePolicy,
                 config: GRPOConfig | None = None):
        self.env_factory = env_factory
        self.policy = policy
        self.config = config or GRPOConfig()
        self.history: List[TrainStats] = []

    def rollout_group(self, sample: Sample) -> tuple[List[Trajectory], List[List[object]]]:
        """对同一样本采样一组轨迹。"""
        trajs, actions = [], []
        for _ in range(self.config.group_size):
            env = self.env_factory()
            self.policy.begin_episode()
            trajs.append(run_episode(env, self.policy, sample))
            actions.append(self.policy.end_episode())
        return trajs, actions

    def train_step(self, samples: Sequence[Sample]) -> TrainStats:
        """对一批样本完成采样、奖励、优势计算与策略更新。"""
        all_actions: List[List[object]] = []
        all_adv: List[float] = []
        rewards, qualities, violations, calls = [], [], [], []
        for sample in samples:
            trajs, actions = self.rollout_group(sample)
            breakdowns = [compute_episode_reward(t, self.config.reward) for t in trajs]
            group_rewards = [b.score for b in breakdowns]
            all_actions.extend(actions)
            all_adv.extend(group_relative_advantages(group_rewards))
            rewards.extend(group_rewards)
            qualities.extend(b.quality_score for b in breakdowns)
            violations.extend(1.0 if b.format_score <= 0 else 0.0 for b in breakdowns)
            calls.extend(float(t.num_tool_calls) for t in trajs)
        policy_stats = self.policy.update(all_actions, all_adv)
        stats = TrainStats(
            step=len(self.history) + 1,
            mean_reward=statistics.fmean(rewards) if rewards else 0.0,
            mean_quality=statistics.fmean(qualities) if qualities else 0.0,
            format_violation_rate=statistics.fmean(violations) if violations else 0.0,
            mean_tool_calls=statistics.fmean(calls) if calls else 0.0,
            policy_stats=policy_stats,
        )
        self.history.append(stats)
        return stats

    def fit(self, samples: Sequence[Sample], batch_size: int = 4,
            on_step: Callable[[TrainStats], None] | None = None) -> List[TrainStats]:
        """按批次遍历样本完成多轮训练，返回训练历史。"""
        for _ in range(self.config.epochs):
            for i in range(0, len(samples), batch_size):
                stats = self.train_step(samples[i:i + batch_size])
                if on_step:
                    on_step(stats)
        return self.history
