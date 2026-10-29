# -*- coding: utf-8 -*-
"""轨迹采样：策略与环境交互直至终止。"""
from __future__ import annotations

from typing import Callable, Optional

from ..data.dataset import Sample
from .env import RestorationEnv
from .policy import Policy
from .trajectory import Trajectory


def run_episode(env: RestorationEnv, policy: Policy, sample: Sample,
                on_step: Optional[Callable[[int, str], None]] = None) -> Trajectory:
    """策略与环境交互直至终止，返回完整轨迹。"""
    obs = env.reset(sample)
    done = False
    while not done:
        response = policy.act(obs)
        obs, done, record = env.step(response)
        if on_step:
            on_step(record.turn, response)
    return env.trajectory
