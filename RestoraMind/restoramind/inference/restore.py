# -*- coding: utf-8 -*-
"""单张图像复原：给定策略与工具箱，执行序列决策复原流程。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from PIL import Image

from ..agent.env import EnvConfig, RestorationEnv
from ..agent.policy import Policy
from ..agent.runner import run_episode
from ..agent.trajectory import Trajectory
from ..data.dataset import Sample
from ..data.synthesize import DegradationSpec
from ..tools.toolbox import Toolbox


@dataclass
class RestoreResult:
    """单图复原结果：复原图像、工具序列、复原日志与完整轨迹。"""
    image: Image.Image
    tool_sequence: List[str]
    restoration_log: List[str]
    num_turns: int
    finished: bool
    trajectory: Trajectory


def restore_image(image: Image.Image, policy: Policy, toolbox: Toolbox,
                  env_config: Optional[EnvConfig] = None,
                  degradation_hint: Optional[List[str]] = None) -> RestoreResult:
    """
    degradation_hint 为可选的退化类型提示；为 None 时环境不提供提示，由智能体自行推理
    （此时 env_config.provide_degradation_hint 被置为 False）。
    """
    cfg = env_config or EnvConfig()
    if degradation_hint is None:
        cfg.provide_degradation_hint = False
        specs: List[DegradationSpec] = []
    else:
        specs = [DegradationSpec(d, "unknown") for d in degradation_hint]
    sample = Sample(degraded=image.convert("RGB"), clean=None, degradations=specs)
    traj = run_episode(RestorationEnv(toolbox, cfg), policy, sample)
    return RestoreResult(
        image=traj.final_image, tool_sequence=traj.executed_tools, restoration_log=traj.restoration_log,
        num_turns=len(traj.steps), finished=traj.finished, trajectory=traj,
    )
