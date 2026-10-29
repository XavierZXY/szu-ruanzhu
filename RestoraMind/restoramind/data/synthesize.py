# -*- coding: utf-8 -*-
"""
复合退化合成：对干净图像随机施加 2~3 种退化（随机类型、随机强度、随机顺序），
并记录真实退化序列，作为训练阶段退化推理奖励的参考。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from .degradations import DEGRADATION_LEVELS, DEGRADATION_TYPES, apply_degradation


@dataclass
class DegradationSpec:
    """一次退化施加的记录：类型 + 强度。"""
    degradation_type: str
    degradation_level: str

    def to_dict(self) -> dict:
        """转换为可序列化的字典。"""
        return {"degradation_type": self.degradation_type, "degradation_level": self.degradation_level}


@dataclass
class SynthesizedSample:
    """合成样本：干净图、退化图、按施加顺序排列的退化序列。"""
    clean: np.ndarray
    degraded: np.ndarray
    degradations: List[DegradationSpec] = field(default_factory=list)

    @property
    def degradation_types(self) -> List[str]:
        """按施加顺序排列的退化类型。"""
        return [d.degradation_type for d in self.degradations]

    @property
    def env_name(self) -> str:
        """与训练数据 ``env_name`` 字段一致的逗号分隔退化类型串；无退化时为 clean。"""
        return ", ".join(self.degradation_types) if self.degradations else "clean"


def synthesize_sample(
    clean: np.ndarray,
    rng: Optional[np.random.Generator] = None,
    num_degradations: Sequence[int] = (2, 3),
    candidate_types: Sequence[str] = DEGRADATION_TYPES,
    clean_prob: float = 0.0,
) -> SynthesizedSample:
    """
    对单张干净图像随机合成复合退化。

    参数:
        clean: RGB uint8 图像
        rng: 随机数发生器，便于复现
        num_degradations: 退化数量候选集合，默认 2 或 3
        candidate_types: 可用退化类型
        clean_prob: 以该概率保留干净图像作为 clean 样本（训练智能体识别无需复原的情况）
    """
    rng = rng if rng is not None else np.random.default_rng()
    if rng.uniform() < clean_prob:
        return SynthesizedSample(clean=clean, degraded=clean.copy(), degradations=[])

    k = int(rng.choice(list(num_degradations)))
    k = min(k, len(candidate_types))
    chosen = rng.choice(list(candidate_types), size=k, replace=False)
    img = clean.copy()
    specs: List[DegradationSpec] = []
    for dtype in chosen:
        level = str(rng.choice(DEGRADATION_LEVELS))
        img = apply_degradation(img, str(dtype), level, rng)
        specs.append(DegradationSpec(str(dtype), level))
    return SynthesizedSample(clean=clean, degraded=img, degradations=specs)
