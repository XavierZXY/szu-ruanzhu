# -*- coding: utf-8 -*-
"""
可训练的表格化工具选择策略。

状态键 = (待处理退化类型, 图像细节丰富度分桶)，动作 = 该退化类型下的候选工具。
策略参数为每个状态键上的工具 logits，按 softmax 采样；
更新采用策略梯度 ∇ log π(a|s)·A，即 GRPO 组相对优势下的 REINFORCE。
退化处理顺序遵循优先级表（与 HeuristicPolicy 一致），顺序决策可由大模型策略学习。
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..agent.env import Observation
from ..agent.parser import ToolCall, format_answer, format_tool_call
from ..agent.policy import DEFAULT_PRIORITY
from ..priors.image_statistics import ImageStatistics
from ..tools.registry import tools_for_degradation
from .grpo import TrainablePolicy


@dataclass
class ActionRecord:
    """训练用动作记录：状态键、所选工具下标与对数概率。"""
    key: Tuple[str, str]
    tool_index: int
    log_prob: float


def detail_bucket(stats: Optional[ImageStatistics]) -> str:
    """按边缘密度与高频能量把图像划分为细节丰富、平滑或中等。"""
    if stats is None:
        return "unknown"
    q = stats.to_qualitative()
    if q["edge_density"] == "high" or q["high_frequency_energy"] == "high":
        return "detail_rich"
    if q["edge_density"] == "low":
        return "smooth"
    return "medium"


class SoftmaxToolPolicy(TrainablePolicy):
    """表格化 softmax 工具选择策略，按策略梯度更新。"""
    def __init__(self, lr: float = 0.5, temperature: float = 1.0, seed: int = 0,
                 priority: Sequence[str] = DEFAULT_PRIORITY, visible_only: bool = False):
        self.lr = lr
        self.temperature = temperature
        self.rng = random.Random(seed)
        self.priority = list(priority)
        self.visible_only = visible_only
        self.logits: Dict[Tuple[str, str], List[float]] = {}
        self._episode: List[ActionRecord] = []

    # ------------------------------------------------------------ 参数
    def candidates(self, degradation: str) -> List[str]:
        """返回处理指定退化类型的候选工具名。"""
        return [s.name for s in tools_for_degradation(degradation, visible_only=self.visible_only)]

    def _logits(self, key: Tuple[str, str]) -> List[float]:
        if key not in self.logits:
            self.logits[key] = [0.0] * len(self.candidates(key[0]))
        return self.logits[key]

    def probabilities(self, degradation: str, bucket: str) -> Dict[str, float]:
        """返回状态键下各候选工具的选择概率。"""
        logits = self._logits((degradation, bucket))
        m = max(logits)
        exp = [math.exp((z - m) / self.temperature) for z in logits]
        total = sum(exp)
        return {name: e / total for name, e in zip(self.candidates(degradation), exp)}

    # ------------------------------------------------------------ 交互
    def begin_episode(self) -> None:
        """清空本轨迹的动作记录。"""
        self._episode = []

    def end_episode(self) -> List[ActionRecord]:
        """返回并清空本轨迹记录的动作。"""
        ep, self._episode = self._episode, []
        return ep

    def act(self, observation: Observation) -> str:
        """按当前策略采样工具并生成响应。"""
        st = observation.state
        remaining = st.remaining if st.remaining is not None else []
        ordered = sorted(remaining, key=lambda d: self.priority.index(d) if d in self.priority else 99)
        if not ordered:
            return format_answer("All hinted degradations have been processed.", st.processed)
        deg = ordered[0]
        names = self.candidates(deg)
        if not names:
            return format_answer("No tool available.", st.processed)
        bucket = detail_bucket(st.statistics)
        probs = self.probabilities(deg, bucket)
        r, acc, idx = self.rng.random(), 0.0, len(names) - 1
        for i, name in enumerate(names):
            acc += probs[name]
            if r <= acc:
                idx = i
                break
        self._episode.append(ActionRecord((deg, bucket), idx, math.log(max(probs[names[idx]], 1e-12))))
        call = ToolCall(name=names[idx], degradation=deg, arguments={})
        return format_tool_call(f"Address '{deg}' (bucket={bucket}) with {names[idx]} sampled from learned policy.", [call])

    # ------------------------------------------------------------ 更新
    def update(self, episodes: Sequence[List[ActionRecord]], advantages: Sequence[float]) -> Dict[str, float]:
        """按优势对 logits 做策略梯度更新。"""
        total_grad = 0.0
        n = 0
        for records, adv in zip(episodes, advantages):
            if adv == 0.0:
                continue
            for rec in records:
                logits = self._logits(rec.key)
                probs = list(self.probabilities(*rec.key).values())
                for i in range(len(logits)):
                    grad = (1.0 if i == rec.tool_index else 0.0) - probs[i]
                    logits[i] += self.lr * adv * grad
                    total_grad += abs(grad * adv)
                n += 1
        return {"num_actions": float(n), "mean_abs_grad": total_grad / n if n else 0.0}

    # ------------------------------------------------------------ 持久化
    def save(self, path: str | Path) -> None:
        """把策略参数保存为 JSON。"""
        data = {f"{k[0]}|{k[1]}": v for k, v in self.logits.items()}
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        """从 JSON 载入策略参数。"""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.logits = {tuple(k.split("|", 1)): v for k, v in data.items()}  # type: ignore[misc]
