# -*- coding: utf-8 -*-
"""轨迹记录：一次完整复原过程中的状态、动作、工具结果序列。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from PIL import Image

from ..data.dataset import Sample
from .parser import ToolCall


@dataclass
class StepRecord:
    """单个决策步：模型响应、解析出的工具调用、执行前后图像索引。"""
    turn: int
    response: str
    tool_calls: List[ToolCall]
    predicted_degradations: List[str]
    executed_tools: List[str]
    image_index_before: int
    image_index_after: int
    tool_cost: float = 0.0
    error: Optional[str] = None
    format_errors: List[str] = field(default_factory=list)
    is_answer: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)   # 策略可写入 log_prob / 状态键等训练信息


@dataclass
class Trajectory:
    """一次完整复原过程的轨迹：步骤记录、图像历史与最终答案。"""
    sample: Sample
    mode: str
    steps: List[StepRecord] = field(default_factory=list)
    image_history: List[Image.Image] = field(default_factory=list)
    answer: Optional[Dict[str, Any]] = None
    finished: bool = False
    truncated: bool = False

    @property
    def responses(self) -> List[str]:
        """各轮模型响应文本列表。"""
        return [s.response for s in self.steps]

    @property
    def concatenated_response(self) -> str:
        """多轮响应拼接后的文本，用于格式与奖励检查。"""
        return "\n".join(self.responses)

    @property
    def executed_tools(self) -> List[str]:
        """按执行顺序排列的已执行工具名。"""
        return [t for s in self.steps for t in s.executed_tools]

    @property
    def num_tool_calls(self) -> int:
        """模型发出的工具调用总数。"""
        return sum(len(s.tool_calls) for s in self.steps)

    @property
    def final_image(self) -> Image.Image:
        """最终图像（图像历史的最后一张）。"""
        return self.image_history[-1]

    @property
    def restoration_log(self) -> List[str]:
        """最终答案中的复原日志，无答案时为空列表。"""
        if self.answer and isinstance(self.answer.get("restoration_log"), list):
            return [str(x) for x in self.answer["restoration_log"]]
        return []
