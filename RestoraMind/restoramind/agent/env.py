# -*- coding: utf-8 -*-
"""
图像复原序列决策环境。

状态 S_t = (当前图像 I_t, 统计先验 F_t, 工具调用历史, 退化预测历史, 任务提示)。
动作 a_t = 工具选择（含参数、目标退化、是否回退）。
转移：执行工具得到 I_{t+1}，重新提取统计先验，更新历史，构造下一状态。
终止：模型输出 <answer>，或达到最大轮数。
环境不计算奖励（由 reward 模块基于轨迹计算），保持单一职责。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from PIL import Image

from ..data.dataset import Sample
from ..priors.image_statistics import ImageStatistics, compute_image_statistics
from ..tools.registry import degradation_of_tool
from ..tools.toolbox import Toolbox, ToolExecutionError
from .parser import ParsedResponse, ToolCall, parse_response
from .prompts import MODES, build_feedback_prompt, build_first_user_prompt, build_system_prompt
from .trajectory import StepRecord, Trajectory


@dataclass
class EnvConfig:
    """环境配置：对话模式、最大轮数、回退次数上限、是否注入统计先验与退化提示。"""
    mode: str = "single_tool_iterative"
    max_turns: int = 6
    max_retries: int = 1              # 同类工具回退重试次数上限（有限回退，而非全局后搜索）
    use_statistics_prior: bool = True
    provide_degradation_hint: bool = True
    max_tools_per_turn: int = 1       # 单工具模式强制为 1；多工具模式 0 表示不限
    stop_on_format_error: bool = False

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"未知对话模式: {self.mode}")
        if self.mode == "single_tool_iterative":
            self.max_tools_per_turn = 1


@dataclass
class AgentState:
    """供策略使用的结构化状态（与自然语言提示词内容一致）。"""
    turn: int
    current_image: Image.Image
    statistics: Optional[ImageStatistics]
    degradation_hint: Optional[List[str]]
    processed: List[str]
    remaining: Optional[List[str]]
    tool_history: List[str]
    retries_used: int
    last_error: Optional[str] = None


@dataclass
class Observation:
    """环境观测：多模态消息列表与结构化状态。"""
    messages: List[Dict[str, Any]]
    state: AgentState


def _text_image_content(text: str, image: Image.Image) -> List[Dict[str, Any]]:
    """把含 <image> 占位的文本转为多模态内容片段。"""
    before, _, after = text.partition("<image>")
    content: List[Dict[str, Any]] = []
    if before.strip():
        content.append({"type": "text", "text": before.strip()})
    content.append({"type": "image", "image": image})
    if after.strip():
        content.append({"type": "text", "text": after.strip()})
    return content


class RestorationEnv:
    """图像复原序列决策环境，负责状态维护、工具执行与状态转移。"""
    def __init__(self, toolbox: Toolbox, config: Optional[EnvConfig] = None):
        self.toolbox = toolbox
        self.config = config or EnvConfig()
        self.system_prompt = build_system_prompt(self.config.mode)
        self._traj: Optional[Trajectory] = None
        self._messages: List[Dict[str, Any]] = []
        self._processed: List[str] = []
        self._retries = 0
        self._turn = 0
        self._last_error: Optional[str] = None

    # ------------------------------------------------------------------ 状态
    @property
    def trajectory(self) -> Trajectory:
        """当前回合的轨迹记录。"""
        assert self._traj is not None, "请先调用 reset()"
        return self._traj

    @property
    def current_image(self) -> Image.Image:
        """当前待复原图像（图像历史的最后一张）。"""
        return self.trajectory.image_history[-1]

    def _stats(self, image: Image.Image) -> Optional[ImageStatistics]:
        return compute_image_statistics(image) if self.config.use_statistics_prior else None

    def _hint(self) -> Optional[List[str]]:
        return list(self.trajectory.sample.degradation_types) if self.config.provide_degradation_hint else None

    def _remaining(self) -> Optional[List[str]]:
        hint = self._hint()
        if hint is None:
            return None
        return [d for d in hint if d not in self._processed]

    def _state(self, stats: Optional[ImageStatistics]) -> AgentState:
        return AgentState(
            turn=self._turn,
            current_image=self.current_image,
            statistics=stats,
            degradation_hint=self._hint(),
            processed=list(self._processed),
            remaining=self._remaining(),
            tool_history=list(self.trajectory.executed_tools),
            retries_used=self._retries,
            last_error=self._last_error,
        )

    # ------------------------------------------------------------------ 交互
    def reset(self, sample: Sample) -> Observation:
        """载入样本并返回首轮观测。"""
        self._traj = Trajectory(sample=sample, mode=self.config.mode, image_history=[sample.degraded.convert("RGB")])
        self._processed = []
        self._retries = 0
        self._turn = 0
        self._last_error = None
        stats = self._stats(self.current_image)
        user_text = build_first_user_prompt(self._hint(), stats.to_prompt_text() if stats else None, self.config.mode)
        self._messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": _text_image_content(user_text, self.current_image)},
        ]
        return Observation(messages=list(self._messages), state=self._state(stats))

    def _apply_tool_chain(self, calls: List[ToolCall], base_image: Image.Image):
        """依次执行工具链，返回 (最终图像, 已执行工具名, 代价, 错误)。"""
        image = base_image
        executed: List[str] = []
        cost = 0.0
        for call in calls:
            if not self.toolbox.has(call.name):
                return image, executed, cost, f"unknown tool '{call.name}'"
            try:
                result = self.toolbox.execute(call.name, image, call.arguments)
            except ToolExecutionError as exc:
                return image, executed, cost, str(exc)
            image = result.image
            executed.append(call.name)
            cost += self.toolbox.spec(call.name).cost
        return image, executed, cost, None

    def step(self, response: str) -> tuple[Observation, bool, StepRecord]:
        """执行一轮模型响应，返回（下一观测，是否终止，本步记录）。"""
        traj = self.trajectory
        self._turn += 1
        parsed: ParsedResponse = parse_response(response)
        idx_before = len(traj.image_history) - 1
        record = StepRecord(
            turn=self._turn, response=response, tool_calls=list(parsed.tool_calls),
            predicted_degradations=[c.degradation or degradation_of_tool(c.name) or "" for c in parsed.tool_calls],
            executed_tools=[], image_index_before=idx_before, image_index_after=idx_before,
            format_errors=list(parsed.errors), is_answer=parsed.is_answer,
        )
        self._last_error = None

        # 终止：模型给出最终答案
        if parsed.is_answer and not parsed.errors:
            traj.answer = parsed.answer
            traj.finished = True
            traj.steps.append(record)
            return Observation(messages=list(self._messages), state=self._state(None)), True, record

        done = False
        calls = list(parsed.tool_calls)
        if self.config.max_tools_per_turn > 0 and len(calls) > self.config.max_tools_per_turn:
            record.format_errors.append(f"too many tools in one turn ({len(calls)} > {self.config.max_tools_per_turn})")
            calls = calls[: self.config.max_tools_per_turn]

        if not calls:
            self._last_error = "no valid tool call"
            if self.config.stop_on_format_error:
                done = True
        else:
            # 选择工具链的起始图像：多工具模式从原图；单工具模式从当前图，回退时从上一张图
            if self.config.mode == "multi_tool_planning":
                base = traj.image_history[0]
            elif calls[0].rollback and self._retries < self.config.max_retries and len(traj.image_history) >= 2:
                base = traj.image_history[-2]
                self._retries += 1
                if self._processed:
                    self._processed.pop()   # 回退撤销上一步的「已处理」记录
            else:
                base = self.current_image
            image, executed, cost, error = self._apply_tool_chain(calls, base)
            record.tool_cost = cost
            if error:
                record.error = error
                self._last_error = error
            if executed:
                traj.image_history.append(image)
                record.image_index_after = len(traj.image_history) - 1
                record.executed_tools = executed
                for name in executed:
                    deg = degradation_of_tool(name)
                    if deg and deg not in self._processed:
                        self._processed.append(deg)

        traj.steps.append(record)
        if self._turn >= self.config.max_turns:
            done = True
            traj.truncated = True

        # 记录助手回复并构造下一状态
        self._messages.append({"role": "assistant", "content": response})
        stats = self._stats(self.current_image)
        feedback = build_feedback_prompt(
            self.config.mode, record.executed_tools, self._processed, self._remaining(),
            stats.to_prompt_text() if stats else None, error=self._last_error,
        )
        self._messages.append({"role": "user", "content": _text_image_content(feedback, self.current_image)})
        return Observation(messages=list(self._messages), state=self._state(stats)), done, record
