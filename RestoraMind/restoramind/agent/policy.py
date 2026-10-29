# -*- coding: utf-8 -*-
"""
策略接口与两种内置策略。

Policy.act(observation) -> 响应文本（严格遵循 <think>/<tool_call>/<answer> 格式）。
    OpenAICompatiblePolicy : 调用 vLLM 等 OpenAI 兼容接口部署的多模态大模型（训练后的智能体）
    HeuristicPolicy        : 基于退化提示 + 图像统计先验的规则策略，用于基线、离线测试与数据流程验证
"""
from __future__ import annotations

import base64
import io
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence

from PIL import Image

from ..priors.image_statistics import ImageStatistics
from ..tools.registry import ToolSpec, tools_for_degradation
from .env import Observation
from .parser import ToolCall, format_answer, format_tool_call

# 默认处理优先级：先去除遮挡/几何类退化，再处理光照与噪声，最后细节恢复
DEFAULT_PRIORITY: Sequence[str] = (
    "rain", "haze", "motion blur", "defocus blur", "dark", "noise", "jpeg compression artifact", "low resolution",
)


class Policy(ABC):
    """策略接口：根据观测输出符合格式的响应文本。"""
    @abstractmethod
    def act(self, observation: Observation) -> str:
        """根据观测生成一轮响应。"""
        ...


def _image_to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def messages_to_openai(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把环境消息（含 PIL 图像）转换为 OpenAI 兼容的多模态消息。"""
    out: List[Dict[str, Any]] = []
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            out.append({"role": m["role"], "content": content})
            continue
        parts: List[Dict[str, Any]] = []
        for c in content:
            if c["type"] == "text":
                parts.append({"type": "text", "text": c["text"]})
            else:
                parts.append({"type": "image_url", "image_url": {"url": _image_to_data_url(c["image"])}})
        out.append({"role": m["role"], "content": parts})
    return out


class OpenAICompatiblePolicy(Policy):
    """通过 OpenAI 兼容接口调用多模态大模型的策略。"""
    def __init__(self, base_url: str = "http://localhost:8000/v1", api_key: str = "EMPTY",
                 model: Optional[str] = None, temperature: float = 0.7, top_p: float = 0.9,
                 max_tokens: int = 2048, client: Any = None):
        if client is None:
            from openai import OpenAI  # 延迟导入，离线环境不依赖
            client = OpenAI(base_url=base_url, api_key=api_key)
        self.client = client
        self.model = model or self._detect_model()
        self.temperature, self.top_p, self.max_tokens = temperature, top_p, max_tokens

    def _detect_model(self) -> str:
        try:
            return self.client.models.list().data[0].id
        except Exception:  # noqa: BLE001
            return "default"

    def act(self, observation: Observation) -> str:
        """把观测转换为多模态消息并请求模型生成响应。"""
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages_to_openai(observation.messages),
            temperature=self.temperature, top_p=self.top_p, max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content or ""


def select_tool_by_statistics(candidates: Sequence[ToolSpec], stats: Optional[ImageStatistics]) -> ToolSpec:
    """
    同类候选工具间的统计先验选择规则：
    边缘密度 / 高频能量高 → 偏细节保持型工具（描述含 detail / texture / fine）；
    否则 → 偏稳健平滑型工具（候选列表中的首个）。
    """
    if len(candidates) == 1 or stats is None:
        return candidates[0]
    q = stats.to_qualitative()
    detail_rich = q["edge_density"] == "high" or q["high_frequency_energy"] == "high"
    for spec in candidates:
        is_detail_tool = any(k in spec.description.lower() for k in ("detail", "texture", "fine"))
        if detail_rich and is_detail_tool:
            return spec
    return candidates[0]


class HeuristicPolicy(Policy):
    """规则策略：按优先级处理剩余退化，同类工具依据统计先验选择。"""

    def __init__(self, priority: Sequence[str] = DEFAULT_PRIORITY, mode: str = "single_tool_iterative",
                 visible_only: bool = True):
        self.priority = list(priority)
        self.mode = mode
        self.visible_only = visible_only

    def _ordered_remaining(self, remaining: Sequence[str]) -> List[str]:
        return sorted(remaining, key=lambda d: self.priority.index(d) if d in self.priority else len(self.priority))

    def _call_for(self, degradation: str, stats: Optional[ImageStatistics]) -> Optional[ToolCall]:
        candidates = tools_for_degradation(degradation, visible_only=self.visible_only)
        if not candidates:
            return None
        spec = select_tool_by_statistics(candidates, stats)
        return ToolCall(name=spec.name, degradation=degradation, arguments=spec.normalize_arguments({}))

    def act(self, observation: Observation) -> str:
        """按优先级处理剩余退化，同类工具依据统计先验选择。"""
        st = observation.state
        remaining = st.remaining if st.remaining is not None else []
        ordered = self._ordered_remaining(remaining)
        if not ordered:
            return format_answer("All hinted degradations have been processed; the image looks clean.", st.processed)
        if self.mode == "multi_tool_planning":
            calls = [c for c in (self._call_for(d, st.statistics) for d in ordered) if c]
            return format_tool_call("Plan the full sequence by priority using image statistics.", calls)
        call = self._call_for(ordered[0], st.statistics)
        if call is None:
            return format_answer("No tool available for remaining degradations.", st.processed)
        why = f"Remaining: {', '.join(ordered)}. Address '{ordered[0]}' first with {call.name} based on image statistics."
        return format_tool_call(why, [call])
