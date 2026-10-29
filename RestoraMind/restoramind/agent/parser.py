# -*- coding: utf-8 -*-
"""
智能体响应解析。

响应格式（严格）：
    <think>推理过程</think>
    <tool_call>[{"name": 工具名, "degradation": 退化类型, "arguments": {...}, "rollback": false}]</tool_call>
或终止：
    <think>...</think>
    <answer>{"restoration_log": ["rain", "dark", ...]}</answer>
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)


@dataclass
class ToolCall:
    """单个工具调用：工具名、目标退化类型、参数、是否回退重试。"""
    name: str
    degradation: Optional[str] = None
    arguments: Dict[str, Any] = field(default_factory=dict)
    rollback: bool = False

    def to_dict(self) -> dict:
        """转换为可序列化的字典。"""
        d = {"name": self.name, "degradation": self.degradation, "arguments": self.arguments}
        if self.rollback:
            d["rollback"] = True
        return d


@dataclass
class ParsedResponse:
    """单轮响应的解析结果，含推理文本、工具调用、最终答案与解析错误。"""
    think: Optional[str]
    tool_calls: List[ToolCall]
    answer: Optional[Dict[str, Any]]
    errors: List[str] = field(default_factory=list)
    raw: str = ""

    @property
    def is_answer(self) -> bool:
        """该轮是否为最终答案。"""
        return self.answer is not None

    @property
    def is_valid(self) -> bool:
        """该轮是否格式有效（无错误且含工具调用或答案）。"""
        return not self.errors and (self.is_answer or bool(self.tool_calls))


def _parse_tool_calls(payload: str) -> List[ToolCall]:
    data = json.loads(payload)
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list) or not data:
        raise ValueError("tool_call 必须是非空 JSON 列表")
    calls: List[ToolCall] = []
    for item in data:
        if not isinstance(item, dict) or "name" not in item:
            raise ValueError("tool_call 元素缺少 name 字段")
        args = item.get("arguments", {})
        if not isinstance(args, dict):
            raise ValueError("arguments 必须是对象")
        calls.append(ToolCall(
            name=str(item["name"]),
            degradation=item.get("degradation"),
            arguments=args,
            rollback=bool(item.get("rollback", False)),
        ))
    return calls


def parse_response(text: str) -> ParsedResponse:
    """解析单轮响应。解析失败时在 errors 中给出原因，不抛异常。"""
    errors: List[str] = []
    think_m = THINK_RE.search(text)
    think = think_m.group(1).strip() if think_m else None
    if think is None:
        errors.append("missing <think>")

    tool_m = TOOL_CALL_RE.search(text)
    ans_m = ANSWER_RE.search(text)
    if tool_m and ans_m:
        errors.append("both <tool_call> and <answer> present")

    tool_calls: List[ToolCall] = []
    answer: Optional[Dict[str, Any]] = None
    if tool_m:
        try:
            tool_calls = _parse_tool_calls(tool_m.group(1))
        except (json.JSONDecodeError, ValueError) as exc:
            errors.append(f"invalid tool_call: {exc}")
    if ans_m:
        try:
            answer = json.loads(ans_m.group(1))
            if not isinstance(answer, dict) or "restoration_log" not in answer:
                errors.append("answer must be an object with restoration_log")
                answer = None
        except json.JSONDecodeError as exc:
            errors.append(f"invalid answer json: {exc}")
    if not tool_m and not ans_m:
        errors.append("neither <tool_call> nor <answer> present")
    return ParsedResponse(think=think, tool_calls=tool_calls, answer=answer, errors=errors, raw=text)


def split_turns(multi_turn_text: str) -> List[str]:
    """把多轮拼接的模型输出按 <think> 起始位置切分为若干轮。"""
    matches = list(THINK_RE.finditer(multi_turn_text))
    turns: List[str] = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(multi_turn_text)
        turns.append(multi_turn_text[m.start():end])
    return turns


def format_tool_call(think: str, calls: List[ToolCall]) -> str:
    """按严格格式生成包含推理与工具调用的响应文本。"""
    body = json.dumps([c.to_dict() for c in calls], ensure_ascii=False)
    return f"<think>{think}</think>\n<tool_call>{body}</tool_call>"


def format_answer(think: str, restoration_log: List[str]) -> str:
    """按严格格式生成包含推理与 restoration_log 的最终答案文本。"""
    body = json.dumps({"restoration_log": restoration_log}, ensure_ascii=False)
    return f"<think>{think}</think>\n<answer>{body}</answer>"
