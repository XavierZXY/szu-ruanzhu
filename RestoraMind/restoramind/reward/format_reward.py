# -*- coding: utf-8 -*-
"""
格式奖励。

enhanced   : 严格二元（1.0 / -1.0），用于模型已掌握基本格式后的强化阶段
progressive: 阶梯式（0.0 → 0.2 → 0.4 → 0.6 → 0.8 → 1.0），用于训练初期温和引导
输入为整条轨迹拼接后的多轮响应文本。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from ..agent.parser import ANSWER_RE, THINK_RE, TOOL_CALL_RE, parse_response, split_turns
from ..tools.registry import TOOL_SPECS


@dataclass
class FormatConfig:
    """格式检查配置：对话模式、单轮工具上限、总工具数上限与允许的工具集合。"""
    mode: str = "single_tool_iterative"
    max_tools_per_turn: int = 1
    enable_total_tools_upper_limit: bool = True   # 总工具数 <= 退化数 + 1
    min_think_chars: int = 10
    allowed_tools: Sequence[str] = field(default_factory=lambda: tuple(TOOL_SPECS.keys()))


def check_format_enhanced(text: str, degradation_count: int = 0, is_clean: bool = False,
                          cfg: Optional[FormatConfig] = None) -> Tuple[float, str]:
    """返回 (1.0, "ok") 或 (-1.0, 违规原因)。"""
    cfg = cfg or FormatConfig()
    turns = split_turns(text)
    if not turns:
        return -1.0, "no <think> block"
    total_tools = 0
    answer_turns: List[int] = []
    for i, turn in enumerate(turns, start=1):
        parsed = parse_response(turn)
        if parsed.think is None or len(parsed.think) < cfg.min_think_chars:
            return -1.0, f"turn {i}: think missing or too short"
        if parsed.errors:
            return -1.0, f"turn {i}: {parsed.errors[0]}"
        if parsed.is_answer:
            answer_turns.append(i)
            if set(parsed.answer.keys()) != {"restoration_log"}:
                return -1.0, f"turn {i}: answer must contain only restoration_log"
            continue
        n = len(parsed.tool_calls)
        if cfg.mode == "single_tool_iterative" and n != 1:
            return -1.0, f"turn {i}: single-tool mode requires exactly 1 tool, got {n}"
        if cfg.max_tools_per_turn > 0 and n > cfg.max_tools_per_turn:
            return -1.0, f"turn {i}: {n} tools exceeds per-turn limit {cfg.max_tools_per_turn}"
        for call in parsed.tool_calls:
            if call.name not in cfg.allowed_tools:
                return -1.0, f"turn {i}: tool '{call.name}' not allowed"
        total_tools += n
    if answer_turns and (len(answer_turns) > 1 or answer_turns[0] != len(turns)):
        return -1.0, "answer must appear only in the final turn"
    if not is_clean and total_tools < 1:
        return -1.0, "degraded sample requires at least one tool call"
    if cfg.enable_total_tools_upper_limit and not is_clean and degradation_count > 0 and total_tools > degradation_count + 1:
        return -1.0, f"{total_tools} tool calls exceed limit {degradation_count + 1}"
    return 1.0, "ok"


def check_format_progressive(text: str, degradation_count: int = 0, is_clean: bool = False,
                             cfg: Optional[FormatConfig] = None) -> Tuple[float, str]:
    """
    阶梯式格式奖励：
        0.0 完全无格式 | 0.2 有 think | 0.4 有 tool_call 或 answer 标签
        0.6 JSON 可解析 | 0.8 工具名合法 | 1.0 通过全部增强检查
    """
    cfg = cfg or FormatConfig()
    if not THINK_RE.search(text):
        return 0.0, "level0: no think"
    if not (TOOL_CALL_RE.search(text) or ANSWER_RE.search(text)):
        return 0.2, "level1: think only"
    try:
        for m in TOOL_CALL_RE.finditer(text):
            json.loads(m.group(1))
        for m in ANSWER_RE.finditer(text):
            json.loads(m.group(1))
    except json.JSONDecodeError:
        return 0.4, "level2: tags present but json invalid"
    for turn in split_turns(text):
        parsed = parse_response(turn)
        for call in parsed.tool_calls:
            if call.name not in cfg.allowed_tools:
                return 0.6, f"level3: illegal tool {call.name}"
    score, reason = check_format_enhanced(text, degradation_count, is_clean, cfg)
    return (1.0, "level5: perfect") if score > 0 else (0.8, f"level4: {reason}")
