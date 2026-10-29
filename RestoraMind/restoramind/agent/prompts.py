# -*- coding: utf-8 -*-
"""
提示词模板。

两种对话模式：
    single_tool_iterative : 每轮恰好一个工具，逐步决策（序列决策式复原，默认）
    multi_tool_planning   : 每轮给出完整工具序列，系统链式执行后反馈结果
每个决策状态都注入当前图像的低层统计先验，用于同类工具间的自适应选择。
"""
from __future__ import annotations

from typing import Iterable, List, Optional

from ..tools.registry import ToolSpec, prompt_visible_tools

MODES = ("single_tool_iterative", "multi_tool_planning")

_FORMAT_COMMON = """## Response Format (STRICT)
Every turn must start with a `<think>` block (at least 10 characters) explaining:
- which degradations you believe remain in the current image, and why (use both the image and its statistics);
- which tool you choose among the candidates for that degradation and why the image statistics support it.

Then output exactly ONE of:
1. `<tool_call>[ {{"name": "<tool>", "degradation": "<type>", "arguments": {{...}}}} ]</tool_call>`
2. `<answer>{{"restoration_log": ["<type1>", "<type2>", ...]}}</answer>`  (only when restoration is complete; list the degradation types in the order they were handled)
Never output both in one turn. Use only tool names from the list above.
"""

_STATS_NOTE = """## Image Statistics
Each image is accompanied by lightweight low-level statistics (brightness, contrast, edge density,
high-frequency energy, noise / blur / blockiness / haze estimates). Use them to (a) confirm which
degradations remain and (b) choose between tools that handle the same degradation, e.g. prefer
detail-preserving tools when edge density is high and robust/smoothing tools when it is low.
"""

_SINGLE_RULES = """## Rules
- One tool per turn. The tool is applied to the CURRENT image; you will then see the result.
- If the previous tool failed to remove its target degradation, you may retry once with an
  alternative tool of the same type on the PREVIOUS image by adding `"rollback": true`.
- Stop with `<answer>` as soon as no significant degradation remains. Avoid redundant calls.
"""

_MULTI_RULES = """## Rules
- Propose the COMPLETE ordered tool sequence as a JSON list in one `<tool_call>`.
- The system executes the whole sequence from the ORIGINAL image and returns the result.
- If the result is unsatisfactory, propose a different complete sequence; otherwise answer.
"""


def _tool_catalog(specs: Iterable[ToolSpec]) -> str:
    by_deg: dict = {}
    for s in specs:
        by_deg.setdefault(s.degradation, []).append(s)
    lines: List[str] = ["## Available Tools (grouped by degradation type)"]
    for deg, items in by_deg.items():
        lines.append(f"### {deg}")
        lines.extend(s.prompt_line() for s in items)
    return "\n".join(lines)


def build_system_prompt(mode: str = "single_tool_iterative", specs: Optional[Iterable[ToolSpec]] = None) -> str:
    """生成系统提示词：工具目录、统计先验说明、模式规则与严格响应格式。"""
    if mode not in MODES:
        raise ValueError(f"未知对话模式: {mode}")
    specs = list(specs) if specs is not None else prompt_visible_tools()
    intro = ("You are RestoraMind, an image restoration agent. You restore images with unknown, "
             "compound degradations by reasoning about the remaining degradations step by step and "
             "selecting the most suitable expert restoration tool at each step.")
    rules = _SINGLE_RULES if mode == "single_tool_iterative" else _MULTI_RULES
    return "\n\n".join([intro, _tool_catalog(specs), _STATS_NOTE, rules, _FORMAT_COMMON])


def build_first_user_prompt(degradation_hint: Optional[List[str]], stats_text: Optional[str],
                            mode: str = "single_tool_iterative") -> str:
    """生成首轮用户提示，可包含退化提示与统计先验文本。"""
    parts = ["<image>"]
    if degradation_hint is not None:
        hint = ", ".join(degradation_hint) if degradation_hint else "none"
        parts.append(f"Image analysis indicates these degradation types (order unknown): {hint}")
    else:
        parts.append("Identify the degradations present in this image yourself.")
    if stats_text:
        parts.append(stats_text)
    if mode == "single_tool_iterative":
        parts.append("Select the FIRST tool to apply, or answer if the image is already clean.")
    else:
        parts.append("Propose the complete restoration sequence now.")
    return "\n".join(parts)


def build_feedback_prompt(mode: str, applied: List[str], processed: List[str], remaining: Optional[List[str]],
                          stats_text: Optional[str], error: Optional[str] = None) -> str:
    """生成工具执行后的反馈提示，包含已处理与剩余退化及新图像统计。"""
    parts: List[str] = []
    if error:
        parts.append(f"Tool execution error: {error}. The image is unchanged.")
    else:
        parts.append("Tool(s) applied successfully: " + ", ".join(applied) + ". Here is the processed image:")
    parts.append("<image>")
    if processed:
        parts.append("Processed degradations so far: " + ", ".join(processed))
    if remaining is not None:
        parts.append("Remaining degradations (hint): " + (", ".join(remaining) if remaining else "none"))
    if stats_text:
        parts.append(stats_text)
    if mode == "single_tool_iterative":
        parts.append("Analyse the new image. Continue with the next tool, retry with rollback, or answer if complete.")
    else:
        parts.append("If satisfactory, answer. Otherwise propose a NEW complete sequence (executed from the original image).")
    return "\n".join(parts)
