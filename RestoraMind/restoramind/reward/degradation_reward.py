# -*- coding: utf-8 -*-
"""退化推理奖励与复原顺序奖励。"""
from __future__ import annotations

from typing import Iterable, List, Sequence


def _clean_list(items: Iterable[str]) -> List[str]:
    return [str(x).strip().lower() for x in items if x is not None and str(x).strip().lower() != "clean"]


def merge_consecutive_duplicates(items: Sequence[str]) -> List[str]:
    """合并相邻的重复项。"""
    out: List[str] = []
    for x in items:
        if not out or out[-1] != x:
            out.append(x)
    return out


def degradation_set_reward(predicted: Iterable[str], truth: Iterable[str]) -> float:
    """
    退化推理奖励 R_deg = |D^ ∩ D| / max(1, |D^|)：
    预测集合中正确类型所占比例（惩罚误报，不惩罚漏报，漏报由后续步骤继续推理）。
    """
    p, t = set(_clean_list(predicted)), set(_clean_list(truth))
    if not p:
        return 0.0
    return len(p & t) / max(1, len(p))


def degradation_type_match(predicted_log: Iterable[str], truth: Iterable[str]) -> float:
    """
    退化类型集合匹配（不考虑顺序）：
    预测含真实集合之外的类型 → 0；完全匹配 → 1；否则为覆盖比例。
    """
    t = set(_clean_list(truth))
    if not t:
        return 0.0
    p = set(merge_consecutive_duplicates(_clean_list(predicted_log)))
    if not p or not p.issubset(t):
        return 0.0
    return 1.0 if p == t else len(p) / len(t)


def restoration_order_reward(predicted_log: Iterable[str], addition_order: Sequence[str]) -> float:
    """
    复原顺序奖励：理想复原顺序为退化施加顺序的逆序（后进先出）。
    完全一致 → 1；按位置匹配比例给部分分；集合正确但顺序错误至少 0.1。
    """
    pred = merge_consecutive_duplicates(_clean_list(predicted_log))
    ideal = list(reversed(_clean_list(addition_order)))
    if not pred or not ideal:
        return 0.0
    if pred == ideal:
        return 1.0
    correct = sum(1 for a, b in zip(pred, ideal) if a == b)
    score = correct / len(ideal)
    if set(pred) == set(ideal):
        score = max(score, 0.1)
    return score


def is_clean_answer(restoration_log: Iterable[str], num_tool_calls: int) -> bool:
    """clean 样本的正确响应：不调用工具，且 restoration_log 为空或仅含 clean。"""
    return num_tool_calls == 0 and len(_clean_list(restoration_log)) == 0
