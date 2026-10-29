# -*- coding: utf-8 -*-
"""训练：GRPO 组相对优势、可训练工具选择策略、VeRL 框架桥接。"""
from .grpo import group_relative_advantages, GRPOConfig, GRPOTrainer, TrainablePolicy
from .softmax_policy import SoftmaxToolPolicy
from .verl_plugin import compute_score, make_verl_tool_classes

__all__ = [
    "group_relative_advantages", "GRPOConfig", "GRPOTrainer", "TrainablePolicy",
    "SoftmaxToolPolicy", "compute_score", "make_verl_tool_classes",
]
