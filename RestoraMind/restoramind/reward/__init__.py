# -*- coding: utf-8 -*-
"""奖励函数：格式奖励、图像质量奖励、退化推理奖励、复合奖励。"""
from .format_reward import FormatConfig, check_format_enhanced, check_format_progressive
from .quality_metrics import psnr, ssim, lpips_distance, normalize_metrics, quality_score, no_reference_score
from .degradation_reward import degradation_set_reward, degradation_type_match, restoration_order_reward, is_clean_answer
from .composite import RewardConfig, RewardBreakdown, compute_step_rewards, compute_episode_reward

__all__ = [
    "FormatConfig", "check_format_enhanced", "check_format_progressive",
    "psnr", "ssim", "lpips_distance", "normalize_metrics", "quality_score", "no_reference_score",
    "degradation_set_reward", "degradation_type_match", "restoration_order_reward", "is_clean_answer",
    "RewardConfig", "RewardBreakdown", "compute_step_rewards", "compute_episode_reward",
]
