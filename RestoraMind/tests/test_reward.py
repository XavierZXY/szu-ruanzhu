# -*- coding: utf-8 -*-
import math

import numpy as np
from PIL import Image

from restoramind.agent.env import EnvConfig, RestorationEnv
from restoramind.agent.policy import HeuristicPolicy
from restoramind.agent.runner import run_episode
from restoramind.reward.composite import RewardConfig, compute_episode_reward, compute_step_rewards
from restoramind.reward.degradation_reward import degradation_set_reward, degradation_type_match, restoration_order_reward
from restoramind.reward.quality_metrics import discretize, no_reference_score, normalize_metrics, psnr, quality_score, ssim


def test_psnr_ssim_identity_and_degraded(clean_image, degraded_sample):
    assert psnr(clean_image, clean_image) == 100.0 and abs(ssim(clean_image, clean_image) - 1.0) < 1e-6
    assert psnr(degraded_sample.degraded, degraded_sample.clean) < 40
    assert ssim(degraded_sample.degraded, degraded_sample.clean) < 1.0


def test_size_mismatch_is_handled(clean_image):
    big = clean_image.resize((clean_image.width * 2, clean_image.height * 2))
    assert psnr(big, clean_image) > 20


def test_normalization_ranges():
    p, s, l = normalize_metrics(25.0, 0.5, 0.3)
    assert abs(p - math.sqrt(0.5)) < 1e-9 and s == 0.75 and l == 0.3
    assert normalize_metrics(5.0, -2.0, 2.0) == (0.0, 0.0, 1.0)


def test_quality_score_without_lpips_monotonic(clean_image, degraded_sample):
    good = quality_score(clean_image, clean_image)["score"]
    bad = quality_score(degraded_sample.degraded, degraded_sample.clean)["score"]
    assert 0 <= bad < good <= 1.0 and math.isnan(quality_score(clean_image, clean_image)["lpips"])


def test_no_reference_score_in_range(clean_image, degraded_sample):
    for im in (clean_image, degraded_sample.degraded):
        assert 0.0 <= no_reference_score(im) <= 1.0


def test_discretize():
    assert discretize(0.57, 10) == 0.5 and discretize(0.57, 0) == 0.57


def test_degradation_rewards():
    assert degradation_set_reward(["noise", "rain"], ["noise", "dark"]) == 0.5
    assert degradation_set_reward([], ["noise"]) == 0.0
    assert degradation_type_match(["noise", "dark"], ["dark", "noise"]) == 1.0
    assert degradation_type_match(["noise"], ["dark", "noise"]) == 0.5
    assert degradation_type_match(["haze"], ["dark", "noise"]) == 0.0
    assert restoration_order_reward(["noise", "dark"], ["dark", "noise"]) == 1.0      # 逆序复原
    assert restoration_order_reward(["dark", "noise"], ["dark", "noise"]) == 0.1      # 集合对、顺序错
    assert restoration_order_reward(["noise", "noise", "dark"], ["dark", "noise"]) == 1.0  # 合并连续重复


def test_episode_and_step_rewards_on_heuristic_trajectory(local_toolbox, degraded_sample):
    env = RestorationEnv(local_toolbox, EnvConfig(max_turns=5))
    traj = run_episode(env, HeuristicPolicy(), degraded_sample)
    cfg = RewardConfig(enable_degradation_type_reward=True, enable_intermediate_reward=True)
    b = compute_episode_reward(traj, cfg)
    assert b.format_score == 1.0, b.format_reason
    assert 0.0 <= b.quality_score <= 1.0 and b.degradation_type_score == 1.0
    assert b.score > 0.3
    steps = compute_step_rewards(traj, cfg)
    assert len(steps) == len(traj.steps)
    assert steps[-1] > 0   # 终止奖励 κ·R_quality


def test_format_violation_gives_negative_reward(local_toolbox, degraded_sample):
    env = RestorationEnv(local_toolbox, EnvConfig(max_turns=2))
    env.reset(degraded_sample)
    env.step("no format at all")
    env.step("still nothing")
    b = compute_episode_reward(env.trajectory)
    assert b.format_score <= 0 and b.score < 0


def test_clean_sample_reward(local_toolbox, clean_sample):
    env = RestorationEnv(local_toolbox, EnvConfig(max_turns=3))
    traj = run_episode(env, HeuristicPolicy(), clean_sample)
    b = compute_episode_reward(traj)
    assert b.clean_accuracy == 1.0 and b.format_score == 1.0 and b.score > 0.9
