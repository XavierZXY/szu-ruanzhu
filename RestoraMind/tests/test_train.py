# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image

from restoramind.agent.env import EnvConfig, RestorationEnv
from restoramind.agent.parser import ToolCall, format_answer, format_tool_call
from restoramind.data.dataset import Sample
from restoramind.data.synthesize import DegradationSpec
from restoramind.reward.composite import RewardConfig
from restoramind.train.grpo import GRPOConfig, GRPOTrainer, group_relative_advantages
from restoramind.train.softmax_policy import SoftmaxToolPolicy
from restoramind.train.verl_plugin import compute_score, make_verl_tool_classes
from restoramind.tools.toolbox import build_toolbox


def test_group_relative_advantages():
    adv = group_relative_advantages([1.0, 2.0, 3.0])
    assert abs(sum(adv)) < 1e-9 and adv[0] < adv[1] < adv[2]
    assert group_relative_advantages([5.0]) == [0.0] and group_relative_advantages([]) == []


def _noise_samples(clean_array, n=3):
    from restoramind.data.degradations import apply_degradation
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        deg = apply_degradation(clean_array, "noise", "high", rng)
        out.append(Sample(Image.fromarray(deg), Image.fromarray(clean_array), [DegradationSpec("noise", "high")], i))
    return out


def test_grpo_trainer_shifts_tool_preference(clean_array, local_toolbox):
    samples = _noise_samples(clean_array)
    policy = SoftmaxToolPolicy(lr=1.0, seed=0)
    trainer = GRPOTrainer(lambda: RestorationEnv(local_toolbox, EnvConfig(max_turns=3)), policy,
                          GRPOConfig(group_size=4, epochs=3, reward=RewardConfig()))
    before = policy.probabilities("noise", "detail_rich")
    history = trainer.fit(samples, batch_size=3)
    assert len(history) == 3 and all(h.format_violation_rate == 0.0 for h in history)
    assert all(h.mean_quality > 0 for h in history)
    after = policy.probabilities("noise", "detail_rich") or policy.probabilities("noise", "medium")
    # 训练后分布不再均匀：某个候选去噪工具被偏好
    probs = [p for key, p in policy.logits.items() if key[0] == "noise" for p in [max(policy.probabilities(*key).values())]]
    assert probs and max(probs) > 1.0 / len(policy.candidates("noise")) + 1e-6
    assert set(before) == set(after)


def test_softmax_policy_save_load(tmp_path):
    p = SoftmaxToolPolicy()
    p._logits(("noise", "smooth"))[0] = 1.5
    p.save(tmp_path / "policy.json")
    q = SoftmaxToolPolicy()
    q.load(tmp_path / "policy.json")
    assert q.logits[("noise", "smooth")][0] == 1.5


def _history(clean_array, local_toolbox):
    from restoramind.data.degradations import apply_degradation
    deg = Image.fromarray(apply_degradation(clean_array, "noise", "high", np.random.default_rng(0)))
    out = local_toolbox.execute("scunet_real_denoising_psnr", deg, {}).image
    return deg, out


def test_verl_compute_score_reference_mode(clean_array, local_toolbox):
    deg, out = _history(clean_array, local_toolbox)
    text = format_tool_call("Noise is heavy; denoise with the PSNR model.", [ToolCall("scunet_real_denoising_psnr", "noise")]) \
        + "\n" + format_answer("Denoising removed the noise completely.", ["noise"])
    gt = {"reward_model": [{"degradation_type": "noise", "degradation_level": "high"}], "env_name": "noise"}
    extra = {"original_image": {"image": [Image.fromarray(clean_array)]}, "image_history": [{"image": [deg]}, {"image": [out]}]}
    r = compute_score(text, gt, extra, enable_degradation_type_reward=True, enable_intermediate_reward=True)
    assert r["format_score"] == 1.0 and 0 < r["quality_score"] <= 1 and r["degradation_type_score"] == 1.0
    assert r["tool_diversity_bonus"] == 0.1 and r["score"] > 0.3 and r["is_clean_sample"] == 0.0


def test_verl_compute_score_env_and_clean(monkeypatch, clean_image):
    monkeypatch.setenv("IMAGE_QUALITY_USE_NO_REFERENCE", "True")
    text = format_tool_call("Rain present, derain first.", [ToolCall("restormer_deraining", "rain")]) + "\n" + format_answer("Rain removed, restoration is complete.", ["rain"])
    extra = {"image_history": [{"image": [clean_image]}, {"image": [clean_image]}]}
    r = compute_score(text, '{"env_name": "rain"}', extra)
    assert r["format_score"] == 1.0 and r["quality_score"] > 0
    r_bad = compute_score("garbage", {"env_name": "rain"}, None)
    assert r_bad["score"] < 0
    r_clean = compute_score(format_answer("Nothing to fix, image is clean.", []), {"env_name": "clean"}, None)
    assert r_clean["clean_accuracy"] == 1.0 and r_clean["score"] > 0.9


def test_make_verl_tool_classes_with_stub_base(clean_image, local_toolbox):
    class StubToolBase:
        registry = {}

        def __init__(self, name, description="", parameters=None, **kwargs):
            self.name, self.description, self.parameters = name, description, parameters or {}

    classes = make_verl_tool_classes(StubToolBase, toolbox=local_toolbox, tool_names=["nafnet_deblur"])
    tool = classes["nafnet_deblur"]("nafnet_deblur", "", {})
    tool.reset(raw_prompt=[], multi_modal_data={"image": [clean_image]})
    obs, reward, done, info = tool.execute(format_tool_call("Deblur the image using NAFNet.", [ToolCall("nafnet_deblur", "motion blur")]))
    assert info["status"] == "success" and not done and "<image>" in obs["prompt"]
    assert isinstance(obs["multi_modal_data"]["image"][0], Image.Image)
    _, _, done2, info2 = tool.execute(format_answer("Finished.", ["motion blur"]))
    assert done2 and info2["status"] == "finished"


def test_verl_entry_signature(clean_array, local_toolbox):
    from restoramind.train.verl_plugin import verl_compute_score
    text = format_tool_call("Rain present, derain first.", [ToolCall("restormer_deraining", "rain")]) + "\n" \
        + format_answer("Rain removed, restoration is complete.", ["rain"])
    r = verl_compute_score("restoramind", text, {"env_name": "rain"}, None)
    assert r["format_score"] == 1.0 and "score" in r
