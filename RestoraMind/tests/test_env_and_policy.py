# -*- coding: utf-8 -*-
import pytest
from PIL import Image

from restoramind.agent.env import EnvConfig, RestorationEnv
from restoramind.agent.parser import ToolCall, format_answer, format_tool_call
from restoramind.agent.policy import HeuristicPolicy, OpenAICompatiblePolicy, messages_to_openai, select_tool_by_statistics
from restoramind.agent.prompts import build_system_prompt
from restoramind.agent.runner import run_episode
from restoramind.priors import compute_image_statistics
from restoramind.tools.registry import prompt_visible_tools, tools_for_degradation


def test_system_prompt_lists_visible_tools_and_rules():
    text = build_system_prompt("single_tool_iterative")
    for spec in prompt_visible_tools():
        assert spec.name in text
    assert "Image Statistics" in text and "rollback" in text
    with pytest.raises(ValueError):
        build_system_prompt("bad_mode")


def test_reset_observation_contains_image_stats_and_hint(local_toolbox, degraded_sample):
    env = RestorationEnv(local_toolbox, EnvConfig())
    obs = env.reset(degraded_sample)
    assert obs.messages[0]["role"] == "system"
    user = obs.messages[1]["content"]
    assert any(c["type"] == "image" for c in user)
    text = " ".join(c["text"] for c in user if c["type"] == "text")
    assert "Image statistics" in text
    for d in degraded_sample.degradation_types:
        assert d in text
    assert obs.state.remaining == degraded_sample.degradation_types


def test_single_tool_step_updates_history_and_processed(local_toolbox, degraded_sample):
    env = RestorationEnv(local_toolbox, EnvConfig(max_turns=4))
    env.reset(degraded_sample)
    deg = degraded_sample.degradation_types[0]
    tool = tools_for_degradation(deg)[0].name
    obs, done, rec = env.step(format_tool_call("Address the first degradation now.", [ToolCall(tool, deg)]))
    assert not done and rec.executed_tools == [tool] and len(env.trajectory.image_history) == 2
    assert obs.state.processed == [deg] and deg not in obs.state.remaining
    assert obs.messages[-2]["role"] == "assistant" and obs.messages[-1]["role"] == "user"


def test_single_tool_mode_truncates_extra_tools(local_toolbox, degraded_sample):
    env = RestorationEnv(local_toolbox, EnvConfig(max_turns=4))
    env.reset(degraded_sample)
    calls = [ToolCall("restormer_deraining", "rain"), ToolCall("swinir_denoising", "noise")]
    _, _, rec = env.step(format_tool_call("Two tools in one turn.", calls))
    assert rec.executed_tools == ["restormer_deraining"] and rec.format_errors


def test_rollback_applies_tool_to_previous_image(local_toolbox, degraded_sample):
    env = RestorationEnv(local_toolbox, EnvConfig(max_turns=5, max_retries=1))
    env.reset(degraded_sample)
    env.step(format_tool_call("First attempt with one deblur tool.", [ToolCall("restormer_motion_deblurring", "motion blur")]))
    _, _, rec = env.step(format_tool_call("Retry on previous image with alternative.",
                                          [ToolCall("nafnet_deblur", "motion blur", rollback=True)]))
    assert rec.executed_tools == ["nafnet_deblur"] and env.trajectory.image_history[-1] is not env.trajectory.image_history[1]
    # 第二次回退超过上限，退化为普通步骤（作用于当前图像）
    _, _, rec2 = env.step(format_tool_call("Another rollback should be ignored.",
                                           [ToolCall("mprnet_motion_deblurring", "motion blur", rollback=True)]))
    assert rec2.executed_tools == ["mprnet_motion_deblurring"]


def test_unknown_tool_reports_error_without_crash(local_toolbox, degraded_sample):
    env = RestorationEnv(local_toolbox, EnvConfig(max_turns=3))
    env.reset(degraded_sample)
    obs, done, rec = env.step(format_tool_call("Call a tool that does not exist.", [ToolCall("ghost_tool", "noise")]))
    assert rec.error and not rec.executed_tools and obs.state.last_error
    assert "error" in " ".join(c["text"] for c in obs.messages[-1]["content"] if c["type"] == "text").lower()


def test_answer_terminates_episode(local_toolbox, degraded_sample):
    env = RestorationEnv(local_toolbox, EnvConfig())
    env.reset(degraded_sample)
    _, done, rec = env.step(format_answer("Image already looks clean enough.", []))
    assert done and rec.is_answer and env.trajectory.finished


def test_max_turns_truncates(local_toolbox, degraded_sample):
    env = RestorationEnv(local_toolbox, EnvConfig(max_turns=2))
    env.reset(degraded_sample)
    env.step("bad")
    _, done, _ = env.step("bad again")
    assert done and env.trajectory.truncated and not env.trajectory.finished


def test_heuristic_policy_completes_episode(local_toolbox, degraded_sample):
    traj = run_episode(RestorationEnv(local_toolbox, EnvConfig(max_turns=6)), HeuristicPolicy(), degraded_sample)
    assert traj.finished and set(traj.restoration_log) == set(degraded_sample.degradation_types)
    assert traj.num_tool_calls == len(degraded_sample.degradation_types)   # 近似线性：每个退化一次调用


def test_multi_tool_planning_mode(local_toolbox, degraded_sample):
    cfg = EnvConfig(mode="multi_tool_planning", max_turns=2, max_tools_per_turn=0)
    traj = run_episode(RestorationEnv(local_toolbox, cfg), HeuristicPolicy(mode="multi_tool_planning"), degraded_sample)
    assert traj.finished and len(traj.steps[0].executed_tools) == len(degraded_sample.degradation_types)


def test_statistics_guided_selection(clean_image):
    stats = compute_image_statistics(clean_image)
    cands = tools_for_degradation("defocus blur")
    chosen = select_tool_by_statistics(cands, stats)
    assert chosen in cands
    assert select_tool_by_statistics(cands, None) is cands[0]


def test_messages_to_openai_encodes_images(local_toolbox, degraded_sample):
    obs = RestorationEnv(local_toolbox).reset(degraded_sample)
    msgs = messages_to_openai(obs.messages)
    parts = msgs[1]["content"]
    assert any(p["type"] == "image_url" and p["image_url"]["url"].startswith("data:image/png;base64,") for p in parts)


class _FakeCompletions:
    def __init__(self, reply):
        self.reply = reply

    def create(self, **kwargs):
        class _Msg:
            content = self.reply
        class _Choice:
            message = _Msg()
        class _Resp:
            choices = [_Choice()]
        return _Resp()


class _FakeClient:
    def __init__(self, reply):
        self.chat = type("Chat", (), {"completions": _FakeCompletions(reply)})()


def test_openai_policy_with_fake_client(local_toolbox, degraded_sample):
    reply = format_answer("Fake model says the image is clean.", [])
    policy = OpenAICompatiblePolicy(client=_FakeClient(reply), model="fake")
    traj = run_episode(RestorationEnv(local_toolbox, EnvConfig(max_turns=2)), policy, degraded_sample)
    assert traj.finished and traj.answer["restoration_log"] == []
