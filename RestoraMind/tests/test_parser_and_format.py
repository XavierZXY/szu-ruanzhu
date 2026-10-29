# -*- coding: utf-8 -*-
from restoramind.agent.parser import ToolCall, format_answer, format_tool_call, parse_response, split_turns
from restoramind.reward.format_reward import FormatConfig, check_format_enhanced, check_format_progressive

GOOD_TURN = format_tool_call("Rain streaks dominate; derain first.", [ToolCall("restormer_deraining", "rain")])
ANSWER = format_answer("All degradations addressed, image is clean now.", ["rain", "noise"])


def test_parse_tool_call_and_answer():
    p = parse_response(GOOD_TURN)
    assert p.is_valid and p.tool_calls[0].name == "restormer_deraining" and p.tool_calls[0].degradation == "rain"
    a = parse_response(ANSWER)
    assert a.is_answer and a.answer["restoration_log"] == ["rain", "noise"]


def test_parse_errors():
    assert "missing <think>" in parse_response("<tool_call>[]</tool_call>").errors[0]
    assert any("invalid tool_call" in e for e in parse_response("<think>abc def ghi jk</think><tool_call>[{}]</tool_call>").errors)
    assert any("both" in e for e in parse_response(GOOD_TURN + ANSWER).errors)
    assert parse_response("<think>only thinking here</think>").errors


def test_parse_rollback_flag():
    text = '<think>retry on previous image</think><tool_call>[{"name":"nafnet_deblur","degradation":"motion blur","arguments":{},"rollback":true}]</tool_call>'
    assert parse_response(text).tool_calls[0].rollback is True


def test_split_turns():
    turns = split_turns(GOOD_TURN + "\n" + GOOD_TURN + "\n" + ANSWER)
    assert len(turns) == 3 and "<answer>" in turns[-1]


def test_enhanced_format_accepts_valid_single_tool_trajectory():
    text = "\n".join([GOOD_TURN, format_tool_call("Now denoise the result.", [ToolCall("scunet_real_denoising_gan", "noise")]), ANSWER])
    score, reason = check_format_enhanced(text, degradation_count=2)
    assert score == 1.0, reason


def test_enhanced_format_rejects_violations():
    cfg = FormatConfig(mode="single_tool_iterative")
    two_tools = format_tool_call("Two tools at once is forbidden here.",
                                 [ToolCall("restormer_deraining", "rain"), ToolCall("nafnet_deblur", "motion blur")])
    assert check_format_enhanced(two_tools + ANSWER, 2, cfg=cfg)[0] == -1.0
    assert check_format_enhanced(ANSWER + "\n" + GOOD_TURN, 1, cfg=cfg)[0] == -1.0      # answer 不在最后一轮
    assert check_format_enhanced(ANSWER, 1, cfg=cfg)[0] == -1.0                           # 退化样本未调用工具
    assert check_format_enhanced(ANSWER, 0, is_clean=True, cfg=cfg)[0] == 1.0            # clean 样本直接作答合法
    too_many = "\n".join([GOOD_TURN] * 4 + [ANSWER])
    assert check_format_enhanced(too_many, degradation_count=1, cfg=cfg)[0] == -1.0      # 超过退化数+1
    bad_tool = format_tool_call("Use a tool that is not registered.", [ToolCall("magic_tool", "noise")])
    assert check_format_enhanced(bad_tool + ANSWER, 1, cfg=cfg)[0] == -1.0


def test_multi_tool_mode_allows_sequences():
    cfg = FormatConfig(mode="multi_tool_planning", max_tools_per_turn=0, enable_total_tools_upper_limit=False)
    plan = format_tool_call("Full plan: derain then denoise.", [ToolCall("restormer_deraining", "rain"), ToolCall("swinir_denoising", "noise")])
    assert check_format_enhanced(plan + "\n" + ANSWER, 2, cfg=cfg)[0] == 1.0


def test_progressive_levels():
    assert check_format_progressive("garbage")[0] == 0.0
    assert check_format_progressive("<think>only thinking text</think>")[0] == 0.2
    assert check_format_progressive("<think>thinking text here</think><tool_call>[{broken</tool_call>")[0] == 0.4
    bad_tool = format_tool_call("Illegal tool name used here.", [ToolCall("magic_tool", "noise")])
    assert check_format_progressive(bad_tool)[0] == 0.6
    assert check_format_progressive(GOOD_TURN + ANSWER, 1)[0] == 1.0
