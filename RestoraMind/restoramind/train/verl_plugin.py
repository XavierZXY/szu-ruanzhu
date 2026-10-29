# -*- coding: utf-8 -*-
"""
VeRL 智能体强化学习框架桥接（框架本身为独立的开源项目，本软件不包含其源码）。

1. compute_score(...)：序列级奖励函数；verl_compute_score(...) 是适配框架奖励管理器
   (data_source, solution_str, ground_truth, extra_info) 调用约定的入口。输入为多轮拼接的模型输出、ground_truth
   （含 reward_model 与 env_name）以及 extra_info（含原图与工具处理后的图像历史）。
2. make_verl_tool_classes(ToolBase)：把注册表中的每个工具动态生成为 VeRL ToolBase
   子类（name 属性对应训练数据 env_name），execute 内部复用本包 Toolbox。
"""
from __future__ import annotations

import io
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Union

from PIL import Image

from ..agent.parser import parse_response, split_turns
from ..reward.degradation_reward import degradation_type_match, is_clean_answer
from ..reward.format_reward import FormatConfig, check_format_enhanced, check_format_progressive
from ..reward.quality_metrics import discretize, no_reference_score, quality_score
from ..tools.registry import TOOL_SPECS, degradation_of_tool
from ..tools.toolbox import Toolbox, build_toolbox


def _env_flag(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _to_image(data: Any) -> Optional[Image.Image]:
    if data is None:
        return None
    if isinstance(data, Image.Image):
        return data.convert("RGB")
    if isinstance(data, dict):
        if "image" in data and isinstance(data["image"], list) and data["image"]:
            return _to_image(data["image"][0])
        if "bytes" in data:
            return Image.open(io.BytesIO(data["bytes"])).convert("RGB")
    if isinstance(data, (bytes, bytearray)):
        return Image.open(io.BytesIO(data)).convert("RGB")
    if isinstance(data, str) and os.path.exists(data):
        return Image.open(data).convert("RGB")
    return None


def _image_history(extra_info: Optional[Dict[str, Any]]) -> List[Image.Image]:
    if not extra_info:
        return []
    hist = extra_info.get("image_history") or []
    images = [_to_image(h) for h in hist]
    return [im for im in images if im is not None]


def compute_score(solution_str: str, ground_truth: Union[str, Dict[str, Any]],
                  extra_info: Optional[Dict[str, Any]] = None, **overrides: Any) -> Dict[str, Any]:
    """
    VeRL 序列级奖励。配置优先级：显式关键字参数 > 环境变量 > 默认值。
    环境变量：FORMAT_REWARD_WEIGHT, QUALITY_REWARD_WEIGHT, ENABLE_DEGRADATION_TYPE_REWARD,
             DEGRADATION_TYPE_REWARD_WEIGHT, ENABLE_INTERMEDIATE_REWARD, INTERMEDIATE_REWARD_WEIGHT,
             IMAGE_QUALITY_USE_NO_REFERENCE, IMAGE_QUALITY_DISCRETIZE_LEVELS, USE_PROGRESSIVE_FORMAT,
             AGENT_CONVERSATION_MODE, MAX_TOOLS_PER_TURN, ENABLE_TOTAL_TOOLS_UPPER_LIMIT
    """
    if isinstance(ground_truth, str):
        try:
            ground_truth = json.loads(ground_truth)
        except json.JSONDecodeError:
            ground_truth = {}
    ground_truth = ground_truth or {}
    reward_model = ground_truth.get("reward_model") or []
    env_name = str(ground_truth.get("env_name", "") or "")
    truth = [d.get("degradation_type") for d in reward_model if isinstance(d, dict)] or \
            [x.strip() for x in env_name.split(",") if x.strip() and x.strip().lower() != "clean"]
    is_clean = env_name.strip().lower() == "clean" or (not truth and not reward_model)

    cfg = {
        "format_weight": overrides.get("format_weight", _env_float("FORMAT_REWARD_WEIGHT", 0.3)),
        "quality_weight": overrides.get("quality_weight", _env_float("QUALITY_REWARD_WEIGHT", 0.7)),
        "enable_degradation_type_reward": overrides.get("enable_degradation_type_reward", _env_flag("ENABLE_DEGRADATION_TYPE_REWARD", False)),
        "degradation_type_weight": overrides.get("degradation_type_weight", _env_float("DEGRADATION_TYPE_REWARD_WEIGHT", 1.0)),
        "enable_intermediate_reward": overrides.get("enable_intermediate_reward", _env_flag("ENABLE_INTERMEDIATE_REWARD", False)),
        "intermediate_weight": overrides.get("intermediate_weight", _env_float("INTERMEDIATE_REWARD_WEIGHT", 0.5)),
        "use_no_reference": overrides.get("use_no_reference", _env_flag("IMAGE_QUALITY_USE_NO_REFERENCE", False)),
        "discretize_levels": int(overrides.get("discretize_levels", _env_float("IMAGE_QUALITY_DISCRETIZE_LEVELS", 0))),
        "progressive": overrides.get("progressive", _env_flag("USE_PROGRESSIVE_FORMAT", False)),
        "mode": overrides.get("mode", os.environ.get("AGENT_CONVERSATION_MODE", "single_tool_iterative")),
        "max_tools_per_turn": int(overrides.get("max_tools_per_turn", _env_float("MAX_TOOLS_PER_TURN", 1))),
        "upper_limit": overrides.get("upper_limit", _env_flag("ENABLE_TOTAL_TOOLS_UPPER_LIMIT", True)),
    }
    fmt_cfg = FormatConfig(mode=cfg["mode"], max_tools_per_turn=cfg["max_tools_per_turn"],
                           enable_total_tools_upper_limit=cfg["upper_limit"])
    checker = check_format_progressive if cfg["progressive"] else check_format_enhanced
    fmt, reason = checker(solution_str, len(truth), is_clean, fmt_cfg)

    turns = [parse_response(t) for t in split_turns(solution_str)]
    num_calls = sum(len(t.tool_calls) for t in turns)
    answer = next((t.answer for t in reversed(turns) if t.answer), None)
    restoration_log = [str(x) for x in (answer or {}).get("restoration_log", [])] if answer else []

    result: Dict[str, Any] = {
        "format_score": fmt, "format_reason": reason, "quality_score": 0.0,
        "degradation_type_score": 0.0, "intermediate_quality_score": 0.0, "tool_diversity_bonus": 0.0,
        "is_clean_sample": 1.0 if is_clean else 0.0,
        "degradation_types_all": ", ".join(truth) if truth else "clean",
    }
    if is_clean:
        acc = 1.0 if is_clean_answer(restoration_log, num_calls) else 0.0
        result["clean_accuracy"] = acc
        result["score"] = cfg["format_weight"] * fmt + (cfg["quality_weight"] * acc if fmt > 0 else 0.0)
        return result
    if fmt <= 0:
        result["score"] = cfg["format_weight"] * fmt
        return result

    history = _image_history(extra_info)
    original = _to_image((extra_info or {}).get("original_image"))
    quality = 0.0
    if len(history) >= 2:   # history[0] 为退化输入，之后为工具输出
        final = history[-1]
        if cfg["use_no_reference"] or original is None:
            quality = no_reference_score(final)
        else:
            quality = quality_score(final, original)["score"]
        quality = discretize(quality, cfg["discretize_levels"])
    result["quality_score"] = quality
    score = cfg["format_weight"] * fmt + cfg["quality_weight"] * quality

    if cfg["enable_degradation_type_reward"]:
        predicted = restoration_log or [c.degradation or degradation_of_tool(c.name) or "" for t in turns for c in t.tool_calls]
        result["degradation_type_score"] = degradation_type_match(predicted, truth)
        score += cfg["degradation_type_weight"] * result["degradation_type_score"]
    if cfg["enable_intermediate_reward"] and len(history) > 2:
        mids = history[1:-1]
        result["intermediate_quality_score"] = sum(no_reference_score(m) for m in mids) / len(mids)
        score += cfg["intermediate_weight"] * result["intermediate_quality_score"]
    tools = [c.name for t in turns for c in t.tool_calls]
    if tools and len(set(tools)) == len(tools) and len(tools) <= len(truth) + 1:
        result["tool_diversity_bonus"] = 0.1
        score += 0.1
    result["score"] = score
    return result


def verl_compute_score(data_source: str, solution_str: str, ground_truth: Union[str, Dict[str, Any]],
                       extra_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """适配框架奖励管理器的调用约定，在框架的 default_compute_score 中按 data_source 分派到这里。"""
    return compute_score(solution_str, ground_truth, extra_info)


def make_verl_tool_classes(tool_base: type, toolbox: Optional[Toolbox] = None,
                           tool_names: Optional[Sequence[str]] = None) -> Dict[str, type]:
    """
    为 VeRL 的 ToolBase 注册表动态生成工具类。
    生成类遵循框架的工具约定：类属性 name；reset(raw_prompt, multi_modal_data, origin_multi_modal_data)；
    execute(action_string) -> (obs, reward, done, info)，obs 含 prompt 与 multi_modal_data。
    """
    toolbox = toolbox or build_toolbox("remote")
    names = list(tool_names) if tool_names else list(TOOL_SPECS.keys())
    classes: Dict[str, type] = {}
    user_prompt = ("The `{tool}` tool was applied successfully. Analyse the new image for any remaining "
                   "degradations and continue.")

    for name in names:
        def _reset(self, raw_prompt, multi_modal_data, origin_multi_modal_data=None, **kwargs):
            self.chatml_history = raw_prompt
            self.multi_modal_data = multi_modal_data or origin_multi_modal_data

        def _execute(self, action_string: str, _tool_name: str = name, **kwargs):
            parsed = parse_response(action_string)
            if parsed.is_answer:
                return "", 0.0, True, {"status": "finished", "answer": parsed.answer}
            calls = [c for c in parsed.tool_calls if c.name == _tool_name]
            if not calls:
                obs = f"\n<|im_start|>user\nError: no valid call for {_tool_name}<|im_end|>\n<|im_start|>assistant\n"
                return obs, -0.1, False, {"status": "failed", "error": "no tool call"}
            try:
                image = self.multi_modal_data["image"][0]
                result = toolbox.execute(_tool_name, image, calls[0].arguments)
            except Exception as exc:  # noqa: BLE001
                obs = f"\n<|im_start|>user\nError: {exc}<|im_end|>\n<|im_start|>assistant\n"
                return obs, -0.1, False, {"status": "failed", "error": str(exc)}
            self.multi_modal_data["image"][0] = result.image
            obs = {
                "prompt": "\n<|im_start|>user\n<tool_response><image>" + user_prompt.format(tool=_tool_name)
                          + "</tool_response><|im_end|>\n<|im_start|>assistant\n",
                "multi_modal_data": {"image": [result.image]},
            }
            return obs, 0.0, False, {"status": "success", "tool_used": _tool_name, "execution_time": result.elapsed}

        def _init(self, _name=None, _desc="", _params=None, **kwargs):
            tool_base.__init__(self, name=self.name, description=_desc, parameters=_params or {})
            self.chatml_history = []
            self.multi_modal_data = None

        cls = type(f"RestoraMind_{name}", (tool_base,), {
            "name": name, "__init__": _init, "reset": _reset, "execute": _execute,
        })
        classes[name] = cls
    return classes
