# -*- coding: utf-8 -*-
"""
复原工具规格注册表。

每个工具对应一个预训练专家模型（或经典算子）的一个任务头，
记录其可处理的退化类型、所属模型家族、远程服务端口、参数模式与相对计算代价。
同一退化类型可配置多个候选工具，由智能体依据图像统计先验自适应选择。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# 远程专家模型服务的默认端口分配（每个模型家族一个服务进程）
FAMILY_PORTS: Dict[str, int] = {
    "SwinIR": 5001,
    "DehazeFormer": 5002,
    "DRBNet": 5003,
    "MPRNet": 5004,
    "FBCNN": 5005,
    "Restormer": 5006,
    "XRestormer": 5007,
    "SCUNet": 5008,
    "Retinexformer": 5009,
    "HAT": 5010,
    "NeRD": 5011,
    "NAFNet": 5012,
    "DiffBIR": 5013,
    "Classical": 0,   # 本地经典算子，无需远程服务
}


@dataclass(frozen=True)
class ToolSpec:
    """复原工具规格：名称、目标退化、模型家族、说明、参数模式与相对代价。"""
    name: str
    degradation: str
    family: str
    description: str
    parameters: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    cost: float = 1.0
    prompt_visible: bool = True

    @property
    def port(self) -> int:
        """该工具所属模型家族的服务端口，0 表示本地算子。"""
        return FAMILY_PORTS[self.family]

    @property
    def is_local(self) -> bool:
        """是否为本地算子。"""
        return self.port == 0

    def normalize_arguments(self, arguments: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """填充默认值并把数值参数裁剪到合法区间；忽略未定义参数。"""
        arguments = arguments or {}
        out: Dict[str, Any] = {}
        for key, schema in self.parameters.items():
            value = arguments.get(key, schema.get("default"))
            if value is None:
                continue
            if schema.get("type") in ("number", "integer"):
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    value = float(schema.get("default", 0))
                if "min" in schema:
                    value = max(float(schema["min"]), value)
                if "max" in schema:
                    value = min(float(schema["max"]), value)
                if "choices" in schema:
                    value = min(schema["choices"], key=lambda c: abs(c - value))
                if schema.get("type") == "integer":
                    value = int(round(value))
            out[key] = value
        return out

    def prompt_line(self) -> str:
        """提示词中的一行工具说明。"""
        if self.parameters:
            params = ", ".join(
                f'"{k}": {s.get("hint", s.get("type"))}' for k, s in self.parameters.items()
            )
            params = "{ " + params + " }"
        else:
            params = "{}"
        return f'- `{self.name}`: `{params}` — For "{self.degradation}". {self.description}'


_STRENGTH = {"strength": {"type": "number", "min": 0.0, "max": 1.0, "default": 0.5, "hint": "0..1"}}
_JPEG_Q = {"jpeg": {"type": "integer", "min": 1, "max": 100, "default": 30, "hint": "1..100"}}
_SCALE = {"scale": {"type": "integer", "choices": [2, 3, 4], "default": 2, "hint": "2|3|4"}}
_GAMMA = {"gamma": {"type": "number", "min": 0.2, "max": 1.0, "default": 0.5, "hint": "0.2..1.0"}}
_SHIFT = {"shift": {"type": "integer", "min": 0, "max": 120, "default": 40, "hint": "0..120"}}

TOOL_SPECS: Dict[str, ToolSpec] = {s.name: s for s in [
    # ---- 去雾 ----
    ToolSpec("dehazeformer_dehaze", "haze", "DehazeFormer", "Transformer dehazing, robust to dense haze.", _STRENGTH, 1.2),
    # ---- 去噪（多候选）----
    ToolSpec("scunet_real_denoising_psnr", "noise", "SCUNet", "Real-noise denoiser optimised for PSNR; smooth output, good for heavy noise.", {}, 1.0),
    ToolSpec("scunet_real_denoising_gan", "noise", "SCUNet", "Real-noise denoiser with GAN loss; preserves texture, good for detailed images.", {}, 1.0),
    ToolSpec("swinir_denoising", "noise", "SwinIR", "Gaussian-noise denoiser; strong on moderate synthetic noise.", {}, 1.1),
    ToolSpec("mprnet_denoising", "noise", "MPRNet", "Multi-stage denoiser; balanced detail and smoothness.", {}, 1.1, prompt_visible=False),
    ToolSpec("scunet_color_denoising", "noise", "SCUNet", "Synthetic Gaussian-noise colour denoiser (levels 15/25/50).", {}, 1.0),
    # ---- 运动去模糊（多候选）----
    ToolSpec("restormer_motion_deblurring", "motion blur", "Restormer", "General motion deblurring, strong on large blur kernels.", {}, 1.3),
    ToolSpec("nafnet_deblur", "motion blur", "NAFNet", "Lightweight motion deblurring; preserves fine edges on sharp-textured images.", {}, 0.8),
    ToolSpec("mprnet_motion_deblurring", "motion blur", "MPRNet", "Multi-stage motion deblurring.", {}, 1.1, prompt_visible=False),
    ToolSpec("xrestormer_motion_deblurring", "motion blur", "XRestormer", "Motion deblurring variant with channel attention.", {}, 1.3, prompt_visible=False),
    # ---- 散焦去模糊（多候选）----
    ToolSpec("restormer_defocus_deblurring", "defocus blur", "Restormer", "Defocus deblurring for smooth, low-edge-density images.", {}, 1.3),
    ToolSpec("drbnet_defocus_deblurring", "defocus blur", "DRBNet", "Defocus deblurring that recovers fine detail on high-edge-density images.", {}, 1.0),
    # ---- 去雨（多候选）----
    ToolSpec("restormer_deraining", "rain", "Restormer", "Rain-streak removal with strong structure preservation.", {}, 1.3),
    ToolSpec("mprnet_deraining", "rain", "MPRNet", "Rain-streak removal; good for dense rain.", {}, 1.1),
    ToolSpec("xrestormer_deraining", "rain", "XRestormer", "Rain-streak removal with channel attention; robust on heavy rain.", {}, 1.3),
    ToolSpec("nerd_deraining", "rain", "NeRD", "Rain removal with directional priors.", {}, 1.2, prompt_visible=False),
    # ---- JPEG 伪影（多候选）----
    ToolSpec("fbcnn_jpeg_artifact_removal", "jpeg compression artifact", "FBCNN", "Flexible blind JPEG artifact removal; quality factor controllable.", _JPEG_Q, 0.9),
    ToolSpec("swinir_jpeg_artifact_removal", "jpeg compression artifact", "SwinIR", "JPEG artifact removal with detail recovery.", {}, 1.1, prompt_visible=False),
    # ---- 超分辨率（多候选）----
    ToolSpec("swinir_super_resolution", "low resolution", "SwinIR", "Robust real-world super-resolution.", _SCALE, 1.5),
    ToolSpec("xrestormer_super_resolution", "low resolution", "XRestormer", "Super-resolution with channel attention; keeps fine texture on detail-rich images.", _SCALE, 1.4),
    ToolSpec("diffbir_super_resolution", "low resolution", "DiffBIR", "Diffusion-based blind super-resolution; hallucinates rich detail at high cost.", _SCALE, 3.0),
    ToolSpec("hat_super_resolution", "low resolution", "HAT", "Super-resolution with strong texture hallucination for detail-rich images.", _SCALE, 1.8, prompt_visible=False),
    # ---- 低光增强（多候选）----
    ToolSpec("retinexformer_sdsd_indoor", "dark", "Retinexformer", "Retinex-based low-light enhancement for indoor/low-noise scenes.", {}, 1.0),
    ToolSpec("retinexformer_fivek", "dark", "Retinexformer", "Retinexformer trained on FiveK; natural exposure for photographic scenes.", {}, 1.0),
    ToolSpec("retinexformer_enhance", "dark", "Retinexformer", "General low-light enhancement.", {}, 1.0, prompt_visible=False),
    ToolSpec("gamma_correction", "dark", "Classical", "Fast gamma brightening without learned priors.", _GAMMA, 0.1, prompt_visible=False),
    ToolSpec("histogram_equalization", "dark", "Classical", "CLAHE contrast/brightness enhancement.", {}, 0.1, prompt_visible=False),
    ToolSpec("constant_shift", "dark", "Classical", "Constant brightness shift.", _SHIFT, 0.05, prompt_visible=False),
]}


def get_tool_spec(name: str) -> ToolSpec:
    """按名称获取工具规格，不存在时抛出 KeyError。"""
    if name not in TOOL_SPECS:
        raise KeyError(f"未注册的工具: {name}")
    return TOOL_SPECS[name]


def tools_for_degradation(degradation: str, visible_only: bool = False) -> List[ToolSpec]:
    """返回处理指定退化类型的候选工具列表。"""
    return [s for s in TOOL_SPECS.values() if s.degradation == degradation and (s.prompt_visible or not visible_only)]


def degradation_of_tool(name: str) -> Optional[str]:
    """返回工具对应的退化类型，未知工具返回 None。"""
    spec = TOOL_SPECS.get(name)
    return spec.degradation if spec else None


def prompt_visible_tools() -> List[ToolSpec]:
    """返回写入智能体提示词的工具列表。"""
    return [s for s in TOOL_SPECS.values() if s.prompt_visible]


def all_degradations() -> List[str]:
    """返回注册表覆盖的全部退化类型。"""
    seen: List[str] = []
    for s in TOOL_SPECS.values():
        if s.degradation not in seen:
            seen.append(s.degradation)
    return seen
