# -*- coding: utf-8 -*-
"""
八类图像退化算子。

每个算子签名统一为 ``fn(img: np.ndarray[H,W,3] uint8 RGB, level: str, rng) -> np.ndarray``，
``level`` 取值 low / medium / high，对应退化强度区间由 ``LEVEL_PARAMS`` 给出。
算子之间可任意组合、任意顺序叠加，用于模拟真实场景的未知复合退化。
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import cv2
import numpy as np

# 退化类型名称与智能体提示词、奖励模块中的名称保持一致
DEGRADATION_TYPES = (
    "haze",
    "rain",
    "defocus blur",
    "low resolution",
    "motion blur",
    "dark",
    "jpeg compression artifact",
    "noise",
)
DEGRADATION_LEVELS = ("low", "medium", "high")

# 每类退化在三个强度等级下的参数采样区间
LEVEL_PARAMS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "haze": {"low": (0.2, 0.4), "medium": (0.4, 0.6), "high": (0.6, 0.85)},          # 透射率衰减
    "rain": {"low": (200, 400), "medium": (400, 800), "high": (800, 1500)},          # 雨线数量
    "defocus blur": {"low": (1.0, 2.0), "medium": (2.0, 3.5), "high": (3.5, 5.0)},   # 高斯 sigma
    "low resolution": {"low": (1.5, 2.0), "medium": (2.0, 3.0), "high": (3.0, 4.0)}, # 下采样倍率
    "motion blur": {"low": (5, 9), "medium": (9, 15), "high": (15, 25)},             # 运动核长度
    "dark": {"low": (1.5, 2.0), "medium": (2.0, 3.0), "high": (3.0, 4.5)},           # gamma
    "jpeg compression artifact": {"low": (40, 60), "medium": (20, 40), "high": (5, 20)},  # JPEG 质量
    "noise": {"low": (5, 15), "medium": (15, 30), "high": (30, 50)},                 # 高斯噪声 sigma
}


def _sample(rng: np.random.Generator, dtype: str, level: str) -> float:
    lo, hi = LEVEL_PARAMS[dtype][level]
    return float(rng.uniform(lo, hi))


def add_haze(img: np.ndarray, level: str, rng: np.random.Generator) -> np.ndarray:
    """大气散射模型 I = J * t + A * (1 - t)，t 为透射率。"""
    beta = _sample(rng, "haze", level)
    h, w = img.shape[:2]
    # 用从上到下的深度渐变模拟远景更浓的雾
    depth = np.linspace(0.6, 1.0, h, dtype=np.float32)[:, None]
    depth = np.repeat(depth, w, axis=1)
    t = np.exp(-beta * depth)[..., None]
    airlight = float(rng.uniform(0.8, 1.0)) * 255.0
    out = img.astype(np.float32) * t + airlight * (1.0 - t)
    return np.clip(out, 0, 255).astype(np.uint8)


def add_rain(img: np.ndarray, level: str, rng: np.random.Generator) -> np.ndarray:
    """绘制随机倾斜雨线并做轻微运动模糊。"""
    n = int(_sample(rng, "rain", level))
    h, w = img.shape[:2]
    layer = np.zeros((h, w), dtype=np.float32)
    angle = float(rng.uniform(-15, 15))
    length = int(rng.integers(8, 20))
    dx = int(round(length * np.sin(np.deg2rad(angle))))
    dy = int(round(length * np.cos(np.deg2rad(angle))))
    for _ in range(n):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        cv2.line(layer, (x, y), (x + dx, y + dy), float(rng.uniform(0.4, 1.0)), 1)
    layer = cv2.GaussianBlur(layer, (3, 3), 0)
    out = img.astype(np.float32) + layer[..., None] * 200.0
    return np.clip(out, 0, 255).astype(np.uint8)


def add_defocus_blur(img: np.ndarray, level: str, rng: np.random.Generator) -> np.ndarray:
    """圆盘核近似的离焦模糊。"""
    sigma = _sample(rng, "defocus blur", level)
    radius = max(1, int(round(sigma * 1.5)))
    k = 2 * radius + 1
    kernel = np.zeros((k, k), dtype=np.float32)
    cv2.circle(kernel, (radius, radius), radius, 1.0, -1)
    kernel /= kernel.sum()
    return cv2.filter2D(img, -1, kernel)


def add_low_resolution(img: np.ndarray, level: str, rng: np.random.Generator) -> np.ndarray:
    """下采样后再上采样回原尺寸，模拟分辨率损失（尺寸保持不变以便指标计算）。"""
    scale = _sample(rng, "low resolution", level)
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(8, int(w / scale)), max(8, int(h / scale))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def add_motion_blur(img: np.ndarray, level: str, rng: np.random.Generator) -> np.ndarray:
    """随机方向的线性运动模糊核。"""
    length = int(_sample(rng, "motion blur", level))
    angle = float(rng.uniform(0, 180))
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot, (length, length))
    kernel /= max(kernel.sum(), 1e-6)
    return cv2.filter2D(img, -1, kernel)


def add_dark(img: np.ndarray, level: str, rng: np.random.Generator) -> np.ndarray:
    """gamma 压暗并叠加轻微泊松噪声模拟低光成像。"""
    gamma = _sample(rng, "dark", level)
    x = img.astype(np.float32) / 255.0
    out = np.power(x, gamma) * 255.0
    out = out + rng.normal(0, 2.0, size=out.shape).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def add_jpeg(img: np.ndarray, level: str, rng: np.random.Generator) -> np.ndarray:
    """低质量 JPEG 编解码产生块效应与振铃伪影。"""
    quality = int(_sample(rng, "jpeg compression artifact", level))
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return img
    dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)


def add_noise(img: np.ndarray, level: str, rng: np.random.Generator) -> np.ndarray:
    """加性高斯白噪声。"""
    sigma = _sample(rng, "noise", level)
    out = img.astype(np.float32) + rng.normal(0, sigma, size=img.shape).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


DEGRADATION_FUNCTIONS: Dict[str, Callable[[np.ndarray, str, np.random.Generator], np.ndarray]] = {
    "haze": add_haze,
    "rain": add_rain,
    "defocus blur": add_defocus_blur,
    "low resolution": add_low_resolution,
    "motion blur": add_motion_blur,
    "dark": add_dark,
    "jpeg compression artifact": add_jpeg,
    "noise": add_noise,
}


def apply_degradation(img: np.ndarray, dtype: str, level: str, rng: np.random.Generator | None = None) -> np.ndarray:
    """对 RGB uint8 图像施加一种指定类型与强度的退化。"""
    if dtype not in DEGRADATION_FUNCTIONS:
        raise ValueError(f"未知退化类型: {dtype}，可选: {DEGRADATION_TYPES}")
    if level not in DEGRADATION_LEVELS:
        raise ValueError(f"未知退化强度: {level}，可选: {DEGRADATION_LEVELS}")
    rng = rng if rng is not None else np.random.default_rng()
    return DEGRADATION_FUNCTIONS[dtype](img, level, rng)
