# -*- coding: utf-8 -*-
"""
图像低层统计先验提取。

在每个决策状态下，以极低计算代价（传统图像处理）提取当前图像的
颜色分布、亮度、对比度、边缘密度、频域能量以及退化估计特征，
组织为结构化文本随图像一起输入多模态智能体，用于
（1）辅助判断剩余退化类型；（2）在同一退化类型的多个候选工具之间做细粒度自适应选择。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import cv2
import numpy as np
from PIL import Image

# 各统计量的「低 / 中 / 高」分界阈值（经验值，可按数据分布调整）
LEVEL_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "brightness_mean": (70.0, 150.0),
    "dark_ratio": (0.15, 0.40),
    "contrast_global": (35.0, 60.0),
    "saturation_mean": (0.15, 0.35),
    "edge_density": (0.04, 0.12),
    "high_freq_energy": (0.05, 0.15),
    "noise_sigma": (4.0, 12.0),
    "blur_score": (50.0, 300.0),   # 拉普拉斯方差，越低越模糊；此项按反向解释
    "blockiness": (0.02, 0.06),
    "haze_score": (0.35, 0.55),
}


def qualitative_level(name: str, value: float) -> str:
    """将连续统计量映射为 low / medium / high 的定性描述。"""
    lo, hi = LEVEL_THRESHOLDS[name]
    if value < lo:
        return "low"
    if value < hi:
        return "medium"
    return "high"


@dataclass
class ImageStatistics:
    """单张图像的低层统计特征。"""
    width: int
    height: int
    brightness_mean: float      # 灰度均值 [0,255]
    dark_ratio: float           # 暗区像素比例（灰度 < 40）
    highlight_ratio: float      # 高光像素比例（灰度 > 220）
    contrast_global: float      # 灰度标准差
    saturation_mean: float      # HSV 饱和度均值 [0,1]
    color_cast: float           # 通道均值最大偏差 / 255，衡量色偏
    edge_density: float         # Canny 边缘像素比例
    high_freq_energy: float     # 高频频谱能量占比
    noise_sigma: float          # 噪声水平估计（中值滤波残差 MAD）
    blur_score: float           # 拉普拉斯方差，越小越模糊
    blockiness: float           # 8x8 块边界梯度不连续度，衡量 JPEG 块效应
    haze_score: float           # 暗通道均值 [0,1]，越高雾越浓

    def to_dict(self) -> dict:
        """转换为字典。"""
        return asdict(self)

    def to_qualitative(self) -> Dict[str, str]:
        """定性描述，供提示词使用。"""
        q = {
            "brightness": qualitative_level("brightness_mean", self.brightness_mean),
            "dark_region_ratio": qualitative_level("dark_ratio", self.dark_ratio),
            "contrast": qualitative_level("contrast_global", self.contrast_global),
            "saturation": qualitative_level("saturation_mean", self.saturation_mean),
            "edge_density": qualitative_level("edge_density", self.edge_density),
            "high_frequency_energy": qualitative_level("high_freq_energy", self.high_freq_energy),
            "noise_estimate": qualitative_level("noise_sigma", self.noise_sigma),
            "jpeg_blockiness": qualitative_level("blockiness", self.blockiness),
            "haze_estimate": qualitative_level("haze_score", self.haze_score),
        }
        # 模糊度反向：拉普拉斯方差低 → 模糊程度高
        inv = {"low": "high", "medium": "medium", "high": "low"}
        q["blur_estimate"] = inv[qualitative_level("blur_score", self.blur_score)]
        return q

    def to_prompt_text(self) -> str:
        """结构化文本，注入智能体的每个决策状态。"""
        q = self.to_qualitative()
        lines = [f"Image statistics ({self.width}x{self.height}):"]
        lines.append(f"- brightness: {q['brightness']} (mean={self.brightness_mean:.1f}, dark_ratio={self.dark_ratio:.2f})")
        lines.append(f"- contrast: {q['contrast']} (std={self.contrast_global:.1f})")
        lines.append(f"- saturation: {q['saturation']} (mean={self.saturation_mean:.2f}, color_cast={self.color_cast:.2f})")
        lines.append(f"- edge density: {q['edge_density']} ({self.edge_density:.3f})")
        lines.append(f"- high-frequency energy: {q['high_frequency_energy']} ({self.high_freq_energy:.3f})")
        lines.append(f"- noise estimate: {q['noise_estimate']} (sigma={self.noise_sigma:.1f})")
        lines.append(f"- blur estimate: {q['blur_estimate']} (laplacian_var={self.blur_score:.1f})")
        lines.append(f"- jpeg blockiness: {q['jpeg_blockiness']} ({self.blockiness:.3f})")
        lines.append(f"- haze estimate: {q['haze_estimate']} (dark_channel={self.haze_score:.2f})")
        return "\n".join(lines)


def _to_rgb_array(image) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return arr[..., :3].astype(np.uint8)


def _estimate_noise_sigma(gray: np.ndarray) -> float:
    """中值滤波残差的 MAD 鲁棒估计噪声标准差。"""
    med = cv2.medianBlur(gray, 3)
    residual = gray.astype(np.float32) - med.astype(np.float32)
    mad = np.median(np.abs(residual - np.median(residual)))
    return float(1.4826 * mad)


def _high_freq_energy(gray: np.ndarray) -> float:
    """高频（半径 > 1/4 频谱宽度）能量占总能量的比例。"""
    f = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
    power = np.abs(f) ** 2
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    mask = radius > min(h, w) / 4.0
    total = float(power.sum()) + 1e-8
    return float(power[mask].sum() / total)


def _blockiness(gray: np.ndarray) -> float:
    """8x8 块边界处与非边界处水平/垂直梯度差异，衡量 JPEG 块效应。"""
    g = gray.astype(np.float32)
    dh = np.abs(np.diff(g, axis=1))
    dv = np.abs(np.diff(g, axis=0))
    if dh.shape[1] < 16 or dv.shape[0] < 16:
        return 0.0
    bh = dh[:, 7::8].mean()
    bv = dv[7::8, :].mean()
    nh = np.delete(dh, np.s_[7::8], axis=1).mean()
    nv = np.delete(dv, np.s_[7::8], axis=0).mean()
    return float(max(0.0, (bh - nh) + (bv - nv)) / 255.0)


def _dark_channel(rgb: np.ndarray, patch: int = 15) -> float:
    """暗通道先验均值，用于雾浓度估计。"""
    mn = rgb.min(axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    dark = cv2.erode(mn, kernel)
    return float(dark.mean() / 255.0)


def compute_image_statistics(image) -> ImageStatistics:
    """计算图像低层统计先验。输入 PIL 图像或 RGB uint8 数组。"""
    rgb = _to_rgb_array(image)
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    brightness = float(gray.mean())
    dark_ratio = float((gray < 40).mean())
    highlight_ratio = float((gray > 220).mean())
    contrast = float(gray.std())
    saturation = float(hsv[..., 1].mean() / 255.0)
    chan_means = rgb.reshape(-1, 3).mean(axis=0)
    color_cast = float((chan_means.max() - chan_means.min()) / 255.0)

    edges = cv2.Canny(gray, 100, 200)
    edge_density = float((edges > 0).mean())
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    return ImageStatistics(
        width=w,
        height=h,
        brightness_mean=brightness,
        dark_ratio=dark_ratio,
        highlight_ratio=highlight_ratio,
        contrast_global=contrast,
        saturation_mean=saturation,
        color_cast=color_cast,
        edge_density=edge_density,
        high_freq_energy=_high_freq_energy(gray),
        noise_sigma=_estimate_noise_sigma(gray),
        blur_score=blur_score,
        blockiness=_blockiness(gray),
        haze_score=_dark_channel(rgb),
    )
