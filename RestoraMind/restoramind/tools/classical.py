# -*- coding: utf-8 -*-
"""
本地经典复原算子。

两个用途：
（1）BrighteningToolbox 等本身即为经典算子的工具；
（2）在无 GPU / 无远程专家模型服务时，为每个注册工具提供可运行的经典近似实现，
    保证训练环境、推理流程与单元测试在离线条件下完整可运行。
同一退化类型下不同候选工具采用不同参数或算法组合，保留「候选工具互补」的语义。
"""
from __future__ import annotations

from typing import Any, Callable, Dict

import cv2
import numpy as np

Op = Callable[[np.ndarray, Dict[str, Any]], np.ndarray]


def _unsharp(img: np.ndarray, sigma: float, amount: float) -> np.ndarray:
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0)


def _clip(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 255).astype(np.uint8)


# ---------------- 去噪 ----------------
def denoise_nlmeans(img, params, h=10.0, post_sharpen=0.0):
    """非局部均值去噪，h 控制强度，可选后置锐化。"""
    out = cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)
    return _unsharp(out, 1.0, post_sharpen) if post_sharpen > 0 else out


def denoise_bilateral(img, params):
    """双边滤波去噪。"""
    return cv2.bilateralFilter(img, 7, 50, 50)


# ---------------- 去模糊 ----------------
def deblur_wiener_like(img, params, radius=1.5, amount=1.2):
    """基于反锐化掩模的去模糊近似。"""
    return _clip(_unsharp(img, radius, amount))


def deblur_richardson_lucy(img, params, iterations=8, ksize=5):
    """简化的 Richardson-Lucy 反卷积（均值核）。"""
    psf = np.ones((ksize, ksize), dtype=np.float32) / (ksize * ksize)
    x = img.astype(np.float32) / 255.0
    est = x.copy()
    for _ in range(iterations):
        conv = cv2.filter2D(est, -1, psf)
        ratio = x / (conv + 1e-6)
        est = est * cv2.filter2D(ratio, -1, psf)
    return _clip(est * 255.0)


# ---------------- 去雾 ----------------
def dehaze_dark_channel(img, params):
    """暗通道先验去雾，strength 控制去雾强度 omega。"""
    strength = float(params.get("strength", 0.5))
    omega = 0.6 + 0.35 * strength
    x = img.astype(np.float32) / 255.0
    mn = x.min(axis=2)
    dark = cv2.erode(mn, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    flat = dark.reshape(-1)
    idx = np.argsort(flat)[-max(1, int(0.001 * flat.size)):]
    airlight = x.reshape(-1, 3)[idx].mean(axis=0)
    t = 1.0 - omega * cv2.erode((x / (airlight + 1e-6)).min(axis=2), cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    t = cv2.GaussianBlur(t, (0, 0), 8)
    t = np.clip(t, 0.1, 1.0)[..., None]
    out = (x - airlight) / t + airlight
    return _clip(out * 255.0)


# ---------------- 去雨 ----------------
def derain_median(img, params, ksize=5):
    """中值滤波去除细雨线。"""
    return cv2.medianBlur(img, ksize)


def derain_guided(img, params):
    """方向中值 + 双边滤波组合去除细雨线。"""
    out = cv2.medianBlur(img, 3)
    return cv2.bilateralFilter(out, 9, 40, 40)


# ---------------- 超分辨率 ----------------
def sr_bicubic_sharpen(img, params, amount=0.6):
    """双三次放大并锐化的超分辨率近似。"""
    scale = int(params.get("scale", 2))
    h, w = img.shape[:2]
    up = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    return _clip(_unsharp(up, 1.0, amount))


def sr_lanczos(img, params):
    """Lanczos 插值放大。"""
    scale = int(params.get("scale", 2))
    h, w = img.shape[:2]
    return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)


# ---------------- 低光增强 ----------------
def enhance_gamma(img, params):
    """gamma 校正增亮。"""
    gamma = float(params.get("gamma", 0.5))
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, lut)


def enhance_clahe(img, params):
    """在 LAB 亮度通道上做限制对比度自适应直方图均衡。"""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def enhance_constant_shift(img, params):
    """常数平移增亮。"""
    shift = int(params.get("shift", 40))
    return _clip(img.astype(np.int16) + shift)


def enhance_retinex(img, params, sigma=30.0):
    """单尺度 Retinex：log(I) - log(blur(I))，再归一化。"""
    x = img.astype(np.float32) + 1.0
    illum = cv2.GaussianBlur(x, (0, 0), sigma)
    r = np.log(x) - np.log(illum + 1.0)
    r = (r - r.min()) / (r.max() - r.min() + 1e-6)
    return _clip(r * 255.0)


# ---------------- JPEG 伪影 ----------------
def deblock_bilateral(img, params):
    """双边滤波去块效应。"""
    return cv2.bilateralFilter(img, 5, 30, 30)


def deblock_nlmeans(img, params):
    """非局部均值去块效应。"""
    return cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)


# 工具名 → 经典近似实现；同类工具使用不同算法/参数以保留互补性
CLASSICAL_OPS: Dict[str, Op] = {
    "dehazeformer_dehaze": dehaze_dark_channel,
    "scunet_real_denoising_psnr": lambda img, p: denoise_nlmeans(img, p, h=12.0),
    "scunet_real_denoising_gan": lambda img, p: denoise_nlmeans(img, p, h=8.0, post_sharpen=0.3),
    "swinir_denoising": lambda img, p: denoise_nlmeans(img, p, h=10.0),
    "mprnet_denoising": denoise_bilateral,
    "restormer_motion_deblurring": lambda img, p: deblur_richardson_lucy(img, p, iterations=10, ksize=7),
    "nafnet_deblur": lambda img, p: deblur_wiener_like(img, p, radius=1.2, amount=1.0),
    "mprnet_motion_deblurring": lambda img, p: deblur_richardson_lucy(img, p, iterations=6, ksize=5),
    "xrestormer_motion_deblurring": lambda img, p: deblur_wiener_like(img, p, radius=2.0, amount=1.4),
    "restormer_defocus_deblurring": lambda img, p: deblur_richardson_lucy(img, p, iterations=8, ksize=5),
    "drbnet_defocus_deblurring": lambda img, p: deblur_wiener_like(img, p, radius=1.5, amount=1.5),
    "restormer_deraining": derain_guided,
    "mprnet_deraining": lambda img, p: derain_median(img, p, ksize=5),
    "xrestormer_deraining": lambda img, p: derain_median(img, p, ksize=3),
    "nerd_deraining": derain_guided,
    "fbcnn_jpeg_artifact_removal": deblock_bilateral,
    "swinir_jpeg_artifact_removal": deblock_nlmeans,
    "swinir_super_resolution": sr_bicubic_sharpen,
    "xrestormer_super_resolution": lambda img, p: sr_bicubic_sharpen(img, p, amount=0.8),
    "diffbir_super_resolution": sr_lanczos,
    "scunet_color_denoising": lambda img, p: denoise_nlmeans(img, p, h=9.0),
    "retinexformer_fivek": lambda img, p: enhance_retinex(img, p, sigma=40.0),
    "hat_super_resolution": lambda img, p: sr_bicubic_sharpen(img, p, amount=1.0),
    "retinexformer_sdsd_indoor": lambda img, p: enhance_retinex(img, p, sigma=30.0),
    "retinexformer_enhance": lambda img, p: enhance_retinex(img, p, sigma=50.0),
    "gamma_correction": enhance_gamma,
    "histogram_equalization": enhance_clahe,
    "constant_shift": enhance_constant_shift,
}


def run_classical(tool_name: str, img: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """按工具名运行对应的经典近似算子。"""
    if tool_name not in CLASSICAL_OPS:
        raise KeyError(f"工具 {tool_name} 没有经典近似实现")
    return CLASSICAL_OPS[tool_name](img, params)
