# -*- coding: utf-8 -*-
"""
图像质量指标与归一化。

有参考：PSNR、SSIM、LPIPS（LPIPS 依赖可选，缺失时自动退化为 PSNR+SSIM）
无参考：基于统计先验的轻量质量分，可通过 ``register_no_reference_metric`` 挂接 NIQE/CLIP-IQA 等外部指标
归一化：PSNR 非线性映射 sqrt((psnr-10)/30)，SSIM 线性 (ssim+1)/2，LPIPS 直接裁剪到 [0,1]
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from ..priors.image_statistics import compute_image_statistics

_lpips_model = None
_no_reference_metric: Optional[Callable[[np.ndarray], float]] = None


def _to_array(img) -> np.ndarray:
    if isinstance(img, Image.Image):
        return np.array(img.convert("RGB"))
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return arr[..., :3].astype(np.uint8)


def _match_size(img: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """工具（如超分）可能改变尺寸，统一缩放到参考图尺寸再计算指标。"""
    if img.shape[:2] != ref.shape[:2]:
        img = cv2.resize(img, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)
    return img


def psnr(img, ref) -> float:
    """峰值信噪比（dB），尺寸不一致时先缩放到参考图尺寸。"""
    a, b = _to_array(img), _to_array(ref)
    a = _match_size(a, b).astype(np.float64)
    mse = float(np.mean((a - b.astype(np.float64)) ** 2))
    if mse <= 1e-10:
        return 100.0
    return 10.0 * math.log10(255.0 ** 2 / mse)


def ssim(img, ref) -> float:
    """灰度 SSIM（高斯窗 11x11, sigma=1.5）。"""
    a, b = _to_array(img), _to_array(ref)
    a = _match_size(a, b)
    x = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float64)
    y = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
    sxx = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x ** 2
    syy = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y ** 2
    sxy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_x * mu_y
    num = (2 * mu_x * mu_y + c1) * (2 * sxy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2)
    return float(np.mean(num / den))


def lpips_distance(img, ref, net: str = "alex") -> Optional[float]:
    """LPIPS 感知距离；依赖缺失或权重不可用时返回 None。"""
    global _lpips_model
    try:
        import lpips  # noqa: WPS433
        import torch
        if _lpips_model is None:
            _lpips_model = lpips.LPIPS(net=net, verbose=False)
            _lpips_model.eval()
        a, b = _to_array(img), _to_array(ref)
        a = _match_size(a, b)
        to_t = lambda z: torch.from_numpy(z).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1.0  # noqa: E731
        with torch.no_grad():
            return float(_lpips_model(to_t(a), to_t(b)).item())
    except Exception:  # noqa: BLE001
        return None


def normalize_metrics(psnr_val: float, ssim_val: float, lpips_val: Optional[float],
                      psnr_range: Tuple[float, float] = (10.0, 40.0)) -> Tuple[float, float, Optional[float]]:
    """把 PSNR、SSIM、LPIPS 归一化到 [0,1]。"""
    lo, hi = psnr_range
    p = max(0.0, min(1.0, (psnr_val - lo) / (hi - lo)))
    p = math.sqrt(p)
    s = max(0.0, min(1.0, (ssim_val + 1.0) / 2.0))
    l = None if lpips_val is None else max(0.0, min(1.0, lpips_val))
    return p, s, l


def quality_score(img, ref, w_psnr: float = 0.3, w_ssim: float = 0.3, w_lpips: float = 0.4,
                  use_lpips: bool = False) -> Dict[str, float]:
    """
    复合质量分 R_quality ∈ [0,1]：
        λ_psnr·PSNR_norm + λ_ssim·SSIM_norm + λ_lpips·(1-LPIPS_norm)
    LPIPS 不可用时权重按比例重新分配给 PSNR 与 SSIM。
    """
    p_raw, s_raw = psnr(img, ref), ssim(img, ref)
    l_raw = lpips_distance(img, ref) if use_lpips else None
    p, s, l = normalize_metrics(p_raw, s_raw, l_raw)
    if l is None:
        total = w_psnr + w_ssim
        score = (w_psnr * p + w_ssim * s) / total if total > 0 else 0.0
    else:
        score = w_psnr * p + w_ssim * s + w_lpips * (1.0 - l)
    return {"psnr": p_raw, "ssim": s_raw, "lpips": l_raw if l_raw is not None else float("nan"),
            "score": float(max(0.0, min(1.0, score)))}


def register_no_reference_metric(fn: Optional[Callable[[np.ndarray], float]]) -> None:
    """挂接外部无参考指标（输入 RGB uint8 数组，输出 [0,1] 分数，越大越好）。"""
    global _no_reference_metric
    _no_reference_metric = fn


def no_reference_score(img) -> float:
    """
    无参考质量分 [0,1]。若注册了外部指标则直接使用；否则由统计先验给出轻量代理：
    清晰度（拉普拉斯方差）↑、噪声估计↓、对比度适中、块效应↓、雾浓度↓。
    """
    arr = _to_array(img)
    if _no_reference_metric is not None:
        return float(max(0.0, min(1.0, _no_reference_metric(arr))))
    st = compute_image_statistics(arr)
    sharp = min(1.0, st.blur_score / 400.0)
    noise = 1.0 - min(1.0, st.noise_sigma / 20.0)
    contrast = 1.0 - abs(st.contrast_global - 55.0) / 55.0
    block = 1.0 - min(1.0, st.blockiness / 0.08)
    haze = 1.0 - min(1.0, max(0.0, (st.haze_score - 0.2) / 0.5))
    return float(max(0.0, min(1.0, 0.3 * sharp + 0.25 * noise + 0.15 * max(0.0, contrast) + 0.15 * block + 0.15 * haze)))


def discretize(score: float, levels: int) -> float:
    """把连续分数量化为 levels 档（0 表示不量化）。"""
    if levels <= 0:
        return score
    return math.floor(score * levels) / levels
