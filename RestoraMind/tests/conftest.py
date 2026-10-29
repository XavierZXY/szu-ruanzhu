# -*- coding: utf-8 -*-
"""测试公共夹具：合成的小尺寸干净图像与复合退化样本（全部离线、CPU）。"""
import numpy as np
import pytest
from PIL import Image

from restoramind.data.dataset import Sample
from restoramind.data.synthesize import synthesize_sample
from restoramind.tools.toolbox import build_toolbox


def make_clean_image(size: int = 96, seed: int = 0) -> np.ndarray:
    """带纹理、边缘与渐变的合成图像，便于统计先验与质量指标产生有意义的数值。"""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    r = 255 * (0.5 + 0.5 * np.sin(8 * np.pi * xx))
    g = 255 * yy
    b = 255 * (0.5 + 0.5 * np.cos(6 * np.pi * yy * xx))
    img = np.stack([r, g, b], axis=-1)
    img[size // 4: size // 2, size // 4: size // 2] = 240   # 亮色方块（强边缘）
    img += rng.normal(0, 3, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


@pytest.fixture(scope="session")
def clean_array() -> np.ndarray:
    return make_clean_image()


@pytest.fixture(scope="session")
def clean_image(clean_array) -> Image.Image:
    return Image.fromarray(clean_array)


@pytest.fixture
def degraded_sample(clean_array) -> Sample:
    syn = synthesize_sample(clean_array, np.random.default_rng(1), num_degradations=(2,))
    return Sample(degraded=Image.fromarray(syn.degraded), clean=Image.fromarray(syn.clean),
                  degradations=syn.degradations, index=0)


@pytest.fixture
def clean_sample(clean_image) -> Sample:
    return Sample(degraded=clean_image, clean=clean_image, degradations=[], index=1)


@pytest.fixture(scope="session")
def local_toolbox():
    return build_toolbox("local")
