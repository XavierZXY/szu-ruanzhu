# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image

from restoramind.data.degradations import apply_degradation
from restoramind.priors import compute_image_statistics, qualitative_level


def test_statistics_fields_and_prompt(clean_image):
    st = compute_image_statistics(clean_image)
    assert st.width == clean_image.width and st.height == clean_image.height
    assert 0 <= st.dark_ratio <= 1 and 0 <= st.edge_density <= 1 and 0 <= st.high_freq_energy <= 1
    text = st.to_prompt_text()
    for key in ("brightness", "edge density", "noise estimate", "blur estimate", "haze estimate"):
        assert key in text
    q = st.to_qualitative()
    assert set(q.values()) <= {"low", "medium", "high"}


def test_statistics_respond_to_degradations(clean_array):
    base = compute_image_statistics(clean_array)
    rng = np.random.default_rng(0)
    dark = compute_image_statistics(apply_degradation(clean_array, "dark", "high", rng))
    noisy = compute_image_statistics(apply_degradation(clean_array, "noise", "high", rng))
    blurred = compute_image_statistics(apply_degradation(clean_array, "defocus blur", "high", rng))
    hazy = compute_image_statistics(apply_degradation(clean_array, "haze", "high", rng))
    assert dark.brightness_mean < base.brightness_mean
    assert noisy.noise_sigma > base.noise_sigma
    assert blurred.blur_score < base.blur_score
    assert hazy.haze_score > base.haze_score


def test_qualitative_level_boundaries():
    assert qualitative_level("brightness_mean", 10) == "low"
    assert qualitative_level("brightness_mean", 100) == "medium"
    assert qualitative_level("brightness_mean", 200) == "high"


def test_accepts_numpy_and_pil(clean_array):
    a = compute_image_statistics(clean_array)
    b = compute_image_statistics(Image.fromarray(clean_array))
    assert abs(a.brightness_mean - b.brightness_mean) < 1e-6
