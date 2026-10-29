# -*- coding: utf-8 -*-
import numpy as np
import pytest

from restoramind.data.degradations import DEGRADATION_LEVELS, DEGRADATION_TYPES, apply_degradation
from restoramind.data.synthesize import synthesize_sample


@pytest.mark.parametrize("dtype", DEGRADATION_TYPES)
@pytest.mark.parametrize("level", DEGRADATION_LEVELS)
def test_each_degradation_changes_image_and_keeps_shape(clean_array, dtype, level):
    out = apply_degradation(clean_array, dtype, level, np.random.default_rng(0))
    assert out.shape == clean_array.shape and out.dtype == np.uint8
    assert np.abs(out.astype(int) - clean_array.astype(int)).mean() > 0.5


def test_unknown_type_raises(clean_array):
    with pytest.raises(ValueError):
        apply_degradation(clean_array, "fog", "low")


def test_synthesize_records_order_and_levels(clean_array):
    syn = synthesize_sample(clean_array, np.random.default_rng(3), num_degradations=(2, 3))
    assert 2 <= len(syn.degradations) <= 3
    assert len(set(syn.degradation_types)) == len(syn.degradation_types)
    assert all(d.degradation_level in DEGRADATION_LEVELS for d in syn.degradations)
    assert syn.env_name == ", ".join(syn.degradation_types)


def test_synthesize_reproducible(clean_array):
    a = synthesize_sample(clean_array, np.random.default_rng(7))
    b = synthesize_sample(clean_array, np.random.default_rng(7))
    assert a.degradation_types == b.degradation_types
    assert np.array_equal(a.degraded, b.degraded)


def test_clean_prob_produces_clean_sample(clean_array):
    syn = synthesize_sample(clean_array, np.random.default_rng(0), clean_prob=1.0)
    assert syn.degradations == [] and syn.env_name == "clean"
