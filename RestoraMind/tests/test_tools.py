# -*- coding: utf-8 -*-
import pytest
from PIL import Image

from restoramind.tools.classical import CLASSICAL_OPS
from restoramind.tools.registry import TOOL_SPECS, all_degradations, get_tool_spec, tools_for_degradation
from restoramind.tools.toolbox import ToolExecutionError, build_toolbox


def test_every_registered_tool_has_classical_fallback():
    assert set(TOOL_SPECS) == set(CLASSICAL_OPS)


def test_registry_covers_eight_degradations_with_alternatives():
    degs = all_degradations()
    assert len(degs) == 8
    for d in ("noise", "motion blur", "defocus blur", "rain", "jpeg compression artifact", "low resolution", "dark"):
        assert len(tools_for_degradation(d)) >= 2, d


def test_normalize_arguments_clamps_and_defaults():
    spec = get_tool_spec("dehazeformer_dehaze")
    assert spec.normalize_arguments({"strength": 5}) == {"strength": 1.0}
    assert spec.normalize_arguments({}) == {"strength": 0.5}
    sr = get_tool_spec("swinir_super_resolution")
    assert sr.normalize_arguments({"scale": 3.4}) == {"scale": 3}
    assert sr.normalize_arguments({"scale": "bad"}) == {"scale": 2}


@pytest.mark.parametrize("name", sorted(TOOL_SPECS))
def test_local_toolbox_executes_every_tool(local_toolbox, clean_image, name):
    result = local_toolbox.execute(name, clean_image, {})
    assert isinstance(result.image, Image.Image) and result.image.mode == "RGB"
    assert result.tool_name == name and result.elapsed >= 0


def test_unknown_tool_raises(local_toolbox, clean_image):
    with pytest.raises(ToolExecutionError):
        local_toolbox.execute("nonexistent_tool", clean_image, {})


def test_super_resolution_scales(local_toolbox, clean_image):
    out = local_toolbox.execute("swinir_super_resolution", clean_image, {"scale": 2}).image
    assert out.size == (clean_image.width * 2, clean_image.height * 2)


def test_prompt_line_lists_parameters():
    line = get_tool_spec("fbcnn_jpeg_artifact_removal").prompt_line()
    assert "fbcnn_jpeg_artifact_removal" in line and "jpeg" in line and "1..100" in line
