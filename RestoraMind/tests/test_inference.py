# -*- coding: utf-8 -*-
from restoramind.agent.env import EnvConfig
from restoramind.agent.policy import HeuristicPolicy
from restoramind.inference.evaluate import evaluate_samples
from restoramind.inference.restore import restore_image


def test_restore_image_with_hint(local_toolbox, degraded_sample):
    res = restore_image(degraded_sample.degraded, HeuristicPolicy(), local_toolbox, EnvConfig(max_turns=5),
                        degradation_hint=degraded_sample.degradation_types)
    assert res.finished and res.image.mode == "RGB"
    assert len(res.tool_sequence) == len(degraded_sample.degradation_types)


def test_restore_image_without_hint_falls_back_to_answer(local_toolbox, degraded_sample):
    res = restore_image(degraded_sample.degraded, HeuristicPolicy(), local_toolbox, EnvConfig(max_turns=3))
    assert res.finished and res.num_turns == 1


def test_evaluate_samples_writes_outputs(tmp_path, local_toolbox, degraded_sample, clean_sample):
    summary = evaluate_samples([degraded_sample, clean_sample], HeuristicPolicy(), local_toolbox,
                               EnvConfig(max_turns=5), output_dir=tmp_path)
    assert summary.num_samples == 2
    assert (tmp_path / "evaluation_results.csv").exists() and (tmp_path / "summary.txt").exists()
    assert (tmp_path / "images" / "sample_0000" / "restored.png").exists()
    assert summary.mean_psnr_restored == summary.mean_psnr_restored  # not NaN
    assert "PSNR" in summary.to_text()
