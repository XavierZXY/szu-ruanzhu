# -*- coding: utf-8 -*-
"""命令行端到端冒烟：合成数据 → 训练 → 评估 → 单图复原（离线）。"""
import pandas as pd
from PIL import Image

from restoramind.cli import main


def test_cli_end_to_end(tmp_path, clean_image):
    clean_dir = tmp_path / "clean"
    clean_dir.mkdir()
    for i in range(3):
        clean_image.save(clean_dir / f"c{i}.png")
    data, policy, ev = tmp_path / "t.parquet", tmp_path / "policy.json", tmp_path / "eval"

    assert main(["make-data", "--clean-dir", str(clean_dir), "--output", str(data), "--samples-per-image", "2"]) == 0
    assert len(pd.read_parquet(data)) == 6
    assert main(["train", "--data", str(data), "--output", str(policy), "--epochs", "1", "--batch-size", "3"]) == 0
    assert policy.exists()
    assert main(["evaluate", "--data", str(data), "--output", str(ev), "--num-samples", "4"]) == 0
    assert (ev / "evaluation_results.csv").exists()

    src = ev / "images" / "sample_0000" / "degraded.png"
    out = tmp_path / "restored.png"
    assert main(["restore", "--input", str(src), "--output", str(out), "--degradations", "noise,dark"]) == 0
    assert Image.open(out).mode == "RGB"


def test_cli_make_data_empty_dir_fails(tmp_path):
    (tmp_path / "empty").mkdir()
    assert main(["make-data", "--clean-dir", str(tmp_path / "empty"), "--output", str(tmp_path / "x.parquet")]) == 1


def test_cli_softmax_policy_file_and_no_hint(tmp_path, clean_image):
    clean_dir = tmp_path / "clean"
    clean_dir.mkdir()
    clean_image.save(clean_dir / "c.png")
    data, policy = tmp_path / "t.parquet", tmp_path / "policy.json"
    assert main(["make-data", "--clean-dir", str(clean_dir), "--output", str(data), "--samples-per-image", "3"]) == 0
    assert main(["train", "--data", str(data), "--output", str(policy), "--epochs", "1", "--batch-size", "3"]) == 0

    out = tmp_path / "eval_softmax"
    assert main(["evaluate", "--data", str(data), "--output", str(out), "--policy", "softmax", "--policy-file", str(policy)]) == 0
    assert (out / "evaluation_results.csv").exists()

    # 不提供退化提示时规则策略无从诊断，应直接作答（零工具调用）
    out2 = tmp_path / "eval_nohint"
    assert main(["evaluate", "--data", str(data), "--output", str(out2), "--no-hint"]) == 0
    import pandas as pd
    assert (pd.read_csv(out2 / "evaluation_results.csv")["num_tool_calls"] == 0).all()


def test_cli_softmax_requires_policy_file(tmp_path):
    import pytest
    with pytest.raises(SystemExit):
        main(["restore", "--input", str(tmp_path / "x.png"), "--output", str(tmp_path / "y.png"), "--policy", "softmax"])
