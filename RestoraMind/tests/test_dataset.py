# -*- coding: utf-8 -*-
import pandas as pd

from restoramind.agent.prompts import build_system_prompt
from restoramind.data.dataset import build_parquet_dataset, load_parquet_samples


def test_build_and_load_parquet(tmp_path, clean_array):
    path = tmp_path / "train.parquet"
    n = build_parquet_dataset([clean_array, clean_array], path, build_system_prompt(),
                              "<image>\nDegradations: {degradations}", seed=0, samples_per_image=2)
    assert n == 4
    df = pd.read_parquet(path)
    assert set(["data_source", "prompt", "images", "env_name", "reward_model", "extra_info"]) <= set(df.columns)
    row = df.iloc[0]
    assert row["prompt"][0]["role"] == "system" and "<image>" in row["prompt"][1]["content"]
    assert row["extra_info"]["use_original"]
    samples = load_parquet_samples(path, num_samples=3)
    assert len(samples) == 3
    s = samples[0]
    assert s.clean is not None and s.degraded.size == s.clean.size
    assert s.degradation_types == [x.strip() for x in row["env_name"].split(",")]
