# -*- coding: utf-8 -*-
"""
训练/评估数据集构建与读取（Parquet 格式）。

字段与 VeRL 智能体训练框架兼容：
    data_source   数据来源标识
    prompt        对话提示（system + user），user 内含 <image> 占位符
    images        [{"bytes": PNG 字节}] 退化图像
    env_name      逗号分隔的真实退化类型（施加顺序），clean 表示无退化
    reward_model  [{"degradation_type", "degradation_level"}, ...]
    extra_info    {"original_image": {"bytes": PNG}, "use_original": True, "index": i}
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from PIL import Image

from .synthesize import DegradationSpec, SynthesizedSample, synthesize_sample

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


@dataclass
class Sample:
    """环境/评估使用的统一样本结构。"""
    degraded: Image.Image
    clean: Optional[Image.Image]
    degradations: List[DegradationSpec] = field(default_factory=list)
    index: int = 0

    @property
    def degradation_types(self) -> List[str]:
        """真实退化类型列表（施加顺序）。"""
        return [d.degradation_type for d in self.degradations]

    @property
    def is_clean(self) -> bool:
        """是否为无退化的干净样本。"""
        return len(self.degradations) == 0


def encode_png(img: np.ndarray | Image.Image) -> bytes:
    """把图像编码为 PNG 字节。"""
    pil = img if isinstance(img, Image.Image) else Image.fromarray(img)
    buf = io.BytesIO()
    pil.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def decode_image(data) -> Image.Image:
    """兼容 {'bytes': ...} / bytes / 路径 / PIL 四种形式。"""
    if isinstance(data, Image.Image):
        return data.convert("RGB")
    if isinstance(data, dict):
        data = data.get("bytes")
    if isinstance(data, (bytes, bytearray)):
        return Image.open(io.BytesIO(data)).convert("RGB")
    if isinstance(data, str):
        return Image.open(data).convert("RGB")
    raise TypeError(f"无法解析的图像数据类型: {type(data)}")


def list_images(root: str | Path) -> List[Path]:
    """递归列出目录下的图像文件。"""
    root = Path(root)
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def load_clean_image(path: str | Path, max_size: int = 512) -> np.ndarray:
    """读取干净图像并限制最长边，避免训练时显存/内存压力。"""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = max(w, h) / float(max_size)
    if scale > 1.0:
        img = img.resize((max(8, int(w / scale)), max(8, int(h / scale))), Image.BICUBIC)
    return np.array(img)


def build_record(sample: SynthesizedSample, index: int, system_prompt: str, user_prompt: str,
                 data_source: str = "restoramind") -> dict:
    """把合成样本转换为一条 Parquet 记录。"""
    return {
        "data_source": data_source,
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "images": [{"bytes": encode_png(sample.degraded)}],
        "env_name": sample.env_name,
        "reward_model": [d.to_dict() for d in sample.degradations],
        "extra_info": {
            "original_image": {"bytes": encode_png(sample.clean)},
            "use_original": True,
            "index": index,
        },
    }


def build_parquet_dataset(
    clean_images: Iterable[str | Path | np.ndarray],
    output_path: str | Path,
    system_prompt: str,
    user_prompt_template: str,
    seed: int = 0,
    num_degradations: Sequence[int] = (2, 3),
    clean_prob: float = 0.0,
    max_size: int = 512,
    samples_per_image: int = 1,
) -> int:
    """
    从干净图像集合合成复合退化数据集并写入 Parquet。

    user_prompt_template 中可使用 {degradations} 占位符注入真实退化类型提示
    （训练时提供退化集合提示，智能体学习顺序与工具选择；评估时可关闭提示）。
    返回写入的样本数。
    """
    rng = np.random.default_rng(seed)
    records: List[dict] = []
    idx = 0
    for item in clean_images:
        clean = item if isinstance(item, np.ndarray) else load_clean_image(item, max_size=max_size)
        for _ in range(samples_per_image):
            syn = synthesize_sample(clean, rng, num_degradations=num_degradations, clean_prob=clean_prob)
            hint = ", ".join(syn.degradation_types) if syn.degradations else "none"
            user_prompt = user_prompt_template.format(degradations=hint)
            records.append(build_record(syn, idx, system_prompt, user_prompt))
            idx += 1
    df = pd.DataFrame(records)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return len(records)


def _as_list(value) -> list:
    """Parquet 读回的列表字段可能是 numpy 数组或 None，统一转为 list。"""
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    return list(value)


def record_to_sample(row: dict, index: int) -> Sample:
    """把 Parquet 记录解析为 Sample。"""
    degraded = decode_image(_as_list(row["images"])[0])
    extra = row.get("extra_info")
    extra = extra if isinstance(extra, dict) else {}
    clean = decode_image(extra["original_image"]) if extra.get("original_image") is not None else None
    specs = [DegradationSpec(d["degradation_type"], d.get("degradation_level", "medium"))
             for d in _as_list(row.get("reward_model")) if isinstance(d, dict)]
    return Sample(degraded=degraded, clean=clean, degradations=specs, index=index)


def load_parquet_samples(path: str | Path, num_samples: Optional[int] = None) -> List[Sample]:
    """读取 Parquet 数据集为 Sample 列表。"""
    df = pd.read_parquet(path)
    if num_samples is not None:
        df = df.head(num_samples)
    return [record_to_sample(row, i) for i, row in enumerate(df.to_dict("records"))]
