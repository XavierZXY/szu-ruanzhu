#!/bin/bash
# 离线端到端演示：合成数据 -> 训练 -> 评估 -> 单图复原（无需 GPU 与远程服务）。
set -euo pipefail
WORK="${WORK:-./demo_output}"
mkdir -p "$WORK/clean"
python3 - <<PY
import numpy as np
from PIL import Image
rng = np.random.default_rng(0)
for i in range(6):
    yy, xx = np.mgrid[0:128, 0:128] / 128.0
    img = np.stack([255*(0.5+0.5*np.sin((6+i)*np.pi*xx)), 255*yy, 255*(0.5+0.5*np.cos(5*np.pi*yy*xx))], -1)
    img[32:64, 32:64] = 235
    Image.fromarray(np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype("uint8")).save("$WORK/clean/img_%d.png" % i)
PY
python3 -m restoramind make-data --clean-dir "$WORK/clean" --output "$WORK/train.parquet" --samples-per-image 2
python3 -m restoramind train --data "$WORK/train.parquet" --output "$WORK/policy.json" --epochs 2
python3 -m restoramind evaluate --data "$WORK/train.parquet" --output "$WORK/eval" --num-samples 6
python3 -m restoramind restore --input "$WORK/eval/images/sample_0000/degraded.png" --output "$WORK/restored.png" \
    --degradations "noise,dark"
