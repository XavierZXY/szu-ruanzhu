# -*- coding: utf-8 -*-
"""
从论文定性对比图中裁出「输入」与「本方法输出」两列，重绘中文标签。
不保留他人方法的输出图，避免第三方成果进入软著材料。
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PAPER = Path(__file__).resolve().parents[2] / "docs" / "papers" / "sec" / "fig" / "sub4"
OUT = Path(__file__).resolve().parent / "figures"
SRC = {"paper_qual_groupC": "fig1.png", "paper_qual_rain": "fig3.png", "paper_qual_real": "fig2.png"}
FONT = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 34)


def column_segments(gray, w, h):
    nonwhite = (gray[: int(h * 0.88)] < 245).mean(axis=0) > 0.15
    segs, start = [], None
    for x, v in enumerate(nonwhite):
        if v and start is None:
            start = x
        if not v and start is not None:
            if x - start > w * 0.05:
                segs.append((start, x))
            start = None
    if start is not None:
        segs.append((start, w))
    return segs


for name, fname in SRC.items():
    im = Image.open(PAPER / fname).convert("RGB")
    w, h = im.size
    gray = np.asarray(im.convert("L")).astype(float)
    segs = column_segments(gray, w, h)
    assert len(segs) == 9, f"{name}: 期望 9 列，检测到 {len(segs)}"
    rows = np.where(((gray[:, segs[0][0]:segs[0][1]] < 245).mean(axis=1) > 0.15)[: int(h * 0.9)])[0]
    bottom = rows.max() + 6
    pieces = [im.crop((max(0, a - 6), 0, b + 6, bottom)) for a, b in (segs[0], segs[-1])]
    gap, label_h = 40, 56
    out = Image.new("RGB", (sum(p.width for p in pieces) + gap, bottom + label_h), (255, 255, 255))
    d, x = ImageDraw.Draw(out), 0
    for p, label in zip(pieces, ("输入", "本方法")):
        out.paste(p, (x, 0))
        d.text((x + (p.width - d.textlength(label, font=FONT)) / 2, bottom + 8), label, fill=(0, 0, 0), font=FONT)
        x += p.width + gap
    out.save(OUT / f"{name}.jpg", quality=90, optimize=True)
    print(name, out.size)

# ---- 三组图并排合成一张，便于版面（左：输入，右：本方法）
panels = [Image.open(OUT / f"{n}.jpg") for n in ("paper_qual_groupC", "paper_qual_rain", "paper_qual_real")]
H = 800
panels = [p.resize((round(p.width * H / p.height), H), Image.LANCZOS) for p in panels]
gap, top = 60, 56
comp = Image.new("RGB", (sum(p.width for p in panels) + gap * 2, H + top), (255, 255, 255))
d, x = ImageDraw.Draw(comp), 0
for p, tag in zip(panels, ("(a) Group C 样例", "(b) 复合退化样例", "(c) 真实场景")):
    comp.paste(p, (x, top))
    d.text((x + (p.width - d.textlength(tag, font=FONT)) / 2, 8), tag, fill=(0, 0, 0), font=FONT)
    x += p.width + gap
comp.save(OUT / "paper_qual_all.jpg", quality=90, optimize=True)
print("paper_qual_all", comp.size)
