# -*- coding: utf-8 -*-
"""
生成软件著作权登记所需的「源程序鉴别材料」PDF。

规则（《计算机软件著作权登记办法》第十条）：
  - 源程序不足 60 页提交全部；超过 60 页提交前连续 30 页与后连续 30 页；
  - 每页不少于 50 行（本脚本固定 50 行/页，空行不计入）；
  - 页眉：软件名称与版本号，右上角页码（从 1 开始）；不设页脚。
"""
import sys
import unicodedata
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from restoramind import __copyright_holder__, __software_name__, __version__  # noqa: E402

FONT_PATH = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
LINES_PER_PAGE = 50
MAX_COLS = 112            # 每行最大显示列数（中文按 2 列）
FONT_SIZE = 8.6
LEADING = 14.6
OUT = Path(__file__).resolve().parent / "source_code_V1.0.pdf"

ORDER = [
    "restoramind/__init__.py",
    "restoramind/data/__init__.py", "restoramind/data/degradations.py", "restoramind/data/synthesize.py",
    "restoramind/data/dataset.py",
    "restoramind/priors/__init__.py", "restoramind/priors/image_statistics.py",
    "restoramind/tools/__init__.py", "restoramind/tools/registry.py", "restoramind/tools/classical.py",
    "restoramind/tools/client.py", "restoramind/tools/toolbox.py",
    "restoramind/tool_server/__init__.py", "restoramind/tool_server/backends.py", "restoramind/tool_server/app.py",
    "restoramind/agent/__init__.py", "restoramind/agent/parser.py", "restoramind/agent/prompts.py",
    "restoramind/agent/trajectory.py", "restoramind/agent/env.py", "restoramind/agent/policy.py",
    "restoramind/agent/runner.py",
    "restoramind/reward/__init__.py", "restoramind/reward/format_reward.py", "restoramind/reward/quality_metrics.py",
    "restoramind/reward/degradation_reward.py", "restoramind/reward/composite.py",
    "restoramind/train/__init__.py", "restoramind/train/grpo.py", "restoramind/train/softmax_policy.py",
    "restoramind/train/verl_plugin.py",
    "restoramind/inference/__init__.py", "restoramind/inference/restore.py", "restoramind/inference/evaluate.py",
    "restoramind/cli.py", "restoramind/__main__.py",
]


def display_width(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def wrap_line(line: str, limit: int = MAX_COLS):
    """按显示宽度折行；续行缩进 4 空格。"""
    line = line.replace("\t", "    ")
    if display_width(line) <= limit:
        return [line]
    out, cur, w = [], "", 0
    for ch in line:
        cw = 2 if unicodedata.east_asian_width(ch) in "WF" else 1
        if w + cw > limit:
            out.append(cur)
            cur, w = "    ", 4
        cur += ch
        w += cw
    out.append(cur)
    return out


def collect_lines():
    lines = []
    for rel in ORDER:
        for raw in (ROOT / rel).read_text(encoding="utf-8").splitlines():
            if raw.strip():                       # 空行不计入页数
                lines.extend(wrap_line(raw.rstrip()))
    return lines


def select_pages(lines):
    pages = [lines[i:i + LINES_PER_PAGE] for i in range(0, len(lines), LINES_PER_PAGE)]
    if len(pages) <= 60:
        return pages, False
    return pages[:30] + pages[-30:], True


def build():
    pdfmetrics.registerFont(TTFont("CJK", FONT_PATH, subfontIndex=0))
    lines = collect_lines()
    pages, truncated = select_pages(lines)
    c = canvas.Canvas(str(OUT), pagesize=A4)
    c.setTitle(f"{__software_name__} V{__version__} 源程序")
    width, height = A4
    left, top = 40, height - 62
    for no, page in enumerate(pages, start=1):
        c.setFont("CJK", 9.5)
        c.drawString(left, height - 36, f"{__software_name__} V{__version__}")
        c.drawRightString(width - left, height - 36, f"第 {no} 页")
        c.setLineWidth(0.5)
        c.line(left, height - 42, width - left, height - 42)
        c.setFont("CJK", FONT_SIZE)
        y = top
        for text in page:
            c.drawString(left, y, text)
            y -= LEADING
        c.showPage()
    c.save()
    return len(lines), len(pages), truncated


if __name__ == "__main__":
    total, n_pages, truncated = build()
    print(f"物理行数={total}  输出页数={n_pages}  前30+后30截取={truncated}  -> {OUT}")
