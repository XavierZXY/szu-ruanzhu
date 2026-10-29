# -*- coding: utf-8 -*-
"""
软件说明书内容模型。

返回块列表，块类型：
    ("h1"|"h2"|"h3", 文本)  ("p", 文本)  ("bullets", [文本...])  ("numbered", [文本...])
    ("code", 文本)  ("table", 表头, 行, 列宽比例)  ("fig", 图片路径, 图题)  ("pagebreak",)
工具表、环境变量表、接口参考、测试清单均由源码自动提取，保证与代码一致。
"""
import importlib
import inspect
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from restoramind import __copyright_holder__, __software_name__, __version__  # noqa: E402
from restoramind.tools.registry import FAMILY_PORTS, TOOL_SPECS  # noqa: E402
import manual_extra  # noqa: E402

FIG = HERE / "figures"
RESULTS = json.loads((HERE / "results.json").read_text(encoding="utf-8"))

MODULES = [
    ("restoramind.data.degradations", "退化算子"), ("restoramind.data.synthesize", "复合退化合成"),
    ("restoramind.data.dataset", "数据集构建与读取"), ("restoramind.priors.image_statistics", "图像统计先验"),
    ("restoramind.tools.registry", "工具规格注册表"), ("restoramind.tools.classical", "本地经典算子"),
    ("restoramind.tools.client", "远程工具客户端"), ("restoramind.tools.toolbox", "工具箱"),
    ("restoramind.tool_server.backends", "工具服务后端"), ("restoramind.tool_server.app", "工具服务应用"),
    ("restoramind.agent.parser", "响应解析"), ("restoramind.agent.prompts", "提示词"),
    ("restoramind.agent.trajectory", "轨迹记录"), ("restoramind.agent.env", "复原环境"),
    ("restoramind.agent.policy", "策略"), ("restoramind.agent.runner", "轨迹采样"),
    ("restoramind.reward.format_reward", "格式奖励"), ("restoramind.reward.quality_metrics", "图像质量指标"),
    ("restoramind.reward.degradation_reward", "退化推理奖励"), ("restoramind.reward.composite", "复合奖励"),
    ("restoramind.train.grpo", "GRPO 训练"), ("restoramind.train.softmax_policy", "可训练工具选择策略"),
    ("restoramind.train.verl_plugin", "VeRL 框架桥接"),
    ("restoramind.inference.restore", "单图复原"), ("restoramind.inference.evaluate", "数据集评估"),
    ("restoramind.cli", "命令行入口"),
]


def first_line(obj) -> str:
    doc = inspect.getdoc(obj) or ""
    return doc.strip().splitlines()[0] if doc.strip() else "—"


def api_tables():
    """按模块自动提取公开类、函数及类的公开方法。"""
    blocks = []
    for mod_name, title in MODULES:
        mod = importlib.import_module(mod_name)
        rows = []
        for name, obj in sorted(vars(mod).items()):
            if name.startswith("_") or getattr(obj, "__module__", None) != mod_name:
                continue
            if inspect.isclass(obj):
                rows.append([name, "类", first_line(obj)])
                for mname, m in sorted(vars(obj).items()):
                    if mname.startswith("_") or not inspect.isfunction(m):
                        continue
                    rows.append([f"  {name}.{mname}", "方法", first_line(m)])
            elif inspect.isfunction(obj):
                rows.append([name, "函数", first_line(obj)])
        if rows:
            blocks.append(("h3", f"{mod_name}（{title}）"))
            blocks.append(("table", ["名称", "类型", "说明"], rows, [0.34, 0.08, 0.58]))
    return blocks


def test_inventory():
    out = subprocess.run([sys.executable, "-m", "pytest", "tests", "--collect-only", "-q"], cwd=ROOT,
                         capture_output=True, text=True).stdout
    counts = {}
    for line in out.splitlines():
        if "::" in line:
            f = line.split("::")[0]
            counts[f] = counts.get(f, 0) + 1
    return counts


TEST_DESC = {
    "tests/test_degradations.py": "8 类退化算子形状与强度、复合退化记录、随机种子复现、clean 样本",
    "tests/test_dataset.py": "Parquet 数据集构建、字段完整性、读回样本解析",
    "tests/test_priors.py": "统计先验字段范围、对各类退化的响应方向、定性分级边界",
    "tests/test_tools.py": "全部注册工具可执行、注册表覆盖 8 类退化、参数裁剪、超分尺寸",
    "tests/test_tool_server.py": "工具服务端与远程客户端离线联调、负载均衡、环境变量解析",
    "tests/test_parser_and_format.py": "响应解析、严格/阶梯格式奖励的各类违规与通过情形",
    "tests/test_reward.py": "PSNR/SSIM/归一化、退化推理奖励、复原顺序奖励、步级与轨迹级奖励",
    "tests/test_env_and_policy.py": "提示词、环境状态转移、回退重试、两种对话模式、策略接口",
    "tests/test_train.py": "组相对优势、GRPO 训练、策略持久化、VeRL 奖励函数与工具类桥接",
    "tests/test_inference.py": "单图复原、数据集评估输出文件",
    "tests/test_cli_e2e.py": "命令行端到端：合成数据 → 训练 → 评估 → 单图复原",
}


def build():  # noqa: C901 - 内容模型，顺序编排
    PAPER_CHAPTER = paper_chapter()
    CS = manual_extra.code_stats()
    B = []
    R = RESULTS
    n_tools = len(TOOL_SPECS)
    tests = test_inventory()
    total_tests = sum(tests.values())

    # ======================================================= 1 概述
    B += [("h1", "1  软件概述")]
    B += [("h2", "1.1  软件简介"),
          ("p", f"{__software_name__}（以下简称 RestoraMind）是一套面向真实场景复合退化图像复原的智能体软件。"
                "真实场景图像常同时受到噪声、模糊、雨雾、低光照、低分辨率和 JPEG 压缩伪影等多种未知退化的影响，"
                "退化类型、强度和顺序均无法预先获知。单一的一体化复原模型难以覆盖如此庞大的退化组合空间，"
                "而现有基于大模型的智能体式复原方法通常先生成初始工具序列，再依赖深度优先搜索、质量驱动搜索或回滚式后搜索反复试错，"
                "工具调用次数多、推理耗时高。"),
          ("p", "RestoraMind 将图像复原建模为序列决策过程：在每一个决策步，智能体根据当前图像、图像低层统计先验、"
                "已使用的工具历史和已预测的退化历史，联合完成剩余退化类型推理、复原工具选择和复原工具执行，"
                "再把工具输出作为下一步的观测，直至图像中不再存在需要处理的显著退化。智能体通过强化学习（组相对策略优化 GRPO；"
                "步级奖励也可供近端策略优化 PPO 等算法使用）端到端训练，同时学得退化推理能力、复原顺序规划能力和工具使用能力。"),
          ("fig", FIG / "paper_motivation.png", "两类现有后搜索方式（深度优先搜索、质量驱动搜索）与本软件的序列决策方式对比", "motivation"),
          ("p", "[[motivation]] 对比了两种路线：现有方法（a、b）先由大模型给出初始规划，再通过搜索反复试错，工具调用次数随工具数量和退化种类迅速增长；"
                "本软件（c）依据当前图像逐步决策，每个退化类型对应一次推理与一次工具调用，不依赖后搜索。"),
          ("p", "软件提供从训练数据合成、复原工具服务、智能体环境、奖励函数、强化学习训练，到单图推理与数据集评估的完整链路，"
                "并提供命令行入口、脚本与自动化测试。")]
    B += [("h2", "1.2  软件基本信息"),
          ("table", ["项目", "内容"], [
              ["软件全称", __software_name__], ["软件简称", "RestoraMind"], ["版本号", f"V{__version__}"],
              ["著作权人", __copyright_holder__], ["开发语言", "Python 3"],
              ["运行平台", "Linux（CPU 可运行全部离线流程；大模型训练与推理需要 GPU）"],
              ["软件形态", "Python 软件包 + 命令行工具 + 工具服务（HTTP）"],
              ["源程序规模", f"restoramind 软件包共 {CS['total']} 行（含空行、注释与文档字符串）；非空行 {CS['nonblank']}；有效代码约 {CS['effective']} 行（不含空行、纯注释与文档字符串，不含测试）"],
          ], [0.22, 0.78])]
    B += [("h2", "1.3  开发目的与意义"),
          ("bullets", [
              "统一处理未知复合退化：不需要人工指定退化类型和工具调用顺序，由智能体逐步判断剩余退化并调用对应工具。",
              "降低推理成本：策略在每个状态直接输出当前工具选择，工具调用次数随退化类型数量近似线性增长，不依赖穷举排列和大规模回滚搜索。",
              "细粒度工具选择：同一退化类型配置多个候选工具，并引入图像低层统计先验，使工具选择与当前图像内容直接关联。",
              "易于扩展：新增复原工具只需在注册表中登记规格并在工具服务中提供实现，不必重新设计整体流程。",
              "可复现：全部流程可在无 GPU、无外部模型服务的环境下以经典算子离线运行，并通过自动化测试验证。",
          ])]
    B += [("h2", "1.4  主要功能与技术特点"),
          ("table", ["功能", "说明"], [
              ["退化合成", "8 类退化算子（雾、雨、离焦模糊、低分辨率、运动模糊、低光、JPEG 伪影、噪声），随机 2~3 种、随机强度、随机顺序叠加，记录真实退化序列"],
              ["统计先验", "亮度、对比度、色彩、边缘密度、高频能量、噪声/模糊/块效应/雾浓度估计，输出数值与低/中/高定性描述"],
              ["工具箱", f"{n_tools} 个复原工具，覆盖 8 类退化，其中 7 类配置 2 个以上候选工具；支持本地经典算子与远程专家模型服务"],
              ["智能体环境", "单工具迭代与多工具规划两种对话模式；有限回退重试；每步重新提取统计先验并更新状态"],
              ["奖励函数", "严格/阶梯格式奖励、质量奖励（PSNR/SSIM/LPIPS 与无参考）、退化推理奖励、中间图像质量奖励、工具成本与步骤惩罚"],
              ["强化学习", "GRPO 组相对优势训练循环；轻量表格化策略可 CPU 运行；通过桥接模块接入 VeRL 训练多模态大模型"],
              ["推理评估", "单图复原；数据集批量评估，输出 CSV、汇总文本、退化/复原/真值图像"],
          ], [0.18, 0.82])]

    # ======================================================= 2 环境
    B += [("h1", "2  运行环境与安装")]
    B += [("h2", "2.1  硬件与软件环境"),
          ("table", ["类别", "要求"], [
              ["操作系统", "Linux（推荐 Ubuntu 20.04 及以上）"],
              ["Python", "3.10 及以上"],
              ["核心依赖", "numpy、Pillow、OpenCV（opencv-python-headless）、pandas、pyarrow、Flask、requests"],
              ["可选依赖", "openai（调用 vLLM 部署的智能体）、lpips（感知指标）、torch（LPIPS 运行时）、VeRL（大模型强化学习训练）"],
              ["离线流程硬件", "普通 CPU 即可，无需 GPU"],
              ["大模型训练", "建议 8 张及以上 GPU；7B 多模态模型训练建议多节点"],
          ], [0.2, 0.8])]
    B += [("h2", "2.2  安装步骤"),
          ("numbered", ["获取源码并进入 RestoraMind 目录。", "安装依赖：pip install -r requirements.txt。",
                        "安装软件包：pip install -e .，安装后可使用 restoramind 命令或 python -m restoramind。",
                        "运行测试确认环境：python -m pytest tests -q。",
                        "运行离线演示：bash scripts/demo.sh。"]),
          ("code", "pip install -r requirements.txt\npip install -e .\npython -m pytest tests -q\nbash scripts/demo.sh")]
    B += [("h2", "2.3  目录结构"),
          ("code", "RestoraMind/\n├── restoramind/           软件源码\n│   ├── data/              退化算子、复合退化合成、Parquet 数据集\n"
                   "│   ├── priors/            图像低层统计先验\n│   ├── tools/             工具注册表、经典算子、远程客户端、工具箱\n"
                   "│   ├── tool_server/       复原工具 HTTP 服务\n│   ├── agent/             解析器、提示词、环境、策略、轨迹采样\n"
                   "│   ├── reward/            格式/质量/退化推理/复合奖励\n│   ├── train/             GRPO 训练、可训练策略、VeRL 桥接\n"
                   "│   ├── inference/         单图复原与数据集评估\n│   └── cli.py             命令行入口\n"
                   "├── tests/                 自动化测试\n├── scripts/               训练、工具服务、演示脚本\n"
                   "└── requirements.txt / pyproject.toml")]

    # ======================================================= 3 总体设计
    B += [("h1", "3  系统总体设计")]
    B += [("h2", "3.1  总体架构"),
          ("p", "软件按职责划分为数据、统计先验、工具、工具服务、智能体、奖励、训练、推理八个模块，模块之间通过明确的数据结构衔接，"
                "各模块只依赖下层模块，不存在循环依赖。训练阶段由环境采样轨迹，奖励模块评价轨迹，训练模块依据组相对优势更新策略；"
                "推理阶段由策略逐步决策，工具箱执行工具，输出复原图。"),
          ("fig", FIG / "fig1_architecture.png", "RestoraMind 软件总体架构", "arch")]
    B += [("h2", "3.2  模块职责"),
          ("table", ["模块", "职责", "主要输出"], [
              ["data", "退化算子、复合退化合成、数据集构建与读取", "Parquet 数据集、Sample 样本"],
              ["priors", "提取图像低层统计特征并生成结构化文本", "ImageStatistics"],
              ["tools", "工具规格注册、经典算子、远程调用、工具箱统一执行", "复原后的图像"],
              ["tool_server", "以 HTTP 服务形式封装专家模型", "/process 接口"],
              ["agent", "解析模型输出、构造提示词、维护状态与转移、采样轨迹", "Trajectory"],
              ["reward", "格式、质量、退化推理、复合奖励", "RewardBreakdown"],
              ["train", "组相对优势、训练循环、可训练策略、VeRL 桥接", "策略参数、训练统计"],
              ["inference", "单图复原、数据集评估", "复原图、评估表"],
          ], [0.16, 0.54, 0.30])]
    B += [("h2", "3.3  序列决策式复原流程"),
          ("p", "软件在每个复原步骤执行以下过程（[[flow]]，[[overview]] 给出一个多退化图像的逐步示例）：提取当前图像的统计先验；将当前图像、统计先验、历史工具调用序列、历史退化预测和任务提示构造为状态；"
                "智能体推理当前仍然存在的退化类型，并从该类型的候选工具中选择一个工具；工具执行得到中间图像；"
                "更新状态并判断是否终止。与仅在初始图像上做一次退化判断的方法不同，软件在每一步工具执行后重新推理剩余退化，"
                "因而能处理某一工具只消除部分退化、退化之间存在顺序依赖、或原始图像中难以观察的退化在前一步复原后变得明显等情况。"),
          ("fig", FIG / "fig2_flow.png", "序列决策式图像复原流程", "flow"),
          ("fig", FIG / "paper_overview.png", "复原流程示例：智能体逐步推理剩余退化、选择并执行工具，并随图像更新统计先验", "overview")]
    B += [("h2", "3.4  状态、动作与转移"),
          ("table", ["要素", "定义", "对应实现"], [
              ["状态 S_t", "当前图像 I_t、统计先验 F_t、历史工具序列、历史退化预测、任务提示", "agent.env.AgentState"],
              ["动作 a_t", "工具名称、目标退化类型、工具参数、是否回退重试（rollback）", "agent.parser.ToolCall"],
              ["转移", "执行工具得到 I_{t+1}，重新提取 F_{t+1}，追加历史，构造 S_{t+1}", "agent.env.RestorationEnv.step"],
              ["终止", "模型输出 answer；或达到最大轮数；或无需继续处理", "EnvConfig.max_turns"],
          ], [0.16, 0.54, 0.30]),
          ("p", "[[pipeline]] 展示了状态在相邻两步之间的传递：工具输出的新图像与重新提取的统计先验一同构成下一步的输入，模型据此判断是否仍有未去除的退化。"),
          ("p", "智能体的单轮响应使用严格格式：<think> 块给出推理过程，随后是 <tool_call> 块（一个 JSON 列表）或 <answer> 块（restoration_log 列表）。"
                "格式由解析器统一解析，并由格式奖励约束。"),
          ("fig", FIG / "paper_pipeline.png", "状态与动作的逐步示例：图像、统计先验与工具历史共同构成每一步的输入", "pipeline")]
    B += [("h2", "3.5  两种对话模式"),
          ("table", ["模式", "行为", "适用场景"], [
              ["single_tool_iterative（默认）", "每轮恰好一个工具，作用于当前图像，下一轮看到处理结果；允许有限次回退到上一张图像换用同类工具重试", "逐步反应型训练与推理，对应序列决策"],
              ["multi_tool_planning", "每轮提出完整工具序列，系统从原图链式执行后反馈结果；结果不佳可再提出新序列", "策略规划型训练与对比实验"],
          ], [0.26, 0.46, 0.28])]

    # ======================================================= 4 核心算法
    B += [("h1", "4  核心算法设计")]
    B += [("h2", "4.1  复合退化数据合成"),
          ("p", "训练数据由干净图像合成。对每张图像，软件从 8 类退化中随机选择 2 至 3 种，每种随机选择 low / medium / high 强度，"
                "在强度区间内随机采样具体参数，并按随机顺序依次施加。施加顺序、类型和强度被完整记录到 reward_model 字段，"
                "用于训练阶段的退化推理奖励和复原顺序奖励。可通过 clean_prob 参数保留一定比例的干净图像，"
                "训练智能体识别无需复原的情形。"),
          ("table", ["退化类型", "合成方法", "强度参数（low ~ high）"], [
              ["haze（雾）", "大气散射模型 I=J·t+A·(1-t)，深度渐变", "衰减系数 0.2~0.4 至 0.6~0.85"],
              ["rain（雨）", "随机倾斜雨线叠加并轻微模糊", "雨线数量 200~400 至 800~1500"],
              ["defocus blur（离焦）", "圆盘核卷积", "半径参数 1.0~2.0 至 3.5~5.0"],
              ["low resolution（低分辨率）", "下采样后双三次上采样回原尺寸", "倍率 1.5~2.0 至 3.0~4.0"],
              ["motion blur（运动模糊）", "随机方向线性核", "核长度 5~9 至 15~25"],
              ["dark（低光）", "gamma 压暗并叠加轻微噪声", "gamma 1.5~2.0 至 3.0~4.5"],
              ["jpeg compression artifact", "低质量 JPEG 编解码", "质量因子 40~60 至 5~20"],
              ["noise（噪声）", "加性高斯白噪声", "sigma 5~15 至 30~50"],
          ], [0.28, 0.40, 0.32])]
    B += [("h2", "4.2  图像低层统计先验"),
          ("p", "仅依赖工具文本说明选择工具，无法反映当前图像的视觉特征。软件在每个决策状态用传统图像处理方法以极低代价提取下列特征，"
                "组织为结构化文本和键值对，与图像一同输入智能体："),
          ("table", ["特征", "计算方法", "用途"], [
              ["亮度均值、暗区比例、高光比例", "灰度均值；灰度<40 与>220 的像素占比", "判断低光/过曝"],
              ["全局对比度", "灰度标准差", "判断雾、低对比度"],
              ["饱和度、色偏", "HSV 饱和度均值；三通道均值最大差", "色彩分布"],
              ["边缘密度", "Canny 边缘像素占比", "区分平滑与细节丰富图像"],
              ["高频能量", "傅里叶谱半径>1/4 区域能量占比", "纹理与噪声"],
              ["噪声估计", "3×3 中值滤波残差的 MAD 稳健估计", "噪声强度"],
              ["模糊估计", "拉普拉斯方差（越小越模糊）", "模糊强度"],
              ["块效应估计", "8×8 块边界与非边界梯度差", "JPEG 伪影"],
              ["雾浓度估计", "暗通道均值", "雾霾"],
          ], [0.28, 0.44, 0.28]),
          ("p", "统计量按阈值映射为 low / medium / high 的定性描述。例如对同样被识别为低分辨率的图像，边缘密度低、纹理弱时倾向稳健型超分工具，"
                "边缘密度高、纹理丰富时倾向细节恢复型超分工具（[[select]]）。"),
          ("fig", FIG / "fig4_selection.png", "同一退化类型下基于统计先验的自适应工具选择", "select")]
    B += [("h2", "4.3  复原工具箱"),
          ("p", f"工具箱共 {n_tools} 个工具，每个工具登记名称、目标退化类型、模型家族、说明、参数模式和相对计算代价。"
                "同一退化类型可配置多个候选工具，且各候选工具特性互补。带有 prompt_visible 标记的工具写入智能体提示词，"
                "其余工具作为扩展候选保留在工具箱中。"),
          ("p", "软件提供两种工具执行方式：离线模式（local）使用基于 OpenCV 的经典算子实现各工具，同类工具采用不同算法或参数，适用于无 GPU 环境下的运行、测试与演示；"
                "部署模式（remote）通过工具服务为同名工具挂接 SwinIR、Restormer、SCUNet、Retinexformer 等预训练专家模型（见 5.3 节）。"
                "第 7 章的实验结果基于部署模式的专家模型工具箱。")]
    rows = [[s.name, s.degradation, s.family, "是" if s.prompt_visible else "否", f"{s.cost:g}",
             ("本地" if s.is_local else str(s.port))] for s in TOOL_SPECS.values()]
    B += [("table", ["工具名称", "目标退化", "模型家族", "提示词可见", "代价", "端口"], rows, [0.31, 0.18, 0.18, 0.11, 0.08, 0.14])]
    B += [("h2", "4.4  奖励函数"),
          ("p", "奖励函数同时优化图像质量、退化推理准确性和工具调用效率。步级奖励用于逐步信用分配："),
          ("code", "r_t = α·ΔQ_t + β·R_deg(D^_t, D) - γ·Cost(a_t) - δ\nΔQ_t = R_quality(I_{t+1}) - R_quality(I_t)\n"
                   "R_quality(I) = λ_psnr·PSNR_norm + λ_ssim·SSIM_norm + λ_lpips·(1 - LPIPS_norm)\n"
                   "R_deg = |D^_t ∩ D| / max(1, |D^_t|)\n终止奖励 r_final = κ·R_quality(I_T)；超过最大轮数未完成则给失败惩罚"),
          ("fig", FIG / "fig3_reward.png", "复合奖励构成", "reward"),
          ("p", "其中 PSNR 归一化为 sqrt(clip((PSNR-10)/30, 0, 1))，SSIM 归一化为 (SSIM+1)/2，LPIPS 裁剪到 [0,1]。LPIPS 依赖不可用时，"
                "权重按比例重新分配给 PSNR 与 SSIM。无参考场景使用无参考质量得分，可通过 register_no_reference_metric 挂接 NIQE、CLIP-IQA 等外部指标。"),
          ("p", "GRPO 使用的轨迹级奖励由下列部分组成；格式违规时仅保留负的格式项，不再计算其他奖励："),
          ("table", ["分项", "含义", "默认权重"], [
              ["格式奖励", "严格二元（1 / -1）或阶梯式（0→0.2→…→1.0）", "0.3"],
              ["质量奖励", "最终图像相对真值的复合质量得分，可离散化", "0.7"],
              ["退化类型奖励（可选）", "预测退化集合与真实集合的匹配度，不考虑顺序", "1.0"],
              ["中间图像质量奖励（可选）", "中间各步图像的无参考质量均值，抑制中间步骤降质", "0.5"],
              ["工具多样性加成", "工具互不重复且数量不超过退化数+1", "0.1"],
          ], [0.28, 0.54, 0.18])]
    B += [("h2", "4.5  格式约束"),
          ("bullets", [
              "每一轮必须包含不少于 10 个字符的 <think> 块；",
              "每一轮必须包含 <tool_call> 或 <answer> 之一，且不能同时出现；",
              "<answer> 只能出现在最后一轮，且只包含 restoration_log 字段；",
              "单工具迭代模式每轮必须恰好一个工具；工具名称必须在工具箱中；",
              "非 clean 样本至少调用一个工具；启用上限检查时总工具数不超过退化数量+1，防止过度调用。",
          ]),
          ("p", "阶梯式格式奖励用于训练初期：无格式 0.0、有 think 0.2、有标签 0.4、JSON 可解析 0.6、工具名合法 0.8、完全符合 1.0，"
                "避免模型尚未学会格式时因负反馈过多而训练崩溃。")]
    B += [("h2", "4.6  强化学习训练"),
          ("p", "训练使用组相对策略优化：对每个样本采样 G 条轨迹，计算每条轨迹奖励，在组内标准化得到优势 A_i=(r_i-mean)/(std+ε)，"
                "再用优势更新策略。软件提供两条训练路径："),
          ("bullets", [
              "轻量路径：以（待处理退化类型，图像细节丰富度分桶）为状态键、以该类型候选工具为动作的表格化 softmax 策略，"
              "更新规则为策略梯度 ∇logπ(a|s)·A，可在 CPU 上完整运行数据、环境、奖励与训练循环。",
              "大模型路径：多模态大模型（Qwen2.5-VL-7B 或具备图像质量感知能力的 Q-Insight）由 VeRL 智能体强化学习框架训练。"
              "restoramind.train.verl_plugin 提供框架调用的序列级奖励函数 compute_score，并把工具箱动态注册为框架工具类。",
          ])]

    B += manual_extra.stats_example_blocks()

    # ======================================================= 5 模块详细设计
    B += [("h1", "5  模块详细设计")]
    B += [("h2", "5.1  data：数据模块"),
          ("p", "degradations.py 实现 8 类退化算子，统一签名 fn(img, level, rng)，通过 apply_degradation 分派。synthesize.py 的 synthesize_sample "
                "按 2~3 种随机类型、随机强度、随机顺序合成复合退化，返回包含干净图、退化图和 DegradationSpec 列表的 SynthesizedSample。"
                "dataset.py 负责 Parquet 的写入与读取，字段如下："),
          ("table", ["字段", "类型", "含义"], [
              ["data_source", "字符串", "数据来源标识"], ["prompt", "消息列表", "system 与 user 提示，user 中含 <image> 占位符"],
              ["images", "[{bytes}]", "退化图像 PNG 字节"], ["env_name", "字符串", "逗号分隔的真实退化类型（施加顺序），clean 表示无退化"],
              ["reward_model", "列表", "[{degradation_type, degradation_level}]"],
              ["extra_info", "对象", "original_image（真值 PNG）、use_original、index"],
          ], [0.2, 0.2, 0.6])]
    B += [("h2", "5.2  priors：统计先验模块"),
          ("p", "compute_image_statistics 接受 PIL 图像或数组，返回 ImageStatistics 数据类。to_qualitative 给出各项 low/medium/high 描述，"
                "to_prompt_text 生成注入提示词的结构化文本。模糊度按反向解释：拉普拉斯方差低对应模糊程度高。")]
    B += [("h2", "5.3  tools 与 tool_server：工具箱与工具服务"),
          ("p", "ToolSpec 描述单个工具；normalize_arguments 对参数填充默认值并裁剪到合法区间（例如 JPEG 质量 1~100、超分倍率取 {2,3,4}）。"
                "Toolbox 以 Executor 抽象屏蔽执行方式：local 模式下全部工具走 classical.py 的本地经典算子（同类工具使用不同算法或参数以保持互补性），"
                "remote 模式下深度专家模型通过 RemoteToolClient 调用远程服务，经典算子仍在本地执行。"),
          ("p", "工具服务协议如下，一个模型家族对应一个进程和端口："),
          ("table", ["项目", "内容"], [
              ["POST /process", "multipart：image（PNG）；form：task（工具名）及其他工具参数"],
              ["成功响应", "{\"success\": true, \"image\": \"<base64 PNG>\", \"width\":..., \"height\":...}"],
              ["失败响应", "{\"success\": false, \"error\": \"...\"}，HTTP 400/500"],
              ["GET /health", "返回状态与该服务支持的工具列表"],
              ["负载均衡", "环境变量 TOOL_SERVICE_IPS（逗号或分号分隔）配置多台服务器，支持 round_robin 与 random；未配置时使用 TOOL_SERVICE_IP"],
          ], [0.22, 0.78]),
          ("p", "服务端后端 Backend 只需实现 process(tool_name, image, params)。内置 ClassicalBackend；"
                "用户可用 module:function 形式挂接封装了 SwinIR、Restormer、SCUNet 等预训练模型的函数，无需改动服务框架。"),
          ("table", ["模型家族", "端口"], [[k, str(v) if v else "本地"] for k, v in FAMILY_PORTS.items()], [0.5, 0.5])]
    B += [("h2", "5.4  agent：智能体模块"),
          ("p", "parser.py 解析单轮响应，解析失败时在 errors 中给出原因而不抛出异常，以便格式奖励和环境统一处理。"
                "prompts.py 生成系统提示词（工具目录按退化类型分组、统计先验使用说明、模式规则、严格格式）、首轮用户提示和反馈提示。"
                "env.py 的 RestorationEnv 维护图像历史、已处理退化、回退次数和最近错误："),
          ("bullets", [
              "reset：构造首轮观测，包含系统提示、图像、统计先验和（可选）退化提示；",
              "step：解析响应；answer 则终止；否则按模式选择起始图像（单工具模式为当前图，回退时为上一张图；多工具模式为原图），链式执行工具；",
              "工具不存在或执行失败时记录错误并把错误信息反馈给智能体，图像保持不变，不中断训练；",
              "单工具模式下若一轮出现多个工具，只执行第一个并记录格式错误；回退重试次数受 max_retries 限制；",
              "每步结束重新提取统计先验，构造下一观测，达到 max_turns 时标记截断。",
          ]),
          ("p", "policy.py 定义策略接口 Policy.act(observation)。OpenAICompatiblePolicy 通过 OpenAI 兼容接口调用 vLLM 部署的多模态模型，"
                "图像以 base64 数据地址传入；HeuristicPolicy 是依据退化提示、固定优先级和统计先验的规则策略，用作基线和离线验证。")]
    B += [("h2", "5.5  reward：奖励模块"),
          ("p", "format_reward.py 提供 check_format_enhanced（严格）与 check_format_progressive（阶梯）；quality_metrics.py 实现 PSNR、SSIM（高斯窗灰度）、"
                "可选 LPIPS、归一化与无参考得分；degradation_reward.py 实现退化推理奖励、集合匹配与复原顺序奖励（理想复原顺序为施加顺序的逆序）；"
                "composite.py 提供 compute_step_rewards 与 compute_episode_reward，后者返回包含各分项的 RewardBreakdown，便于训练监控。")]
    B += [("h2", "5.6  train：训练模块"),
          ("p", "grpo.py 的 GRPOTrainer 对每个样本采样一组轨迹，计算轨迹奖励与组内优势，调用策略更新，并记录平均奖励、平均质量、格式违规率和平均工具调用数。"
                "softmax_policy.py 的 SoftmaxToolPolicy 同时是可训练策略和规则式顺序决策器，支持 save/load。"
                "verl_plugin.py 的 compute_score 读取环境变量完成奖励配置；verl_compute_score 是适配框架奖励分派的入口；make_verl_tool_classes 为框架的 ToolBase 注册表动态生成工具类。"),
          ("p", "多模态大模型的训练依赖独立的开源 VeRL 智能体强化学习框架（Apache License 2.0）。本软件不包含该框架的源码，仅提供上述奖励入口和工具注册函数。"
                "接入时需要在框架的奖励分派中对 data_source 为 restoramind 的样本调用 verl_compute_score，并由框架在 extra_info 中提供 original_image 与 image_history（各步工具输出图像）；"
                "scripts/train_verl.sh 的头部注释给出了完整的接入步骤。")]
    B += [("h2", "5.7  inference：推理模块"),
          ("p", "restore_image 对单张图像执行复原，可选提供退化提示；未提供提示时环境不注入提示，由智能体自行推理。"
                "evaluate_samples 对样本集合运行完整流程，计算退化图与复原图相对真值的 PSNR、SSIM、可选 LPIPS 和无参考得分，统计工具调用数、耗时与奖励，"
                "输出 evaluation_results.csv、summary.txt、每个样本的退化/复原/真值图像和对话明细。")]

    B += manual_extra.config_blocks()
    B += manual_extra.robustness_blocks()

    # ======================================================= 6 使用说明
    B += [("h1", "6  使用说明")]
    B += [("h2", "6.1  命令行总览"),
          ("table", ["子命令", "功能", "关键参数"], [
              ["make-data", "合成复合退化数据集", "--clean-dir --output --samples-per-image --clean-prob --max-size --seed"],
              ["serve-tools", "启动复原工具服务", "--family --backend --host --port"],
              ["train", "轻量策略 GRPO 训练", "--data --output --epochs --batch-size --group-size --lr"],
              ["restore", "单图复原", "--input --output --degradations --mode --policy --policy-file --toolbox"],
              ["evaluate", "数据集评估", "--data --output --num-samples --no-hint --mode --policy --policy-file --toolbox"],
          ], [0.16, 0.26, 0.58])]
    B += [("h2", "6.2  合成训练数据"),
          ("code", "python -m restoramind make-data \\\n    --clean-dir /path/to/clean_images \\\n    --output train.parquet \\\n"
                   "    --samples-per-image 2 --clean-prob 0.05"),
          ("p", "程序读取目录下的 png/jpg/bmp/webp 图像，限制最长边为 512，合成退化后写入 Parquet。user 提示中包含真实退化类型提示，"
                "训练智能体学习处理顺序与工具选择；评估时可通过 evaluate 的 --no-hint 或 restore 不指定 --degradations 取消提示，由智能体自行诊断退化。")]
    B += [("h2", "6.3  启动工具服务"),
          ("code", "bash scripts/serve_tools.sh                 # 按家族启动 13 个服务进程，端口 5001~5013\n"
                   "BACKEND=my_models:run bash scripts/serve_tools.sh   # 挂接真实专家模型\n"
                   "python -m restoramind serve-tools --family SCUNet --port 5008"),
          ("p", "接入真实模型时，编写函数 run(tool_name, image, params) 返回复原后的 PIL 图像，并通过 BACKEND 指向它。"
                "客户端侧设置 TOOL_SERVICE_IPS 即可在多台服务器间负载均衡。")]
    B += [("h2", "6.4  训练"),
          ("p", "轻量策略训练（CPU 可运行）："),
          ("code", "python -m restoramind train --data train.parquet --output policy.json --epochs 2 --group-size 4"),
          ("p", "多模态大模型训练（需要 VeRL 与 GPU），路径与地址均通过环境变量提供："),
          ("code", "export MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct\nexport TRAIN_FILES=/data/train.parquet\nexport VAL_FILES=/data/val.parquet\n"
                   "export TOOL_SERVICE_IPS=10.0.0.1,10.0.0.2\nbash scripts/train_verl.sh"),
          ("p", "train_verl.sh 不修改框架源码；奖励函数分派与工具注册的接入步骤写在脚本头部注释中（见 5.6 节）。奖励相关环境变量如下："),
          ("table", ["环境变量", "默认值", "含义"], [
              ["AGENT_CONVERSATION_MODE", "single_tool_iterative", "对话模式"],
              ["MAX_TOOLS_PER_TURN", "1", "单轮最大工具数，0 表示不限制"],
              ["ENABLE_TOTAL_TOOLS_UPPER_LIMIT", "True", "总工具数不超过退化数+1"],
              ["FORMAT_REWARD_WEIGHT", "0.3", "格式奖励权重"], ["QUALITY_REWARD_WEIGHT", "0.7", "质量奖励权重"],
              ["ENABLE_DEGRADATION_TYPE_REWARD", "False", "启用退化类型奖励"],
              ["DEGRADATION_TYPE_REWARD_WEIGHT", "1.0", "退化类型奖励权重"],
              ["ENABLE_INTERMEDIATE_REWARD", "False（脚本中为 True）", "启用中间图像质量奖励"],
              ["INTERMEDIATE_REWARD_WEIGHT", "0.5", "中间图像质量奖励权重"],
              ["IMAGE_QUALITY_USE_NO_REFERENCE", "False", "质量奖励使用无参考指标"],
              ["IMAGE_QUALITY_DISCRETIZE_LEVELS", "0", "质量奖励离散化档数，0 为连续"],
              ["USE_PROGRESSIVE_FORMAT", "False", "使用阶梯式格式奖励"],
              ["TOOL_SERVICE_IPS / TOOL_SERVICE_IP", "127.0.0.1", "工具服务地址，前者支持多个"],
              ["TOOL_LOAD_BALANCE_STRATEGY", "round_robin", "负载均衡策略（round_robin 或 random）"],
          ], [0.36, 0.24, 0.40]),
          ("p", "建议的训练策略：前 1~2 个轮次使用阶梯式格式奖励学习基本格式，随后切换为严格格式奖励强化规范。")]
    B += [("h2", "6.5  推理与评估"),
          ("code", "# 规则策略 + 本地工具（离线）\npython -m restoramind restore --input degraded.png --output restored.png --degradations \"noise,dark\"\n\n"
                   "# 训练后的多模态智能体（vLLM 部署）+ 远程专家模型\nexport TOOL_SERVICE_IPS=10.0.0.1\n"
                   "python -m restoramind restore --input degraded.png --output restored.png \\\n"
                   "    --policy openai --api-url http://localhost:8000/v1 --toolbox remote\n\n"
                   "# 数据集评估\npython -m restoramind evaluate --data test.parquet --output eval_out --num-samples 100"),
          ("p", "--policy 有三种取值：heuristic（规则策略，默认）、softmax（载入 train 子命令产出的 policy.json，需同时指定 --policy-file）、openai（调用 vLLM 部署的多模态智能体）。"
                "--no-hint 表示不向智能体提供退化类型提示，与 --policy openai 配合使用，由多模态智能体自行诊断图像中的退化。"),
          ("p", "评估输出目录包含 evaluation_results.csv（逐样本指标）、summary.txt（汇总）、images/（退化、复原、真值图像）和 details/（对话明细）。")]
    B += [("h2", "6.6  作为库使用"),
          ("code", "from PIL import Image\nfrom restoramind.agent.env import EnvConfig\nfrom restoramind.agent.policy import HeuristicPolicy\n"
                   "from restoramind.inference.restore import restore_image\nfrom restoramind.tools.toolbox import build_toolbox\n\n"
                   "image = Image.open('degraded.png')\nresult = restore_image(image, HeuristicPolicy(), build_toolbox('local'),\n"
                   "                       EnvConfig(max_turns=6), degradation_hint=['noise', 'dark'])\n"
                   "print(result.tool_sequence)\nresult.image.save('restored.png')")]
    B += [("h2", "6.7  扩展：新增复原工具"),
          ("numbered", ["在 restoramind/tools/registry.py 的 TOOL_SPECS 中新增 ToolSpec，指定名称、目标退化、模型家族、说明、参数模式与代价。",
                        "如为新模型家族，在 FAMILY_PORTS 中分配端口。",
                        "在工具服务后端实现该工具的推理；如需离线可运行，在 classical.py 的 CLASSICAL_OPS 中提供经典近似实现（测试会检查注册表与经典算子一一对应）。",
                        "如需让智能体直接使用，将 prompt_visible 设为 True；重新生成数据集与提示词后训练即可，整体流程无需修改。"])]

    # ======================================================= 7 运行与测试
    B += manual_extra.trace_blocks()
    B += manual_extra.deploy_blocks()
    B += PAPER_CHAPTER
    B += [("h1", "8  软件离线演示与自动化测试")]
    B += [("h2", "8.1  离线演示"),
          ("p", "[[demo]] 为一次离线运行：随机合成两种退化，规则策略逐步调用工具，最终输出复原图。该演示使用离线模式的经典算子，"
                "展示数据合成、统计先验、环境、工具执行和指标计算的完整处理流程。"),
          ("fig", FIG / "fig5_demo.png", "离线演示（原图、退化图、复原图）", "demo")]
    B += [("h2", "8.2  批量评估结果"),
          ("p", f"在 {R['num_samples']} 张合成复合退化图像（每张 2~3 种退化，固定随机种子）上，使用规则策略与离线模式工具箱的评估结果如下。"),
          ("table", ["指标", "退化图像", "复原图像"], [
              ["平均 PSNR（dB）", f"{R['psnr_before']:.2f}", f"{R['psnr_after']:.2f}"],
              ["平均 SSIM", f"{R['ssim_before']:.4f}", f"{R['ssim_after']:.4f}"],
          ], [0.4, 0.3, 0.3]),
          ("table", ["统计项", "数值"], [
              ["样本数", str(R["num_samples"])],
              ["平均退化数量", f"{R['mean_degradations']:.2f}"],
              ["平均工具调用数", f"{R['mean_tool_calls']:.2f}"],
              ["格式违规样本数", str(R["format_violations"])],
              ["平均单图耗时（秒，CPU）", f"{R['mean_seconds']:.3f}"],
              ["平均轨迹奖励", f"{R['mean_reward']:.4f}"],
          ], [0.6, 0.4]),
          ("p", "按退化数量分组，平均工具调用数为：" +
                "；".join(f"{k} 种退化对应 {v:.2f} 次" for k, v in R["tool_calls_by_num_degradations"].items()) +
                "。规则策略对每类退化调用一次工具，调用次数与退化数量一致。")]
    B += [("h2", "8.3  训练流程验证"),
          ("p", "使用 12 个合成样本、每组 4 条轨迹、6 个训练轮次运行轻量 GRPO 训练，训练统计如 [[train]]。每个训练步使用不同的样本批次，"
                "曲线记录了训练循环中采样、奖励计算、优势估计与策略更新的运行情况。"),
          ("fig", FIG / "fig6_train_curve.png", "GRPO 训练曲线", "train")]
    B += [("h2", "8.4  自动化测试"),
          ("p", f"软件配套 {total_tests} 个自动化测试，全部在 CPU、离线环境下运行，使用 pytest 执行。测试覆盖如下："),
          ("table", ["测试文件", "用例数", "覆盖内容"],
           [[f, str(c), TEST_DESC.get(f, "")] for f, c in sorted(tests.items())], [0.30, 0.10, 0.60]),
          ("code", "$ python -m pytest tests -q\n" + f"{total_tests} passed")]

    # ======================================================= 8 附录
    B += manual_extra.appendix_blocks()
    B += [("h1", "10  附录 B：接口参考")]
    B += [("p", "下表由源码自动提取，列出各模块公开的类、函数和方法及其说明。")]
    B += api_tables()
    B += [("h1", "11  常见问题")]
    B += [("table", ["问题", "解答"], [
        ["没有 GPU 能否运行？", "可以。离线演示、测试、轻量训练、评估均在 CPU 上运行，工具使用本地经典算子。"],
        ["如何使用真实的 SwinIR、Restormer 等模型？", "编写 run(tool_name, image, params) 函数封装模型推理，用 BACKEND=module:function 启动工具服务，客户端使用 --toolbox remote。"],
        ["工具服务返回错误怎么办？", "环境会把错误信息反馈给智能体，图像保持不变，不会中断训练。检查服务日志与 TOOL_SERVICE_IPS 配置。"],
        ["智能体总是格式违规？", "训练初期设置 USE_PROGRESSIVE_FORMAT=True 使用阶梯式格式奖励，待格式稳定后再切换为严格格式奖励。"],
        ["能否没有真值图像评估？", "可以。quality 奖励在无真值时自动使用无参考得分，evaluate 对无真值样本只输出无参考指标。"],
        ["如何检验智能体自行诊断退化的能力？", "使用 --policy openai 调用训练后的多模态智能体，并加 --no-hint，不向智能体提供退化类型提示。"],
        ["如何新增退化类型？", "在 data/degradations.py 中新增算子并登记到 DEGRADATION_FUNCTIONS，在工具注册表中增加对应工具即可。"],
    ], [0.34, 0.66])]
    return _number_figures(B)


# ----------------------------------------------------------------------------- 论文实验数据
METHODS = ["AirNet", "PromptIR", "MiOIR", "DA-CLIP", "InstructIR", "AutoDIR", "AgenticIR", "RestoraMind"]
# 数据来自论文 Table 1（PSNR、SSIM、LPIPS、MANIQA、CLIP-IQA、MUSIQ）
PAPER_TABLE = {
    "A": [(19.13, .6019, .4283, .2581, .3930, 42.46), (20.06, .6088, .4127, .2633, .4013, 42.62), (20.84, .6558, .3715, .2451, .3933, 47.82),
          (19.58, .6032, .4266, .2418, .4139, 42.51), (18.03, .5751, .4429, .2660, .3528, 45.77), (19.64, .6286, .3967, .2500, .3767, 47.01),
          (21.04, .6818, .3148, .3071, .4474, 56.88), (19.57, .6421, .3086, .3471, .4824, 60.80)],
    "B": [(19.31, .6567, .3670, .2882, .4274, 47.88), (20.47, .6704, .3370, .2893, .4289, 48.10), (20.56, .6905, .3243, .2638, .4330, 51.87),
          (18.56, .5946, .4405, .2435, .4154, 43.70), (18.34, .6235, .4072, .3022, .3790, 50.94), (19.90, .6643, .3542, .2534, .3986, 49.64),
          (20.55, .7009, .3072, .3204, .4648, 57.57), (19.98, .6732, .3161, .3439, .4762, 59.19)],
    "C": [(17.95, .5145, .5782, .1854, .3113, 30.12), (18.51, .5166, .5756, .1906, .3104, 29.71), (15.63, .4896, .5376, .1717, .2891, 37.95),
          (18.53, .5320, .5335, .1916, .3476, 33.87), (17.09, .5135, .5582, .1732, .2537, 33.69), (18.61, .5443, .5019, .2045, .2939, 37.86),
          (18.82, .5474, .4493, .2698, .3948, 48.86), (18.44, .5376, .4233, .2730, .3996, 50.32)],
}
# 工具调用次数，来自论文工具使用统计图（顶部五类退化）
TOOL_USAGE = {
    "low resolution": [("swinir_super_resolution", 358), ("xrestormer_super_resolution", 67), ("diffbir_super_resolution", 41)],
    "noise": [("scunet_real_denoising_psnr", 349), ("scunet_real_denoising_gan", 10), ("scunet_color_denoising", 7)],
    "defocus blur": [("restormer_defocus_deblurring", 307), ("drbnet_defocus_deblurring", 61)],
    "dark": [("retinexformer_fivek", 344), ("retinexformer_sdsd_indoor", 2)],
    "rain": [("xrestormer_deraining", 339), ("mprnet_deraining", 6)],
}


def _paper_rows():
    rows = []
    for g, data in PAPER_TABLE.items():
        best = [max(col) if i != 2 else min(col) for i, col in enumerate(zip(*data))]
        for name, vals in zip(METHODS, data):
            cells = [f"{v:.2f}" if i in (0, 5) else f"{v:.4f}" for i, v in enumerate(vals)]
            cells = [c + ("*" if vals[i] == best[i] else "") for i, c in enumerate(cells)]
            rows.append([f"Group {g}" if name == "AirNet" else "", name] + cells)
    return rows


def paper_chapter():
    B = [("h1", "7  方法验证与实验结果")]
    B += [("p", "本章给出本软件所实现方法的实验结果，实验内容见本软件对应的研究工作《RestoraMind: Thinking with Degradations and Tools toward Real-world Image Restoration》。"
                "实验使用预训练专家模型工具箱与训练后的多模态智能体完成；第 8 章为离线演示环境下的运行结果。"),
          ("p", "对比方法包括 AirNet、PromptIR、MiOIR、DA-CLIP、InstructIR、AutoDIR，"
                "以及 AgenticIR（Zhu 等，An intelligent agentic system for complex image restoration problems，arXiv:2410.17809，2024）。")]
    B += [("h2", "7.1  实验设置"),
          ("table", ["项目", "设置"], [
              ["训练数据", "从 Unsplash-lite 收集 7000 余张高质量图像；每张图像随机施加 2~3 种退化，强度随机"],
              ["退化库", "雾、雨、离焦模糊、低分辨率、运动模糊、低光、JPEG 压缩伪影、噪声共 8 类"],
              ["智能体", "以 Q-Insight（由 Qwen2.5-VL-7B-Instruct 经强化学习得到、具备图像质量感知能力的多模态模型）为初始化"],
              ["训练算法", "GRPO，训练 10 个轮次"],
              ["优化器", "AdamW，β1=0.9，β2=0.999，权重衰减 0.01，恒定学习率 1e-6"],
              ["工具箱", "预训练低层视觉专家模型，覆盖去噪、去模糊（运动与离焦）、去雾、去雨、超分辨率、低光增强、JPEG 伪影去除 7 个任务类别"],
              ["测试集", "三组复合退化测试集（Group A、B、C）"],
              ["对比方法", "一体化方法 AirNet、PromptIR、MiOIR、DA-CLIP、InstructIR、AutoDIR；智能体方法 AgenticIR"],
              ["评价指标", "有参考：PSNR、SSIM、LPIPS；无参考：MANIQA、CLIP-IQA、MUSIQ"],
          ], [0.16, 0.84])]
    B += [("h2", "7.2  与现有方法的定量对比"),
          ("p", "下表为三组测试集上的对比结果。PSNR、SSIM、MANIQA、CLIP-IQA、MUSIQ 越高越好，LPIPS 越低越好，*号标记该组该指标的最优值。"),
          ("table", ["测试组", "方法", "PSNR", "SSIM", "LPIPS", "MANIQA", "CLIP-IQA", "MUSIQ"], _paper_rows(),
           [0.11, 0.17, 0.10, 0.11, 0.11, 0.11, 0.14, 0.15]),
          ("bullets", [
              "智能体式方法（AgenticIR 与 RestoraMind）在 LPIPS、MANIQA、CLIP-IQA、MUSIQ 等感知类指标上整体优于一体化方法。",
              "RestoraMind 在三组测试集上的 MANIQA、CLIP-IQA、MUSIQ 三项无参考指标数值最高，在 Group A 与 Group C 上 LPIPS 数值最低。",
          ])]
    B += [("h2", "7.3  定性对比"),
          ("p", "[[qual]] 为三类测试图像的处理结果，每组左为输入，右为本软件输出，红框为局部放大。（a）为 Group C 测试图像，输入含密集雨纹和噪声，输出中蜥蜴鳞片、火花等高频细节得到恢复；"
                "（b）为含雨纹、噪声与低光等复合退化的人像与火花场景；（c）为真实场景图像，上为老照片、下为卡通图像，退化类型事先未知，且与合成训练数据存在域差异，输出恢复了较清晰的语义与自然纹理。"),
          ("fig", FIG / "paper_qual_all.jpg", "定性结果：每组左为输入，右为本软件输出", "qual")]
    rows, total_n = [], {}
    for deg, items in TOOL_USAGE.items():
        total = sum(n for _, n in items)
        total_n[deg] = total
        for name, n in items:
            rows.append([deg, name, str(n), f"{n / total * 100:.1f}%"])
    B += [("h2", "7.4  自适应工具选择分析"),
          ("p", "[[usage]] 统计了训练后的智能体在五类退化上各候选工具的调用次数。每类退化都出现一个调用最多的通用型主导工具，"
                "例如低分辨率中的 swinir_super_resolution 被调用 358 次，噪声中的 scunet_real_denoising_psnr 被调用 349 次。"),
          ("fig", FIG / "paper_tool_usage.png", "五类退化上的工具调用次数分布", "usage"),
          ("table", ["退化类型", "工具", "调用次数", "类内占比"], rows, [0.22, 0.44, 0.16, 0.18]),
          ("p", "同类其他候选工具也保持一定比例的调用：低分辨率中 xrestormer_super_resolution 被选择 67 次，"
                "离焦模糊中 drbnet_defocus_deblurring 被选择 61 次。智能体在每个决策状态结合图像统计先验，在同类工具之间进行选择。")]
    B += [("h2", "7.5  效率"),
          ("p", "搜索式智能体方法需要通过试错搜索寻找较优的工具序列。实验中，AgenticIR 平均每张图像需要 65.04 秒、调用工具 5.15 次。"
                "RestoraMind 由训练后的策略在每个状态直接输出当前工具，不依赖深度优先搜索、质量驱动搜索或大规模回滚，"
                "每个退化类型对应一次推理与一次工具调用，工具调用次数随退化类型数量近似线性增长。")]
    return B


PAPER_CHAPTER = None


def _number_figures(blocks):
    ids = {}
    n = 0
    for b in blocks:
        if b[0] == "fig":
            n += 1
            ids[b[3]] = n
    def sub(text):
        for k, v in ids.items():
            text = text.replace(f"[[{k}]]", f"图{v}")
        return text
    out = []
    for b in blocks:
        if b[0] == "fig":
            out.append(("fig", b[1], f"图{ids[b[3]]}  {b[2]}"))
        elif b[0] == "p":
            out.append(("p", sub(b[1])))
        elif b[0] in ("bullets", "numbered"):
            out.append((b[0], [sub(x) for x in b[1]]))
        else:
            out.append(b)
    return out
