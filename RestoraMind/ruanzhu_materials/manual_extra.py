# -*- coding: utf-8 -*-
"""
说明书补充章节：均由真实运行或源码自动生成。
    stats_example_blocks   统计先验数值示例（对同一幅图施加 8 类退化后的实测统计量）
    config_blocks          配置项参考（由 dataclass 字段自动提取）
    robustness_blocks      异常处理与鲁棒性设计
    trace_blocks           一次完整交互轨迹（真实运行的提示、响应、反馈与奖励）
    deploy_blocks          部署与运维
    appendix_blocks        系统提示词全文、工具提示词条目、数据样例
"""
import dataclasses
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from restoramind.agent.env import EnvConfig, RestorationEnv  # noqa: E402
from restoramind.agent.policy import HeuristicPolicy  # noqa: E402
from restoramind.agent.prompts import build_system_prompt  # noqa: E402
from restoramind.agent.runner import run_episode  # noqa: E402
from restoramind.data.degradations import DEGRADATION_TYPES, apply_degradation  # noqa: E402
from restoramind.data.dataset import Sample, build_record  # noqa: E402
from restoramind.data.synthesize import synthesize_sample  # noqa: E402
from restoramind.priors.image_statistics import compute_image_statistics  # noqa: E402
from restoramind.reward.composite import RewardConfig, compute_episode_reward, compute_step_rewards  # noqa: E402
from restoramind.reward.format_reward import FormatConfig  # noqa: E402
from restoramind.tools.registry import TOOL_SPECS, prompt_visible_tools  # noqa: E402
from restoramind.tools.toolbox import build_toolbox  # noqa: E402
from restoramind.train.grpo import GRPOConfig  # noqa: E402


def demo_clean(size: int = 160) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size] / float(size)
    img = np.stack([255 * (0.5 + 0.5 * np.sin(7 * np.pi * xx)), 255 * yy, 255 * (0.5 + 0.5 * np.cos(5 * np.pi * yy * xx))], -1)
    img[40:90, 40:90] = 235
    return np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- 统计先验示例
def stats_example_blocks():
    clean = demo_clean()
    rng = np.random.default_rng(0)
    cols = ["图像", "亮度均值", "对比度", "边缘密度", "高频能量", "噪声估计", "拉普拉斯方差", "块效应", "雾浓度"]
    rows = []
    items = [("原始清晰图", clean)] + [(f"{d}（high）", apply_degradation(clean, d, "high", rng)) for d in DEGRADATION_TYPES]
    for name, img in items:
        s = compute_image_statistics(img)
        rows.append([name, f"{s.brightness_mean:.1f}", f"{s.contrast_global:.1f}", f"{s.edge_density:.3f}",
                     f"{s.high_freq_energy:.3f}", f"{s.noise_sigma:.2f}", f"{s.blur_score:.0f}", f"{s.blockiness:.3f}", f"{s.haze_score:.2f}"])
    stats = compute_image_statistics(apply_degradation(clean, "noise", "high", np.random.default_rng(1)))
    return [
        ("h2", "4.7  统计先验数值示例"),
        ("p", "为直观说明统计先验对各类退化的响应，下表对同一幅 160×160 的合成图像分别施加 8 类退化（强度均为 high，固定随机种子），"
              "列出提取到的主要统计量。数值由 priors.compute_image_statistics 计算得到。"),
        ("table", cols, rows, [0.21, 0.09, 0.09, 0.09, 0.10, 0.10, 0.12, 0.09, 0.11]),
        ("p", "可以看到：低光退化使亮度均值显著下降；噪声使噪声估计和高频能量显著上升；离焦与运动模糊使拉普拉斯方差下降；"
              "雾使雾浓度（暗通道均值）上升；JPEG 压缩使块效应指标上升。这些方向性变化为智能体判断剩余退化、"
              "选择合适工具提供了廉价且可解释的依据。"),
        ("p", "注入智能体提示词的结构化文本如下（对加噪图像，由 to_prompt_text 生成）："),
        ("code", stats.to_prompt_text()),
        ("p", "文本同时给出数值与 low / medium / high 的定性描述。数值便于模型比较细微差异，定性描述便于模型快速建立判断，两者互为补充。"),
    ]


# ---------------------------------------------------------------- 配置参考
_DESC = {
    ("EnvConfig", "mode"): "对话模式：single_tool_iterative 或 multi_tool_planning",
    ("EnvConfig", "max_turns"): "最大对话轮数，达到后回合被截断",
    ("EnvConfig", "max_retries"): "同类工具回退重试的次数上限（有限回退）",
    ("EnvConfig", "use_statistics_prior"): "是否在每个状态提取并注入图像统计先验",
    ("EnvConfig", "provide_degradation_hint"): "是否在提示中给出样本的退化类型（训练常开，检验自行诊断能力时关闭）",
    ("EnvConfig", "max_tools_per_turn"): "单轮最大工具数，0 表示不限制；单工具模式强制为 1",
    ("EnvConfig", "stop_on_format_error"): "出现无有效工具调用的响应时是否立即终止回合",
    ("RewardConfig", "alpha"): "步级奖励中图像质量增益 ΔQ 的系数 α",
    ("RewardConfig", "beta"): "步级奖励中退化推理奖励的系数 β",
    ("RewardConfig", "gamma"): "工具成本惩罚系数 γ",
    ("RewardConfig", "delta"): "每步固定惩罚 δ，鼓励更短的复原路径",
    ("RewardConfig", "kappa"): "终止奖励系数 κ，作用于最终图像质量",
    ("RewardConfig", "failure_penalty"): "未在最大轮数内完成复原的失败惩罚",
    ("RewardConfig", "format_weight"): "轨迹级格式奖励权重",
    ("RewardConfig", "quality_weight"): "轨迹级图像质量奖励权重",
    ("RewardConfig", "enable_degradation_type_reward"): "是否启用退化类型集合匹配奖励",
    ("RewardConfig", "degradation_type_weight"): "退化类型奖励权重",
    ("RewardConfig", "enable_intermediate_reward"): "是否启用中间图像质量奖励",
    ("RewardConfig", "intermediate_weight"): "中间图像质量奖励权重",
    ("RewardConfig", "tool_diversity_bonus"): "工具互不重复且数量合理时的加成",
    ("RewardConfig", "use_no_reference"): "质量奖励是否使用无参考指标（无真值时自动启用）",
    ("RewardConfig", "use_lpips"): "质量奖励是否加入 LPIPS（依赖 lpips 库）",
    ("RewardConfig", "discretize_levels"): "质量奖励离散化档数，0 表示连续",
    ("RewardConfig", "format_mode"): "格式奖励模式：enhanced（严格二元）或 progressive（阶梯式）",
    ("RewardConfig", "format"): "格式检查配置（FormatConfig）",
    ("FormatConfig", "mode"): "对话模式，单工具模式要求每轮恰好一个工具",
    ("FormatConfig", "max_tools_per_turn"): "单轮最大工具数，0 表示不限制",
    ("FormatConfig", "enable_total_tools_upper_limit"): "是否限制总工具数不超过退化数量+1",
    ("FormatConfig", "min_think_chars"): "think 块的最少字符数",
    ("FormatConfig", "allowed_tools"): "允许调用的工具名集合，默认为注册表全部工具",
    ("GRPOConfig", "group_size"): "每个样本采样的轨迹数 G",
    ("GRPOConfig", "epochs"): "遍历训练集的轮数",
    ("GRPOConfig", "reward"): "奖励配置（RewardConfig）",
}


def config_blocks():
    blocks = [("h2", "5.8  配置项参考"),
              ("p", "软件的行为通过四个配置数据类控制，字段、默认值和含义由源码自动提取。命令行与环境变量最终都映射到这些字段。")]
    for cls in (EnvConfig, RewardConfig, FormatConfig, GRPOConfig):
        rows = []
        for f in dataclasses.fields(cls):
            if f.default is not dataclasses.MISSING:
                default = repr(f.default)
            elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
                v = f.default_factory()  # type: ignore[misc]
                default = f"{type(v).__name__}(...)" if dataclasses.is_dataclass(v) else (f"（{len(v)} 个工具）" if f.name == "allowed_tools" else repr(v))
            else:
                default = "—"
            desc = _DESC[(cls.__name__, f.name)]
            rows.append([f.name, default, desc])
        blocks += [("h3", cls.__name__), ("table", ["字段", "默认值", "含义"], rows, [0.28, 0.20, 0.52])]
    blocks.append(("p", "SoftmaxToolPolicy 的构造参数为学习率 lr（默认 0.5）、温度 temperature（默认 1.0）、随机种子 seed 和退化处理优先级 priority；"
                        "OpenAICompatiblePolicy 的构造参数为 base_url、api_key、model、temperature（默认 0.7）、top_p（默认 0.9）和 max_tokens（默认 2048）。"))
    return blocks


# ---------------------------------------------------------------- 异常处理
def robustness_blocks():
    return [
        ("h2", "5.9  异常处理与鲁棒性设计"),
        ("p", "强化学习训练会产生大量不规范的模型输出，工具服务也可能出现网络波动或显存不足。软件遵循「错误不中断回合、错误可反馈、错误可度量」的原则："),
        ("table", ["场景", "处理方式", "对训练/推理的影响"], [
            ["响应缺少 think、tool_call 或 answer，或 JSON 非法", "解析器返回错误列表而不抛异常；环境记录格式错误并把无有效工具调用的提示反馈给模型", "格式奖励给出负分（严格模式）或较低阶梯分（阶梯模式）"],
            ["单工具模式下一轮出现多个工具", "只执行第一个工具并记录格式错误", "格式奖励判违规"],
            ["调用了不存在的工具", "不执行，图像保持不变，错误信息反馈给模型", "回合继续，奖励由格式检查体现"],
            ["工具执行失败（模型报错、显存不足）", "Toolbox 统一包装为 ToolExecutionError，环境记录错误并反馈，不中断训练", "该步无图像更新，模型可换用同类工具"],
            ["远程服务返回 HTTP 错误或超时", "客户端抛出异常，由工具箱捕获并按工具失败处理；负载均衡下次请求自动轮转到其他服务器", "单台服务器故障不会使训练崩溃"],
            ["工具改变图像尺寸（如超分）", "质量指标先把结果缩放到参考图尺寸再计算", "指标可比，不会因尺寸不一致报错"],
            ["回退重试次数超限", "回退请求退化为普通步骤，作用于当前图像", "限制重试开销，避免无限搜索"],
            ["达到最大轮数仍未作答", "回合标记为截断，步级奖励最后一步叠加失败惩罚", "鼓励在限定轮数内完成复原"],
            ["clean 样本（无退化）", "正确响应是不调用工具、直接作答；调用工具则质量项为 0", "避免模型对干净图像过度处理"],
            ["无真值图像", "质量奖励自动改用无参考得分；评估只输出无参考指标", "可在真实无标注数据上继续使用"],
            ["LPIPS 或外部指标库不可用", "自动退化为 PSNR+SSIM，权重按比例重新分配", "离线环境仍可完整运行"],
        ], [0.26, 0.44, 0.30]),
        ("p", "此外，所有随机过程（退化合成、策略采样）都接受随机种子，保证实验可复现；奖励函数不依赖全局状态，可在多进程环境中并行调用。"),
    ]


# ---------------------------------------------------------------- 交互轨迹
def _msg_text(content) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for c in content:
        parts.append(c["text"] if c["type"] == "text" else "[图像：当前待处理图像]")
    return "\n".join(parts)


def trace_blocks():
    clean = demo_clean()
    syn = synthesize_sample(clean, np.random.default_rng(11), num_degradations=(3,))
    sample = Sample(Image.fromarray(syn.degraded), Image.fromarray(clean), syn.degradations, 0)
    env = RestorationEnv(build_toolbox("local"), EnvConfig(max_turns=6))
    traj = run_episode(env, HeuristicPolicy(), sample)
    obs = RestorationEnv(build_toolbox("local"), EnvConfig(max_turns=6)).reset(sample)
    cfg = RewardConfig(enable_degradation_type_reward=True, enable_intermediate_reward=True)
    bd = compute_episode_reward(traj, cfg)
    steps = compute_step_rewards(traj, cfg)

    blocks = [
        ("h2", "6.8  完整交互轨迹示例"),
        ("p", f"下面展示一次真实运行的完整交互：对一幅干净图像随机合成 3 种退化（真实施加顺序：{' → '.join(syn.degradation_types)}），"
              "由规则策略逐步复原。示例运行在离线模式，工具由经典算子实现（见 4.3 节）。"),
        ("h3", "第 0 步：首轮用户提示（系统提示词见附录 A）"),
        ("code", _msg_text(obs.messages[1]["content"])),
    ]
    msgs = env.trajectory
    # 重放消息以取得每步反馈文本
    env2 = RestorationEnv(build_toolbox("local"), EnvConfig(max_turns=6))
    o = env2.reset(sample)
    for rec in traj.steps:
        blocks += [("h3", f"第 {rec.turn} 步：模型响应"), ("code", rec.response)]
        o, done, _ = env2.step(rec.response)
        if not rec.is_answer:
            blocks += [("p", f"环境执行工具 {', '.join(rec.executed_tools) or '（无）'}，得到新图像并重新提取统计先验，返回如下反馈："),
                       ("code", _msg_text(o.messages[-1]["content"]))]
    rows = [[str(i + 1), ", ".join(s.executed_tools) or "（作答）", f"{r:+.4f}"] for i, (s, r) in enumerate(zip(traj.steps, steps))]
    blocks += [
        ("h3", "轨迹奖励"),
        ("p", "该轨迹的步级奖励与轨迹级奖励明细如下。步级奖励按 r_t=α·ΔQ_t+β·R_deg-γ·Cost-δ 计算，最后一步为终止奖励 κ·R_quality；"
              "轨迹级奖励启用了退化类型奖励与中间图像质量奖励。"),
        ("table", ["步", "动作", "步级奖励"], rows, [0.1, 0.65, 0.25]),
        ("table", ["分项", "数值"], [
            ["格式得分", f"{bd.format_score:+.1f}（{bd.format_reason}）"], ["质量得分", f"{bd.quality_score:.4f}"],
            ["退化类型得分", f"{bd.degradation_type_score:.4f}"], ["中间图像质量得分", f"{bd.intermediate_quality_score:.4f}"],
            ["工具多样性加成", f"{bd.tool_diversity_bonus:.2f}"], ["轨迹总奖励", f"{bd.score:.4f}"],
        ], [0.4, 0.6]),
    ]
    return blocks


# ---------------------------------------------------------------- 部署
def deploy_blocks():
    return [
        ("h2", "6.9  部署与运维"),
        ("h3", "6.9.1  典型部署拓扑"),
        ("table", ["角色", "运行内容", "资源建议"], [
            ["工具服务节点（1~N 台）", "scripts/serve_tools.sh 启动 13 个家族服务，端口 5001~5013；各节点部署相同的工具集合", "每个深度模型家族占用一份显存；节点数越多吞吐越高"],
            ["智能体推理节点", "vLLM 部署训练后的多模态智能体，对外提供 OpenAI 兼容接口", "单卡可部署 7B 模型；显存按上下文长度和图像数量预留"],
            ["训练节点", "VeRL 的 Actor/Rollout/Reference，调用工具服务采集轨迹", "建议 8 卡及以上；图像较大时需要较大的内存"],
            ["评估与数据节点", "make-data、evaluate 等命令行任务", "CPU 即可，评估时调用推理节点与工具服务"],
        ], [0.22, 0.46, 0.32]),
        ("h3", "6.9.2  启动顺序"),
        ("numbered", ["在每台工具服务节点上启动工具服务：bash scripts/serve_tools.sh（或 BACKEND=module:function 指向真实模型）。",
                      "用健康检查确认服务可用：curl http://<ip>:5008/health，应返回 status 为 ok 及该服务支持的工具列表。",
                      "在训练或推理侧设置 TOOL_SERVICE_IPS 为全部工具服务地址，并选择 round_robin 或 random 负载均衡策略。",
                      "推理场景：用 vLLM 启动智能体模型，再运行 restore 或 evaluate 子命令；训练场景：设置 MODEL_PATH、TRAIN_FILES 等环境变量后运行 scripts/train_verl.sh。"]),
        ("code", "# 健康检查\ncurl http://10.0.0.1:5008/health\n# 单次工具请求（命令行等价于客户端调用）\n"
                 "curl -X POST http://10.0.0.1:5008/process -F image=@degraded.png -F task=scunet_real_denoising_gan"),
        ("h3", "6.9.3  日志与监控"),
        ("bullets", [
            "工具服务日志：scripts/serve_tools.sh 将每个家族的输出写入 logs/tool_<家族>.log，便于定位单个模型的错误；",
            "训练日志：train_verl.sh 将控制台输出写入 logs/<实验名>.log；轨迹奖励的各分项（格式、质量、退化类型、中间质量、工具多样性）由 compute_score 返回，可接入训练监控面板；",
            "评估输出：evaluate 子命令生成 evaluation_results.csv、summary.txt 和每个样本的 details/*.json，便于离线分析。",
        ]),
        ("h3", "6.9.4  常见故障排查"),
        ("table", ["现象", "可能原因", "处理办法"], [
            ["训练初期格式奖励长期为负", "模型尚未掌握严格格式", "启用 USE_PROGRESSIVE_FORMAT=True 的阶梯式格式奖励，稳定后再切换"],
            ["所有样本都没有工具被执行", "工具服务不可达或工具名不在允许集合", "检查 TOOL_SERVICE_IPS 与健康检查；确认响应中的工具名与注册表一致"],
            ["质量奖励长期为 0", "未调用任何工具，或样本缺少真值且未启用无参考指标", "检查格式得分；无真值时设置 IMAGE_QUALITY_USE_NO_REFERENCE=True"],
            ["工具调用次数明显多于退化数量", "模型过度调用或反复回退", "启用 ENABLE_TOTAL_TOOLS_UPPER_LIMIT，降低 max_retries"],
            ["单台服务器响应变慢", "该节点显存不足或排队", "增加节点数并使用多 IP 负载均衡；降低并发 concurrent_workers"],
            ["指标计算报尺寸不一致", "工具改变了图像尺寸", "已自动缩放到参考尺寸；若仍报错请检查真值图像是否与输入同源"],
        ], [0.28, 0.30, 0.42]),
    ]


# ---------------------------------------------------------------- 附录
def appendix_blocks():
    system_prompt = build_system_prompt("single_tool_iterative")
    multi_prompt_head = build_system_prompt("multi_tool_planning").split("## Rules")[1].split("## Response Format")[0].strip()
    rows = [[s.name, s.degradation, s.prompt_line().split("— ", 1)[1] if "— " in s.prompt_line() else "", ", ".join(s.parameters) or "—"]
            for s in prompt_visible_tools()]
    clean = demo_clean()
    syn = synthesize_sample(clean, np.random.default_rng(5), num_degradations=(2,))
    rec = build_record(syn, 0, "<system prompt，见附录 A.1>", "<image>\nImage analysis indicates these degradation types (order unknown): "
                       + ", ".join(syn.degradation_types) + "\nSelect the FIRST tool to apply.")
    shown = {
        "data_source": rec["data_source"],
        "prompt": [{"role": "system", "content": "<system prompt，见附录 A.1>"}, rec["prompt"][1]],
        "images": [{"bytes": f"<PNG 字节，{len(rec['images'][0]['bytes'])} B>"}],
        "env_name": rec["env_name"],
        "reward_model": rec["reward_model"],
        "extra_info": {"original_image": {"bytes": f"<PNG 字节，{len(rec['extra_info']['original_image']['bytes'])} B>"},
                       "use_original": True, "index": 0},
    }
    return [
        ("h1", "9  附录 A：提示词与数据样例"),
        ("h2", "A.1  单工具迭代模式的系统提示词全文"),
        ("p", "以下为 agent.prompts.build_system_prompt 在默认模式下生成的完整系统提示词（工具目录由注册表自动生成，仅包含对智能体可见的工具）。"),
        ("code", system_prompt),
        ("h3", "中文说明"),
    ] + chinese_prompt_summary() + [
        ("h2", "A.2  多工具规划模式的规则段"),
        ("p", "多工具规划模式与上面的提示词共用工具目录、统计先验说明和响应格式，仅规则段不同："),
        ("code", multi_prompt_head),
        ("h2", "A.3  智能体可见工具一览"),
        ("table", ["工具", "目标退化", "特点", "参数"], rows, [0.30, 0.17, 0.43, 0.10]),
        ("h2", "A.4  训练数据记录样例"),
        ("p", f"下面是一条由 build_record 生成的 Parquet 记录（图像字节已省略），真实退化施加顺序为 {' → '.join(syn.degradation_types)}："),
        ("code", json.dumps(shown, ensure_ascii=False, indent=2)),
    ]


# ---------------------------------------------------------------- 代码规模统计
def code_stats():
    """统计 restoramind 包：总行数、非空行、有效代码行（不含空行、纯注释行与文档字符串）。"""
    import ast
    total = nonblank = effective = 0
    for p in sorted((HERE.parent / "restoramind").rglob("*.py")):
        text = p.read_text(encoding="utf-8")
        lines = text.splitlines()
        doc_lines = set()
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                body = getattr(node, "body", [])
                if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                        and isinstance(body[0].value.value, str):
                    doc_lines.update(range(body[0].lineno, body[0].end_lineno + 1))
        total += len(lines)
        for i, line in enumerate(lines, 1):
            if not line.strip():
                continue
            nonblank += 1
            if line.strip().startswith("#") or i in doc_lines:
                continue
            effective += 1
    return {"total": total, "nonblank": nonblank, "effective": effective}


def chinese_prompt_summary():
    """系统提示词的中文说明（提示词本身为英文，属软件运行时内容）。"""
    return [
        ("p", "系统提示词面向多模态大模型，使用英文书写以便与通用模型的指令跟随能力对齐，属于软件运行时内容。各段含义如下："),
        ("table", ["段落", "中文含义"], [
            ["开头身份说明", "你是 RestoraMind 图像复原智能体，需要通过逐步推理剩余退化并为每一步选择最合适的专家复原工具，来复原含未知复合退化的图像"],
            ["Available Tools", "按退化类型分组列出可调用工具，每个工具给出名称、参数模式和适用特点（例如偏细节保持或偏稳健平滑）"],
            ["Image Statistics", "每张图像都会附带轻量低层统计量；用它们确认还剩哪些退化，并在同类工具之间选择，如边缘密度高时偏细节保持型工具、低时偏稳健平滑型工具"],
            ["Rules", "单工具模式：每轮一个工具并作用于当前图像；上一步未去除目标退化时，可加 rollback 在上一张图像上换同类工具重试一次；无显著退化时立即作答，避免冗余调用。多工具规划模式：一次给出完整有序工具序列，系统从原图链式执行"],
            ["Response Format", "每轮必须先写不少于 10 个字符的 think 说明推理，再二选一输出 tool_call（JSON 列表）或 answer（restoration_log，按处理顺序列出退化类型）；不能同时输出二者，只能使用列表中的工具名"],
        ], [0.2, 0.8]),
    ]
