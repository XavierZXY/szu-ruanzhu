# RestoraMind 基于多模态推理与强化学习的图像复原智能体软件 V1.0

著作权人：深圳理工大学 邹翔宇、马智恒

RestoraMind 把真实场景复合退化图像的复原建模为**序列决策过程**。智能体在每一步联合完成
「剩余退化推理 → 复原工具选择 → 工具执行」，并借助图像低层统计先验在同一退化类型的多个候选工具之间自适应选择。
智能体通过强化学习（GRPO）端到端训练，工具调用次数随退化类型数量近似线性增长，无需回滚式后搜索。

## 功能

| 模块 | 说明 |
|---|---|
| `restoramind.data` | 8 类退化算子（雾、雨、离焦模糊、低分辨率、运动模糊、低光、JPEG 伪影、噪声）、随机 2~3 种复合退化合成、Parquet 数据集构建与读取 |
| `restoramind.priors` | 图像低层统计先验：亮度、对比度、色彩、边缘密度、高频能量、噪声/模糊/块效应/雾浓度估计 |
| `restoramind.tools` | 28 个复原工具的规格注册表、本地经典算子、远程专家模型客户端（多 IP 负载均衡）、统一工具箱 |
| `restoramind.tool_server` | 复原工具 HTTP 服务（Flask），可挂接 SwinIR、Restormer、SCUNet 等任意预训练模型 |
| `restoramind.agent` | 响应解析、提示词、序列决策环境（单工具迭代 / 多工具规划两种模式）、策略接口、轨迹采样 |
| `restoramind.reward` | 格式奖励（严格/阶梯）、PSNR·SSIM·LPIPS 质量奖励、无参考质量、退化推理奖励、步级与轨迹级复合奖励 |
| `restoramind.train` | GRPO 组相对优势训练循环、可训练工具选择策略、VeRL/DeepEyes 框架桥接（奖励函数与工具类） |
| `restoramind.inference` | 单图复原、数据集评估（CSV / 汇总 / 图像） |

## 安装

```bash
pip install -r requirements.txt
pip install -e .
```

Python ≥ 3.10。核心流程仅依赖 numpy、Pillow、OpenCV、pandas、pyarrow、Flask，可在 CPU 上离线运行。
大模型训练与推理需另装 VeRL 与 vLLM（见下文）。

## 快速开始（离线演示）

```bash
bash scripts/demo.sh          # 合成数据 → 训练 → 评估 → 单图复原
python -m pytest tests -q     # 运行全部测试
```

## 命令行

```bash
# 1. 合成复合退化数据集（Parquet）
python -m restoramind make-data --clean-dir /path/to/clean_images --output train.parquet --samples-per-image 2

# 2. 启动复原工具服务（经典算子后端；接入真实模型见下）
bash scripts/serve_tools.sh

# 3. 轻量策略 GRPO 训练（CPU 可运行，用于验证流程）
python -m restoramind train --data train.parquet --output policy.json --epochs 2

# 4. 单图复原
python -m restoramind restore --input degraded.png --output restored.png --degradations "noise,dark"

# 5. 数据集评估
python -m restoramind evaluate --data test.parquet --output eval_out
```

使用训练后的多模态智能体推理时，用 vLLM 部署模型，并指定 `--policy openai --api-url http://host:8000/v1`；
工具改为远程专家模型服务时加 `--toolbox remote`，并设置环境变量 `TOOL_SERVICE_IPS=ip1,ip2`。

## 接入真实专家模型

工具服务的后端是一个函数 `fn(tool_name, image, params) -> PIL.Image`：

```python
# my_models.py
def run(tool_name, image, params):
    ...  # 按 tool_name 加载并调用 SwinIR / Restormer / SCUNet 等预训练模型
    return restored_pil_image
```

```bash
BACKEND=my_models:run bash scripts/serve_tools.sh
```

各模型家族与端口对应关系见 `restoramind/tools/registry.py` 中的 `FAMILY_PORTS`。

## 大模型强化学习训练（VeRL）

多模态智能体（Qwen2.5-VL-7B 或 Q-Insight）的 GRPO 训练基于 VeRL/DeepEyes 智能体框架：

```bash
export MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct
export TRAIN_FILES=/data/train.parquet VAL_FILES=/data/val.parquet
export TOOL_SERVICE_IPS=10.0.0.1,10.0.0.2
bash scripts/train_verl.sh
```

`restoramind.train.verl_plugin.compute_score` 是框架调用的序列级奖励函数，
`make_verl_tool_classes(ToolBase)` 把工具箱注册为框架工具类。奖励权重等通过环境变量配置，见脚本头部。

## 目录结构

```
restoramind/      软件源码
tests/            单元测试与端到端冒烟测试
scripts/          训练、工具服务、演示脚本
configs/          默认配置
ruanzhu_materials/ 软件著作权登记材料生成脚本与成品
```
