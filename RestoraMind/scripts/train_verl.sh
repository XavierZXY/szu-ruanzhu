#!/bin/bash
# 基于 VeRL / DeepEyes 智能体强化学习框架训练多模态智能体（GRPO）。
# 前置条件：已安装 VeRL 智能体框架；已启动复原工具服务；已用 make-data 生成 Parquet 数据集。
# 所有路径、地址均通过环境变量提供，脚本不含任何密钥或内网地址。
#
# 与框架的接口（需要在所用的 VeRL 智能体框架中各做一次，本脚本不修改框架源码）：
#   1. 奖励：在框架的 reward_score 分派函数（default_compute_score）中，对 data_source == "restoramind"
#      调用 restoramind.train.verl_plugin.verl_compute_score(data_source, solution_str, ground_truth, extra_info)；
#      该函数需要框架在 extra_info 中提供 original_image 与 image_history（各步工具输出图像）。
#   2. 工具：启动前执行
#        from restoramind.train.verl_plugin import make_verl_tool_classes
#        make_verl_tool_classes(ToolBase)      # 把工具箱注册为框架工具类，名称对应数据中的 env_name
#   3. 数据：使用 python -m restoramind make-data 生成的 Parquet。
#   上述接口的轻量替代与单元测试见 tests/test_train.py（使用桩 ToolBase 验证）。
set -euo pipefail

: "${MODEL_PATH:?请设置 MODEL_PATH（Qwen2.5-VL-7B-Instruct 或 Q-Insight 权重目录）}"
: "${TRAIN_FILES:?请设置 TRAIN_FILES（Parquet 训练集，逗号分隔）}"
: "${VAL_FILES:?请设置 VAL_FILES（Parquet 验证集，逗号分隔）}"
: "${TOOL_SERVICE_IPS:?请设置 TOOL_SERVICE_IPS（工具服务地址，逗号分隔）}"

export SAVE_DIR="${SAVE_DIR:-./checkpoints}"
export PROJECT_NAME="${PROJECT_NAME:-RestoraMind}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-restoramind_grpo}"
export NNODES="${NNODES:-1}"
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"

# 智能体与奖励配置（由 restoramind.train.verl_plugin.compute_score 读取）
export AGENT_CONVERSATION_MODE="${AGENT_CONVERSATION_MODE:-single_tool_iterative}"
export MAX_TOOLS_PER_TURN="${MAX_TOOLS_PER_TURN:-1}"
export ENABLE_TOTAL_TOOLS_UPPER_LIMIT="${ENABLE_TOTAL_TOOLS_UPPER_LIMIT:-True}"
export FORMAT_REWARD_WEIGHT="${FORMAT_REWARD_WEIGHT:-0.3}"
export QUALITY_REWARD_WEIGHT="${QUALITY_REWARD_WEIGHT:-0.7}"
export ENABLE_DEGRADATION_TYPE_REWARD="${ENABLE_DEGRADATION_TYPE_REWARD:-False}"
export ENABLE_INTERMEDIATE_REWARD="${ENABLE_INTERMEDIATE_REWARD:-True}"
export IMAGE_QUALITY_USE_NO_REFERENCE="${IMAGE_QUALITY_USE_NO_REFERENCE:-False}"
export USE_PROGRESSIVE_FORMAT="${USE_PROGRESSIVE_FORMAT:-False}"

mkdir -p logs
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files="[${TRAIN_FILES}]" \
    data.val_files="[${VAL_FILES}]" \
    data.train_batch_size=32 \
    data.max_prompt_length=8192 \
    data.max_response_length=20480 \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
    actor_rollout_ref.rollout.agent.activate_agent=True \
    actor_rollout_ref.rollout.agent.tool_name_key=env_name \
    actor_rollout_ref.rollout.agent.single_response_max_tokens=10240 \
    actor_rollout_ref.rollout.agent.max_turns=4 \
    actor_rollout_ref.rollout.agent.concurrent_workers=4 \
    trainer.logger="['console']" \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.default_local_dir="${SAVE_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}" \
    trainer.total_epochs=10 2>&1 | tee "./logs/${EXPERIMENT_NAME}.log"
