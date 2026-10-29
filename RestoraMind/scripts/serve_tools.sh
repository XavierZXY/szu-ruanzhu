#!/bin/bash
# 按模型家族启动复原工具服务，端口与 restoramind.tools.registry.FAMILY_PORTS 一致。
# 默认使用经典算子后端；接入真实专家模型时设置 BACKEND=your_package.module:function。
set -euo pipefail
BACKEND="${BACKEND:-classical}"
HOST="${HOST:-0.0.0.0}"
declare -A PORTS=( [SwinIR]=5001 [DehazeFormer]=5002 [DRBNet]=5003 [MPRNet]=5004 [FBCNN]=5005 [Restormer]=5006
                   [XRestormer]=5007 [SCUNet]=5008 [Retinexformer]=5009 [HAT]=5010 [NeRD]=5011 [NAFNet]=5012 [DiffBIR]=5013 )
mkdir -p logs
for family in "${!PORTS[@]}"; do
  python3 -m restoramind serve-tools --family "$family" --backend "$BACKEND" --host "$HOST" --port "${PORTS[$family]}" \
      > "logs/tool_${family}.log" 2>&1 &
  echo "started $family on port ${PORTS[$family]} (pid $!)"
done
wait
