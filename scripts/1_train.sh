#!/bin/bash
set -e
PROJECT_DIR="/root/private_data/trus-report-generation"
cd "$PROJECT_DIR"

echo "============================================"
echo " TRUS 前列腺癌报告生成 - LoRA 训练"
echo " 模型: Qwen2.5-VL-7B + QLoRA (4bit NF4)"
echo "============================================"

export MAX_PIXELS=112896
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH=/opt/conda/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH

/opt/conda/bin/python3 -m training.train_swift
if [ $? -ne 0 ]; then
    echo "[错误] 训练失败，请检查错误信息。"
    exit 1
fi

echo ""
echo "训练完成！模型保存在 output/ 目录。"
