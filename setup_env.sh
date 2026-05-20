#!/bin/bash
# TRUS 项目环境激活脚本

# 激活conda
source /opt/conda/etc/profile.d/conda.sh
conda activate base

# 设置代理（如果需要下载模型）
export HTTP_PROXY=http://127.0.0.1:12080
export HTTPS_PROXY=http://127.0.0.1:12080

# CUDA 环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MAX_PIXELS=112896

# 项目路径
export PROJECT_DIR=/root/private_data/trus-report-generation
cd $PROJECT_DIR

echo '========================================'
echo ' TRUS 前列腺癌报告生成环境'
echo '========================================'
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null)"
echo "CUDA:    $(python3 -c 'import torch; print(torch.version.cuda)' 2>/dev/null)"
echo "GPU:     $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null)"
echo "项目路径: $PROJECT_DIR"
echo "数据路径: /root/private_data/图生文/数据集/第23批数据"
echo '========================================'

# bitsandbytes CUDA 13 库路径
export LD_LIBRARY_PATH=/opt/conda/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
