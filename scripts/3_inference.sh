#!/bin/bash
set -e
PROJECT_DIR="/root/private_data/trus-report-generation"
cd "$PROJECT_DIR"

echo "============================================"
echo " TRUS 报告生成 - 推理演示"
echo "============================================"
/opt/conda/bin/python3 -m inference.inference
