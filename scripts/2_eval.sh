#!/bin/bash
set -e
PROJECT_DIR="/root/private_data/trus-report-generation"
cd "$PROJECT_DIR"

echo "============================================"
echo " TRUS 报告生成模型评估"
echo "============================================"
/opt/conda/bin/python3 -m evaluation.eval_metrics
