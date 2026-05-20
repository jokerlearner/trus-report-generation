#!/bin/bash
set -e
PROJECT_DIR="/root/private_data/trus-report-generation"
cd "$PROJECT_DIR"

echo "============================================"
echo " TRUS 数据预处理 - 第1步：DICOM转PNG"
echo "============================================"
python3 -m data.dicom_to_png
if [ $? -ne 0 ]; then
    echo "[错误] DICOM转换失败，请检查错误信息。"
    exit 1
fi

echo ""
echo "============================================"
echo " TRUS 数据预处理 - 第2步：构建训练JSON"
echo "============================================"
python3 -m data.build_training_json
if [ $? -ne 0 ]; then
    echo "[错误] 训练数据构建失败，请检查错误信息。"
    exit 1
fi

echo ""
echo "============================================"
echo " 数据预处理完成！"
echo " 训练数据: processed/trus_train.json"
echo " 验证数据: processed/trus_val.json"
echo " 测试数据: processed/trus_test.json"
echo "============================================"
