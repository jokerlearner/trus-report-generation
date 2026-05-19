@echo off
chcp 65001 >nul
echo ============================================
echo  TRUS 报告生成 - 推理演示
echo ============================================
cd /d "D:\全新科研\图生报告\代码"
python -m inference.inference
pause
