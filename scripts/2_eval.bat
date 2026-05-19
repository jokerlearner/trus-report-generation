@echo off
chcp 65001 >nul
echo ============================================
echo  TRUS 报告生成模型评估
echo ============================================
cd /d "D:\全新科研\图生报告\代码"
python -m evaluation.eval_metrics
pause
