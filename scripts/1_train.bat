@echo off
chcp 65001 >nul
echo ============================================
echo  TRUS 前列腺癌报告生成 - LoRA 训练
echo  模型: Qwen2.5-VL-7B + QLoRA (4bit NF4)
echo ============================================
cd /d "D:\全新科研\图生报告\代码"

REM 设置环境变量
set MAX_PIXELS=112896
set CUDA_VISIBLE_DEVICES=0
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m training.train_swift
if %ERRORLEVEL% NEQ 0 (
    echo [错误] 训练失败，请检查错误信息。
    pause
    exit /b 1
)
echo.
echo 训练完成！模型保存在 output\ 目录。
pause
