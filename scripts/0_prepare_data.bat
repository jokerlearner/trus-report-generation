@echo off
chcp 65001 >nul
echo ============================================
echo  TRUS 数据预处理 - 第1步：DICOM转PNG
echo ============================================
cd /d "D:\全新科研\图生报告\代码"
python -m data.dicom_to_png
if %ERRORLEVEL% NEQ 0 (
    echo [错误] DICOM转换失败，请检查错误信息。
    pause
    exit /b 1
)
echo.
echo ============================================
echo  TRUS 数据预处理 - 第2步：构建训练JSON
echo ============================================
python -m data.build_training_json
if %ERRORLEVEL% NEQ 0 (
    echo [错误] 训练数据构建失败，请检查错误信息。
    pause
    exit /b 1
)
echo.
echo ============================================
echo  数据预处理完成！
echo  训练数据: processed\trus_train.json
echo  验证数据: processed\trus_val.json
echo  测试数据: processed\trus_test.json
echo ============================================
pause
