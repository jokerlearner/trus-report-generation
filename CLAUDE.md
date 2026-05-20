# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

基于 EchoVLM 架构的 TRUS（经直肠超声）前列腺癌医学报告自动生成系统。基座模型 Qwen2.5-VL-7B + QLoRA (4-bit NF4)，训练框架 ms-swift 4.2.1，适配 RTX 4090 24GB。

## 双环境架构

项目在本地和远程两套环境中并行运行，`config.py` 路径需要区分：

| | 本地 (Windows) | 远程 (Linux GPU 服务器) |
|---|---|---|
| 项目路径 | `D:\全新科研\图生报告\代码` | `/root/private_data/trus-report-generation/` |
| 数据路径 | `E:\第23批数据` | `/root/private_data/图生文/数据集/第23批数据/` |
| SSH 连接 | — | `ssh ksai.scnet.cn` (端口 50475, 免密) |
| conda/Python | — | `/opt/conda/bin/python3` |

**远程服务器无法访问外网**，安装包或下载模型时需先建立 SSH 反向隧道：
```bash
ssh -f -N -R 12080:127.0.0.1:7890 -o ServerAliveInterval=10 ksai.scnet.cn
# 远程命令前加: HTTP_PROXY=http://127.0.0.1:12080 HTTPS_PROXY=http://127.0.0.1:12080
```

远程修改代码后必须同步回本地再 commit 和 push 到 GitHub（`git@github.com:jokerlearner/trus-report-generation.git`）。

## 常用命令

```bash
# 数据预处理（DICOM→PNG → 构建训练JSON）
python -m data.dicom_to_png
python -m data.build_training_json

# 训练
python -m training.train_swift

# 推理
python -m inference.inference --mode single --image <path>   # 单张
python -m inference.inference --mode batch                   # 批量（测试集）
python -m inference.inference --mode webui                   # Gradio WebUI

# 评估（需先跑 batch 推理生成 test_predictions.json）
python -m evaluation.eval_metrics

# 远程一键脚本（在远程服务器上）
bash scripts/0_prepare_data.sh   # 数据预处理
bash scripts/1_train.sh          # 训练
bash scripts/2_eval.sh           # 评估
bash scripts/3_inference.sh      # 推理
```

## 架构

### 数据管线 (`data/`)
```
Excel + DICOM文件
  → dicom_to_png.py          : DICOM→PNG + 患者名匹配，输出 image_mapping.json
  → build_training_json.py   : Excel元数据 + TRUS报告 + 病理标签 → ms-swift兼容JSON
  → split_dataset.py         : 分层划分 train/val/test
```
关键依赖：`report_parser.py`（正则提取结构化字段）、`pathology_labeler.py`（Gleason/ISUP分级）

### 训练模块 (`training/`)
- `dataset_register.py`：向 ms-swift 注册本地 JSON 数据集（侧效应导入，由 train_swift.py import 触发）
- `train_swift.py`：从 `config.TRAINING_CONFIG` 读取参数，构建 `SftArguments`，调用 `sft_main()`

### 推理 (`inference/`)
- `TRUSReportGenerator` 类封装模型加载（QLoRA + 4-bit）和推理
- 三种模式：single / batch / webui（Gradio）

### 评估 (`evaluation/`)
- NLG 指标：BLEU-1~4、ROUGE-L、BERTScore（METEOR 需 wordnet，不可用时跳过）
- 结构化字段准确率：形态/回声/钙化/血流/包膜
- 癌症诊断分类：灵敏度/特异度/AUC

### 路径管理
所有路径集中在 `config.py`，其他模块通过 `sys.path.insert(0, ...)` + `from config import ...` 引用，禁止硬编码路径。

## 重要注意事项

- **ms-swift 4.2.1 API 已变更**：`from swift import sft_main, SftArguments`（非旧版 `swift.llm`），数据集注册用 `dataset_path`（非 `ms_dataset_id`），训练类型用 `tuner_type`（非 `train_type`），需显式设置 `quant_bits=4`
- **flash-attn 不可用**：编译耗时过长，改用 `sdpa`（config.py 和 inference.py 均已设置）。`xformers` 已安装作为备选但 C++ 扩展版本可能不匹配
- **PSA 值解析**：Excel 中 PSA 是多行文本（含单位），`build_training_json.py` 用正则 `总前列腺特异性抗原\s*([\d.]+)` 提取首个 TPSA 数值
- **Excel 列名**：TRUS 报告列名为 `经直肠超声影像报告`（无多余后缀），对应 `config.EXCEL_COLUMNS["trus_report"]`
- **Windows 路径限制**：`flash-attn` 源码包文件名超过 260 字符，无法在 Windows 上下载后 SCP，需在远程直接编译或用替代方案
