# TRUS 前列腺癌自动诊断报告生成

基于 EchoVLM 架构的经直肠超声（TRUS）前列腺癌医学报告自动生成系统。

> 基座模型: Qwen2.5-VL-7B + QLoRA | 框架: ms-swift | 适配 RTX 4090 24GB

## 项目结构

```
├── config.py                  # 集中配置（路径、模型、训练参数）
├── requirements.txt           # Python 依赖
├── data/                      # 数据处理模块
│   ├── dicom_utils.py         #   DICOM 读取/归一化
│   ├── dicom_to_png.py        #   批量 DICOM→PNG
│   ├── report_parser.py       #   TRUS 报告结构化解析
│   ├── pathology_labeler.py   #   穿刺病理标签提取
│   ├── build_training_json.py #   核心管线：构建训练数据集
│   └── split_dataset.py       #   分层划分 train/val/test
├── training/                  # 训练模块
│   ├── dataset_register.py    #   ms-swift 数据集注册
│   └── train_swift.py         #   训练入口
├── evaluation/
│   └── eval_metrics.py        # NLG + 结构化字段 + 分类评估
├── inference/
│   └── inference.py           # 推理 + Gradio WebUI
├── utils/
│   └── helpers.py             # 工具函数
└── scripts/                   # 一键脚本
    ├── 0_prepare_data.bat
    ├── 1_train.bat
    ├── 2_eval.bat
    └── 3_inference.bat
```

## 环境要求

- **操作系统**: WSL2 Ubuntu（推荐）或 Linux
- **GPU**: NVIDIA RTX 4090 24GB 或同等显存
- **Python**: 3.10+
- **CUDA**: 12.1+

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置路径

编辑 `config.py` 中的路径：

```python
DATA_DIR = Path("/mnt/e/第23批数据")          # 原始数据目录
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"     # HuggingFace 模型
MODEL_LOCAL_PATH = None                         # 或指定本地路径
```

### 3. 准备数据

```bash
# 步骤1：DICOM 转 PNG
python -m data.dicom_to_png

# 步骤2：构建训练数据集
python -m data.build_training_json
```

输出：`processed/trus_train.json`, `processed/trus_val.json`, `processed/trus_test.json`

### 4. 训练

```bash
python -m training.train_swift
```

训练参数在 `config.py` 的 `TRAINING_CONFIG` 中配置，默认针对 RTX 4090 24GB 优化。

### 5. 评估

```bash
# 先批量推理
python -m inference.inference --mode batch

# 再评估
python -m evaluation.eval_metrics
```

### 6. 推理

```bash
# 单张图像
python -m inference.inference --mode single --image /path/to/image.png

# Web 演示
python -m inference.inference --mode webui
```

## 训练策略

| 参数 | 值 |
|------|-----|
| 基座模型 | Qwen2.5-VL-7B-Instruct |
| 量化 | 4-bit NF4 (bitsandbytes) |
| 微调 | LoRA rank=8, alpha=16 |
| 有效批大小 | 8 (1×8 梯度累积) |
| 学习率 | 2e-4, cosine 衰减 |
| 最大序列长度 | 4096 |
| 图像分辨率 | 336×336 (112896 pixels) |
| 预估显存 | ~10-13 GB / 24 GB |

## 数据格式

训练 JSON 格式（ms-swift 兼容）：

```json
{
  "id": "患者名_图像标识",
  "images": ["images/batch1_张三_300954508_01.png"],
  "messages": [
    {"role": "system", "content": "你是一名专注于TRUS的医学顾问..."},
    {"role": "user", "content": "<image>请根据这张TRUS图像生成前列腺超声诊断报告..."},
    {"role": "assistant", "content": "前列腺（经直肠）：左右径56mm..."}
  ],
  "metadata": {
    "patient_name": "张三",
    "psa": 20.9,
    "has_cancer": false,
    "gleason_score": null,
    "isup_grade": null
  }
}
```

## 参考文献

本代码基于以下工作：

- **EchoVLM**: [Dynamic Mixture-of-Experts Vision-Language Model for Universal Ultrasound Intelligence](https://arxiv.org/abs/2509.14977)
- **Qwen2.5-VL**: [Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923)
- **ms-swift**: [Scalable Lightweight Infrastructure for Fine-Tuning](https://github.com/modelscope/ms-swift)
