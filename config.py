"""
TRUS前列腺癌报告生成 — 集中式路径与超参数配置
所有路径通过此模块统一管理，不硬编码在其他模块中。
"""

import os
from pathlib import Path

# ======================== 基础路径 ========================
BASE_DIR = Path(r"D:\全新科研\图生报告")
CODE_DIR = Path(r"D:\全新科研\图生报告\代码")
DATA_DIR = Path(r"E:\第23批数据")

# 处理后数据输出
PROCESSED_DIR = BASE_DIR / "processed"
IMAGE_DIR = PROCESSED_DIR / "images"
TRAIN_JSON_PATH = PROCESSED_DIR / "trus_train.json"
VAL_JSON_PATH = PROCESSED_DIR / "trus_val.json"
TEST_JSON_PATH = PROCESSED_DIR / "trus_test.json"
FULL_JSON_PATH = PROCESSED_DIR / "trus_all.json"

# 模型
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"

# ======================== 原始数据 ========================
EXCEL_PATH = DATA_DIR / "多模态数据整集表.xlsx"
BATCH1_DIR = DATA_DIR / "多模态数据整集-第1批"
BATCH2_DIR = DATA_DIR / "多模态数据整集-第2批"
BATCH3_DIR = DATA_DIR / "多模态数据整集-第3批"

BATCH_DIRS = {
    "batch1": BATCH1_DIR,
    "batch2": BATCH2_DIR,
    "batch3": BATCH3_DIR,
}

# ======================== Excel 列名映射 ========================
EXCEL_COLUMNS = {
    "影像号": "影像号",
    "姓名": "姓名",
    "门诊号": "门诊号",
    "住院号": "住院号",
    "手术结果": "手术结果",
    "穿刺结果1": "穿刺结果1",
    "穿刺结果2": "穿刺结果2",
    "经腹部超声报告": "经腹部超声影像报告",
    "trus_report": "经直肠超声影像报告",
    "mri_report": "MRI影像报告",
    "psa": "PSA值",
}

# ======================== 模型配置 ========================
# HuggingFace模型ID（首次运行自动下载，也可手动放到 MODEL_DIR 下）
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
# 若已手动下载到本地，设置此路径；否则设为 None 从 HuggingFace 下载
MODEL_LOCAL_PATH = None  # 例: Path("/mnt/d/全新科研/图生报告/models/Qwen2.5-VL-7B-Instruct")

# ======================== 训练超参数 (RTX 4090 24GB) ========================
TRAINING_CONFIG = {
    # 量化
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": "bfloat16",
    "bnb_4bit_use_double_quant": True,

    # LoRA (降低容量防过拟合)
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.2,
    "lora_target_modules": "all-linear",

    # 训练循环
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "num_train_epochs": 1,
    "learning_rate": 1e-4,
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine",

    # 精度与计算
    "torch_dtype": "bfloat16",
    "attn_implementation": "sdpa",
    "gradient_checkpointing": True,
    "tf32": True,  # Ampere+ 架构

    # 序列长度
    "max_length": 4096,
    "max_pixels": 401408,  # 634×634

    # 检查点 + 早停
    "save_steps": 50,
    "save_total_limit": 3,
    "eval_steps": 50,
    "logging_steps": 5,

    # Windows/WSL2 优化
    "dataloader_num_workers": 2,
    "dataloader_persistent_workers": False,
    "torch_empty_cache_steps": 50,

    # 损失与正则 (强力防过拟合)
    "neftune_noise_alpha": 10.0,
    "weight_decay": 0.05,
}

# ======================== 推理配置 ========================
INFERENCE_CONFIG = {
    "max_new_tokens": 512,
    "temperature": 0.1,
    "do_sample": True,
    "top_p": 0.9,
}

# ======================== CLOVER 架构配置 ========================
CLOVER_CONFIG = {
    # 损失权重 (适度降低辅助损失权重防过拟合)
    "lambda_cls": 0.2,              # 0.3 → 0.2
    "lambda_contrast": 0.1,         # 0.15 → 0.1

    # Warmup (延长warmup让模型先学好LM再引入辅助任务)
    "cls_warmup_steps": 80,         # 50 → 80
    "contrast_warmup_steps": 150,   # 100 → 150

    # 温度参数
    "temperature_init": 0.5,        # InfoNCE 初始温度
    "temperature_min": 0.1,         # InfoNCE 最低温度 (cosine 退火)

    # 防过拟合 (全面增强)
    "label_smoothing": 0.15,        # BCE 标签平滑 0.1 → 0.15
    "clinical_dropout": 0.2,        # ClinicalMLP/Heads dropout 0.1 → 0.2
    "feature_mask_prob": 0.2,       # 随机遮蔽临床特征 0.05 → 0.2

    # 维度
    "projection_dim": 256,          # 对比投影空间维度
    "clinical_mlp_hidden": 64,      # ClinicalMLP 隐藏层维度

    # Multi-Crop
    "local_crop_ratio": 0.6,        # 局部视图的中心裁剪比例
    "local_crop_size": 256,         # 局部视图 resize 大小

    # MoCo 对比队列 (增大队列提供更多负样本)
    "moco_queue_size": 64,          # 32 → 64
    "moco_momentum": 0.999,         # 队列动量更新系数
}

# ======================== 确保目录存在 ========================
for _dir in [PROCESSED_DIR, IMAGE_DIR, OUTPUT_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)
