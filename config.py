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
    "trus_report": "经直肠超声影像报告(TRUS)",
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

    # LoRA
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "lora_target_modules": "all-linear",

    # 训练循环
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "num_train_epochs": 3,
    "learning_rate": 2e-4,
    "warmup_ratio": 0.05,
    "lr_scheduler_type": "cosine",

    # 精度与计算
    "torch_dtype": "bfloat16",
    "attn_implementation": "flash_attention_2",
    "gradient_checkpointing": True,
    "tf32": True,  # Ampere+ 架构

    # 序列长度
    "max_length": 4096,
    "max_pixels": 112896,  # 336×336

    # 检查点
    "save_steps": 100,
    "save_total_limit": 3,
    "eval_steps": 50,
    "logging_steps": 5,

    # Windows/WSL2 优化
    "dataloader_num_workers": 2,
    "dataloader_persistent_workers": False,
    "torch_empty_cache_steps": 50,

    # 损失与正则
    "neftune_noise_alpha": 5.0,
    "weight_decay": 0.01,
}

# ======================== 推理配置 ========================
INFERENCE_CONFIG = {
    "max_new_tokens": 512,
    "temperature": 0.1,
    "do_sample": True,
    "top_p": 0.9,
}

# ======================== 确保目录存在 ========================
for _dir in [PROCESSED_DIR, IMAGE_DIR, OUTPUT_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)
