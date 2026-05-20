"""
TRUS 前列腺癌报告生成 — ms-swift LoRA 训练入口。
基于 EchoVLM swift_part/ms-swift_train_sft.py 精简适配。
适配 RTX 4090 24GB：Qwen2.5-VL-7B + QLoRA (4-bit NF4)。
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    MODEL_NAME, MODEL_LOCAL_PATH, OUTPUT_DIR, TRAINING_CONFIG, PROCESSED_DIR
)

# 注册 TRUS 数据集到 ms-swift
import training.dataset_register  # noqa: F401 — dataset registration side-effect

from swift import sft_main, SftArguments


def build_args():
    cfg = TRAINING_CONFIG
    model_path = str(MODEL_LOCAL_PATH) if MODEL_LOCAL_PATH else MODEL_NAME

    return SftArguments(
        # ==================== 模型 ====================
        model=model_path,
        model_type="qwen2_5_vl",
        template="qwen2_5_vl",

        # ==================== 数据集 ====================
        dataset=["trus_train"],
        val_dataset=["trus_val"],
        dataset_num_proc=4,
        load_from_cache_file=True,

        # ==================== 训练类型 ====================
        tuner_type="lora",
        lora_rank=cfg["lora_rank"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["lora_target_modules"],
        freeze_vit=False,
        freeze_llm=False,
        freeze_aligner=False,

        # ==================== 量化 ====================
        quant_method="bnb",
        quant_bits=4,
        bnb_4bit_compute_dtype=cfg["bnb_4bit_compute_dtype"],
        bnb_4bit_quant_type=cfg["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=cfg["bnb_4bit_use_double_quant"],
                # ==================== 训练循环 ====================
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        weight_decay=cfg["weight_decay"],

        # ==================== 精度与注意力 ====================
        torch_dtype=cfg["torch_dtype"],
        attn_impl=cfg["attn_implementation"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
        bf16=True if cfg["torch_dtype"] == "bfloat16" else False,
        fp16=True if cfg["torch_dtype"] == "float16" else False,

        # ==================== 序列长度 ====================
        max_length=cfg["max_length"],
        max_pixels=cfg["max_pixels"],
        truncation_strategy="right",

        # ==================== 检查点 ====================
        save_steps=cfg["save_steps"],
        save_total_limit=cfg["save_total_limit"],
        eval_steps=cfg["eval_steps"],
        logging_steps=cfg["logging_steps"],
        save_only_model=True,

        # ==================== 输出 ====================
        output_dir=str(OUTPUT_DIR / "trus_lora"),
        report_to="tensorboard",

        # ==================== NeFTune ====================
        neftune_noise_alpha=cfg["neftune_noise_alpha"],

        # ==================== 数据加载 ====================
        dataloader_num_workers=cfg["dataloader_num_workers"],
        dataloader_persistent_workers=cfg["dataloader_persistent_workers"],
        dataloader_drop_last=True,
        remove_unused_columns=False,
        torch_empty_cache_steps=cfg["torch_empty_cache_steps"],

        # ==================== 杂项 ====================
        seed=42,
        data_seed=42,
        do_train=True,
        do_eval=True,
        eval_strategy="steps",
        save_strategy="steps",
        ddp_timeout=1800,
    )


def main():
    args = build_args()
    return sft_main(args)


if __name__ == "__main__":
    os.environ.setdefault("MAX_PIXELS", str(TRAINING_CONFIG["max_pixels"]))
    os.environ.setdefault("ROOT_IMAGE_DIR", str(PROCESSED_DIR))
    main()
