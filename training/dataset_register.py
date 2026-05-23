"""
向 ms-swift 注册 TRUS 数据集。
基于 EchoVLM swift_part/my_register.py 的精简适配版本。
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import TRAIN_JSON_PATH, VAL_JSON_PATH, TEST_JSON_PATH, PROCESSED_DIR

# ms-swift 注册数据集
from swift.dataset import DatasetMeta, register_dataset
from swift.dataset.preprocessor.core import MessagesPreprocessor


class TRUSPreprocessor(MessagesPreprocessor):
    """TRUS 数据集预处理器：确保图像路径正确、处理系统提示。"""

    def preprocess(self, row):
        images = row.get("images", [])
        if images:
            resolved = []
            for img_path in images:
                img_path = str(img_path)
                if not os.path.isabs(img_path):
                    img_path = os.path.join(str(PROCESSED_DIR), img_path)
                resolved.append(img_path)
            row["images"] = resolved

        messages = row.get("messages", [])
        has_system = any(msg.get("role") == "system" for msg in messages)
        if not has_system:
            messages.insert(0, {
                "role": "system",
                "content": "你是一名专注于经直肠超声（TRUS）的医学顾问，擅长前列腺超声影像分析与结构化报告生成。请用专业、客观的语言进行描述。"
            })
            row["messages"] = messages

        # --- CLOVER: 将临床特征从 metadata 提升到顶层字段 ---
        # 配合 remove_unused_columns=False，这些字段会被保留在训练 batch 中
        metadata = row.get("metadata", {})
        for key in ["has_cancer", "psa", "volume"]:
            row[key] = metadata.get(key)

        return super().preprocess(row)


# 注册训练集
register_dataset(DatasetMeta(
    dataset_path=str(TRAIN_JSON_PATH),
    dataset_name="trus_train",
    preprocess_func=TRUSPreprocessor(),
))

# 注册验证集
register_dataset(DatasetMeta(
    dataset_path=str(VAL_JSON_PATH),
    dataset_name="trus_val",
    preprocess_func=TRUSPreprocessor(),
))

# 注册测试集
register_dataset(DatasetMeta(
    dataset_path=str(TEST_JSON_PATH),
    dataset_name="trus_test",
    preprocess_func=TRUSPreprocessor(),
))
