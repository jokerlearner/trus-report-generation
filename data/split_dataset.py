"""
按患者分层划分 train/val/test，确保同患者的所有图像在同一集合。
基于癌症标签分层抽样。
"""

import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def group_by_patient(samples: List[Dict]) -> Dict[str, List[Dict]]:
    """将样本按患者名分组。"""
    groups = defaultdict(list)
    for s in samples:
        patient_name = s.get("metadata", {}).get("patient_name", s.get("id", ""))
        groups[patient_name].append(s)
    return dict(groups)


def stratified_split(
    patient_groups: Dict[str, List[Dict]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[List[Dict], List[Dict], List[Dict]]:
    """
    按患者级分层抽样划分数据集。
    优先按 has_cancer 标签分层，无标签的患者按无标签处理。
    """
    random.seed(seed)
    np.random.seed(seed)

    cancer_patients = []
    benign_patients = []
    unknown_patients = []

    for pname, samples in patient_groups.items():
        has_cancer = None
        for s in samples:
            hc = s.get("metadata", {}).get("has_cancer")
            if hc is not None:
                has_cancer = hc
                break
        if has_cancer is True:
            cancer_patients.append(pname)
        elif has_cancer is False:
            benign_patients.append(pname)
        else:
            unknown_patients.append(pname)

    random.shuffle(cancer_patients)
    random.shuffle(benign_patients)
    random.shuffle(unknown_patients)

    def _split_patients(patients, tr, vr):
        n = len(patients)
        n_train = max(1, int(n * tr))
        n_val = max(1, int(n * vr))
        n_train = min(n_train, n - 2) if n > 2 else n_train
        n_val = min(n_val, n - n_train - 1) if n - n_train > 1 else n_val
        train = patients[:n_train]
        val = patients[n_train:n_train + n_val]
        test = patients[n_train + n_val:]
        return train, val, test

    c_train, c_val, c_test = _split_patients(cancer_patients, train_ratio, val_ratio)
    b_train, b_val, b_test = _split_patients(benign_patients, train_ratio, val_ratio)
    u_train, u_val, u_test = _split_patients(unknown_patients, train_ratio, val_ratio)

    train_patients = set(c_train + b_train + u_train)
    val_patients = set(c_val + b_val + u_val)
    test_patients = set(c_test + b_test + u_test)

    train_data = []
    val_data = []
    test_data = []

    for pname, samples in patient_groups.items():
        if pname in train_patients:
            train_data.extend(samples)
        elif pname in val_patients:
            val_data.extend(samples)
        elif pname in test_patients:
            test_data.extend(samples)

    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    return train_data, val_data, test_data


def split_and_save(
    samples: List[Dict],
    train_path: Optional[Path] = None,
    val_path: Optional[Path] = None,
    test_path: Optional[Path] = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
):
    """完整划分+保存。"""
    if not samples:
        logger.warning("无样本可划分")
        return

    patient_groups = group_by_patient(samples)
    logger.info("患者总数: %d, 样本总数: %d", len(patient_groups), len(samples))

    train_data, val_data, test_data = stratified_split(
        patient_groups, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed
    )

    logger.info("划分结果 — Train: %d (%d患者), Val: %d (%d患者), Test: %d (%d患者)",
                len(train_data), len(set(g["metadata"]["patient_name"] for g in train_data)),
                len(val_data), len(set(g["metadata"]["patient_name"] for g in val_data)),
                len(test_data), len(set(g["metadata"]["patient_name"] for g in test_data)))

    for path, data in [(train_path, train_data), (val_path, val_data), (test_path, test_data)]:
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info("已保存: %s (%d 条)", path, len(data))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import FULL_JSON_PATH, TRAIN_JSON_PATH, VAL_JSON_PATH, TEST_JSON_PATH

    if not FULL_JSON_PATH.exists():
        print(f"未找到 {FULL_JSON_PATH}，请先运行 build_training_json.py")
    else:
        with open(FULL_JSON_PATH, "r", encoding="utf-8") as f:
            all_samples = json.load(f)
        split_and_save(all_samples, TRAIN_JSON_PATH, VAL_JSON_PATH, TEST_JSON_PATH)
