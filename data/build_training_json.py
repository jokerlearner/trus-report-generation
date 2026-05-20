"""
构建 TRUS 训练 JSON 数据集。
整合 Excel 元数据 + DICOM 映射 + TRUS报告 + 病理标签 → ms-swift 兼容的 JSON。
"""

import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    EXCEL_PATH, PROCESSED_DIR, IMAGE_DIR, FULL_JSON_PATH, EXCEL_COLUMNS,
    TRAIN_JSON_PATH, VAL_JSON_PATH, TEST_JSON_PATH,
)
from data.report_parser import parse_trus_report, clean_report_text
from data.pathology_labeler import parse_pathology

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ======================== 患者匹配 ========================

def _normalize_name(name: str) -> str:
    """清洗姓名：去空格、括号、多余字符。"""
    if not name or not isinstance(name, str):
        return ""
    name = re.sub(r"[（(].*?[）)]", "", name)
    name = re.sub(r"\s+", "", name)
    return name.strip()


def _extract_chinese_name(dirname: str) -> str:
    """从目录名提取中文姓名部分（去掉数字、拼音后缀）。"""
    chars = []
    for ch in dirname:
        if '一' <= ch <= '鿿':
            chars.append(ch)
    return ''.join(chars) if chars else dirname


def match_patients_to_images(excel_df: pd.DataFrame) -> Dict[str, List[Path]]:
    """
    将 Excel 患者与 DICOM 图像匹配。
    返回: {excel_patient_name: [image_path_list]}
    """
    mapping_path = PROCESSED_DIR / "image_mapping.json"
    if mapping_path.exists():
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
    else:
        logger.warning("image_mapping.json 不存在，请先运行 dicom_to_png.py")
        return {}

    img_by_patient = defaultdict(list)
    for out_name, info in mapping.items():
        img_by_patient[info["patient_name"]].append(IMAGE_DIR / out_name)

    excel_names = set(_normalize_name(n) for n in excel_df[EXCEL_COLUMNS["姓名"]].dropna())

    result: Dict[str, List[Path]] = defaultdict(list)
    unmatched = set()
    matched_count = 0

    for ename in excel_names:
        if not ename:
            continue
        found = False
        # 精确匹配
        mapping_names = set(img_by_patient.keys())
        if ename in mapping_names:
            result[ename] = img_by_patient[ename]
            matched_count += 1
            found = True
        else:
            # 模糊匹配：Excel名包含于mapping名 或 反之
            for mname in mapping_names:
                if ename in mname or mname in ename:
                    result[ename].extend(img_by_patient[mname])
                    found = True
            if found:
                matched_count += 1
            else:
                unmatched.add(ename)

    logger.info("Excel患者: %d, DICOM目录名: %d", len(excel_names), len(img_by_patient))
    logger.info("匹配成功: %d (精确 %d, 模糊 %d)", matched_count, matched_count - len(unmatched), matched_count)

    if unmatched:
        logger.warning("未匹配患者 (%d): %s", len(unmatched), ", ".join(sorted(unmatched)[:30]))
        unmatched_path = PROCESSED_DIR / "unmatched_patients.txt"
        with open(unmatched_path, "w", encoding="utf-8") as f:
            f.writelines(f"{name}\n" for name in sorted(unmatched))
        logger.info("未匹配清单已保存至: %s", unmatched_path)

    return result


# ======================== 训练数据构建 ========================

# TRUS报告生成提示词模板
REPORT_PROMPTS = [
    "请根据这张经直肠超声（TRUS）图像，生成一份结构化的前列腺超声诊断报告，包括前列腺尺寸、形态、包膜完整性、内部回声特征、CDFI血流信号及超声诊断印象。",
    "作为超声科医生，请为这张TRUS图像撰写一份完整的前列腺超声报告，涵盖所见与诊断。",
    "这是一张经直肠超声图像，请用中文写出结构化的前列腺超声报告（尺寸、形态、回声、血流、诊断）。",
    "请对这张TRUS超声图像进行专业解读，生成前列腺超声诊断报告。",
    "假设你是超声科医师，请根据此图像撰写前列腺超声所见与诊断。",
]

SYSTEM_PROMPT = "你是一名专注于经直肠超声（TRUS）的医学顾问，擅长前列腺超声影像分析与结构化报告生成。请用专业、客观的语言进行描述。"


def _pick_prompt(image_count: int) -> str:
    """轮换提示词模板，增加多样性。"""
    import random
    random.seed(42)
    return random.choice(REPORT_PROMPTS)


def build_training_data(
    excel_df: pd.DataFrame,
    patient_images: Dict[str, List[Path]],
    output_path: Optional[Path] = None,
) -> List[Dict]:
    """
    主函数：构建完整的训练 JSON 数据集。
    """
    all_samples = []
    stats = {"total_patients": 0, "with_report": 0, "with_pathology": 0, "with_cancer": 0, "total_images": 0}

    for _, row in excel_df.iterrows():
        name = _normalize_name(str(row.get(EXCEL_COLUMNS["姓名"], "")))
        if not name or name not in patient_images:
            continue

        stats["total_patients"] += 1
        images = patient_images[name]
        if not images:
            continue

        trus_report_text = row.get(EXCEL_COLUMNS["trus_report"], "")
        if pd.isna(trus_report_text) or not str(trus_report_text).strip():
            continue

        stats["with_report"] += 1
        trus_report_text = clean_report_text(str(trus_report_text))

        pathology_text = row.get(EXCEL_COLUMNS["穿刺结果1"], "")
        pathology_info = parse_pathology(str(pathology_text)) if not pd.isna(pathology_text) else {}

        if pathology_info.get("has_cancer") is not None:
            stats["with_pathology"] += 1
        if pathology_info.get("has_cancer"):
            stats["with_cancer"] += 1

        psa_val = row.get(EXCEL_COLUMNS["psa"], None)
        psa = None
        if pd.notna(psa_val):
            psa_str = str(psa_val)
            # Extract first TPSA value: "总前列腺特异性抗原 20.929 ↑ng/mL"
            tpsa_match = re.search(r"总前列腺特异性抗原\s*([\d.]+)", psa_str)
            if tpsa_match:
                try:
                    psa = float(tpsa_match.group(1))
                except ValueError:
                    pass

        structured = parse_trus_report(trus_report_text)

        patient_id = str(row.get(EXCEL_COLUMNS["影像号"], ""))

        for img_path in sorted(images, key=lambda p: p.stem):
            stats["total_images"] += 1

            prompt = _pick_prompt(len(images))

            sample = {
                "id": f"{name}_{img_path.stem}",
                "images": [str(img_path.relative_to(PROCESSED_DIR))],
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"<image>{prompt}"},
                    {"role": "assistant", "content": trus_report_text},
                ],
                "metadata": {
                    "patient_name": name,
                    "patient_id": patient_id,
                    "psa": psa,
                    "has_cancer": pathology_info.get("has_cancer"),
                    "gleason_score": pathology_info.get("gleason_score"),
                    "isup_grade": pathology_info.get("isup_grade"),
                    "dimensions": structured.get("dimensions"),
                    "source_batch": "unknown",
                },
            }
            all_samples.append(sample)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_samples, f, ensure_ascii=False, indent=2)
        logger.info("训练JSON已保存: %s (%d 条)", output_path, len(all_samples))

    logger.info("数据统计: 患者总数=%d, 有报告=%d, 有病理=%d, 癌症=%d, 总样本=%d",
                stats["total_patients"], stats["with_report"], stats["with_pathology"],
                stats["with_cancer"], stats["total_images"])

    if stats["with_cancer"] > 0:
        cancer_ratio = stats["with_cancer"] / stats["with_pathology"] * 100 if stats["with_pathology"] > 0 else 0
        logger.info("癌症占比: %.1f%% (%d/%d)", cancer_ratio, stats["with_cancer"], stats["with_pathology"])

    return all_samples


def run():
    """完整数据管线入口。"""
    if not EXCEL_PATH.exists():
        logger.error("Excel文件不存在: %s", EXCEL_PATH)
        return

    excel_df = pd.read_excel(EXCEL_PATH, dtype=str)
    logger.info("Excel加载完成: %d 行, %d 列", len(excel_df), len(excel_df.columns))

    patient_images = match_patients_to_images(excel_df)

    all_samples = build_training_data(excel_df, patient_images, output_path=FULL_JSON_PATH)

    from data.split_dataset import split_and_save
    split_and_save(all_samples, TRAIN_JSON_PATH, VAL_JSON_PATH, TEST_JSON_PATH)

    return all_samples


if __name__ == "__main__":
    run()
