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

# TRUS报告生成提示词模板 — 分4类，每类多种表述，强制模型关注图像内容
REPORT_PROMPTS = {
    # 完整报告（基础）
    "full": [
        "请输出这张经直肠超声（TRUS）图像的前列腺超声所见与诊断：",
        "请写出这张TRUS图像对应的前列腺超声报告正文：",
        "请直接输出以下TRUS图像的超声诊断报告内容：",
    ],
    # 强调尺寸测量
    "measure": [
        "请输出这张TRUS图像的前列腺超声报告，务必精确写出前列腺的左右径、前后径和上下径尺寸：",
        "根据这张超声图像，写出前列腺尺寸（三径）及超声所见：",
        "请从这张TRUS图像中读取前列腺的具体测量值，并写出完整报告：",
    ],
    # 强调回声和形态特征
    "texture": [
        "请描述这张TRUS图像中前列腺的内部回声特征和形态，写出超声所见与诊断：",
        "根据这张图像，详细描述前列腺的回声分布、有无钙化、包膜完整性，然后写出诊断：",
        "请输出这张前列腺超声图像的所见描述（重点关注回声和形态特征）及诊断结论：",
    ],
    # 强调诊断结论
    "diagnosis": [
        "请根据这张TRUS图像写出前列腺超声报告，特别关注最终的诊断结论和临床建议：",
        "请生成这张前列腺超声图像的检查所见，并在末尾给出明确的超声诊断印象：",
        "输出这张TRUS图像的报告，确保诊断结论准确反映图像中的异常发现：",
    ],
    # 简单直接
    "simple": [
        "请输出这张TRUS图像的超声所见与诊断。",
        "写出这张前列腺超声报告的正文。",
        "输出以下图像的TRUS报告：",
    ],
}

SYSTEM_PROMPT = "你是前列腺超声报告系统的文本输出模块。你只输出超声所见与诊断的正文内容，不加任何标题、前缀、解释或格式标记，直接输出报告原文。"


def _pick_prompts(image_count: int, samples_per_image: int = 5) -> List[str]:
    """为每张图像生成多个不同角度的prompt，通过prompt多样性迫使模型关注图像内容。"""
    import random
    random.seed(42)
    prompts = []
    categories = list(REPORT_PROMPTS.keys())
    # 确保每张图至少覆盖4个类别 + 1个随机
    for _ in range(samples_per_image):
        cat = random.choice(categories)
        prompts.append(random.choice(REPORT_PROMPTS[cat]))
    return prompts


# 诊断结论的多种自然变体（用于数据增强）
DIAGNOSIS_VARIANTS = [
    "前列腺增大伴钙化 前列腺穿刺活检，待病理",
    "前列腺增大伴钙化灶 已行穿刺活检，等待病理结果",
    "前列腺增生伴钙化 前列腺穿刺术后，病理待回报",
    "前列腺增大，内见钙化 已行前列腺穿刺活检，待病理回报",
    "前列腺增生并钙化 穿刺活检已做，待病理结果",
    "前列腺增大伴钙化 建议结合病理结果综合评估",
    "前列腺增生伴钙化灶形成 穿刺活检后待病理确认",
]

def _augment_report_text(report: str, pathology_info: dict) -> List[str]:
    """为同一份报告生成多个措辞变体，增加训练数据文本多样性。"""
    import random
    random.seed(42)
    variants = [report]  # 原始版本

    # 1. 诊断结论变体：替换最后一句为随机变体
    endings = ["。", "；", ";"]
    last_end = -1
    for end_char in endings:
        pos = report.rfind(end_char)
        if pos > last_end:
            last_end = pos
    if last_end > len(report) * 0.6:  # 最后一句在后60%位置
        base = report[:last_end]
        for _ in range(2):  # 生成2个诊断变体
            new_ending = random.choice(DIAGNOSIS_VARIANTS)
            if new_ending != report[last_end+1:].strip():
                variants.append(base + "。" + new_ending)

    # 2. 癌症病例：增加包含病理发现的版本
    if pathology_info.get("has_cancer"):
        gleason = pathology_info.get("gleason_score", "")
        isup = pathology_info.get("isup_grade", "")
        if gleason:
            path_add = f"。病理补充：前列腺腺泡腺癌，Gleason评分{gleason}"
            if isup:
                path_add += f"，ISUP分级{isup}级组"
            variants.append(report + path_add)
    else:
        # 良性病例偶尔也加病理确认
        if random.random() < 0.3:
            variants.append(report + "。病理补充：良性前列腺组织")

    return variants[:4]  # 最多4个变体


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

            prompts = _pick_prompts(len(images), samples_per_image=5)
            report_variants = _augment_report_text(trus_report_text, pathology_info)

            for pi, prompt in enumerate(prompts):
                # 每个prompt配不同的report变体
                variant = report_variants[pi % len(report_variants)]
                sample = {
                    "id": f"{name}_{img_path.stem}_p{pi}",
                    "images": [str(img_path.relative_to(PROCESSED_DIR))],
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"<image>{prompt}"},
                        {"role": "assistant", "content": variant},
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
