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
    TRAIN_JSON_PATH, VAL_JSON_PATH, TEST_JSON_PATH, CLOVER_CONFIG,
)
from data.report_parser import parse_trus_report, clean_report_text
from data.pathology_labeler import parse_pathology
from data.crop_utils import generate_local_crop, get_crop_rel_path

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


# ======================== CLOVER: 临床参数辅助函数 ========================

def _compute_volume(dimensions: dict) -> Optional[float]:
    """前列腺体积估算（椭球公式）：V = 0.52 × LR × AP × SI / 1000 (mL)。"""
    lr = dimensions.get("lr_diameter")
    ap = dimensions.get("ap_diameter")
    si = dimensions.get("si_diameter")
    if all(v is not None for v in [lr, ap, si]):
        return 0.52 * lr * ap * si / 1000.0
    return None


def _build_clinical_prefix(psa: Optional[float], dimensions: dict, volume: Optional[float]) -> str:
    """构建结构化临床参数前缀字符串。"""
    parts = []
    if psa is not None:
        parts.append(f"PSA={psa:.2f}ng/mL")
    if volume is not None and volume > 0:
        parts.append(f"体积={volume:.1f}mL")
    if dimensions:
        lr = dimensions.get("lr_diameter")
        ap = dimensions.get("ap_diameter")
        si = dimensions.get("si_diameter")
        if all(v is not None for v in [lr, ap, si]):
            parts.append(f"三径={lr:.0f}×{ap:.0f}×{si:.0f}mm")
    if not parts:
        return ""
    return "【临床参数】" + " | ".join(parts)


# ======================== 影像特征 → 差异化诊断结论 ========================

def _extract_image_features(report_text: str) -> dict:
    """从TRUS报告文本中提取可区分的影像特征。"""
    features = {
        "has_hypoechoic": False,
        "hypoechoic_location": "",
        "has_cyst": False,
        "has_nodule": False,
        "has_peripheral_zone_abnormal": False,
        "capsule_incomplete": False,
        "calcification_type": "点状",
        "has_calcification": True,
    }
    # 低回声区/灶（最需关注的特征）
    if re.search(r'低回声', report_text):
        features["has_hypoechoic"] = True
        # 提取解剖位置（不包含"低回声"本身，模板会补上）
        side = ""
        zone = ""
        if m := re.search(r'右侧|右叶', report_text):
            side = m.group(0)
        elif re.search(r'左侧|左叶', report_text):
            side = "左侧"
        if re.search(r'外周带', report_text):
            zone = "外周带"
        elif re.search(r'移行带', report_text):
            zone = "移行带"
        features["hypoechoic_location"] = (side + zone) if (side or zone) else ""
    # 囊肿/无回声
    if re.search(r'囊肿|无回声[区灶]', report_text):
        features["has_cyst"] = True
    # 结节
    if re.search(r'结节', report_text):
        features["has_nodule"] = True
    # 外周带异常
    if re.search(r'外周带', report_text):
        features["has_peripheral_zone_abnormal"] = True
    # 包膜
    if re.search(r'包膜不完整|包膜欠完整|包膜中断|包膜不光', report_text):
        features["capsule_incomplete"] = True
    # 钙化类型
    if '团状强回声' in report_text:
        features["calcification_type"] = "团状"
    elif '点状强回声' in report_text:
        features["calcification_type"] = "点状"
    elif '强回声' not in report_text and '钙化' not in report_text:
        features["has_calcification"] = False
        features["calcification_type"] = "无"
    return features


def _get_psa_level(psa) -> str:
    """PSA风险分层。"""
    if psa is None:
        return "unknown"
    if psa < 4:
        return "low"
    if psa < 10:
        return "intermediate"
    if psa < 20:
        return "high"
    return "very_high"


def _generate_differentiated_conclusion(features: dict, psa=None, variant_idx: int = 0) -> str:
    """根据影像特征 + PSA联合生成差异化诊断结论。variant_idx控制措辞变体。"""
    psa_level = _get_psa_level(psa)
    psa_mod = {"low": "", "intermediate": "PSA轻度升高，", "high": "PSA显著升高，",
                "very_high": "PSA明显升高，", "unknown": ""}[psa_level]

    # --- 优先级1：低回声 → 恶性风险最高 ---
    if features["has_hypoechoic"]:
        loc = features["hypoechoic_location"]
        loc_text = (loc + "见") if loc else "局部见"
        tmpl = [
            f"前列腺增大伴钙化，{psa_mod}{loc_text}低回声区，建议靶向穿刺活检",
            f"前列腺增生伴钙化，{psa_mod}{loc_text}低回声，不排除恶性可能，建议穿刺",
            f"前列腺增大，{psa_mod}{loc_text}低回声灶，建议穿刺活检明确诊断",
        ]
        return tmpl[variant_idx % len(tmpl)]

    # --- 优先级2：外周带异常 ---
    if features["has_peripheral_zone_abnormal"]:
        tmpl = [
            f"前列腺增大伴钙化，{psa_mod}外周带异常回声区，建议进一步检查",
            f"前列腺增生伴钙化，外周带回声异常，{psa_mod}建议靶向穿刺",
            f"前列腺增大，外周带见异常信号，{psa_mod}建议穿刺活检",
        ]
        return tmpl[variant_idx % len(tmpl)]

    # --- 优先级3：结节 ---
    if features["has_nodule"]:
        tmpl = [
            f"前列腺增大伴钙化，{psa_mod}前列腺结节，建议穿刺活检",
            f"前列腺增生，内见结节样改变，{psa_mod}建议进一步检查明确性质",
            f"前列腺增大伴钙化及结节形成，{psa_mod}建议穿刺活检",
        ]
        return tmpl[variant_idx % len(tmpl)]

    # --- 优先级4：包膜不完整 ---
    if features["capsule_incomplete"]:
        tmpl = [
            f"前列腺增大伴钙化，包膜局部不完整，{psa_mod}建议穿刺活检",
            f"前列腺增生，包膜不完整，{psa_mod}不排除恶性可能，建议穿刺",
        ]
        return tmpl[variant_idx % len(tmpl)]

    # --- 优先级5：囊肿 → 良性倾向 ---
    if features["has_cyst"]:
        tmpl = [
            f"前列腺增大伴钙化，内见无回声区（囊肿可能），{psa_mod}建议定期随访",
            f"前列腺增生伴钙化，囊肿形成，{psa_mod}建议年度复查",
            f"前列腺增大，前列腺囊肿，{psa_mod}建议随访观察",
        ]
        return tmpl[variant_idx % len(tmpl)]

    # --- 优先级6：仅钙化+增大 → 随访为主，PSA高则加急 ---
    if features["has_calcification"]:
        if psa_level in ("high", "very_high"):
            tmpl = [
                f"前列腺增生伴钙化，{psa_mod}建议穿刺活检排除恶性",
                f"前列腺增大伴钙化，{psa_mod}强烈建议穿刺活检",
            ]
        else:
            tmpl = [
                f"前列腺增生伴钙化，{psa_mod}未见明确占位性病变，建议定期随访PSA",
                f"前列腺增大伴钙化，{psa_mod}建议结合临床定期复查",
                f"前列腺增生伴钙化灶，{psa_mod}建议年度随访",
            ]
        return tmpl[variant_idx % len(tmpl)]

    # --- 默认 ---
    tmpl = [
        f"前列腺增大，{psa_mod}建议结合临床综合评估",
        f"前列腺增生，{psa_mod}建议定期随访",
    ]
    return tmpl[variant_idx % len(tmpl)]


def _augment_report_text(report: str, pathology_info: dict, psa=None) -> List[str]:
    """为同一份报告生成基于影像特征差异化的诊断结论变体。"""
    import random

    features = _extract_image_features(report)
    variants = []

    # 找到原始报告"所见"部分和"结论"部分的分界
    endings = ["。", "；", ";"]
    last_end = -1
    for end_char in endings:
        pos = report.rfind(end_char)
        if pos > last_end:
            last_end = pos

    if last_end > len(report) * 0.6:
        base = report[:last_end + 1]  # 保留句号
    else:
        base = report

    # 生成3个不同措辞的结论变体
    for vi in range(3):
        new_conclusion = _generate_differentiated_conclusion(features, psa, variant_idx=vi)
        variant = base + new_conclusion
        if variant != report:
            variants.append(variant)

    # 去重后最多保留
    seen = set()
    unique = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            unique.append(v)
    variants = unique[:3]

    # 癌症病例：额外追加病理确认版本（保持医学术语覆盖）
    if pathology_info.get("has_cancer"):
        gleason = pathology_info.get("gleason_score", "")
        isup = pathology_info.get("isup_grade", "")
        if gleason:
            path_add = f"。病理补充：前列腺腺泡腺癌，Gleason评分{gleason}"
            if isup:
                path_add += f"，ISUP分级{isup}级组"
            variants.append(base + path_add)
    else:
        if random.random() < 0.3:
            variants.append(base + "。病理补充：良性前列腺组织")

    return variants[:4]


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
        volume = _compute_volume(structured.get("dimensions", {}))
        clinical_prefix = _build_clinical_prefix(psa, structured.get("dimensions", {}), volume)

        patient_id = str(row.get(EXCEL_COLUMNS["影像号"], ""))

        for img_path in sorted(images, key=lambda p: p.stem):
            stats["total_images"] += 1

            img_rel_global = str(img_path.relative_to(PROCESSED_DIR))

            # CLOVER B': 生成 Multi-Crop 局部视图 (作为数据增强备选)
            crop_rel = get_crop_rel_path(img_rel_global)
            crop_path = IMAGE_DIR / Path(crop_rel).name
            if not crop_path.exists():
                generate_local_crop(
                    img_path, IMAGE_DIR,
                    crop_ratio=CLOVER_CONFIG["local_crop_ratio"],
                    crop_size=CLOVER_CONFIG["local_crop_size"],
                )

            prompts = _pick_prompts(len(images), samples_per_image=5)
            report_variants = _augment_report_text(trus_report_text, pathology_info, psa)

            # CLOVER A'': 结构化临床前缀 → 嵌入 system prompt (不影响标签掩码)
            if clinical_prefix:
                system_content = f"{SYSTEM_PROMPT}\n\n{clinical_prefix}"
            else:
                system_content = SYSTEM_PROMPT

            for pi, prompt in enumerate(prompts):
                variant = report_variants[pi % len(report_variants)]

                # 50%概率使用局部裁剪视图 (数据增强，防过拟合)
                use_crop = (crop_path.exists() and (pi % 2 == 0))
                img_list = [crop_rel] if use_crop else [img_rel_global]

                # 单 <image> 格式 — 与原始工作版本一致，确保标签掩码正确
                user_content = f"<image>\n{prompt}"

                sample = {
                    "id": f"{name}_{img_path.stem}_p{pi}",
                    "images": img_list,
                    "messages": [
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": variant},
                    ],
                    "metadata": {
                        "patient_name": name,
                        "patient_id": patient_id,
                        "psa": psa,
                        "volume": volume,
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
