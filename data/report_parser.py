"""
TRUS（经直肠超声）前列腺报告文本解析模块。
从中文超声报告文本中提取结构化字段。
"""

import re
from typing import Optional


def clean_report_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", "", text)
    return text


def extract_dimensions(text: str) -> dict[str, Optional[float]]:
    result: dict[str, Optional[float]] = {
        "lr_diameter": None,
        "ap_diameter": None,
        "si_diameter": None,
    }

    patterns = {
        "lr_diameter": r"左右径\s*[：:]\s*(\d+\.?\d*)\s*mm",
        "ap_diameter": r"前后径\s*[：:]\s*(\d+\.?\d*)\s*mm",
        "si_diameter": r"上下径\s*[：:]\s*(\d+\.?\d*)\s*mm",
    }

    for key, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            result[key] = float(m.group(1))

    return result


def extract_diagnosis(text: str) -> Optional[str]:
    patterns = [
        r"诊断意见[：:]\s*(.+?)(?:$|\n)",
        r"超声提示[：:]\s*(.+?)(?:$|\n)",
        r"提示[：:]\s*(.+?)(?:$|\n)",
        r"诊断[：:]\s*(.+?)(?:$|\n)",
        r"检查结论[：:]\s*(.+?)(?:$|\n)",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1).strip("。，,.")
    return None


def _extract_shape(text: str) -> Optional[str]:
    m = re.search(r"前列腺\s*(?:形态)?(增大|缩小)", text)
    if m:
        return m.group(1)
    if re.search(r"前列腺\s*(?:形态)?正常", text):
        return "正常"
    if re.search(r"前列腺大小?形态?正常", text):
        return "正常"
    if re.search(r"前列腺形态未见异常", text):
        return "正常"
    return None


def _extract_capsule(text: str) -> Optional[str]:
    m = re.search(r"包膜\s*(完整|不完整|欠完整|光滑|欠光滑)", text)
    if m:
        return m.group(1)
    return None


def _extract_echotexture(text: str) -> Optional[str]:
    if re.search(r"回声分布?\s*(欠均匀|不均匀|欠均)", text):
        return "欠均匀"
    if re.search(r"回声分布?\s*(均匀)", text):
        return "均匀"
    return None


def _extract_calcification(text: str) -> Optional[str]:
    if re.search(r"团状强回声", text):
        return "团状强回声"
    if re.search(r"点状强回声", text):
        return "点状强回声"
    if re.search(r"钙化", text):
        return "钙化"
    if re.search(r"内可见[强高]回声", text):
        return "点状强回声"
    return "无"


def _extract_blood_flow(text: str) -> Optional[str]:
    if re.search(r"(未见|无明显|未见明显)\s*(明显)?\s*异常血[流信]|CDFI[：:]\s*未见明显异常", text):
        return "未见明显异常血流信号"
    if re.search(r"异常血[流信]", text):
        return "异常血流信号"
    return None


def _extract_biopsy_info(text: str) -> Optional[str]:
    m = re.search(r"(经\s*(?:会阴|直肠)[^。，,\.\n]*?(?:系统)?穿刺[^。，,\.\n]*)", text)
    if m:
        return re.sub(r"\s+", "", m.group(1))
    return None


def parse_trus_report(text: str) -> dict:
    if not text or not isinstance(text, str):
        return {
            "dimensions": {"lr_diameter": None, "ap_diameter": None, "si_diameter": None},
            "shape": None,
            "capsule": None,
            "echotexture": None,
            "calcification": None,
            "blood_flow": None,
            "diagnosis": None,
            "biopsy_info": None,
        }

    cleaned = clean_report_text(text)

    return {
        "dimensions": extract_dimensions(cleaned),
        "shape": _extract_shape(cleaned),
        "capsule": _extract_capsule(cleaned),
        "echotexture": _extract_echotexture(cleaned),
        "calcification": _extract_calcification(cleaned),
        "blood_flow": _extract_blood_flow(cleaned),
        "diagnosis": extract_diagnosis(cleaned),
        "biopsy_info": _extract_biopsy_info(cleaned),
    }
