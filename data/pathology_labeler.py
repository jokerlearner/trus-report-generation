"""
前列腺穿刺活检病理结果解析模块。
从中文病理报告中提取 Gleason 评分、ISUP 分级等结构化字段。
"""

import re
from typing import Optional


def infer_isup(gleason_primary: int, gleason_secondary: int) -> Optional[int]:
    score = gleason_primary + gleason_secondary

    if gleason_primary == 3 and gleason_secondary == 3:
        return 1
    if gleason_primary == 3 and gleason_secondary == 4:
        return 2
    if gleason_primary == 4 and gleason_secondary == 3:
        return 3
    if score == 8:
        return 4
    if score in (9, 10):
        return 5
    return None


def detect_cancer(text: str) -> bool:
    if not text or not isinstance(text, str):
        return False

    benign_patterns = [
        r"未见[明确]*(?:癌|恶性)",
        r"未见[明确]*肿瘤",
        r"未见[明确]*肿物",
        r"未见[异常病]",
        r"未见明确病变",
        r"未见恶性",
        r"^未见",
    ]
    negative_phrases = [
        r"(增生|炎症|炎性|肉芽肿|结石|钙化).*(?:未见癌|未见明确癌|未见恶性)",
        r"(未见癌|未见明确癌|未见恶性).*(?:增生|炎症|炎性)",
        r"良性(?:前列腺)?增生",
        r"前列腺增生症",
        r"慢性(?:前列?腺)?炎",
        r"急性(?:前列?腺)?炎",
        r"送检[组]*织[^癌]*?未见[明]?确癌",
        r"送检组织[^。]*?(?:均|全部)?为良性",
        r"未见明确[恶性肿]",
    ]

    cancer_patterns = [
        r"(?:前列腺)?腺癌",
        r"(?:前列腺)?癌(?!.*未见)",
        r"腺泡腺癌",
        r"导管腺癌",
        r"导管内癌",
        r"尿路上皮癌",
        r"鳞状细胞癌",
        r"小细胞癌",
        r"肉瘤样癌",
        r"神经内分泌癌",
        r"基底细胞癌",
        r"浸润性癌",
        r"恶性(?:肿瘤|病变)",
        r"Gleason",
        r"ISUP",
    ]

    for pat in negative_phrases:
        if re.search(pat, text):
            return False

    for pat in benign_patterns:
        if re.search(pat, text):
            return False

    for pat in cancer_patterns:
        if re.search(pat, text):
            return True

    return False


def extract_gleason(text: str) -> dict[str, Optional[int | str]]:
    result: dict[str, Optional[int | str]] = {
        "gleason_primary": None,
        "gleason_secondary": None,
        "gleason_score": None,
    }

    if not text or not isinstance(text, str):
        return result

    patterns = [
        r"Gleason\s*(?:评分|score)?[：:]*\s*(\d)\s*\+\s*(\d)\s*=\s*(\d+)",
        r"gleason\s*(?:评分|score)?[：:]*\s*(\d)\s*\+\s*(\d)\s*=\s*(\d+)",
        r"(\d)\s*\+\s*(\d)\s*=\s*(\d+)\s*(?:分|级|组)?",
    ]

    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            result["gleason_primary"] = int(m.group(1))
            result["gleason_secondary"] = int(m.group(2))
            result["gleason_score"] = f"{m.group(1)}+{m.group(2)}={m.group(3)}"
            break

    return result


def _extract_isup(text: str, gleason_primary: Optional[int], gleason_secondary: Optional[int]) -> Optional[int]:
    m = re.search(r"ISUP\s*(?:分级|grade)?[：:]*\s*(\d)", text, re.IGNORECASE)
    if m:
        return int(m.group(1))

    m = re.search(r"WHO/?ISUP\s*分级[：:]*\s*(\d)\s*组", text)
    if m:
        return int(m.group(1))

    if gleason_primary is not None and gleason_secondary is not None:
        return infer_isup(gleason_primary, gleason_secondary)

    return None


def _extract_summary(text: str) -> Optional[str]:
    if not text or not isinstance(text, str):
        return None

    text = re.sub(r"\s+", "", text)

    m = re.search(r"病理诊断[：:]\s*(.+?)(?:\n|$|(?=免疫组化)|(?=特殊染色))", text)
    if m:
        return m.group(1).strip("。，,.")

    m = re.search(r"(?:病理)?诊断[：:]\s*(.+?)(?:\n|$|(?=免疫组化)|(?=特殊染色))", text)
    if m:
        return m.group(1).strip("。，,.")

    m = re.search(r"镜[下检]所见[：:]*\s*(.+?)(?=诊断|病理诊断|意见|$)", text)
    if m and len(m.group(1)) < 30:
        return m.group(1).strip("。，,.")

    return text[:120]


def parse_pathology(text) -> dict[str, Optional[bool | int | str]]:
    if not text or not isinstance(text, str) or text.strip().lower() in ("nan", "none", "null", ""):
        return {
            "has_cancer": None,
            "is_benign": None,
            "gleason_primary": None,
            "gleason_secondary": None,
            "gleason_score": None,
            "isup_grade": None,
            "summary": None,
        }

    gleason_info = extract_gleason(text)
    gleason_primary = gleason_info["gleason_primary"]
    gleason_secondary = gleason_info["gleason_secondary"]

    has_cancer = detect_cancer(text)
    isup_grade = _extract_isup(text, gleason_primary, gleason_secondary)

    is_benign: Optional[bool] = None
    if has_cancer is True:
        is_benign = False
    elif has_cancer is False and not gleason_info["gleason_score"]:
        is_benign = True

    summary = _extract_summary(text)

    return {
        "has_cancer": has_cancer,
        "is_benign": is_benign,
        "gleason_primary": gleason_primary,
        "gleason_secondary": gleason_secondary,
        "gleason_score": gleason_info["gleason_score"],
        "isup_grade": isup_grade,
        "summary": summary,
    }
