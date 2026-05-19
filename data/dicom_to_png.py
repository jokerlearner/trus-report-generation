import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import logging

from tqdm import tqdm

from config import IMAGE_DIR, BATCH_DIRS
from data.dicom_utils import dicom_to_pil, get_dicom_info

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _extract_patient_name(dirname):
    chars = []
    for ch in dirname:
        if '一' <= ch <= '鿿' or '㐀' <= ch <= '䶿':
            chars.append(ch)
        else:
            break
    return ''.join(chars) if chars else dirname


def run():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    total_dcm = 0
    all_dcm = []
    for batch_name, batch_dir in BATCH_DIRS.items():
        if not batch_dir.exists():
            logger.warning("批次目录不存在: %s", batch_dir)
            continue
        for dcm_path in batch_dir.rglob("*.dcm"):
            total_dcm += 1
            if "MR" in str(dcm_path):
                continue
            all_dcm.append((batch_name, dcm_path))

    skipped_mr = total_dcm - len(all_dcm)
    logger.info("扫描到 %d 个DCM，跳过 %d 个MR，待转换 %d 个", total_dcm, skipped_mr, len(all_dcm))

    success, failed = 0, 0
    mapping = {}

    for batch_name, dcm_path in tqdm(all_dcm, desc="转换DICOM"):
        try:
            rel_parts = dcm_path.relative_to(BATCH_DIRS[batch_name]).parts
            patient_dir = rel_parts[0]
            patient_name = _extract_patient_name(patient_dir)

            info = get_dicom_info(dcm_path)
            image_id = info["image_id"]
            series_num = info["series_number"]

            out_name = f"{batch_name}_{patient_name}_{image_id}_{series_num}.png"
            out_path = IMAGE_DIR / out_name

            img = dicom_to_pil(dcm_path)
            img.save(out_path, "PNG")

            mapping[out_name] = {
                "batch": batch_name,
                "patient_name": patient_name,
                "patient_id": image_id,
                "original_dcm": str(dcm_path),
            }

            success += 1
        except Exception:
            logger.warning("转换失败: %s", dcm_path, exc_info=True)
            failed += 1

    mapping_path = IMAGE_DIR.parent / "image_mapping.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    logger.info("完成 — 成功: %d, 失败: %d, 跳过MR: %d", success, failed, skipped_mr)
    logger.info("映射文件: %s", mapping_path)


if __name__ == "__main__":
    run()
