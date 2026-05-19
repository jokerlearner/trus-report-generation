import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pydicom
from PIL import Image


def read_dicom(path):
    px = pydicom.dcmread(path, force=True).pixel_array
    return px.astype(np.float32)


def normalize_to_uint8(pixel_array):
    p_min, p_max = pixel_array.min(), pixel_array.max()
    if p_max == p_min:
        return np.zeros_like(pixel_array, dtype=np.uint8)
    return ((pixel_array - p_min) / (p_max - p_min) * 255).astype(np.uint8)


def grayscale_to_rgb(img_array):
    if img_array.ndim == 3:
        return img_array
    return np.stack([img_array] * 3, axis=-1)


def anonymize_dicom(ds):
    phi_tags = [
        (0x0010, 0x0010),  # PatientName
        (0x0010, 0x0020),  # PatientID
        (0x0010, 0x0030),  # PatientBirthDate
        (0x0010, 0x0040),  # PatientSex
        (0x0010, 0x1010),  # PatientAge
        (0x0010, 0x1030),  # PatientWeight
        (0x0010, 0x1040),  # PatientAddress
        (0x0010, 0x2154),  # PatientTelephoneNumbers
        (0x0008, 0x0090),  # ReferringPhysicianName
        (0x0010, 0x1000),  # OtherPatientIDs
    ]
    for tag in phi_tags:
        if tag in ds:
            ds[tag].value = ""
    return ds


def dicom_to_pil(dcm_path):
    px = read_dicom(dcm_path)
    px_uint8 = normalize_to_uint8(px)
    px_rgb = grayscale_to_rgb(px_uint8)
    return Image.fromarray(px_rgb)


def get_dicom_info(dcm_path):
    dcm_path = Path(dcm_path)
    filename = dcm_path.stem
    parts = filename.split("_")

    info = {
        "image_id": parts[0] if len(parts) >= 1 else "",
        "series_number": parts[2] if len(parts) >= 3 else "",
        "frame_number": parts[3] if len(parts) >= 4 else "",
    }

    try:
        ds = pydicom.dcmread(dcm_path, force=True, specific_tags=[
            (0x0010, 0x0020),
            (0x0010, 0x0010),
            (0x0008, 0x0020),
        ])
        info["patient_id"] = str(getattr(ds, 'PatientID', ''))
        info["patient_name"] = str(getattr(ds, 'PatientName', ''))
        info["study_date"] = str(getattr(ds, 'StudyDate', ''))
    except Exception:
        info["patient_id"] = ""
        info["patient_name"] = ""
        info["study_date"] = ""

    return info
