from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd


SUBJECT_REGEX = re.compile(r"\d{3}_S_\d{4}")


DEFAULT_DIAGNOSIS_TO_CLASS = {
    "CN": 0,
    "NORMAL": 0,
    "CONTROL": 0,
    "MCI": 1,
    "EMCI": 1,
    "LMCI": 1,
    "AD": 2,
    "ALZHEIMER": 2,
}


@dataclass
class LabelLookup:
    subject_to_class: Dict[str, int]
    num_classes: int


def extract_subject_id_from_path(path: Path) -> Optional[str]:
    m = SUBJECT_REGEX.search(str(path))
    return m.group(0) if m else None


def _normalize_col(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def build_label_lookup(
    labels_csv: Path,
    diagnosis_to_class: Optional[dict[str, int]] = None,
    subject_id_columns: Iterable[str] = ("subject_id", "ptid", "participant_id", "subject"),
    diagnosis_columns: Iterable[str] = ("diagnosis", "dx", "group", "label"),
) -> LabelLookup:
    diagnosis_to_class = diagnosis_to_class or DEFAULT_DIAGNOSIS_TO_CLASS

    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_csv}")

    df = pd.read_csv(labels_csv)
    normalized = {_normalize_col(c): c for c in df.columns}

    subj_col = next((normalized.get(_normalize_col(c)) for c in subject_id_columns if _normalize_col(c) in normalized), None)
    dx_col = next((normalized.get(_normalize_col(c)) for c in diagnosis_columns if _normalize_col(c) in normalized), None)

    if subj_col is None or dx_col is None:
        raise ValueError(
            f"Could not find required columns in {labels_csv}. "
            f"Need subject-id in {list(subject_id_columns)} and diagnosis in {list(diagnosis_columns)}."
        )

    subject_to_class: Dict[str, int] = {}
    for _, row in df.iterrows():
        subj_raw = str(row[subj_col]).strip()
        dx_raw = str(row[dx_col]).strip().upper()
        if dx_raw not in diagnosis_to_class:
            continue
        subject_to_class[subj_raw] = int(diagnosis_to_class[dx_raw])

    if not subject_to_class:
        raise ValueError(f"No usable labels found in {labels_csv} after normalization/mapping.")

    num_classes = max(subject_to_class.values()) + 1
    return LabelLookup(subject_to_class=subject_to_class, num_classes=num_classes)


def lookup_subject_class(volume_path: Path, lookup: LabelLookup) -> Optional[int]:
    subject_id = extract_subject_id_from_path(volume_path)
    if subject_id is None:
        return None
    return lookup.subject_to_class.get(subject_id)
