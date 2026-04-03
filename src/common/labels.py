from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd


SUBJECT_REGEX = re.compile(r"\d{3}_S_\d{4}")


DEFAULT_DIAGNOSIS_TO_CLASS = {
    "CN": 0,
    "NL": 0,
    "NORMAL": 0,
    "CONTROL": 0,
    "SMC": 0,
    "SIGNIFICANT MEMORY CONCERN": 0,
    "MCI": 1,
    "EMCI": 1,
    "LMCI": 1,
    "AD": 2,
    "ALZHEIMER": 2,
    "DEMENTIA": 2,
    "DEMENTED": 2,
}


DXCHANGE_TO_CLASS = {
    1: 0,  # NL to NL
    2: 1,  # MCI to MCI
    3: 2,  # AD to AD
    4: 1,  # NL to MCI
    5: 2,  # MCI to AD
    6: 2,  # NL to AD
    7: 0,  # MCI to NL
    8: 0,  # AD to NL
    9: 1,  # AD to MCI
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


def _normalize_diagnosis_value(raw: object) -> str:
    s = str(raw).strip().upper()
    s = re.sub(r"\s+", " ", s)
    return s


def _diagnosis_value_to_class(value: object, diagnosis_to_class: dict[str, int]) -> Optional[int]:
    s = _normalize_diagnosis_value(value)
    if not s or s in {"NAN", "NONE", "NULL"}:
        return None

    # Direct dictionary mapping first.
    if s in diagnosis_to_class:
        return int(diagnosis_to_class[s])

    # DXCHANGE numeric fallback.
    try:
        code = int(float(s))
        if code in DXCHANGE_TO_CLASS:
            return int(DXCHANGE_TO_CLASS[code])
    except Exception:
        pass

    # Keyword fallback for noisy text values.
    if "DEMENT" in s or "ALZ" in s:
        return int(diagnosis_to_class.get("DEMENTIA", 2))
    if "MCI" in s:
        return int(diagnosis_to_class.get("MCI", 1))
    if s in {"SMC", "NL", "CN", "NORMAL", "CONTROL"}:
        return int(diagnosis_to_class.get("CN", 0))

    return None


def _choose_best_diagnosis_column(df: pd.DataFrame, candidate_cols: list[str], diagnosis_to_class: dict[str, int]) -> Optional[str]:
    best_col = None
    best_score = -1
    for col in candidate_cols:
        score = 0
        for v in df[col].head(5000):  # bounded scan for speed on large tables
            if _diagnosis_value_to_class(v, diagnosis_to_class) is not None:
                score += 1
        if score > best_score:
            best_col = col
            best_score = score
    return best_col


def build_label_lookup(
    labels_csv: Path,
    diagnosis_to_class: Optional[dict[str, int]] = None,
    subject_id_columns: Iterable[str] = ("subject_id", "ptid", "participant_id", "subject"),
    diagnosis_columns: Iterable[str] = ("diagnosis", "dx_bl", "dx", "dxchange", "group", "label"),
) -> LabelLookup:
    diagnosis_to_class = diagnosis_to_class or DEFAULT_DIAGNOSIS_TO_CLASS

    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_csv}")

    df = pd.read_csv(labels_csv, low_memory=False)
    normalized = {_normalize_col(c): c for c in df.columns}

    subj_col = next((normalized.get(_normalize_col(c)) for c in subject_id_columns if _normalize_col(c) in normalized), None)
    dx_candidates = [normalized[_normalize_col(c)] for c in diagnosis_columns if _normalize_col(c) in normalized]
    dx_col = _choose_best_diagnosis_column(df, dx_candidates, diagnosis_to_class) if dx_candidates else None

    if subj_col is None or dx_col is None:
        raise ValueError(
            f"Could not find required columns in {labels_csv}. "
            f"Need subject-id in {list(subject_id_columns)} and diagnosis in {list(diagnosis_columns)}."
        )

    subject_to_class: Dict[str, int] = {}
    for _, row in df.iterrows():
        subj_raw = str(row[subj_col]).strip()
        cls = _diagnosis_value_to_class(row[dx_col], diagnosis_to_class)
        if cls is None:
            continue
        # Keep first non-null class assignment for stability across repeated visits.
        if subj_raw not in subject_to_class:
            subject_to_class[subj_raw] = int(cls)

    if not subject_to_class:
        raise ValueError(f"No usable labels found in {labels_csv} after normalization/mapping.")

    num_classes = max(subject_to_class.values()) + 1
    return LabelLookup(subject_to_class=subject_to_class, num_classes=num_classes)


def lookup_subject_class(volume_path: Path, lookup: LabelLookup) -> Optional[int]:
    subject_id = extract_subject_id_from_path(volume_path)
    if subject_id is None:
        return None
    return lookup.subject_to_class.get(subject_id)
