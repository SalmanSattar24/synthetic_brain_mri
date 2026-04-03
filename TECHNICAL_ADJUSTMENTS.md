# TECHNICAL_ADJUSTMENTS

This document summarizes the technical mapping fixes applied for ADNI label ingestion and conditional training readiness.

## 1) Diagnosis mapping fixes

Updated `src/common/labels.py` default diagnosis mapping to better align with ADNI exports:

- Added CN-like aliases:
  - `NL -> 0`
  - `SMC -> 0`
  - `SIGNIFICANT MEMORY CONCERN -> 0`
- Added AD-like aliases:
  - `DEMENTIA -> 2`
  - `DEMENTED -> 2`

Existing mappings retained:

- `CN`/`NORMAL`/`CONTROL -> 0`
- `MCI`/`EMCI`/`LMCI -> 1`
- `AD`/`ALZHEIMER -> 2`

## 2) ADNI DXCHANGE numeric support

Added numeric fallback mapping for `DXCHANGE` values:

- `1,7,8 -> CN (0)`
- `2,4,9 -> MCI (1)`
- `3,5,6 -> AD (2)`

This allows ingestion when diagnosis is encoded numerically.

## 3) Smarter diagnosis column selection

Label ingestion now prefers the most usable diagnosis column among available candidates:

`diagnosis`, `dx_bl`, `dx`, `dxchange`, `group`, `label`

Selection logic scores columns by number of rows that can be mapped to valid classes and picks the best one.

## 4) Stable per-subject label assignment

For repeated visits per subject, the first valid class assignment is kept to avoid unstable overwrites across rows.

## 5) Parsing robustness

Added normalization and keyword fallback behavior for noisy text values:

- whitespace normalization
- case normalization
- keyword fallback for dementia/alzheimer/mci/cn-like values

## 6) Validation and tests

Extended tests in `tests/test_labels.py` to cover:

- `DEMENTIA` and `SMC` mapping behavior
- `DX_bl` preference behavior
- `DXCHANGE` numeric mapping behavior

These tests ensure the mapping fixes remain stable.

## 7) Operational impact

- Conditional Step 2/3 can now ingest a wider range of ADNI diagnosis representations.
- Preflight + conditional path checks remain unchanged and continue to fail fast if labels path is missing when conditional mode is enabled.
