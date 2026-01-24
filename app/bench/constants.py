from __future__ import annotations

import os

WHOLE_DS_PATH = os.path.join("data", "eval_questions_final.xlsx")

# Canonical mapping name (used across the project).
DATASETS_MAPPING: dict[str, str] = {
    "generated_ds": "Sheet1",
    "real_ds": "Sheet2",
}

# Backwards-compatible alias (README previously used DATASET_MAPPING).
DATASET_MAPPING = DATASETS_MAPPING
