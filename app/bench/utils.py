"""Backward-compatible façade.

Historically `test_base.py` imported everything from `app.bench.utils`.
After refactor, the implementation is split into small modules, but we keep the old
imports working by re-exporting public symbols from here.
"""

from .constants import WHOLE_DS_PATH, DATASETS_MAPPING, DATASET_MAPPING
from .rag_app import RagFSBench
from .io import dump_eval_result
from .cases import sync_build_test_cases, async_build_test_cases

__all__ = [
    "WHOLE_DS_PATH",
    "DATASETS_MAPPING",
    "DATASET_MAPPING",
    "RagFSBench",
    "dump_eval_result",
    "sync_build_test_cases",
    "async_build_test_cases",
]