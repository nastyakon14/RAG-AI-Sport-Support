"""Backward-compatible façade.

Historically `test_base.py` imported everything from `app.bench.utils`.
After refactor, the implementation is split into small modules, but we keep the old
imports working by re-exporting public symbols from here.
"""

from .cases import async_build_test_cases, sync_build_test_cases
from .constants import DATASET_MAPPING, DATASETS_MAPPING, WHOLE_DS_PATH
from .io import dump_eval_result
from .rag_app import RagFSBench

__all__ = [
    "WHOLE_DS_PATH",
    "DATASETS_MAPPING",
    "DATASET_MAPPING",
    "RagFSBench",
    "dump_eval_result",
    "sync_build_test_cases",
    "async_build_test_cases",
]
