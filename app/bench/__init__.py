"""Benchmarking helpers for the RAG pipeline.

Public API is re-exported here for convenience:
- Metrics builders: `mk_*`
- Test-case builders: `sync_build_test_cases`, `async_build_test_cases`
- Result serialization: `dump_eval_result`
- RAG wrapper used in bench: `RagFSBench`
"""

from .cases import async_build_test_cases, sync_build_test_cases
from .constants import DATASET_MAPPING, DATASETS_MAPPING, WHOLE_DS_PATH
from .io import dump_eval_result
from .rag_app import RagFSBench

__all__ = [
    "WHOLE_DS_PATH",
    "DATASETS_MAPPING",
    "DATASET_MAPPING",
    "sync_build_test_cases",
    "async_build_test_cases",
    "dump_eval_result",
    "RagFSBench",
]
