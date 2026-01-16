from __future__ import annotations

import json
from typing import Any


def dump_eval_result(result: Any, path: str) -> None:
    """Serialize deepeval `evaluate()` output to a stable JSON artifact."""
    out: list[dict[str, Any]] = []

    # deepeval returns EvaluateResult with `.test_results`
    for tr in getattr(result, "test_results", []) or []:
        out.append(
            {
                "name": getattr(tr, "name", None),
                "success": getattr(tr, "success", None),
                "input": getattr(tr, "input", None),
                "actual_output": getattr(tr, "actual_output", None),
                "expected_output": getattr(tr, "expected_output", None),
                "retrieval_context": getattr(tr, "retrieval_context", None),
                "metrics": [
                    {
                        "name": mr.name,
                        "score": mr.score,
                        "success": getattr(mr, "success", None),
                        "reason": getattr(mr, "reason", None),
                    }
                    for mr in getattr(tr, "metrics_data", []) or []
                ],
            }
        )

    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

