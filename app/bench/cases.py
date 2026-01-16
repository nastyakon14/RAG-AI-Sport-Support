from __future__ import annotations

import asyncio
import time
from typing import Iterable

from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from deepeval.test_case import LLMTestCase

try:
    # openai>=1.0
    from openai import RateLimitError  # type: ignore
except Exception:  # pragma: no cover
    RateLimitError = None  # type: ignore


def _is_rate_limit_exc(exc: Exception) -> bool:
    if RateLimitError is not None and isinstance(exc, RateLimitError):  # type: ignore[arg-type]
        return True
    msg = str(exc).lower()
    return "rate limit" in msg or "429" in msg


def sync_build_test_cases(rows, rag_app) -> list[LLMTestCase]:
    """Build LLMTestCase list sequentially (stable & easiest to debug)."""

    def one(row) -> LLMTestCase:
        try:
            actual_output, retrieved_contexts = rag_app.find_answer(row.Сам_запрос)
        except Exception as e:
            if _is_rate_limit_exc(e):
                time.sleep(65)
                actual_output, retrieved_contexts = rag_app.find_answer(row.Сам_запрос)
            else:
                raise

        return LLMTestCase(
            input=row.Сам_запрос,
            actual_output=actual_output,
            retrieval_context=retrieved_contexts,
            expected_output=row.Ожидаемый_ответ,
        )

    test_cases: list[LLMTestCase] = []
    # NOTE: `rows` is typically a pandas.DataFrame
    for _, row in tqdm(enumerate(rows.itertuples(index=False)), total=len(rows)):
        test_cases.append(one(row))
    return test_cases


async def async_build_test_cases(rows, rag_app, concurrency: int = 2) -> list[LLMTestCase]:
    """Build LLMTestCase list concurrently using threads (safer for sync RAG code)."""
    sem = asyncio.Semaphore(concurrency)

    async def one(row) -> LLMTestCase:
        async with sem:
            actual_output, retrieved_contexts = await asyncio.to_thread(
                rag_app.find_answer, row.Сам_запрос
            )
            return LLMTestCase(
                input=row.Сам_запрос,
                actual_output=actual_output,
                retrieval_context=retrieved_contexts,
                expected_output=row.Ожидаемый_ответ,
            )

    tasks = [one(row) for row in rows.itertuples(index=False)]
    test_cases = await tqdm_asyncio.gather(*tasks)
    return list(test_cases)

