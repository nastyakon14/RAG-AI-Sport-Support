import os
import asyncio
import json
from tqdm.asyncio import tqdm_asyncio
import time
from tqdm import tqdm

from deepeval.test_case import LLMTestCase
from openai import RateLimitError

WHOLE_DS_PATH = os.path.join('data', 'eval_questions_final.xlsx')
DATASETS_MAPPING = {
  'generated_ds': 'Sheet1',
  'real_ds': 'Sheet2'
}

def dump_eval_result(result, path: str):
    out = []

    for tr in result.test_results:  # [web:9]
        out.append({
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
                for mr in tr.metrics_data
            ],
        })

    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

def sync_build_test_cases(rows, rag_app):
    def one(row) -> LLMTestCase:
        try:
            actual_output, retrieved_contexts = rag_app.find_answer(row.Сам_запрос)
        except RateLimitError:
            time.sleep(65)
            actual_output, retrieved_contexts = rag_app.find_answer(row.Сам_запрос)

        return LLMTestCase(
            input=row.Сам_запрос,
            actual_output=actual_output,
            retrieval_context=retrieved_contexts,
            expected_output=row.Ожидаемый_ответ
        )
    test_cases = []
    for i, row in tqdm(enumerate(rows.itertuples(index=False)), total=len(rows)):
        test_cases.append(one(row))
    return test_cases

async def async_build_test_cases(rows, rag_app, concurrency: int = 2):
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
                expected_output=row.Ожидаемый_ответ
            )

    tasks = [one(row) for row in rows.itertuples(index=False)]
    test_cases = await tqdm_asyncio.gather(*tasks)
    return test_cases
    