from __future__ import annotations

import os

import pandas as pd
from deepeval import evaluate
from deepeval.evaluate import AsyncConfig, ErrorConfig
from dotenv import load_dotenv

from app.bench import (
    DATASETS_MAPPING,
    WHOLE_DS_PATH,
    RagFSBench,
    dump_eval_result,
    sync_build_test_cases,
)
from app.bench.metrics import (
    mk_brevity_0_2,
    mk_completeness_0_2,
    mk_correctness_0_2,
    mk_faithfulness_0_2,
)


def test_base():
    load_dotenv()
    load_dotenv(".env.local", override=True)

    rag_app = RagFSBench()

    dataset_name = os.getenv("DATASET", "real_ds")
    kind = os.getenv("KIND", None)

    if dataset_name not in DATASETS_MAPPING:
        raise KeyError(
            f"Unknown DATASET={dataset_name!r}. Available: {sorted(DATASETS_MAPPING.keys())}"
        )

    dataset = pd.read_excel(WHOLE_DS_PATH, sheet_name=DATASETS_MAPPING[dataset_name])
    dataset.dropna(subset=["Сам_запрос", "Ожидаемый_ответ"], how="any", inplace=True)

    if "Вид_катания" in dataset.columns:
        dataset["Вид_катания"] = dataset["Вид_катания"].replace(
            {"Танцы на льду": "танцы", "Общее": "общее"}
        )

    if kind is not None:
        if "Вид_катания" not in dataset.columns:
            raise KeyError("KIND is set but dataset has no 'Вид_катания' column")
        dataset = dataset[dataset["Вид_катания"] == kind]

    # dataset = dataset.iloc[:2]
    if len(dataset) == 0:
        return

    print(f"Inference for {dataset_name} of kind {kind} are started")
    test_cases = sync_build_test_cases(dataset, rag_app)

    results = evaluate(
        test_cases,
        metrics=[
            mk_correctness_0_2(),
            mk_faithfulness_0_2(),
            mk_completeness_0_2(),
            mk_brevity_0_2(),
        ],
        async_config=AsyncConfig(run_async=True, max_concurrent=1, throttle_value=2),
        error_config=ErrorConfig(ignore_errors=True),
        identifier=f"{dataset_name}_{kind}",
    )

    # TODO: сохранение score происходит в нормализованном виде (score из [0,1])
    dump_eval_result(
        results, os.path.join("data", f"test_results_{dataset_name}_{kind}.json")
    )


if __name__ == "__main__":
    test_base()
