import os
from dotenv import load_dotenv
import pandas as pd
import json
import dataclasses

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig, ErrorConfig

# from app.config import settings
from bench.utils import sync_build_test_cases, dump_eval_result, WHOLE_DS_PATH
from bench.utils import DATASETS_MAPPING
from bench.metrics import (
  mk_correctness_0_2,
  mk_faithfulness_0_2,
  mk_completeness_0_2,
  mk_brevity_0_2,
)

from app.rag.pipeline import RagFS

def test_base():
    # print('new version are confirmed')
    rag_app = RagFS() 
    dataset_name = os.getenv("DATASET", 'real_ds')
    real_ds = pd.read_excel(WHOLE_DS_PATH, sheet_name=DATASETS_MAPPING[dataset_name])
    # real_ds = real_ds.iloc[:2]
    test_cases = sync_build_test_cases(real_ds, rag_app)

    correctness_0_2 = mk_correctness_0_2()
    faithfulness_0_2 = mk_faithfulness_0_2()
    completeness_0_2 = mk_completeness_0_2()
    brevity_0_2 = mk_brevity_0_2()

    results = evaluate(test_cases,
        metrics=[
          correctness_0_2,
          faithfulness_0_2,
          completeness_0_2,
          brevity_0_2
        ],
        async_config=AsyncConfig(run_async=True, max_concurrent=1, throttle_value=2),
        error_config=ErrorConfig(ignore_errors=True),
    )
    # TODO: сохранение score происходит в нормализованном виде!!!(score из [0,1]), если это критично нужно переписать!
    dump_eval_result(results, os.path.join('data', f'test_results_{dataset_name}.json'))

if __name__ == "__main__":
    test_base()
