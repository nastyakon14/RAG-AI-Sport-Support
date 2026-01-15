## Bench: Deepeval-оценка RAG пайплайна

Этот модуль запускает **оценку качества ответов RAG** через [`deepeval`](https://github.com/confident-ai/deepeval):
- генерирует `LLMTestCase`-кейсы, прогоняя ваш `RagFS` на вопросах из Excel
- считает метрики (0–2): correctness, faithfulness/groundedness, completeness, brevity
- сохраняет результаты в JSON, чтобы их можно было анализировать отдельно

### Что именно тестируется

Точка входа: `bench_test.py` (функция `test_base()`).

Используется:
- **RAG**: `app.rag.pipeline.RagFS` (метод `find_answer(query)` должен возвращать `(actual_output, retrieval_contexts)`).
- **Датасет**: Excel-файл `data/eval_questions_final.xlsx`
  - лист выбирается через `DATASET`
  - ожидаются колонки: `Сам_запрос`, `Ожидаемый_ответ`

Маппинг датасетов: `bench/utils.py` → `DATASETS_MAPPING`:
- `generated_ds` → `Sheet1`
- `real_ds` → `Sheet2` (значение по умолчанию)

### Требования

- активированное окружение и зависимости:

```bash
pip install -r requirements.txt
```

- переменные окружения для LLM-оценки (`GEval`):
  - `OPENAI_API_KEY`
  - (опционально) `.env` в корне проекта — если используете `python-dotenv`

> Примечание: `pandas.read_excel(...)` обычно требует `openpyxl`. Если будет ошибка импорта — установите `pip install openpyxl`.

### Быстрый запуск (без Confident AI UI)

Из корня проекта:

```bash
# выбрать датасет (опционально)
export DATASET=real_ds

python bench_test.py
```

На Windows PowerShell:

```powershell
$env:DATASET = "real_ds"
python .\bench_test.py
```

Результат сохранится в:
- `data/test_results_<DATASET>.json`

### Запуск через Deepeval CLI (и UI через Confident AI)

Из корня проекта:

```bash
deepeval login                 # введи API key Confident AI (опционально, для UI)
export DATASET=real_ds          # или generated_ds
deepeval test run bench_test.py
deepeval view                   # откроет UI/вьюер (если используете Confident AI)
```

### Частые проблемы

- **Очень долго на первых шагах / таймауты HuggingFace**: это обычно скачивание модели эмбеддингов (например, `BAAI/bge-m3`) и прогрев кэша.
  - можно увеличить таймауты (пример):

```powershell
$env:HF_HUB_READ_TIMEOUT="120"
$env:HF_HUB_ETAG_TIMEOUT="120"
```

- **RateLimit от OpenAI**: в `bench/utils.py` есть простой retry с `sleep(65)` для `RateLimitError`.