### Основная инструкция

Основной тест живёт в `app/bench/test_base.py`.
Чтобы запустить тест, в корне папки `data` должен лежать `eval_questions_final.xlsx`.

В data так же будут сохраняться сырые результаты после `deepeval.evaluate()`.

Тест рассматривает пока два датасэта: generated_ds - Sheet1 в eval_questions_final.xlsx и real_ds - Sheet2 там же.

Можно добавить новые, через обновление файла .xlsx и дополнение `app.bench.constants.DATASETS_MAPPING`. По умолчанию, `DATASET=real_ds`.

Можно запустить тестирование для определенного датасэта сгруппированное по виду, для этого выполни из корня проекта:

```bash
deepeval login  # введи свой API key
env DATASET=name_of_required_dataset # список доступных: keys() из app.bench.constants.DATASETS_MAPPING
chmod +x app/bench/scripts/test_bench_run_grouped_by_kinds.sh # если будут проблемы с доступом к скрипту
sed -i 's/\r$//' app/bench/scripts/test_bench_run_grouped_by_kinds.sh # если будут проблемы с чем-то вроде syntax error near unexpected token `$'do\r''
bash app/bench/scripts/test_bench_run_grouped_by_kinds.sh
```

После этого в Confident AI будут test runs для каждого вида.

Если хочешь запустить одно общее тестирование тестирование на одном датасэте выполни из корня проекта:

```bash
deepeval login  # введи свой API key
env DATASET=name_of_required_dataset # список доступных: keys() из app.bench.constants.DATASETS_MAPPING
bash app/bench/scripts/test_bench_run.sh
```

! При запуске из google colab это нужно выполнять из терминала, а не из ячейки.
