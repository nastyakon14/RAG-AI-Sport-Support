import os
import re
from collections import defaultdict
from typing import List, Dict, Any

import pandas as pd
import camelot

from langchain_community.document_loaders import WebBaseLoader, UnstructuredPDFLoader
from langchain_core.documents import Document


def concat_tables(raw_tables: camelot.core.TableList) -> List[Dict[str, Any]]:
    """
    Объединяет фрагменты таблиц, растянутых на несколько страниц, в логические таблицы.
    Логика: если на следующей странице таблица с тем же числом колонок — считаем продолжением.
    """
    merged_tables: List[Dict[str, Any]] = []
    current = None  # текущая собираемая многостраничная таблица

    for t in raw_tables:
        df = t.df
        page = int(t.page)  # страницы, нумерация с 1
        ncols = df.shape[1]

        if current is None:  # инициализация первой таблицы
            current = {
                "df": df.copy(),
                "pages": [page],
                "ncols": ncols,
            }
            continue

        prev_page = current["pages"][-1]
        prev_ncols = current["ncols"]

        # это продолжение:
        # - такое же число колонок
        # - страница сразу после предыдущей
        if (ncols == prev_ncols) and (page == prev_page + 1):
            new_df = df.copy()

            # если первая строка нового куска == первой строке общей таблицы,
            # считаем её дублирующим заголовком и выбрасываем
            if (new_df.iloc[0] == current["df"].iloc[0]).all():
                new_df = new_df.iloc[1:]

            current["df"] = pd.concat(
                [current["df"], new_df],
                ignore_index=True
            )
            current["pages"].append(page)
        else:
            # предыдущая логическая таблица закончилась
            merged_tables.append(current)
            current = {
                "df": df.copy(),
                "pages": [page],
                "ncols": ncols,
            }

    # последний элемент
    if current is not None:
        merged_tables.append(current)

    return merged_tables


def normalize_header_and_data(
    df: pd.DataFrame,
    header_len_threshold: int = 15,
) -> pd.DataFrame:
    """
    Приводит "сырую" таблицу к виду:
    - первая строка -> заголовки столбцов,
    - остальные строки -> данные (все строки привели к str и обрезали пробелы),
    - выбрасываем полностью пустые строки/столбцы.
    Дополнительно:
    - если заголовки столбцов слишком длинные (таблица, вероятно, «боком»),
      транспонируем таблицу и повторяем нормализацию.
    """

    # заполняем пропуски для объединенных ячеек значениями из предыдущих строк
    df.fillna(method='ffill', inplace=True)

    # убираем полностью пустые строки/колонки
    df = df.dropna(axis=0, how="all")
    df = df.dropna(axis=1, how="all")

    if df.empty:
        return df

    # -----------------------------
    # 1. Проверяем "нормальную" ориентацию
    # -----------------------------
    header_row = df.iloc[0].astype(str).str.strip()
    avg_header_len = header_row.apply(len).mean()

    # если заголовки слишком длинные — считаем, что таблица боком, и транспонируем
    if avg_header_len > header_len_threshold:
        # Транспонируем таблицу
        df = df.T

        # После транспонирования ещё раз почистим пустые строки/колонки
        df.fillna(method='ffill', inplace=True)
        df = df.dropna(axis=0, how="all")
        df = df.dropna(axis=1, how="all")

        if df.empty:
            return df

        # новая "первая строка" как заголовок
        header_row = df.iloc[0].astype(str).str.strip()
        data = df.iloc[1:].copy()
    else:
        # всё нормально — используем первую строку как заголовок как есть
        data = df.iloc[1:].copy()

    # Названия колонок
    data.columns = header_row

    # всё в строковый формат и удаляем пробелы
    data = data.applymap(lambda x: str(x).strip())

    # убираем полностью пустые строки (все ячейки == "")
    mask_not_empty = ~data.apply(lambda r: all(v == "" for v in r), axis=1)
    data = data[mask_not_empty].reset_index(drop=True)

    return data



def df_to_rowwise_json(df: pd.DataFrame) -> List[Dict[str, str]]:
    """
    Превращает датафрейм в список словарей (строки таблицы).
    Любые таблицы, любые столбцы.
    """
    # df уже нормализован в normalize_header_and_data
    records: List[Dict[str, str]] = df.to_dict(orient="records")
    return records


def clean_pdf_text(text: str) -> str:
    """
    Чистит PDF-текст от мусорных строк:
    - одиночные цифры и номера страниц
    - пустые строки
    - строки с dtype/object (артефакты таблиц)
    - строки, состоящие только из пунктуации
    - отдельные мусорные столбцы и артефакты
    """

    cleaned_lines = []

    for line in text.splitlines():
        original = line
        line = line.strip()

        # 1) Пустая строка -> удалить
        if not line:
            continue

        # 2) Только цифры (номера страниц и т.п.)
        if re.fullmatch(r"\d{1,4}", line):
            continue

        # 3) dtype/object, NaN, Series/Index артефакты от pandas
        if re.search(r"(dtype|Series|Name:|object)", line):
            continue

        # 4) Строки вида ":" или ": 0"
        if re.fullmatch(r":\s*\d*", line):
            continue

        # 5) Только пунктуация/символы
        if re.fullmatch(r"[\W_]+", line):
            continue

        # 6) Строки типа "М Ж М Ж", "II III II III"
        if re.fullmatch(r"([IVXМЖ]\s*){2,}", line):
            continue

        # 7) Только римская цифра (уровень разряда)
        if re.fullmatch(r"(I|II|III|IV|V|VI|VII|VIII|IX|X)$", line):
            continue

        # 8) Короткие обрывки
        if len(line) < 3:
            continue

        # 9) 1,2   1,12   12,12
        if re.fullmatch(r"\d{1,},\d{1,}", line):
            continue

        # 10) 1-2   1-12   12-15
        if re.fullmatch(r"\d{1,}-\d{1,}", line):
            continue

        cleaned_lines.append(original.strip())

    return "\n".join(cleaned_lines)



def infer_pdf_metadata(pdf_path: str) -> Dict[str, Any]:
    """
    Метаданные:
    - audience: pro / amateur / isu
    - discipline: singles / pairs / ice_dance / synchro / general
    - season: '2024-2025', '2025', ...
    - level: adult_amateur / kids_amateur / unspecified
    - doc_type: rules / technical_requirements / handbook / ranks / other
    """
    meta: Dict[str, Any] = {}
    parts = pdf_path.split(os.sep)
    folder = parts[1] if len(parts) > 1 else ""
    filename = os.path.basename(pdf_path)
    fname_lower = filename.lower()

    # 1) Аудитория / уровень
    if "профессионалы" in folder.lower():
        meta["audience"] = "pro"
    elif "любители" in folder.lower():
        meta["audience"] = "amateur"
    elif "isu" in folder.lower():
        meta["audience"] = "isu"
    else:
        meta["audience"] = "unknown"

    # 2) Дисциплина
    if "одиноч" in fname_lower:
        meta["discipline"] = "singles"
    elif "парн" in fname_lower:
        meta["discipline"] = "pairs"
    elif "танц" in fname_lower:
        meta["discipline"] = "ice_dance"
    elif "синхрон" in fname_lower:
        meta["discipline"] = "synchro"
    else:
        meta["discipline"] = "general"

    # 3) Сезон / год (форматы вида 2024-2025, 2024-25, просто 2024)
    season_match = re.search(r"(20\d{2})\s*[-–]\s*(20\d{2})", filename)
    if season_match:
        meta["season"] = f"{season_match.group(1)}-{season_match.group(2)}"
    else:
        year_match = re.search(r"(20\d{2})", filename)
        if year_match:
            meta["season"] = year_match.group(1)

    # 4) Уровень внутри любителей (дети / взрослые)
    if "взросл" in fname_lower:
        meta["level"] = "adult_amateur"
    elif "дет" in fname_lower and "любител" in fname_lower:
        meta["level"] = "kids_amateur"
    else:
        meta["level"] = "unspecified"

    # 5) Тип документа
    if "правила" in fname_lower:
        meta["doc_type"] = "rules"
    elif "технические требования" in fname_lower or "техтребован" in fname_lower:
        meta["doc_type"] = "technical_requirements"
    elif "руководств" in fname_lower:
        meta["doc_type"] = "handbook"
    elif "разряд" in fname_lower:
        meta["doc_type"] = "ranks"
    else:
        meta["doc_type"] = "other"

    meta["source_file"] = pdf_path
    return meta



# ------------------------------------------------------------------------------------------
#  ЗАГРУЗЧИК PDF: ТЕКСТ + ТАБЛИЦЫ (JSON)

def load_pdf(pdf_file: str) -> List[Document]:
    print(f"Processing: {pdf_file}")

    # 1. Текст
    try:
        loader = UnstructuredPDFLoader(pdf_file, mode="paged")
        page_docs = loader.load()
    except Exception as e:
        print(f"Ошибка чтения текста PDF {pdf_file}: {e}")
        return []

    # 2. Таблицы
    try:
        raw_tables = camelot.read_pdf(pdf_file, pages="all", flavor="lattice")
    except Exception as e:
        print(f"Camelot не смог прочитать таблицы (или их нет): {e}")
        raw_tables = []

    merged_tables = concat_tables(raw_tables) if raw_tables else []
    tables_by_page = defaultdict(list)

    for idx, t in enumerate(merged_tables, start=1):
        df_clean = normalize_header_and_data(t["df"])
        if df_clean.empty:
            continue

        table_json = df_to_rowwise_json(df_clean)
        table_md = df_clean.to_markdown(index=False)

        item = {
            "table_index": idx,
            "columns": list(df_clean.columns),
            "rows": table_json,
            "markdown": table_md
        }
        tables_by_page[t["pages"][0]].append(item)

    # метаданные файла
    pdf_level_meta = infer_pdf_metadata(pdf_file)

    final_docs: List[Document] = []

    for page_idx, page_doc in enumerate(page_docs):
        page_number = page_idx + 1

        raw_text = page_doc.page_content or ""
        text_clean = clean_pdf_text(raw_text)

        page_tables = tables_by_page.get(page_number, [])

        content_parts = []
        if text_clean:
            content_parts.append(text_clean)

        for tbl in page_tables:
            content_parts.append(
                f"\n--- Table {tbl['table_index']} ---\n{tbl['markdown']}\n------------------\n"
            )

        full_page_content = "\n\n".join(content_parts)

        if not full_page_content.strip():
            continue

        metadata = dict(page_doc.metadata) if page_doc.metadata else {}
        metadata["page_number"] = page_number
        metadata["source"] = pdf_file
        metadata.update(pdf_level_meta)  # <--- важная строка

        final_docs.append(Document(page_content=full_page_content, metadata=metadata))

    return final_docs