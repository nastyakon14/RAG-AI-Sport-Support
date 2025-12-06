import os
import re
import pandas as pd
import camelot
from collections import defaultdict


from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import UnstructuredPDFLoader,PyPDFLoader
from langchain_core.documents import Document


# website for parsing
url_isu = 'https://www.isu.org/figure-skating-rules/?tab=ISU%20Judging%20System'
# pdf for parsing
INPUT_PDF_PATH = 'pdf_rules'
PROF_PATH = os.path.join(INPUT_PDF_PATH,'Профессионалы')
LOVER_PATH = os.path.join(INPUT_PDF_PATH,'Любители')
ISU_PATH = os.path.join(INPUT_PDF_PATH,'ISU')

# all pdf files
prof_pdf_rules = os.listdir(PROF_PATH)
lover_pdf_rules = os.listdir(LOVER_PATH)
isu_pdf_rules = os.listdir(ISU_PATH)

prof_pdf_rules = [os.path.join(PROF_PATH, f)  for f in prof_pdf_rules]
lover_pdf_rules = [os.path.join(PROF_PATH, f)  for f in lover_pdf_rules]
isu_pdf_rules = [os.path.join(PROF_PATH, f)  for f in isu_pdf_rules]


# ------------------------------------------------------------------------------------
#   WEBSITES


# Функция для извлечения информации с веб-сайтов (с сайта ISU)
def load_url(url):
    '''DataLoader for url
    Input: url
    Output: split'''
    
    loader_web = WebBaseLoader(url_isu)
    docs = loader_web.load()
     # разбиваем склеенные слова по заглавной букве
    # SustainabilityPressAnti-dopingSafeguardingISU  --> Sustainability Press Anti-doping Safeguarding ISU
    
    for i in range(len(docs)):
        s = docs[i].page_content 
        docs[i].page_content = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', ' ', s)
    return docs


# ------------------------------------------------------------------------------------
# PDF вспомогательные функции

def concat_tables(raw_tables):
    merged_tables = []   # список логических таблиц
    current = None       # текущая собираемая многостраничная таблица

    for t in raw_tables:
        df = t.df
        page = int(t.page)       # страницы, нумерация с 1
        ncols = df.shape[1]

        if current is None:  # инициализация первой таблицы
            # начинаем новую логическую таблицу
            current = {
                "df": df.copy(),
                "pages": [page],
                "ncols": ncols,
            }
            continue

        prev_page = current["pages"][-1] 
        prev_ncols = current["ncols"]

        # проверка, что это продолжение предыдущей таблицы:
        #   - такое же число колонок
        #   - страница сразу после предыдущей (prev_page + 1)
        if (ncols == prev_ncols) and (page == prev_page + 1):
            new_df = df.copy()

            # На случай, если заголовок всё‑таки повторяется на новой странице:
            # если первая строка нового куска == первой строке общей таблицы,
            # то считаем её дублирующим заголовком и выбрасываем
            # текущая таблица является продолжением предыдущей таблицы
            if (new_df.iloc[0] == current["df"].iloc[0]).all():
                new_df = new_df.iloc[1:]

            current["df"] = pd.concat([current["df"], new_df], ignore_index=True)   # объединяем их
            current["pages"].append(page)
        else:
            # предыдущая логическая таблица закончилась
            # на новой странице новая таблица
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

#очистка заголовков 
def normalize_header_and_data(df: pd.DataFrame) -> pd.DataFrame:
    
    # заполняем пропуски для объединенных ячеек значениями из предыдущих строк
    df.fillna(method='ffill', inplace = True)   
    
    """Первая строка — заголовок, остальное — данные"""
    # Убираем полностью пустые строки/колонки
    df = df.dropna(axis=0, how="all")
    df = df.dropna(axis=1, how="all")

    if df.empty:
        return df

    # Берём первую строку как заголовок
    header = df.iloc[0].astype(str).str.strip()
    data = df.iloc[1:].copy()

    # Названия колонок
    data.columns = header

    # переводим все в стринговый формат и удаляем пробелы по краям строк
    data = data.applymap(lambda x: str(x).strip())

    # Убираем полностью пустые строки (все ячейки == "")
    mask_not_empty = ~data.apply(lambda r: all(v == "" for v in r), axis=1)
    data = data[mask_not_empty].reset_index(drop=True)

    return data

def df_to_rowwise_text(df: pd.DataFrame) -> str:
    """
    Превращает таблицу в текст:
    'Колонка1: значение; Колонка2: значение; ...'
    по одной строке на каждую запись.
    """
    headers = list(df.columns)
    lines = []

    for i, row in df.iterrows():
        parts = []
        for col in headers:
            value = str(row[col]).strip()
            if value:  # пропускаем пустые ячейки
                parts.append(f"{col}: {value}")
        if not parts:
            continue

        line = "; ".join(parts)  
        lines.append(line)

    return "\n".join(lines)


def clean_pdf_text(text: str) -> str:
    """
    Чистит PDF-текст от мусорных строк:
    - одиночные цифры и номера страниц
    - пустые строки
    - строки с dtype/object (артефакты таблиц)
    - строки состоящие только из пробелов, табов или пунктуации
    - отдельные мусорные столбцы таблиц
    """

    cleaned_lines = []
    
    for line in text.splitlines():
        original = line
        line = line.strip()

        # 1) Пустая строка -> удалить
        if not line:
            continue
        
        # 2) Строка содержит только цифру или номер страницы (1–3 символа)
        #    Пример: "1" "12" "141"
        if re.fullmatch(r"\d{1,4}", line):
            continue
        
        # 3) dtype/object, NaN, Series/Index артефакты от pandas
        if re.search(r"(dtype|Series|Name:|object)", line):
            continue
        
        # 4) Строки вида ":" или ": 0"
        if re.fullmatch(r":\s*\d*", line):
            continue
        
        # 5) Строки состоящие только из пунктуации или спецсимволов
        if re.fullmatch(r"[\W_]+", line):
            continue
        
        # 6) Строки с набором одиночных букв из таблиц (артефакты)
        # Например: "М Ж М Ж", "II III II III"
        if re.fullmatch(r"([IVXМЖ]\s*){2,}", line):
            continue
        
        # 7) Удаляем строки, где только уровень разряда без значения
        #    Например: "II", "III", "IV"
        if re.fullmatch(r"(I|II|III|IV|V|VI|VII|VIII|IX|X)$", line):
            continue
        
        # 8) Строки из односложных обрывков колонок
        if len(line) < 3:
            continue
        
        # 9) 1,2   1,12   12,12
        if re.fullmatch(r"\d{1,},\d{1,}", line):
            continue
        
        # 10) 1-2   1-12   12-15
        if re.fullmatch(r"\d{1,}-\d{1,}", line):
            continue
        
        # Если строка нормальная → вернуть
        cleaned_lines.append(original.strip())

    return "\n".join(cleaned_lines)



# --------------------------------------------------------------------------------------
# pdf итоговый загрузчик

def load_pdf(pdf_file):
    # извлекаем информацию из pdf
    loader = UnstructuredPDFLoader(pdf_file, mode="paged") # постранично
    page_docs = loader.load()        
    num_pages = len(page_docs)

    # извлекаем табличные данные из pdf в структурированном виде
    raw_tables = camelot.read_pdf(pdf_file, pages="all", flavor="lattice")  # ищет таблицы по линиям (сетке)
    
    # объединяем таблицы по страницам, если одна и та же таблица на разных страницах 
    merged_tables = concat_tables(raw_tables)

    tables_text_by_page = defaultdict(list)

    # преобразовываем таблцицы
    for idx, t in enumerate(merged_tables, start=1):
        df_raw = t["df"]  # отдельный датафрейм
        pages = t["pages"]          # список страниц, на которых тянется таблица
        start_page = pages[0]   # первая страницы, откуда начинается таблица
        
        df_clean = normalize_header_and_data(df_raw)  # преобразовываем названия 
        if df_clean.empty:
            continue

        table_text = df_to_rowwise_text(df_clean)   # таблица в текст
        
        tables_text_by_page[start_page].append(
            {
                "table_index": idx,
                "pages": pages,
                "text": table_text,
            }
        )
    
    # объединяем текст и таблицы
    final_docs = []

    for page_idx, page_doc in enumerate(page_docs): # то что извлекли из unstructedpdfloader
        page_number = page_idx + 1  # нумерация с 0, начинаем с 1

        parts = []

        # Текст страницы, очищаем от ненужных строк 
        raw_page_text = page_doc.page_content or ""
        page_text = raw_page_text.strip()    
        if page_text:
            parts.append(page_text)

        # Таблицы, начинающиеся на этой странице
        for tinfo in tables_text_by_page.get(page_number, []):
            header_line = (
                f"Таблица {tinfo['table_index']} "
                f"(страницы {', '.join(map(str, tinfo['pages']))}):"
            )
            parts.append(header_line)
            parts.append(tinfo["text"])

        # Если на странице вообще ничего нет (ни текста, ни таблиц) — пропускаем
        if not parts:
            continue

        combined_text = "\n\n".join(parts)
        combined_text_clean = clean_pdf_text(combined_text)
        # combined_text_clen =  clean_pdf_text(combined_text)
        # сохраняем метаданные
        metadata = dict(page_doc.metadata)
        # корректный номер страницы в метаданных
        metadata["page_number"] = page_number
        metadata.setdefault("source", pdf_file)

        final_docs.append(
            Document(
                page_content=combined_text_clean,
                metadata=metadata,
            )
        )

    return final_docs