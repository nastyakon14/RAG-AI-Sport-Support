import os
import re
from collections import defaultdict
from typing import Any, Dict, List

import camelot
import pandas as pd
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_core.documents import Document

ISU_links = {  # медиа
    "ISU Figure Skating Media Guide 2024-2025.pdf": "https://isu-d8g8b4b7ece7aphs.a03.azurefd.net/isudamcontainer/uploads/images/press/mediaaccreditationguides/Figure_Skating_Media_Guide_2024-25.pdf",
    # компоненты - личное, парное, танцы, синхронное
    "Program Components – Single & Pairs, Ice Dance and Synchronized Skating 2024.pdf": "https://isu-d8g8b4b7ece7aphs.a03.azurefd.net/isudamcontainer/CMS/sportrulesdisciplinecategory/pdf/Componentschartupdated2024July086344800_17327068456450.pdf",
    # компоненты и презентация
    "Composition, Presentation and Skating Skills Charts.pdf": "https://www.isu.org/figure-skating-rules/?tab=Handbooks%20Single%20%26%20Pair%20Skating",
    #  Объяснение Символов На Судейских Данных
    "Explanation Of Symbols On The Judges Details Per Skater_2023.pdf": "https://isu-d8g8b4b7ece7aphs.a03.azurefd.net/isudamcontainer/CMS/sportrulesdisciplinecategory/pdf/EXPLANATIONOFSYMBOLSONTHEJUDGESDETAILSPERSKATERJuniorandSenior056186400_17327128487461.pdf",
    # техническое судейство - одиночное + парное
    "Handbook For Referees And Judges Single And Pair Skating 2025-2026.pdf": "https://isu-d8g8b4b7ece7aphs.a03.azurefd.net/isudamcontainer/CMS/Corporate-Site/Sports-Rules/Figure-Skating-Rules/Handbooks-Single-&-Pair-Skating/Handbook-for-Referees-and-Judges-2025-26-July-1753088479-0379.pdf",
    # техническое руководство - парное катание
    "Technical Panel Handbook Pair Skating 2025-2026.pdf": "https://isu-d8g8b4b7ece7aphs.a03.azurefd.net/isudamcontainer/CMS/Corporate-Site/Sports-Rules/Figure-Skating-Rules/Handbooks-Single-&-Pair-Skating/TP-Handbook-Pair-Skating-2025-26-25July-1754657063-3208.pdf",
    # техническое руководство - одиночное катание
    "Technical Panel Handbook Single Skating 2025-2026.pdf": "https://isu-d8g8b4b7ece7aphs.a03.azurefd.net/isudamcontainer/CMS/Corporate-Site/Sports-Rules/Figure-Skating-Rules/Handbooks-Single-&-Pair-Skating/TP-Handbook-Singles-25-26-FINAL-21-July-2025-update-25-July-1753703999-2708.pdf",
    # технические требования - одиночное, парное, танцы
    "Special Regulations & Technical Rules Single & Pair Skating And Ice Dance 2024.pdf": "https://isu-d8g8b4b7ece7aphs.a03.azurefd.net/isuproduction/uploads/images/isustatutes/documents/2024_Special_Regulation_SP_and_Ice_Dance_and_Technical_Rules_SP__and_ID_Final_rev.pdf",
    # технические требования - танцы на льду
    "Requirements for Technical Rules season Ice Dance 2025-2026.pdf": "https://isu-d8g8b4b7ece7aphs.a03.azurefd.net/isudamcontainer/CMS/Corporate-Site/Governance/Transparency/ISU-Communications/2704-ID-Requirements-for-Technical-Rules-season-2025-26-updated-post-Frankfurt-final-1754983899-0624.pdf",
}

prof_links = {
    "Правила Вида Спорта «Фигурное Катание На Коньках» 2024.pdf": "https://fsrussia.ru/files/docs/fs_rules_rus_16_10_24_1025.pdf",
    "Нормы, требования и условия их выполнения по виду спорта «фигурное катание на коньках» 2024.pdf": "https://fsrussia.ru/files/docs/evsk_fs_2326_311024.pdf",
    "Руководство Технических бригад Одиночное катание 2024-2025.pdf": "https://fsrussia.ru/files/docs/tp_handbook_singles_2425.pdf",
    "Руководство Технических бригад Парное катание 2024-2025.pdf": "https://fsrussia.ru/files/docs/tp_handbook_pairs_2425.pdf",
    "Руководство технических бригад Синхронное катание 2024-2025.pdf": "https://fsrussia.ru/files/docs/archive/synchro/tp_handbook_sys_2425.pdf",
    "Технические требования по одиночному и парному катанию.pdf": "https://calculatorfs.ru/wp-content/uploads/2023/03/fs_tech_requirement.pdf",
    "Указания по судейству элемента «хореографическая спираль».pdf": "https://fsrussia.ru/files/docs/chspl_2023_description_upd.pdf",
}

lovers_link = {
    "Специальные требования к проведению соревнований 2025.pdf": "https://docs.yandex.ru/docs/view?url=ya-disk-public%3A%2F%2FFkh4OgSnJhwc5nJukWnMo7IzHesR4Z5bG%2BVeJ5ft%2BiJsBtue%2F9pCsw3pugwbsIbEq%2FJ6bpmRyOJonT3VoXnDag%3D%3D&name=Специальные%20требования%20к%20проведению%20соревнований%20по%20фигурному%20катанию%20на%20коньках%20среди%20взрослых-любителей.pdf",
    "Технические требования для фигуристов-любителей с РАС и другими ментальными нарушениями в сезоне 2025-2026.pdf": "https://docs.yandex.ru/docs/view?url=ya-disk-public%3A%2F%2Fe13tSDA7bt762%2F47%2FQdfFl1dumxmKDLCScXmRYaWk%2FX2i55FcMdNAnNlRbzqGYiXq%2FJ6bpmRyOJonT3VoXnDag%3D%3D&name=ПРЕДВАРИТЕЛЬНЫЕ_Технические_требования_2025_2026_послед.pdf&nosw=1",
    "Технические требования для соревнований по фигурному катанию на коньках среди взрослых любителей 2025-2026.pdf": "https://docs.yandex.ru/docs/view?url=ya-disk-public%3A%2F%2F0pInUV91oCnIOUM9jZItc7rRn1WDQ0dvhnYdOCUUj3oVR7fuhTWg1dXp2%2BnXmbuIq%2FJ6bpmRyOJonT3VoXnDag%3D%3D&name=Технические%20требования%20для%20соревнований%20среди%20взрослых-любителей%202025-2026.pdf&nosw=1",
    "Технические требования «фигурное катание на коньках» для детей-любителей 2025-2026.pdf": "https://docs.yandex.ru/docs/view?url=ya-disk-public%3A%2F%2FYVsYquFcRbB%2FuJvY84kGBpAce%2BciRmh628iO6R%2Fwj6pLk7ctwW3tS8QuTgE89XTdq%2FJ6bpmRyOJonT3VoXnDag%3D%3D&name=Технические_требования_для_детей_любителей_2025%5C2026.pdf&nosw=1",
    "Танцы на льду соло содержание программ и дополнительные требования  2025-2026.pdf": "https://docviewer.yandex.ru/?url=ya-disk-public%3A%2F%2FyWRe4fGGsUkEY1HxCE7ovBkYiwN7mHVDIciIgtA25Vc4mvYgzwHhFsvhdwyPN04Fq%2FJ6bpmRyOJonT3VoXnDag%3D%3D&name=Технические%20требования%20для%20танцев%20на%20льду%20соло%202025-2026.pdf",
    "Технические требования для танцев на льду соло 2025-2026.pdf": "https://docs.yandex.ru/docs/view?url=ya-disk-public%3A%2F%2FyWRe4fGGsUkEY1HxCE7ovBkYiwN7mHVDIciIgtA25Vc4mvYgzwHhFsvhdwyPN04Fq%2FJ6bpmRyOJonT3VoXnDag%3D%3D&name=Технические%20требования%20для%20танцев%20на%20льду%20соло%202025-2026.pdf",
    "Программа тестирования для любителей.pdf": "https://docs.yandex.ru/docs/view?url=ya-disk-public%3A%2F%2F2U%2FIy32gcoHICJEBicheA8ogi8xCVlEt3h8rdxccYjy2EgojSTRhnlkym2%2FO808Rq%2FJ6bpmRyOJonT3VoXnDag%3D%3D&name=Программа_тестирования_для_любителей.pdf&nosw=1",
}


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

            current["df"] = pd.concat([current["df"], new_df], ignore_index=True)
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
    df.fillna(method="ffill", inplace=True)

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
        df.fillna(method="ffill", inplace=True)
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
    - source_link: link
    """
    meta: Dict[str, Any] = {}
    parts = pdf_path.split(os.sep)
    folder = parts[-2] if len(parts) > 1 else ""
    filename = os.path.basename(pdf_path)
    fname_lower = filename.lower()

    # 1) Аудитория / уровень
    if "профессионалы" in folder.lower():
        meta["audience"] = "pro"
        source_link = prof_links
    elif "любители" in folder.lower():
        meta["audience"] = "amateur"
        source_link = lovers_link
    elif "isu" in folder.lower():
        meta["audience"] = "isu"
        source_link = ISU_links
    else:
        meta["audience"] = "unknown"

    # источник откуда скачан файл (веб ресурс)
    meta["source_link"] = source_link[filename]

    # 2) Дисциплина
    if "одиноч" in fname_lower or "singl" in fname_lower:
        meta["discipline"] = "singles"
    elif "парн" in fname_lower or "pair" in fname_lower:
        meta["discipline"] = "pairs"
    elif "танц" in fname_lower or "dance" in fname_lower:
        meta["discipline"] = "ice_dance"
    elif "синхрон" in fname_lower or "synchro" in fname_lower:
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
    if "правила" in fname_lower or "requirement" in fname_lower:
        meta["doc_type"] = "rules"
    elif (
        "технические требования" in fname_lower
        or "техтребован" in fname_lower
        or "technical" in fname_lower
    ):
        meta["doc_type"] = "technical_requirements"
    elif "руководств" in fname_lower or "handbook" in fname_lower:
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
            "markdown": table_md,
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
