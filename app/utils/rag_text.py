import re
from typing import Tuple

_HEADER_RE = re.compile(
    r"""(?im)^[ \t]*
        (?:<\s*(?:b|strong)\s*>\s*)?
        (ссылки|источники|sources|references)\s*:\s*
        (?:<\s*/\s*(?:b|strong)\s*>\s*)?
        $""",
    re.IGNORECASE | re.MULTILINE | re.VERBOSE,
)


def split_answer_and_links(text: str) -> Tuple[str, str | None]:
    """
    Возвращает:
      (текст_без_ссылок, блок_ссылок_или_None)

    Если найден заголовок "<b>Ссылки:</b>" (или Источники/Sources/References),
    то заголовок + всё ниже уходит во вторую часть.

    Если после заголовка ничего нет — вернём links=None, но заголовок из ответа уберём.
    """
    if not text:
        return "", None

    m = _HEADER_RE.search(text)
    if not m:
        return text, None

    answer = text[: m.start()].rstrip()

    tail = text[m.end() :].strip()
    if not tail:
        return answer, None

    links = text[m.start() :].strip()
    return answer, links
