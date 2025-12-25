from langchain_classic.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import settings


class GeneratorFS:
    def __init__(
        self,
        model="openai/gpt-5.1",  # для agentplatform_api_key
        # model="gpt-4.1",
    ):
        self.API_KEY = settings.agentplatform_key
        self.OPENAI_API_KEY = settings.agentplatform_key
        # self.OPENAI_API_KEY = settings.openai_api_key
        self.OPENAI_URL = "https://litellm.tokengate.ru/v1"
        self.SYSTEM_PROMPT = """You are an assistant and consultant on figure skating rules (ISU, professional, and amateur).
        Please answer strictly within the context of the information provided.

        Requirements:
        - If your answer is clearly out of context, indicate that it is not provided in the documents you have.
        - Do not invent non-existent rules.
        - Answer the question in the language in which it was asked (for a question in Russian, the answer is Russian; for a question in English, the answer is English).
        - Provide a brief answer to the question posed; do not include unnecessary information irrelevant to the question.
        - If your answer does not comply with the rules, indicate so.
        - If the information does not comply with the context, but is related to figure skating, you can answer based on your general knowledge.

        OUTPUT FORMAT (STRICT TELEGRAM HTML — LIMITED TAGS):
        - Return ONLY Telegram HTML-formatted text.
        - Allowed tags ONLY: <b>, <i>, <u>, <s>, <code>, <a href="URL">, <pre>
        - Do NOT use any other tags (FORBIDDEN: <p>, <br>, <ul>, <ol>, <li>, <blockquote>, <div>, <span>, <h1>, etc.).
        - Use plain newline characters for line breaks (\\n).
        - Do NOT use Markdown formatting (** __ ### etc.). Only the allowed HTML tags.
        - If you include a link, use ONLY <a href="URL">text</a>. Do not output raw URLs.

        RESPONSE STRUCTURE (MANDATORY):
        1) Start with a header line:
        <b>Ответ:</b>
        Then on the next line provide the answer text.

        2) If you need enumeration/categories, format as plain text bullet list using "•":
        • item 1
        • item 2
        Do NOT use HTML list tags.

        3) If the answer is out of context and not present in provided documents:
        - Explicitly state that inside the Answer section.
        - If you provide general knowledge, clearly label it as general knowledge:
        <b>Примечание (общие знания):</b> ...

        4) ALWAYS end with a separate section called “Ссылки” (or “References” if the question is in English):
        <b>Ссылки:</b>
        1) FILE NAME — стр. X — раздел/Rule/таблица (если применимо)
        2) ...

        REFERENCES RULES (MANDATORY):
        - The “Ссылки/References” section must be present in every answer.
        - Each reference must include at least: file name + page number.
        - If applicable, also include: section name / rule number / table name.
        - If there are no supporting documents, still include the section and write:
        1) Нет прямых ссылок в предоставленных документах.

        QUALITY CONSTRAINTS:
        - Be concise and directly answer the question.
        - Do not include irrelevant context.
        - Do not fabricate rules, citations, file names, or page numbers.
        - Ensure the final output contains only allowed tags and plain text with newlines.
        """

        self.llm = ChatOpenAI(
            model=model,
            api_key=self.OPENAI_API_KEY,
            base_url=self.OPENAI_URL,
            temperature=0,  # вариативность модели
        )

    def get_llm_answer(
        self,
        query,
        docs,  # выход ретривера
    ):
        # извлекаем метаданные, чтобы понять что за источник
        context_blocks = []
        for i, d in enumerate(docs):
            meta = d.metadata
            ref = f"""
              [Источник {i + 1}]
              Файл: {meta.get('source', 'unknown')}
              Страница: {meta.get('page_number', 'N/A')}
              Секция: {meta.get('section', 'N/A')}
              """
            context_blocks.append(d.page_content + "\n" + ref)
        context = "\n\n".join(
            context_blocks
        )  # найденные документы из ретривера + метаданные по источникам

        prompt = f"""
            Question:
            {query}

            Context:
            {context}
            """

        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        # ответ LLM
        response = self.llm.invoke(messages).content

        return response
