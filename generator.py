import os
from langchain_openai import ChatOpenAI
from langchain_classic.schema import HumanMessage, SystemMessage


class GeneratorFS:

  def __init__(self,
              #  model = "openai/gpt-5.1"  # для agentplatform_api_key
               model = 'gpt-4.1'
               ):

    # self.API_KEY = os.getenv("AGENTPLATFORM_KEY")
    # self.OPENAI_API_KEY = os.getenv("AGENTPLATFORM_KEY")
    self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    # self.OPENAI_URL = "https://litellm.tokengate.ru/v1"
    self.SYSTEM_PROMPT = '''You are an assistant and consultant on figure skating rules (ISU, professional, and amateur).
                    Please answer strictly within the context of the information provided.

                    Requirements:
                    - If your answer is clearly out of context, indicate that it is not provided in the documents you have.
                    - Do not invent non-existent rules.
                    - At the end, provide a list of references, including the file name, page, and (if applicable) section.
                    - Answer the question in the language in which it was asked (for a question in Russian, the answer is Russian; for a question in English, the answer is English).
                    - Provide a brief answer to the question posed; do not include unnecessary information irrelevant to the question.
                    - If your answer does not comply with the rules, indicate so.
                    - If the information does not comply with the context, but is related to figure skating, you can answer based on your general knowledge.
                '''

    self.llm = ChatOpenAI(
        model = model,
        api_key = self.OPENAI_API_KEY,
        # base_url = self.OPENAI_URL,
        temperature = 0.2, # вариативность модели
    )

  def get_llm_answer(self, query,
                     docs # выход ретривера
                     ):

    # извлекаем метаданные, чтобы понять что за источник
    context_blocks = []
    for i, d in enumerate(docs):
      meta = d.metadata
      ref = f"""
              [Источник {i+1}]
              Файл: {meta.get('source', 'unknown')}
              Страница: {meta.get('page_number', 'N/A')}
              Секция: {meta.get('section', 'N/A')}
              """
      context_blocks.append(d.page_content + "\n" + ref)
    context = '\n\n'.join(context_blocks)  # найденные документы из ретривера + метаданные по источниками

    prompt = f"""
            Question:
            {query}

            Context:
            {context}
            """

    messages = [
        SystemMessage(content=self.SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]

    # ответ LLM
    response = self.llm.invoke(messages).content

    return response