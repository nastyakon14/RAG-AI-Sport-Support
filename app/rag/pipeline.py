from app.rag.generator import GeneratorFS
from app.rag.retriever import RetrieverFS


class RagFS:
    def __init__(self):
        self.retriever = RetrieverFS()
        self.generator = GeneratorFS()

    def find_answer(self, query):
        docs = self.retriever.get_retriever_answer(query, metadata_flag=False)
        result_answer = self.generator.get_llm_answer(query=query, docs=docs)

        return result_answer
