from __future__ import annotations

from ..rag.pipeline import RagFS


class RagFSBench(RagFS):
    """RagFS wrapper with a stable interface for benchmark/test building."""

    def find_answer(self, query: str) -> tuple[str, list[str]]:
        docs = self.retriever.get_retriever_answer(query, metadata_flag=False)
        result_answer = self.generator.get_llm_answer(query=query, docs=docs)

        docs_str = [doc.page_content for doc in docs]
        return result_answer, docs_str
