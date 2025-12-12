from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS


class RetrieverFS:
  def __init__(self, top_k = 100):    
  
    model_name = "BAAI/bge-m3"
    # model_kwargs = {'device': 'cuda'}
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True, 'batch_size': 32}

    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    
    self.vectorstore = FAISS.load_local(
                "faiss_index_bge_m3_2700",
                embeddings,
                allow_dangerous_deserialization=True
            )
    self.top_k = top_k
    # self.reranker = FlagReranker("BAAI/bge-reranker-large", use_fp16=True)



  def extract_query_metadata(self, question):
    """
    Извлечение метаданных из вопроса пользователя.
    discipline: single / pairs / ice_dance / synchro / None
    category: pro / amateur / None
    """
    q = question.lower()

    # дисциплина
    discipline = 'general'   # по умолчанию общий вопрос
    if "одиноч" in q or "single" in q:
        discipline = "singles"
    elif "парн" in q or "pairs" in q:
        discipline = "pairs"
    elif "танц" in q or "dance" in q:
        discipline = "ice_dance"
    elif "синхрон" in q or "synchro" in q:
        discipline = "synchro"

    # статус — профессионалы / любители
    category = 'pro'  # по умолчанию профессионалы
     # профессионалы
    if any(
        k in q
        for k in [
            "профессионал", "профи", "спорт высших достижений"
        ]
    ):
        category = "pro"

    # любители
    if any(
        k in q
        for k in [
            "любител", "массовое катание", "массовый прокат", "хобби",
            "дворов", "каток для всех", "public skating", "recreational"
        ]
    ):
        category = "amateur"

    if any(
        k in q
        for k in [
            "чемпионат", "олимп", "isu", "гран-при", "grand prix", "worlds"
        ]
    ):
        category = "isu"

    return {
        "discipline": discipline,
        "audience": category,
    }

  
  def get_answer_retriever(self,
    query,
    metadata_flag = False,
    reranker_flag = False
      ):

    if metadata_flag:
      # извлекаем метаданные из запроса 
      metadata_filter = self.extract_query_metadata(query)
      # print(metadata_filter)

      # фильтруемся по метаданным
      docs_answer = self.vectorstore.similarity_search(
            query = query,
            k = self.top_k, filter = metadata_filter
          )
    else:
      # без метаданных
      docs_answer = self.vectorstore.similarity_search(
            query = query,
            k = self.top_k
          )
      
    # if reranker_flag:   # очениваем качество реранкера
    #   pairs = [(query, d.page_content) for d in docs_answer]  
    #   scores = self.reranker.compute_score(pairs)
    #   ranked_docs = [
    #       doc for _, doc in sorted(zip(scores, docs_answer), key=lambda x: x[0], reverse=True)
    #   ]

    #   reranked_docs = ranked_docs[:5]  # топовые 5
    #   return reranked_docs
        
    # else:
    #   return docs_answer

    docs = [doc.page_content for doc in docs_answer]
    return docs