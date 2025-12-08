import re
from langchain_community.document_loaders import WebBaseLoader

# Функция для извлечения информации с веб-сайтов (с сайта ISU)
def load_url(url):
    '''DataLoader for url
    Input: url
    Output: split'''
    
    loader_web = WebBaseLoader(url)
    docs = loader_web.load()
     # разбиваем склеенные слова по заглавной букве
    # SustainabilityPressAnti-dopingSafeguardingISU  --> Sustainability Press Anti-doping Safeguarding ISU
    
    for i in range(len(docs)):
        s = docs[i].page_content 
        docs[i].page_content = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', ' ', s)
    return docs