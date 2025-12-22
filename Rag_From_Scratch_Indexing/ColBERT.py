import requests
from ragatouille import RAGPretrainedModel

RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")


def get_wikipedia_page(title: str):

    # Wikipedia API endpoint
    URL = "https://en.wikipedia.org/w/api.php"

    # Parameters for the API request
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
    }

    # Custom User-Agent header to comply with Wikipedia's best practices
    headers = {"User-Agent": "RAGatouille_tutorial/0.0.1 (ben@clavie.eu)"}

    response = requests.get(URL, params=params, headers=headers)

    data = response.json()

    # Extracting page content
    page = next(iter(data["query"]["pages"].values()))
    return page["extract"] if "extract" in page else None


full_document = get_wikipedia_page("Hayao_Miyazaki")

# 自动建库 索引
RAG.index(
    collection=[full_document],
    index_name="Hayao_Miyazaki",
    max_document_length=1024,
    split_documents=True,
)


# 取出
retriever = RAG.as_langchain_retriever(k=3)
result = retriever.invoke("What animation studio did Miyazaki found?")
print(result)
