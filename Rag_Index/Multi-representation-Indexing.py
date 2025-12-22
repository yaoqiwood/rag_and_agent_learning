import uuid

from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryByteStore
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from tools.models import get_tongyi_embeddings, get_tongyi_llm

# 读数据
url = ("https://lilianweng.github.io/posts/2023-06-23-agent/",)
loader = WebBaseLoader(url)
docs = loader.load()
url2 = ("https://lilianweng.github.io/posts/2024-02-05-human-data-quality/",)
loader = WebBaseLoader(url2)
docs.extend(loader.load())

llm = get_tongyi_llm(temperature=0)

chain = (
    {"doc": lambda x: x.page_content}
    | ChatPromptTemplate.from_template("请总结以下文档的内容：{doc}")
    | llm
    | StrOutputParser()
)

summarise = chain.batch(docs, {"max_concurrency": 5})
print(summarise)

vectorstore = Chroma(
    collection_name="summarise", embedding_function=get_tongyi_embeddings()
)
# 运行内存存储
store = InMemoryByteStore()

id_key = "doc_id"

# 巡回器 开始准备索引
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key,
)

# 对每一个文档都生成一个绝对唯一的 id
doc_ids = [str(uuid.uuid4()) for _ in docs]

# 关联 id
summary_docs = [
    Document(page_content=s, metadata={id_key: doc_ids[i]})
    for i, s in enumerate(summarise)
]

# 向量化 并将结果存储到 vectorstore 中
retriever.vectorstore.add_documents(summary_docs)
# 将键值对存储到     byte_store=store 中
retriever.docstore.mset(list(zip(doc_ids, docs)))
query = "Memory in agents"
sub_docs = retriever.vectorstore.similarity_search(query)
