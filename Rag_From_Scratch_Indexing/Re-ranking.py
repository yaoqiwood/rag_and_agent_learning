import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.load import dumps, loads
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from tools.models import get_tongyi_embeddings, get_tongyi_llm

web_paths = ("https://lilianweng.github.io/posts/2023-06-23-agent/",)
loader = WebBaseLoader(
    web_paths=web_paths,
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)

blog_docs = loader.load()

# split
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300, chunk_overlap=50
)

splits = text_splitter.split_documents(blog_docs)

embeddings = get_tongyi_embeddings()

# 分批处理：通义 embedding 接口一次最多支持 10 条
batch_size = 10
vectorstore = None
for i in range(0, len(splits), batch_size):
    batch = splits[i : i + batch_size]
    if vectorstore is None:
        # 第一批直接创建
        vectorstore = Chroma.from_documents(documents=batch, embedding=embeddings)
    else:
        # 后续批次增量添加
        vectorstore.add_documents(documents=batch)

# vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

retriever = vectorstore.as_retriever()


# RAG-Fusion
template = """You are a helpful assistant that generates multiple search
queries based on a single input query. \n
Generate multiple search queries related to: {question} \n
Output (4 queries):"""

prompt_rag_fusion = ChatPromptTemplate.from_template(template)

llm = get_tongyi_llm(temperature=0)

generate_queries = (
    prompt_rag_fusion | llm | StrOutputParser() | (lambda x: x.split("\n"))
)


# 重排序
def reciprocal_rank_fusion(results: list[list], k=60):
    # 初始化一个空字典来存储每个文档的融合分数
    fused_scores = {}
    # 遍历每个子列表（每个子列表代表一个查询的文档结果）
    for docs in results:
        # 遍历每个文档，计算其融合分数
        for rank, doc in enumerate(docs):
            # 序列化文档
            doc_str = dumps(doc)
            # 如果不在 fused_scores 中，初始化其分数为 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # 计算当前文档的融合分数并更新 fused_scores rank 是i, 出现的相关性次数越多, 分数越高
            fused_scores[doc_str] += 1 / (rank + k)

    # 重排文档列表，根据融合分数从高到低排序
    rerank_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return rerank_results


question = "What is task decomposition for LLM agents?"
retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion
docs = retrieval_chain_rag_fusion.invoke({"question": question})
print(docs[:2])
