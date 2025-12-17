import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.load import dumps, loads
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from tools.models import get_tongyi_embeddings, get_tongyi_llm

embeddings = get_tongyi_embeddings()

# 创建 WebBaseLoader 实例，用于加载指定网页内容
loader = WebBaseLoader(
    # 目标网页地址
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    # BeautifulSoup 解析参数，仅提取指定 class 的元素
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            # 仅保留 post-content、post-title、post-header 三类元素
            class_=("post-content", "post-title", "post-header")
        )
    ),
)

doc = loader.load()

# print(doc)

# Split 文档
text_splitter = RecursiveCharacterTextSplitter(
    # 每个文档的最大字符数
    chunk_size=300,
    # 每个文档的最大字符数
    chunk_overlap=50,
)

splits = text_splitter.split_documents(doc)


# 从文档中创建 Chroma 向量数据库 并且存入本地
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# 从 Chroma 向量数据库中创建检索器
retriever = vectorstore.as_retriever()

template = """
    你是一个 AI 语言模型助手。
    你的任务是针对给定的用户问题生成五个不同的版本，
    以便从向量数据库中检索相关文档。通过从多个角度重构用户问题，
    帮助用户克服基于距离的相似性搜索的一些局限性。
    请用换行符分隔这些替代问题。原始问题：{question}
    """

# 编写 prompt 模板
prompt_perspectives = ChatPromptTemplate.from_template(template)

# 定义 llm
llm = get_tongyi_llm()


# 成chain
generate_queries = (
    prompt_perspectives | llm | StrOutputParser() | (lambda x: x.split("\n"))
)


def get_unique_union(documents: list[list]):
    # 将内部每个元素进行字符串处理
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # 去重
    unique_docs = list(set(flattened_docs))
    # 转换回文档对象
    return [loads(doc) for doc in unique_docs]


# 写第二条 chain 将生成的问题丢给检索器 然后去重
retrieval_chain = generate_queries | retriever.map() | get_unique_union

# 得到数据后 进行再次提交给大模型
template = """
    根据已经给出的上下文去回答下面的问题
    上下文：{context}
    问题：{question}

    输出要求：请在原文回答的基础上，再进行一次中文翻译
"""

prompt = ChatPromptTemplate.from_template(template)

# 定义 chain
final_rag_chain = (
    {"context": retrieval_chain, "question": lambda x: x["question"]}
    | prompt
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    result = final_rag_chain.invoke(
        {"question": "What is task decomposition for LLM agents?"}
    )
    print(result)
