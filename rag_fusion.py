import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.load import dumps, loads
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from tools.models import get_tongyi_embeddings, get_tongyi_llm

# 文档数据切片


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

# 获取 embeddings 模型
embeddings = get_tongyi_embeddings()

# 从文档中创建 Chroma 向量数据库 并且存入本地
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# 从 Chroma 向量数据库中创建检索器
retriever = vectorstore.as_retriever()


# 定义 prompt 模板
template = """
    你是一个 AI 助手，任务是根据用户输入的单一的问题，生成若干个不同的搜索词。
    生成若干个与问题{question}相关的搜索词。
    严格按照下面要求生成：
    输出：按照上面的要求输出（4 个）搜索词
    输出格式：每个搜索词占一行
"""

rag_fusion_prompt = ChatPromptTemplate.from_template(template)

# 定义 llm
llm = get_tongyi_llm(temperature=0)

generate_queries_chain = (
    rag_fusion_prompt | llm | StrOutputParser() | (lambda x: x.split("\n"))
)


# 设定一个 累加 1/k+rank的算法 然后进行重排
def reciprocal_rank_fusion(results: list[list], k=60):
    # 保存每个文档的分数
    fused_scores = {}
    # 遍历循环
    for docs in results:
        for i, doc in enumerate(docs):
            # 对文档进行序列化
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                # 不存在的记录则初始化分数为 0
                fused_scores[doc_str] = 0
            # 进行算法计算 1/rank+k 并且累加
            fused_scores[doc_str] += 1 / (i + k)
    # 重排元素 并且用元组存储 (文档, 分数)
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results


new_template = """
    根据已经给出的上下文去回答下面的问题
    上下文：{context}
    问题：{question}

    输出要求：请在原文回答的基础上，再进行一次中文翻译
"""

final_template = ChatPromptTemplate.from_template(new_template)

# 定义查询chain 流程 1.将问题丢给 llm 生成 5 个不同的版本 2.将这 5 个版本丢给检索器 3.将检索器返回的结果进行去重 4.将去重后的结果丢给 reciprocal_rank_fusion 函数 5.返回重排后的结果
final_chain = (
    {
        "context": generate_queries_chain | retriever.map() | reciprocal_rank_fusion,
        "question": lambda x: x["question"],
    }
    | final_template
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    print(
        final_chain.invoke({"question": "What is task decomposition for LLM agents?"})
    )
