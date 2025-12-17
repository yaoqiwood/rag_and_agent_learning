from operator import itemgetter

import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from tools.models import get_tongyi_embeddings, get_tongyi_llm

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

template = """
    你是一个AI 助手，可以生成多个与输入的问题相关的子问题。
    你的目标是将输入的问题拆分成一组多个子问题/子任务，这些子问题/子任务是一组可以被独立回答的问题。
    生成多个与问题相关的子问题/子任务:{question}
    输出(3个)，并且不带序号:
"""

prompt_decomposition = ChatPromptTemplate.from_template(template)
llm = get_tongyi_llm(temperature=0)

decomposition_chain = (
    prompt_decomposition | llm | StrOutputParser() | (lambda x: x.split("\n"))
)

question = "What are the main components of an LLM-powered autonomous agent system?"

questions = decomposition_chain.invoke({"question": question})

# decomposition_retriever_results = retriever.invoke(questions)


# 流程上是这样的 1.现将问题通过提示词，拆解成 3 个不同的子问/子任务 2.将这三个问题丢入向量数据库进行查询
# 3.将 Q1 的结果丢给 LLM 进行回答，然后和 Q2 的结果进行合并作为上下文继续丢给 LLM 进行回答，再将 Q3 的结果也如此做（如果有更多的问题
# 则继续丢给上下文合并，然后丢给 LLM 以此类推
final_template = """
    这是给出的问题你需要去回答：
    \n --- \n {question} \n --- \n
    这是结合了背景问题和回答的内容：
    \n --- \n {q_a_pairs} \n --- \n
    这是其余的与问题相关的上下文内容：
    \n --- \n {context} \n --- \n
    请利用以上上下文和背景问题和回答的内容去回答问题：{question}
"""

final_template_decomposition = ChatPromptTemplate.from_template(final_template)

# 写chain
final_chain = (
    {
        "question": itemgetter("question"),
        "q_a_pairs": itemgetter("q_a_pairs"),
        "context": itemgetter("context"),
    }
    | final_template_decomposition
    | llm
    | StrOutputParser()
)
q_a_pairs = ""

if __name__ == "__main__":
    # result = ""
    # # 把刚刚巡回的检索内容进行遍历
    # for i, doc in enumerate(decomposition_retriever_results):
    #     result = final_chain.invoke(
    #         {"question": question, "q_a_pairs": q_a_pairs, "context": doc}
    #     )
    #     q_a_pair = f"Question:{questions[i]}\nAnswer:{result}"
    #     q_a_pairs += "\n --- \n" + q_a_pair
    # print(result)

    context = ""
    for i, q in enumerate(questions):
        docs = retriever.invoke(q)
        context = "\n".join([p.page_content for p in docs])
        print(context)
        result = final_chain.invoke(
            {"question": question, "q_a_pairs": q_a_pairs, "context": context}
        )

        q_a_pair = f"Question:{q}\nAnswer:{result}"
        q_a_pairs += "\n --- \n" + q_a_pair
    print(result)
    pass
