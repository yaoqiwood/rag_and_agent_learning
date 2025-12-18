from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from tools.models import get_tongyi_llm
from vectorDb.rag_test_db import VectorDBManager

web_path = ("https://lilianweng.github.io/posts/2023-06-23-agent/",)

# 初始化数据库管理器 并且传入目标网页地址
db_manager = VectorDBManager(web_path)
# 加载并切分文档
db_manager.load_and_split()
# 构建数据库
db_manager.build_db()
# 获得检索器
retriever = db_manager.get_retriever()

# 定义 template
template = """
    please write a scientific paper passage to answer the question: {question}
    Passage:
"""

prompt_hyde = ChatPromptTemplate.from_template(template)

llm = get_tongyi_llm()

generate_docs_for_retrieval_chain = prompt_hyde | llm | StrOutputParser()

question = "What is task decomposition for LLM agents?"
# hyde_answer = generate_docs_for_retrieval_chain.invoke({"question": question})

# print(hyde_answer)

retrieval_chain = generate_docs_for_retrieval_chain | retriever

# docs = retrieval_chain.invoke({"question": question})

# 将文档作为上下文填入prompt
# 定义 template
template = """
    回答问题基于上下文:
    {context}
    问题: {question}
"""

final_prompt = ChatPromptTemplate.from_template(template)

final_rag_chain = (
    {
        "context": RunnableLambda[Any, list[str]](lambda x: {"question": x["question"]})
        | retrieval_chain,
        "question": lambda x: x["question"],
    }
    | final_prompt
    | llm
    | StrOutputParser()
)

result = final_rag_chain.invoke({"question": question})

print(result)
