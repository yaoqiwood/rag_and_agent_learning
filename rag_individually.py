import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
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
# rag_prompt
# prompt_rag = hub.Client().pull("rlm/rag-prompt")


# 先将问题decomposition
template = """
    你是一个AI 助手，可以生成多个与输入的问题相关的子问题。
    你的目标是将输入的问题拆分成一组多个子问题/子任务，这些子问题/子任务是一组可以被独立回答的问题。
    生成多个与问题相关的子问题/子任务:{question}
    输出(3个)，并且不带序号:
    """

decomposition_prompt = ChatPromptTemplate.from_template(template)

llm = get_tongyi_llm(temperature=0)
chain_decomposition = (
    decomposition_prompt | llm | StrOutputParser() | (lambda x: x.split("\n"))
)

question = "What is task decomposition for LLM agents?"

# 调用 chain_decomposition 进行问题分解
sub_questions = chain_decomposition.invoke({"question": question})

llm_prompt = """
    你是一个AI 助手，你的任务是根据输入的问题和上下文，回答问题。
    问题:{question}
    上下文:{context}
    请根据上下文回答问题。
"""

# 创建 LLM 提示模板
llm_prompt_template = ChatPromptTemplate.from_template(llm_prompt)

rag_chain = llm_prompt_template | llm | StrOutputParser()

answers = []

# 根据问题获得文档
for q in sub_questions:
    retrieved_docs = retriever.invoke(q)
    answer = rag_chain.invoke({"question": q, "context": retrieved_docs})
    answers.append(answer)

new_template = """
    这是一组回答 Q+A 对：{context}
    请针对当前这个问题，一句上面的这组 Q+A 的问答对，综合合成后回答这个问题：{question}
"""

new_prompt_template = ChatPromptTemplate.from_template(new_template)

# Format Q 和 A 对
strQ_A = ""
for i, (q, a) in enumerate(zip(sub_questions, answers), start=1):
    strQ_A += f"Question {i}: {q} \n Answer {i}: {a} \n "

final_rag_chain = (
    {
        "context": lambda x: x["context"],
        "question": lambda x: x["question"],
    }
    | new_prompt_template
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    print(final_rag_chain.invoke({"context": strQ_A, "question": question}))
