import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.runnables import RunnableLambda
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

examples = [
    {
        "input": "Could the members of The Police perform lawful arrests?",
        "output": "what can the members of The Police do?",
    },
    {
        "input": "Jan Sindel’s was born in what country?",
        "output": "what is Jan Sindel’s personal history?",
    },
]

# messages 上下文模板对应
template = [("human", "{input}"), ("ai", "{output}")]

example_prompt = ChatPromptTemplate.from_messages(template)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,  # 每个example的模板
    examples=examples,  # 示例列表
)

# 将few_shot_prompt 加入到 prompt 中
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert at world knowledge. Your task is to step back and paraphrase a question\
                to a more generic step-back question, which is easier to answer. Here are a few examples:",
        ),
        few_shot_prompt,
        ("user", "{question}"),
    ]
)

llm = get_tongyi_llm(temperature=0)

generate_queries_step_back = prompt | llm | StrOutputParser()

question = "What is task decomposition for LLM agents?"

step_back_question = generate_queries_step_back.invoke({"question": question})

response_prompt_template = """
    You are an expert of world knowledge. I am going to ask you a question.
    Your response should be comprehensive and not contradicted with the following context if the are relevant.
    Otherwise, ignore them if they are not relevant.
    # {normal_context}
    # {step_back_context}
    # Original Question: {question}
    # Answer:
"""

final_prompt = ChatPromptTemplate.from_template(response_prompt_template)

final_chain = (
    {
        "normal_context": RunnableLambda(lambda x: x["question"]) | retriever,
        "step_back_context": generate_queries_step_back | retriever,
        "question": lambda x: x["question"],
    }
    | final_prompt
    | llm
    | StrOutputParser()
)

result = final_chain.invoke({"question": question})

if __name__ == "__main__":
    print(result)
