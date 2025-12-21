from langchain_community.utils.math import cosine_similarity
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.passthrough import RunnablePassthrough

from tools.models import get_tongyi_embeddings, get_tongyi_llm

physics_template = """You are a very smart physics professor. \
You are great at answering questions about physics in a concise and easy to understand manner. \
When you don't know the answer to a question you admit that you don't know.

Here is a question:
{question}"""

math_template = """You are a very good mathematician. You are great at answering math questions. \
You are so good because you are able to break down hard problems into their component parts, \
answer the component parts, and then put them together to answer the broader question.

Here is a question:
{question}"""

embeddings = get_tongyi_embeddings()

prompt_templates = [physics_template, math_template]

# 嵌入为向量
prompt_embeddings = embeddings.embed_documents(prompt_templates)

# print(prompt_embeddings)

llm = get_tongyi_llm(temperature=0)


def prompt_router(input):
    print(input)
    # 将问题嵌入
    query_embedding = embeddings.embed_query(input["question"])
    # 比对相似度
    similarities = cosine_similarity([query_embedding], prompt_embeddings)[0]
    # 找到相似度最高的模板
    most_similar = prompt_templates[similarities.argmax()]
    print("Using MATH" if most_similar == math_template else "Using PHYSICS")
    # print(most_similar)
    return PromptTemplate.from_template(most_similar)


final_chain = (
    {"question": RunnablePassthrough()}
    | RunnableLambda(prompt_router)
    | llm
    | StrOutputParser()
)


if __name__ == "__main__":
    print(final_chain.invoke("What's a black hole?"))
