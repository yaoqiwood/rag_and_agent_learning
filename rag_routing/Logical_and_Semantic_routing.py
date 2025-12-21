from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from pydantic.v1 import BaseModel, Field
from sqlalchemy.util.typing import Literal

from tools.models import get_tongyi_llm

# 把当前文件的上一级目录加入搜索路径


class RouteQuery(BaseModel):
    datasource: Literal["python_docs", "js_docs", "golang_docs"] = Field(
        description="Given a user question choose which datasource would be most relevant for answering their question"
    )


llm = get_tongyi_llm(temperature=0)

structured_llm = llm.with_structured_output(RouteQuery)

system_prompt = """
    You are an expert at routing a user question to the appropriate datasource.
    Based on the programming language the question is referring to,
    route it to the relevant data source.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}"),
    ]
)
router = prompt | structured_llm

question = """Why doesn't the following code work:

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(["human", "speak in {language}"])
prompt.invoke("french")
"""

res = router.invoke({"question": question})
print(res.datasource)


def choose_route(result):
    if "python_docs" in result.datasource.lower():
        # 逻辑：如果问题中包含Python相关的关键词，路由到Python文档
        return "chain for python_docs"
    elif "js_docs" in result.datasource.lower():
        # 逻辑：如果问题中包含JS相关的关键词，路由到JS文档
        return "chain for js_docs"
    else:
        # 逻辑：如果问题中不包含Python或JS相关的关键词，路由到Golang文档
        return "golang_docs"


temp_chain = router | RunnableLambda(choose_route)

print(temp_chain.invoke({"question": question}))
