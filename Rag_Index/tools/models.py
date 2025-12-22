from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings

from .conf import TONGYI_EMBEDDING_API_KEY

# print(TONGYI_EMBEDDING_API_KEY)


def get_tongyi_embeddings():
    return DashScopeEmbeddings(
        model="text-embedding-v4",
        dashscope_api_key=TONGYI_EMBEDDING_API_KEY,
    )


def get_tongyi_llm(temperature: float = 0.7):
    return ChatTongyi(
        model_name="qwen-plus",
        api_key=TONGYI_EMBEDDING_API_KEY,
        temperature=temperature,
    )
