from datetime import date, datetime

from datas.fake_datas import docs
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from models.TutorialSearch import TutorialSearch

from tools.models import get_tongyi_embeddings, get_tongyi_llm
from tools.proxy import set_proxy

set_proxy()
# 因为 youtube 反爬机制数据不能轻易获取 所以改成从 fake_datas 中加载


# youtube_url = "https://www.youtube.com/watch?v=pbAd8O1Lvm4"
# docs = YoutubeLoader.from_youtube_url(
#     "https://www.youtube.com/watch?v=pbAd8O1Lvm4",
#     add_video_info=False,  # 重点：设为 False，跳过 pytube 获取元数据的逻辑
# ).load()

# print(docs)

# 将假数据存储到 chroma 中
embeddings = get_tongyi_embeddings()
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="tutorial_videos",
)

system = """You are an expert at converting user questions into database queries. \
You have access to a database of tutorial videos about a software library for building LLM-powered applications. \
Given a question, return a database query optimized to retrieve the most relevant results.

If there are acronyms or words you are not familiar with, do not try to rephrase them."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)
llm = get_tongyi_llm(temperature=0)
# 结构化输出
structured_llm = llm.with_structured_output(TutorialSearch)
query_analyzer_chain = prompt | structured_llm

question = "videos on chat langchain published in 2023"
result = query_analyzer_chain.invoke({"question": question})
# print(result)


# 最后去chroma查相应的数据即可
# 假设你的 TutorialSearch 模型定义的字段对应文档中的 metadata
# result 的内容是：content_search='building a rag system from scratch' title_search='RAG from scratch' ...


def search_chroma(vectorstore, search_params: TutorialSearch):
    # 1. 准备过滤条件 (Chroma 使用 MongoDB 风格的语法)
    filters = []

    # 视图数量过滤
    if search_params.min_view_count is not None:
        filters.append({"view_count": {"$gte": search_params.min_view_count}})
    if search_params.max_view_count is not None:
        filters.append({"view_count": {"$lte": search_params.max_view_count}})

    # 时长过滤
    if search_params.min_length_sec is not None:
        filters.append({"length": {"$gte": search_params.min_length_sec}})
    if search_params.max_length_sec is not None:
        filters.append({"length": {"$lte": search_params.max_length_sec}})

    # 日期过滤 (假设 metadata 里的日期是 timestamp 或 可比较字符串)
    if search_params.earliest_publish_date:
        d = search_params.earliest_publish_date

        # 1. 安全转换：不管它是字符串还是日期对象，都统一转为 YYYYMMDD 格式的整数
        if isinstance(d, (datetime, date)):
            date_int = int(d.strftime("%Y%m%d"))
        else:
            # 如果是字符串 "2023-01-01"，先去掉横杠再转整数
            date_int = int(str(d).replace("-", "").split(" ")[0])

        filters.append({"publish_date": {"$gte": date_int}})

    # 合并过滤条件
    where_filter = None
    if len(filters) == 1:
        where_filter = filters[0]
        print("filters:", filters)
    elif len(filters) > 1:
        where_filter = {"$and": filters}
    print("where_filter:", where_filter)

    # 2. 执行查询
    # 优先使用 content_search 进行向量搜索，如果没有则用 title_search
    query_str = search_params.content_search or search_params.title_search or ""

    final_docs = vectorstore.similarity_search(query_str, k=4, filter=where_filter)
    return final_docs


# 执行查询
final_results = search_chroma(vectorstore, result)
print(final_results)
