from langchain_core.documents import Document

# 1. 直接手动定义抓取回来的数据
# 这里 page_content 填入一段模拟的字幕内容，用于后续的向量搜索练习
mock_content = """
In this video, we explore Self-reflective RAG using LangGraph, focusing on Self-RAG and Corrective RAG (CRAG). 
We discuss how to use evaluators to check the quality of retrieved documents and 
how to iterate on the generation process.
Self-RAG allows the model to grade its own answers, while CRAG helps in correcting 
the retrieval path if the initial results are irrelevant.
"""

# 2. 按照你提供的格式伪造 Metadata
docs = [
    Document(
        page_content=mock_content,
        metadata={
            "source": "pbAd8O1Lvm4",
            "title": "Self-reflective RAG with LangGraph: Self-RAG and CRAG",
            "description": "Unknown",
            "view_count": 11922,
            "thumbnail_url": "https://i.ytimg.com/vi/pbAd8O1Lvm4/hq720.jpg",
            "publish_date": "2024-02-07 00:00:00",
            "length": 1058,
            "author": "LangChain",
        },
    )
]
