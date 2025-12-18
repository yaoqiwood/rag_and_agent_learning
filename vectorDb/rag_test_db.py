# 创建 WebBaseLoader 实例，用于加载指定网页内容
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from tools.models import get_tongyi_embeddings


class VectorDBManager:
    def __init__(self, web_path, persist_directory="./chroma_db"):
        """
        初始化配置
        :param web_path: 目标网页地址元组
        :param persist_directory: Chroma 数据库本地存储路径
        """
        self.web_path = web_path
        self.persist_directory = persist_directory
        self.embeddings = self._get_embeddings()
        self.vectorstore = None

    def _get_embeddings(self):
        """私有方法：获取 Embeddings 模型"""
        # 假设你的 get_tongyi_embeddings 已经在外部定义好
        return get_tongyi_embeddings()

    def load_and_split(self):
        """第一步：加载网页并切分文档"""
        loader = WebBaseLoader(
            web_paths=self.web_path,
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header")
                )
            ),
        )
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
        )
        return text_splitter.split_documents(docs)

    def build_db(self):
        """第二步：创建并持久化向量数据库"""
        splits = self.load_and_split()

        # 创建并存入本地
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
        )
        print(f"数据库已保存至: {self.persist_directory}")
        return self.vectorstore

    def get_retriever(self, search_kwargs={"k": 3}):
        """第三步：获取检索器"""
        if not self.vectorstore:
            # 如果内存中没有 vectorstore，尝试从本地加载
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
            )

        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)
