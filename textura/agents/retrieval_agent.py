from __future__ import annotations

from typing import List

from llama_index.core import Document, ServiceContext, VectorStoreIndex
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.vector_stores.milvus import MilvusVectorStore


class RetrievalAgent:
    """Simple retrieval agent backed by Milvus via LlamaIndex."""

    def __init__(self, collection_name: str = "textura", host: str = "localhost", port: str = "19530") -> None:
        self.vector_store = MilvusVectorStore(collection_name=collection_name, host=host, port=port)
        self.index = VectorStoreIndex.from_vector_store(self.vector_store)
        self.llm = GoogleGenAI(model="models/gemini-pro") # Updated class and model name if necessary
        self.query_engine = self.index.as_chat_engine(service_context=ServiceContext.from_defaults(llm=self.llm))

    def insert_documents(self, texts: List[str]) -> None:
        docs = [Document(text=t) for t in texts]
        self.index.insert_documents(docs)

    def query(self, question: str) -> str:
        response = self.query_engine.chat(question)
        return str(response)
