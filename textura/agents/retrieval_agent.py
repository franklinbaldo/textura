from __future__ import annotations

from llama_index.core import Document, ServiceContext, VectorStoreIndex
from llama_index.vector_stores.milvus import MilvusVectorStore

try:
    from llama_index.llms.google_genai import GoogleGenAI
except Exception:  # pragma: no cover - optional dependency
    GoogleGenAI = None


class RetrievalAgent:
    """Simple retrieval agent backed by Milvus via LlamaIndex."""

    def __init__(
        self,
        collection_name: str = "textura",
        host: str = "localhost",
        port: str = "19530",
    ) -> None:
        self.vector_store = MilvusVectorStore(
            collection_name=collection_name, host=host, port=port
        )
        self.index = VectorStoreIndex.from_vector_store(self.vector_store)
        if GoogleGenAI is not None:
            self.llm = GoogleGenAI(model="models/gemini-pro")
        else:  # pragma: no cover - fallback if LlamaIndex lacks this class
            self.llm = None
        self.query_engine = self.index.as_chat_engine(
            service_context=ServiceContext.from_defaults(llm=self.llm)
        )

    def insert_documents(self, texts: list[str]) -> None:
        docs = [Document(text=t) for t in texts]
        self.index.insert_documents(docs)

    def query(self, question: str) -> str:
        response = self.query_engine.chat(question)
        return str(response)
