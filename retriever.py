import os
import json
from typing import List, Any
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Si la importación de LangChain falla, se usa una clase EnsembleRetriever basada en RRF
try:
    from langchain.retrievers import EnsembleRetriever
except (ImportError, ModuleNotFoundError):
    class EnsembleRetriever(BaseRetriever):
        retrievers: List[Any]
        weights: List[float]
        c: int = 60  # Constante RRF

        def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
            doc_scores = {}
            for retriever, weight in zip(self.retrievers, self.weights):
                docs = retriever.invoke(query)
                for rank, doc in enumerate(docs):
                    content_key = doc.page_content
                    score = weight * (1.0 / (rank + 1 + self.c))
                    if content_key not in doc_scores:
                        doc_scores[content_key] = {"doc": doc, "score": 0.0}
                    doc_scores[content_key]["score"] += score

            sorted_docs = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
            return [item["doc"] for item in sorted_docs]

load_dotenv()

def get_embedding_model():
    """Selecciona el modelo de embeddings según la variable EMBEDDING_PROVIDER en el .env."""
    provider = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
    
    if provider == "huggingface":
        print("ℹ️ Modo: Usando embeddings de HuggingFace (all-MiniLM-L6-v2 - 384 dims)")
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    else:
        print("ℹ️ Modo: Usando embeddings de OpenAI (text-embedding-3-small - 384 dims)")
        return OpenAIEmbeddings(model="text-embedding-3-small", dimensions=384)

class RAGSystem:
    def __init__(self, top_k: int = 5):
        raw_index_name = os.getenv("INDEX_NAME", "rag-coderhouse-index-384")
        self.index_name = raw_index_name.split("=")[-1].strip()
        
        self.top_k = top_k
        self.ensemble_retriever = self._build_ensemble_retriever()

    def _build_ensemble_retriever(self):
        # 1. Recuperador Vectorial (Pinecone con selección dinámica de embeddings)
        embeddings = get_embedding_model()
        
        vectorstore = PineconeVectorStore(
            index_name=self.index_name,
            embedding=embeddings,
            namespace="coderhouse-namespace"  # Alineado con ingest.py
        )
        pinecone_retriever = vectorstore.as_retriever(
            search_kwargs={"k": self.top_k}
        )

        # 2. Recuperador Léxico (BM25)
        chunks_path = os.path.join("data", "processed_chunks.json")
        if not os.path.exists(chunks_path):
            raise FileNotFoundError(
                f"No se encontró el archivo {chunks_path}. Verificá que exista en la carpeta 'data'."
            )

        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)

        bm25_docs = [
            Document(page_content=c["page_content"], metadata=c.get("metadata", {}))
            for c in chunks_data
        ]
        bm25_retriever = BM25Retriever.from_documents(bm25_docs)
        bm25_retriever.k = self.top_k

        # 3. Ensemble Retriever (60% Vectorial, 40% BM25)
        ensemble = EnsembleRetriever(
            retrievers=[pinecone_retriever, bm25_retriever],
            weights=[0.6, 0.4]
        )
        return ensemble

    def search(self, query: str):
        """Retorna los Top-K documentos usando la búsqueda híbrida."""
        return self.ensemble_retriever.invoke(query)

if __name__ == "__main__":
    rag = RAGSystem(top_k=5)
    query_prueba = "¿Cómo hacer un JOIN en Pandas?"
    resultados = rag.search(query_prueba)
    
    print(f"\nResultados para: '{query_prueba}'\n")
    for i, doc in enumerate(resultados, 1):
        print(f"[{i}] ID: {doc.metadata.get('doc_id', 'N/A')} | Texto: {doc.page_content[:150]}...")