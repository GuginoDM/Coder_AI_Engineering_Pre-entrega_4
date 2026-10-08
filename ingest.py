import os
import json
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

def get_embedding_model():
    """Selecciona el modelo de embeddings según la variable EMBEDDING_PROVIDER en el .env."""
    provider = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
    
    if provider == "huggingface":
        print("ℹ️ Modo: Usando embeddings de HuggingFace (all-MiniLM-L6-v2 - 384 dims)")
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    else:
        print("ℹ️ Modo: Usando embeddings de OpenAI (text-embedding-3-small - 384 dims)")
        # Forzamos 384 dimensiones para ser compatible con tu índice actual de Pinecone
        return OpenAIEmbeddings(model="text-embedding-3-small", dimensions=384)

def run_ingestion():
    raw_index_name = os.getenv("INDEX_NAME", "rag-coderhouse-index-384")
    index_name = raw_index_name.split("=")[-1].strip()
    
    # 1. Cargar documentos desde JSON
    json_path = os.path.join("data", "raw_docs.json")
    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    documents = []
    for item in raw_data:
        # Guardamos el texto original y la metadata requerida
        doc = Document(
            page_content=item["text"],
            metadata={
                "doc_id": item["doc_id"],
                "source": item["source"],
                "category": item["category"],
                "text": item["text"]  # Guardamos el texto original en metadata
            }
        )
        documents.append(doc)

    # 2. Splitter por caracteres
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Documentos procesados: {len(documents)} -> Chunks generados: {len(chunks)}")

    # 3. Guardar chunks procesados localmente para el recuperador BM25
    os.makedirs("data", exist_ok=True)
    chunks_dict = [
        {"page_content": c.page_content, "metadata": c.metadata} 
        for c in chunks
    ]
    with open(os.path.join("data", "processed_chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks_dict, f, ensure_ascii=False, indent=2)

    # 4. Generar embeddings usando la función dinámica y subir a Pinecone
    embeddings = get_embedding_model()
    
    print(f"Cargando vectores en Pinecone (Índice: {index_name})...")
    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name,
        namespace="coderhouse-namespace"
    )
    print("Ingesta completada exitosamente.")

if __name__ == "__main__":
    run_ingestion()