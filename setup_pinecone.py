import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

def init_pinecone_index():
    """Verifica si el índice existe en Pinecone Serverless y lo crea si es necesario."""
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("INDEX_NAME", "rag-coderhouse-index-384")
    dimension = 384  # Dimensión requerida para text-embedding-3-small

    if not api_key:
        raise ValueError("Error: PINECONE_API_KEY no configurada en el archivo .env")

    pc = Pinecone(api_key=api_key)
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if index_name not in existing_indexes:
        print(f"Creando índice Serverless '{index_name}' en AWS (us-east-1)...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"Índice '{index_name}' creado exitosamente.")
    else:
        print(f"El índice '{index_name}' ya existe. Listo para usar.")

if __name__ == "__main__":
    init_pinecone_index()