import os
import json
from retriever import RAGSystem

# Dataset de prueba con consultas y los IDs de documentos relevantes esperados
TEST_BENCHMARK = [
    {
        "query": "¿Cómo hacer un JOIN en Pandas?",
        "expected_doc_ids": ["doc_pd_02", "doc_pd_01"]
    },
    {
        "query": "¿Qué función modifica el contexto de filtro en DAX?",
        "expected_doc_ids": ["doc_pbi_01"]
    },
    {
        "query": "¿Cómo funciona LCEL en LangChain?",
        "expected_doc_ids": ["doc_lc_01"]
    },
    {
        "query": "¿Cómo almacena vectores Pinecone Serverless?",
        "expected_doc_ids": ["doc_pc_01"]
    }
]

def calculate_metrics(retrieved_ids, expected_ids, k=5):
    """Calcula Precision@k y Recall@k."""
    retrieved_k = retrieved_ids[:k]
    relevant_retrieved = set(retrieved_k).intersection(set(expected_ids))
    
    precision = len(relevant_retrieved) / k
    recall = len(relevant_retrieved) / len(expected_ids) if expected_ids else 0.0
    
    return precision, recall

def run_evaluation():
    rag = RAGSystem(top_k=5)
    total_precision = 0.0
    total_recall = 0.0
    num_queries = len(TEST_BENCHMARK)

    print("=" * 65)
    print(" EVALUACIÓN DEL SISTEMA RAG HÍBRIDO (Precision@5 & Recall@5)")
    print("=" * 65)

    for idx, item in enumerate(TEST_BENCHMARK, 1):
        query = item["query"]
        expected_ids = item["expected_doc_ids"]
        
        results = rag.search(query)
        retrieved_ids = [doc.metadata.get("doc_id", "N/A") for doc in results]
        
        p_at_k, r_at_k = calculate_metrics(retrieved_ids, expected_ids, k=5)
        
        total_precision += p_at_k
        total_recall += r_at_k
        
        print(f"\n[Consulta {idx}] '{query}'")
        print(f"  • Recuperados (Top-5): {retrieved_ids}")
        print(f"  • Esperados:          {expected_ids}")
        print(f"  • Precision@5: {p_at_k:.2f} | Recall@5: {r_at_k:.2f}")

    avg_precision = total_precision / num_queries
    avg_recall = total_recall / num_queries

    print("\n" + "=" * 65)
    print(" RESULTADOS GLOBALES DEL BENCHMARK")
    print("=" * 65)
    print(f" Precision@5 Promedio: {avg_precision:.4f} ({avg_precision * 100:.1f}%)")
    print(f" Recall@5 Promedio:    {avg_recall:.4f} ({avg_recall * 100:.1f}%)")
    print("=" * 65)

if __name__ == "__main__":
    run_evaluation()