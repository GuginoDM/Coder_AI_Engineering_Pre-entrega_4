# Coder_AI_Engineering_Pre-entrega_4
Pre-entrega 4: Sistema RAG escalable en la nube con Pinecone
Qué construir
Debes entregar un Módulo de Recuperación Escalable integrado en un repositorio de código. El artefacto principal es un servicio (o conjunto de scripts organizados) en Python que ejecute el flujo completo de un sistema RAG en la nube.

Componentes obligatorios:
Pipeline de Ingesta en Pinecone: Un script que tome un conjunto de documentos (PDFs, Markdown o JSON), los procese y los suba a un índice de Pinecone Serverless utilizando metadatos avanzados (fuente, página, etiquetas de categoría).
Recuperador Híbrido (Hybrid Retriever): Una implementación que combine búsqueda por similitud de vectores con búsqueda léxica (BM25) para mejorar la precisión en términos técnicos o nombres propios.
Script de Evaluación: Una utilidad que calcule al menos dos métricas fundamentales (Precision@k y Recall@k) utilizando un pequeño "Golden Set" de preguntas y respuestas de prueba.
Pasos sugeridos
Preparación de Infraestructura: Crea un índice Serverless en Pinecone (usa la dimensión 1536 si usas OpenAI text-embedding-3-small).
Ingesta Inteligente: No subas el texto plano. Crea un esquema donde guardes el texto original dentro de los metadatos de Pinecone para evitar consultas adicionales a una base de datos relacional.
Configuración de LangChain: Utiliza el PineconeVectorStore de LangChain o el SDK nativo de Pinecone para configurar el motor de búsqueda.
Implementación BM25: Configura un recuperador de LangChain que use BM25Retriever y combínalo con el de Pinecone usando un EnsembleRetriever.
Evaluación Local: Crea un pequeño archivo JSON con pares {"pregunta": "...", "documento_id_esperado": "..."} y mide cuántos de esos documentos aparecen efectivamente en el Top-5 recuperado.
Errores comunes a evitar
Mismatch de Dimensiones: Intentar subir embeddings de 1536 dimensiones a un índice configurado con 512 o 768.
Ignorar el Namespace: En aplicaciones multi-inquilino o con distintos tipos de datos, no usar namespaces en Pinecone hará que la búsqueda sea ruidosa y lenta.
Subestimar el Chunking: Chunks muy pequeños pierden el contexto semántico; muy grandes diluyen la precisión del embedding. Busca un punto medio (~500-800 tokens).
