"""
run_phase4.py
--------------
Runs the full Phase 4 RAG pipeline:
  1. Document chunking — convert Gold data to text
  2. Embed and index — generate vectors, store in ChromaDB
  3. Retrieval chain — answer natural language queries
"""

from rag.document_chunker  import run_document_chunking
from rag.embed_and_index    import run_embedding_and_indexing
from rag.retrieval_chain    import run_retrieval_chain

if __name__ == "__main__":
    print("\n PHASE 4 — RAG PIPELINE\n")

    print("STEP 1: Chunking documents...")
    run_document_chunking()

    print("\nSTEP 2: Embedding and indexing into ChromaDB...")
    run_embedding_and_indexing()

    print("\nSTEP 3: Running retrieval chain demo...")
    run_retrieval_chain()

    print("\nPhase 4 Complete!")
    print("  ChromaDB index saved to lake/chromadb/")
    print("  Query examples saved to lake/rag_documents/query_examples.json")