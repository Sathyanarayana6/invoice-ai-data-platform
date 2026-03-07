"""
retrieval_chain.py
-------------------
The RAG pipeline — answers natural language questions using our invoice data.

RAG = Retrieval Augmented Generation
Step 1: RETRIEVE  — find relevant chunks from ChromaDB
Step 2: AUGMENT   — add those chunks to the prompt as context
Step 3: GENERATE  — LLM answers based on the retrieved context

We use a local LLM (no API key needed):
- Uses sentence-transformers for retrieval (already installed)
- Uses a simple extractive approach for answering
- This means answers come DIRECTLY from the data, not from LLM hallucination

For production you'd swap in OpenAI/Claude API — 
the retrieval logic stays identical, only the generation step changes.
"""

import json
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict
import os

# ── Config ─────────────────────────────────────────────────────────────────────

CHROMA_PATH     = "lake/chromadb"
COLLECTION      = "invoice_knowledge_base"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K           = 5   # How many chunks to retrieve per query

# ── ChromaDB Connection ────────────────────────────────────────────────────────

def get_collection():
    """Connects to existing ChromaDB collection using OpenAI embeddings."""
    import os
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set.")

    from rag.embed_and_index import OpenAIEmbeddingFunction
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    ef     = OpenAIEmbeddingFunction(api_key=api_key)

    collection = client.get_collection(
        name=COLLECTION,
        embedding_function=ef
    )
    return collection


# ── Retrieval ──────────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    collection,
    n_results: int = TOP_K,
    filter_dict: Dict = None
) -> List[Dict]:
    """
    Retrieves the most semantically relevant chunks for a query.

    filter_dict examples:
    - {"doc_type": "vendor_summary"}     → only vendor documents
    - {"risk_level": "CRITICAL"}         → only critical anomalies
    - {"doc_type": "monthly_trend"}      → only trend documents

    This is metadata filtering — powerful for scoping queries.
    """
    query_params = {
        "query_texts": [query],
        "n_results"  : n_results,
    }
    if filter_dict:
        query_params["where"] = filter_dict

    results   = collection.query(**query_params)
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    retrieved = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        retrieved.append({
            "text"      : doc,
            "metadata"  : meta,
            "similarity": round(1 - dist, 4)  # Convert distance to similarity
        })

    return retrieved


# ── Simple Answer Generation ───────────────────────────────────────────────────

def generate_answer(query: str, retrieved_chunks: List[Dict]) -> str:
    """
    Generates an answer using GPT-4o-mini with retrieved chunks as context.
    This is the core of RAG — the LLM only answers from our data.
    """
    from openai import OpenAI

    if not retrieved_chunks:
        return "No relevant information found in the knowledge base."

    api_key = os.environ.get("OPENAI_API_KEY")
    client  = OpenAI(api_key=api_key)

    # Build context string from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        doc_type   = chunk["metadata"].get("doc_type", "unknown")
        similarity = chunk["similarity"]
        context_parts.append(
            f"[Source {i} | Type: {doc_type} | Relevance: {similarity}]\n"
            f"{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    # Call GPT-4o-mini with context + question
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an invoice analytics assistant. "
                    "Answer questions using ONLY the provided context. "
                    "Be concise and specific. "
                    "If the context doesn't contain the answer, say so."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}"
            }
        ],
        temperature=0,       # 0 = deterministic, factual answers
        max_tokens=500
    )

    answer_text = response.choices[0].message.content

    # Format full response
    answer = f"""
Query: {query}
{'='*60}
{answer_text}
{'='*60}
Sources used: {len(retrieved_chunks)} chunks
    """.strip()

    return answer


# ── RAG Pipeline ───────────────────────────────────────────────────────────────

def ask(query: str, collection, filter_dict: Dict = None) -> str:
    """
    Full RAG pipeline in one function:
    1. Retrieve relevant chunks
    2. Generate answer from context
    """
    print(f"\n  Query: '{query}'")
    if filter_dict:
        print(f"  Filter: {filter_dict}")

    # Step 1: Retrieve
    chunks = retrieve(query, collection, filter_dict=filter_dict)
    print(f"  Retrieved {len(chunks)} chunks")

    # Step 2: Generate
    answer = generate_answer(query, chunks)
    return answer


# ── Demo Queries ───────────────────────────────────────────────────────────────

def run_demo_queries(collection):
    """Runs a set of example queries to demonstrate the RAG pipeline."""

    queries = [
        {
            "question": "Which vendors have the highest overdue rates?",
            "filter"  : {"doc_type": "vendor_summary"}
        },
        {
            "question": "What invoices are flagged as critical risk?",
            "filter"  : {"doc_type": "anomaly_alert"}
        },
        {
            "question": "What were the invoice trends in 2024?",
            "filter"  : {"doc_type": "monthly_trend"}
        },
        {
            "question": "Show me vendors with dispute problems",
            "filter"  : None
        },
        {
            "question": "Which invoices have zero dollar amounts?",
            "filter"  : {"doc_type": "anomaly_alert"}
        },
    ]

    answers = []
    for q in queries:
        print("\n" + "=" * 60)
        answer = ask(q["question"], collection, q["filter"])
        print(answer)
        answers.append({
            "question": q["question"],
            "answer"  : answer
        })

    # Save query examples to disk (good for GitHub README)
    os.makedirs("lake/rag_documents", exist_ok=True)
    with open("lake/rag_documents/query_examples.json", "w") as f:
        json.dump(answers, f, indent=2)
    print(f"\n  Saved query examples to lake/rag_documents/query_examples.json")

    return answers


# ── Main ───────────────────────────────────────────────────────────────────────

def run_retrieval_chain():
    print("=" * 50)
    print("RAG RETRIEVAL CHAIN — Starting")
    print("=" * 50)

    print("\n[1/2] Connecting to ChromaDB...")
    collection = get_collection()
    print(f"      Collection '{COLLECTION}' loaded: {collection.count()} vectors")

    print("\n[2/2] Running demo queries...")
    answers = run_demo_queries(collection)

    print("\n" + "=" * 50)
    print("RAG RETRIEVAL CHAIN — Complete")
    print(f"  Answered {len(answers)} queries from local knowledge base")
    print("=" * 50)

    return answers


if __name__ == "__main__":
    run_retrieval_chain()