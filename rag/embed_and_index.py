"""
embed_and_index.py
-------------------
Converts text chunks into vector embeddings using OpenAI
and stores them in ChromaDB.

Embedding model: text-embedding-3-small
- Fast, cheap (~$0.00002 per 1K tokens)
- 1536 dimensions
- Best for semantic search tasks
"""

import json
import os
import chromadb
from typing import List, Dict
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────

DOCS_PATH       = "lake/rag_documents/chunks.json"
CHROMA_PATH     = "lake/chromadb"
COLLECTION      = "invoice_knowledge_base"
EMBEDDING_MODEL = "text-embedding-3-small"

# ── OpenAI Embedding Function ──────────────────────────────────────────────────

class OpenAIEmbeddingFunction:
    """
    Custom embedding function that uses OpenAI's embedding API.
    ChromaDB calls this automatically when adding or querying documents.
    """
    def __init__(self, api_key: str, model: str = EMBEDDING_MODEL):
        self.client = OpenAI(api_key=api_key)
        self.model  = model

    def __call__(self, input: List[str]) -> List[List[float]]:
        """Embeds a list of texts and returns their vectors."""
        response = self.client.embeddings.create(
            input=input,
            model=self.model
        )
        return [item.embedding for item in response.data]


# ── ChromaDB Setup ─────────────────────────────────────────────────────────────

def get_chroma_client():
    """Creates a persistent ChromaDB client."""
    return chromadb.PersistentClient(path=CHROMA_PATH)


# ── Indexing ───────────────────────────────────────────────────────────────────

def index_chunks(chunks: List[Dict], client, ef):
    """Indexes all chunks into ChromaDB with OpenAI embeddings."""

    try:
        client.delete_collection(COLLECTION)
        print(f"      Deleted existing collection: {COLLECTION}")
    except:
        pass

    collection = client.create_collection(
        name=COLLECTION,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )

    BATCH_SIZE = 50
    total      = len(chunks)

    for i in range(0, total, BATCH_SIZE):
        batch     = chunks[i : i + BATCH_SIZE]
        ids       = [c["chunk_id"]   for c in batch]
        documents = [c["chunk_text"] for c in batch]

        metadatas = []
        for c in batch:
            meta = {
                "doc_id"   : str(c.get("doc_id",   "")),
                "doc_type" : str(c.get("doc_type", "")),
                "chunk_idx": int(c.get("chunk_idx", 0)),
            }
            if "vendor_id"  in c: meta["vendor_id"]  = str(c["vendor_id"])
            if "risk_level" in c: meta["risk_level"]  = str(c["risk_level"])
            if "year_month" in c: meta["year_month"]  = str(c["year_month"])
            if "invoice_id" in c: meta["invoice_id"]  = str(c["invoice_id"])
            metadatas.append(meta)

        collection.add(
            ids       = ids,
            documents = documents,
            metadatas = metadatas
        )

        print(f"      Indexed batch {i//BATCH_SIZE + 1} "
              f"({min(i+BATCH_SIZE, total)}/{total} chunks)")

    return collection


# ── Main ───────────────────────────────────────────────────────────────────────

def run_embedding_and_indexing():
    print("=" * 50)
    print("EMBEDDING & INDEXING — Starting")
    print("=" * 50)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    print("\n[1/4] Loading chunks...")
    with open(DOCS_PATH, "r") as f:
        chunks = json.load(f)
    print(f"      Loaded {len(chunks)} chunks")

    print(f"\n[2/4] Initializing ChromaDB at {CHROMA_PATH}...")
    os.makedirs(CHROMA_PATH, exist_ok=True)
    client = get_chroma_client()

    print(f"\n[3/4] Initializing OpenAI embedding function ({EMBEDDING_MODEL})...")
    ef = OpenAIEmbeddingFunction(api_key=api_key)
    print(f"      Ready")

    print(f"\n[4/4] Indexing {len(chunks)} chunks into ChromaDB...")
    collection = index_chunks(chunks, client, ef)

    count = collection.count()
    print(f"\n      ChromaDB collection '{COLLECTION}' contains {count} vectors")

    print(f"\n  Test query: 'overdue invoices vendor risk'")
    results = collection.query(
        query_texts=["overdue invoices vendor risk"],
        n_results=3
    )
    print(f"  Top 3 results:")
    for i, doc in enumerate(results["documents"][0]):
        print(f"    [{i+1}] {doc[:100]}...")

    print("\n" + "=" * 50)
    print("EMBEDDING & INDEXING — Complete")
    print("=" * 50)

    return collection


if __name__ == "__main__":
    run_embedding_and_indexing()