"""
EA AI Assistant — RAG Pipeline
Loads documents → chunks → embeds → stores in ChromaDB → answers queries
"""

import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()  # загружает .env автоматически

import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "ea_documents"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def _openai_ef():
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"],
        model_name="text-embedding-3-small",
    )


def get_collection():
    db = chromadb.PersistentClient(path=CHROMA_PATH)
    return db.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_openai_ef(),
    )


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end].strip())
        start += size - overlap
    return [c for c in chunks if c]


def ingest_document(file_path: str) -> int:
    """Load a .txt or .md file and store chunks in ChromaDB."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    text = path.read_text(encoding="utf-8")
    chunks = chunk_text(text)

    collection = get_collection()
    ids = [f"{path.stem}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": path.name, "chunk": i} for i in range(len(chunks))]

    collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)
    print(f"Ingested {len(chunks)} chunks from '{path.name}'")
    return len(chunks)


def query(question: str, n_results: int = 3) -> str:
    """Retrieve relevant chunks and generate an answer."""
    collection = get_collection()

    if collection.count() == 0:
        return "No documents ingested yet. Run ingest_document() first."

    results = collection.query(query_texts=[question], n_results=n_results)
    context_chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]

    context = "\n\n---\n\n".join(context_chunks)

    system_prompt = (
        "You are an Enterprise Architecture AI assistant. "
        "Answer questions using only the provided context. "
        "If the answer is not in the context, say so clearly. "
        "Be concise and structured."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    answer = response.choices[0].message.content
    unique_sources = list(dict.fromkeys(sources))
    return f"{answer}\n\nSources: {', '.join(unique_sources)}"


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m rag.pipeline ingest <file.txt>")
        print("  python -m rag.pipeline query \"What is TOGAF?\"")
        sys.exit(1)

    command = sys.argv[1]

    if command == "ingest":
        if len(sys.argv) < 3:
            print("Provide a file path.")
            sys.exit(1)
        ingest_document(sys.argv[2])

    elif command == "query":
        if len(sys.argv) < 3:
            print("Provide a question.")
            sys.exit(1)
        question = " ".join(sys.argv[2:])
        print(f"\nQ: {question}\n")
        print(query(question))

    else:
        print(f"Unknown command: {command}")