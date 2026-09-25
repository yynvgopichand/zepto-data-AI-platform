
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from config import DOCS_DIR, CHROMA_DIR


COLLECTION_NAME = "zepto_policies"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_documents():
    documents = []

    for file_path in sorted(DOCS_DIR.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8").strip()

        if not text:
            continue

        documents.append(
            {
                "id": file_path.stem,
                "filename": file_path.name,
                "text": text,
            }
        )

    return documents


def build_collection():
    documents = load_documents()

    if not documents:
        raise ValueError("No documents found in the docs folder.")

    model = SentenceTransformer(EMBEDDING_MODEL)

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    embeddings = model.encode(
        [doc["text"] for doc in documents]
    ).tolist()

    collection.upsert(
        ids=[doc["id"] for doc in documents],
        documents=[doc["text"] for doc in documents],
        metadatas=[
            {"filename": doc["filename"]}
            for doc in documents
        ],
        embeddings=embeddings,
    )

    print(f"Indexed {len(documents)} documents.")

    return collection


if __name__ == "__main__":
    build_collection()