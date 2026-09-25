
import chromadb
from sentence_transformers import SentenceTransformer

from .config import CHROMA_DIR, DATA_DIR


COLLECTION_NAME = "zepto_policy_chunks"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def load_and_chunk_documents():
    """Load all policy documents from the docs directory."""

    chunks = []

    print(f"Looking for documents in: {DATA_DIR}")

    # Check whether the docs directory exists.
    if not DATA_DIR.exists():
        print(f"ERROR: Document directory does not exist: {DATA_DIR}")
        return chunks

    # Find all TXT files in the docs directory.
    txt_files = sorted(DATA_DIR.glob("*.txt"))

    print(f"Found {len(txt_files)} TXT files.")

    for file_path in txt_files:
        print(f"Reading document: {file_path.name}")

        text = file_path.read_text(
            encoding="utf-8"
        ).strip()

        # Skip empty documents.
        if not text:
            print(f"Skipping empty document: {file_path.name}")
            continue

        # One chunk per document.
        chunks.append(
            {
                "id": f"{file_path.stem}_chunk_00",
                "text": text,
                "metadata": {
                    "document": file_path.name
                },
            }
        )

    print(f"Total document chunks created: {len(chunks)}")

    return chunks


def get_embedding_model():
    """Load the Sentence Transformer embedding model."""

    return SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )


def get_chroma_collection():
    """Create or retrieve the ChromaDB collection."""

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        configuration={
            "hnsw": {
                "space": "cosine"
            }
        },
    )


def build_index():
    """Load documents, generate embeddings, and store them in ChromaDB."""

    chunks = load_and_chunk_documents()

    if not chunks:
        raise ValueError(
            f"No document chunks found in: {DATA_DIR}. "
            "Make sure the docs folder contains TXT files."
        )

    print("Loading embedding model...")

    model = get_embedding_model()

    collection = get_chroma_collection()

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print("Generating embeddings...")

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).tolist()

    print("Saving documents to ChromaDB...")

    collection.upsert(
        ids=[
            chunk["id"]
            for chunk in chunks
        ],
        documents=texts,
        metadatas=[
            chunk["metadata"]
            for chunk in chunks
        ],
        embeddings=embeddings,
    )

    return {
        "documents": len(chunks),
        "collection": COLLECTION_NAME,
    }


def retrieve_chunks(
    query: str,
    top_k: int = 3
):
    """Retrieve the most relevant policy chunks from ChromaDB."""

    model = get_embedding_model()

    collection = get_chroma_collection()

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    retrieved = []

    # Handle an empty ChromaDB result.
    if not results.get("ids") or not results["ids"][0]:
        return retrieved

    for i, chunk_id in enumerate(
        results["ids"][0]
    ):
        retrieved.append(
            {
                "id": chunk_id,
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": float(
                    results["distances"][0][i]
                ),
            }
        )

    return retrieved


if __name__ == "__main__":
    result = build_index()

    print("Indexing completed successfully:")
    print(result)