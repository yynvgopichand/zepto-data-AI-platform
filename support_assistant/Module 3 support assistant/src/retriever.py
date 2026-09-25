
import chromadb
from sentence_transformers import SentenceTransformer

from config import CHROMA_DIR
from ingest import COLLECTION_NAME, EMBEDDING_MODEL


class Retriever:
    def __init__(self, top_k=3):
        self.top_k = top_k

        self.model = SentenceTransformer(
            EMBEDDING_MODEL
        )

        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR)
        )

        self.collection = self.client.get_collection(
            name=COLLECTION_NAME
        )

    def search(self, query: str):
        query_embedding = self.model.encode(
            [query]
        ).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=self.top_k,
            include=["documents", "metadatas", "distances"],
        )

        retrieved = []

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for document, metadata, distance in zip(
            documents, metadatas, distances
        ):
            retrieved.append(
                {
                    "text": document,
                    "filename": metadata["filename"],
                    "distance": float(distance),
                }
            )

        return retrieved