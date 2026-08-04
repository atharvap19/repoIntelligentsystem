"""ChromaDB vector store."""

import chromadb
from chromadb.config import Settings


class VectorStore:
    """
    Handles storage and retrieval of repository embeddings using ChromaDB.
    """

    def __init__(self):

        self.client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )

        self.collection = self.client.get_or_create_collection(
            name="repositories"
        )

    def add_chunk(self, chunk, embedding):
        """
        Store a single chunk.
        """

        chunk_id = (
            f"{chunk['relative_path']}_{chunk['chunk_index']}"
        )

        self.collection.add(
            ids=[chunk_id],
            documents=[chunk["content"]],
            embeddings=[embedding],
            metadatas=[{
                "repository": chunk["repository"],
                "language": chunk["language"],
                "file_name": chunk["file_name"],
                "relative_path": chunk["relative_path"],
                "extension": chunk["extension"],
                "chunk_index": chunk["chunk_index"],
            }]
        )

    def add_chunks(self, chunks, embeddings):
        """
        Store multiple chunks in one ChromaDB request.
        """

        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:

            ids.append(
                f"{chunk['relative_path']}_{chunk['chunk_index']}"
            )

            documents.append(
                chunk["content"]
            )

            metadatas.append(
                {
                    "repository": chunk["repository"],
                    "language": chunk["language"],
                    "file_name": chunk["file_name"],
                    "relative_path": chunk["relative_path"],
                    "extension": chunk["extension"],
                    "chunk_index": chunk["chunk_index"],
                }
            )

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def search(self, embedding, top_k=5):

        return self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

    def count(self):
        return self.collection.count()