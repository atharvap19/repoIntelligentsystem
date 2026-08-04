"""Repository retriever."""

from app.rag.vector_store import VectorStore


class Retriever:
    """
    Retrieves the most relevant repository chunks.
    """

    def __init__(self):
        self.vector_store = VectorStore()

    def retrieve(
        self,
        query_embedding: list[float],
        top_k: int = 5
    ) -> list[dict]:

        results = self.vector_store.search(
            embedding=query_embedding,
            top_k=top_k
        )

        retrieved_chunks = []

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances,
        ):

            retrieved_chunks.append(
                {
                    "content": document,
                    "distance": distance,
                    **metadata,
                }
            )

        return retrieved_chunks