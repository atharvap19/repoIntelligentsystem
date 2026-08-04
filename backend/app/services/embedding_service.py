"""Embedding service using Ollama."""

from ollama import Client


class EmbeddingService:
    """
    Service for generating embeddings using Ollama.
    """

    def __init__(self):
        self.client = Client(host="http://localhost:11434")
        self.model = "nomic-embed-text"

    def generate_embedding(self, text: str) -> list[float]:
        """
        Generate an embedding for a single text.
        Used by the Retriever.
        """

        response = self.client.embed(
            model=self.model,
            input=text
        )

        return response["embeddings"][0]

    def generate_embeddings(
        self,
        texts: list[str]
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Used during repository indexing.
        """

        response = self.client.embed(
            model=self.model,
            input=texts
        )

        return response["embeddings"]