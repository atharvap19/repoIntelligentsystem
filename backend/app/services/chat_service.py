"""Chat service."""

from app.rag.retriever import Retriever
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService


class ChatService:
    """
    Coordinates the complete RAG pipeline.
    """

    def __init__(self):

        self.embedding_service = EmbeddingService()
        self.retriever = Retriever()
        self.llm_service = LLMService()

    def chat(
        self,
        question: str,
        top_k: int = 5,
    ) -> dict:

        # Generate embedding for the question
        query_embedding = self.embedding_service.generate_embedding(
            question
        )

        # Retrieve relevant chunks
        retrieved_chunks = self.retriever.retrieve(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        # Generate answer
        answer = self.llm_service.generate_response(
            question=question,
            retrieved_chunks=retrieved_chunks,
        )

        return {
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "file": chunk["relative_path"],
                    "distance": chunk["distance"],
                }
                for chunk in retrieved_chunks
            ],
        }