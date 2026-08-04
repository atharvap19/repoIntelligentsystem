"""Test repository retrieval."""

from app.rag.retriever import Retriever
from app.services.embedding_service import EmbeddingService


def main():

    query = "How does FastAPI handle routing?"

    embedding_service = EmbeddingService()
    retriever = Retriever()

    # Generate query embedding
    query_embedding = embedding_service.generate_embedding(query)

    # Retrieve relevant chunks
    results = retriever.retrieve(
        query_embedding=query_embedding,
        top_k=3
    )

    print("\n========== SEARCH RESULTS ==========\n")

    for i, result in enumerate(results, start=1):

        print(f"Result {i}")
        print(f"Distance   : {result['distance']:.4f}")
        print(f"File       : {result['relative_path']}")
        print("-" * 60)
        print(result["content"][:400])
        print()


if __name__ == "__main__":
    main()