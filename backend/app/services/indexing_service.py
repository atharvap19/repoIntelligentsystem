"""Indexing service logic."""

from app.rag.parser import RepositoryParser
from app.rag.chunker import RepositoryChunker
from app.rag.vector_store import VectorStore
from app.services.embedding_service import EmbeddingService


class IndexingService:
    """
    Handles the complete repository indexing pipeline.

    Pipeline:
        Repository
            ↓
        Parser
            ↓
        Chunker
            ↓
        Embedding Service
            ↓
        ChromaDB
    """

    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size

        self.parser = RepositoryParser()
        self.chunker = RepositoryChunker()
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()

    def index_repository(self, repository_path: str):
        """
        Index an entire repository into ChromaDB.

        Args:
            repository_path (str): Local path to the cloned repository.

        Returns:
            dict: Indexing statistics.
        """

        print("Parsing repository...")
        files = self.parser.parse_repository(repository_path)
        print(f"Parsed {len(files)} files")

        print("Chunking files...")
        chunks = self.chunker.chunk_repository(files)
        print(f"Created {len(chunks)} chunks")

        print("Generating embeddings...")

        for start in range(0, len(chunks), self.batch_size):

            batch = chunks[start:start + self.batch_size]

            texts = [
                chunk["content"]
                for chunk in batch
            ]

            embeddings = self.embedding_service.generate_embeddings(
                texts
            )

            # Store the entire batch in one ChromaDB request
            self.vector_store.add_chunks(
                batch,
                embeddings
            )

            processed = min(
                start + self.batch_size,
                len(chunks)
            )

            print(
                f"Embedded {processed}/{len(chunks)} chunks"
            )

        print("Repository indexing complete.")

        return {
            "files": len(files),
            "chunks": len(chunks),
            "stored_vectors": self.vector_store.count(),
        }