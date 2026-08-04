from app.services.embedding_service import EmbeddingService
import time

service = EmbeddingService()

texts = [
    "FastAPI is a Python framework.",
    "Repository Intelligence Platform.",
    "Vector databases store embeddings.",
    "Ollama generates embeddings.",
    "Python programming language."
]

start = time.time()

embeddings = service.generate_embeddings(texts)

end = time.time()

print(f"Generated {len(embeddings)} embeddings")
print(f"Embedding dimension: {len(embeddings[0])}")
print(f"Time taken: {end-start:.2f} seconds")