"""Test repository indexing."""

from app.services.indexing_service import IndexingService


def main():

    # Path to the cloned repository
    repository_path = "repositories/fastapi"

    # Create indexing service
    service = IndexingService(batch_size=32)

    # Index repository
    stats = service.index_repository(repository_path)

    print("\n========== INDEXING COMPLETE ==========")
    print(f"Files Parsed   : {stats['files']}")
    print(f"Chunks Created : {stats['chunks']}")
    print(f"Vectors Stored : {stats['stored_vectors']}")
    print("=======================================")


if __name__ == "__main__":
    main()