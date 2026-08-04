"""Repository chunker."""


class RepositoryChunker:
    """
    Splits parsed repository files into overlapping chunks.
    """

    def __init__(self, chunk_size=1000, overlap=100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_repository(self, parsed_files):
        """
        Split parsed repository files into chunks.

        Args:
            parsed_files (list): Output from RepositoryParser.

        Returns:
            list[dict]
        """

        chunks = []

        for file in parsed_files:

            content = file["content"].strip()

            # Skip empty files
            if not content:
                continue

            # Small files become a single chunk
            if len(content) <= self.chunk_size:

                chunks.append(
                    {
                        "repository": file["repository"],
                        "language": file["language"],
                        "file_name": file["file_name"],
                        "relative_path": file["relative_path"],
                        "extension": file["extension"],
                        "chunk_index": 0,
                        "content": content,
                    }
                )

                continue

            start = 0
            chunk_index = 0

            while start < len(content):

                end = start + self.chunk_size

                chunk_text = content[start:end].strip()

                if chunk_text:

                    chunks.append(
                        {
                            "repository": file["repository"],
                            "language": file["language"],
                            "file_name": file["file_name"],
                            "relative_path": file["relative_path"],
                            "extension": file["extension"],
                            "chunk_index": chunk_index,
                            "content": chunk_text,
                        }
                    )

                start += self.chunk_size - self.overlap
                chunk_index += 1

        return chunks