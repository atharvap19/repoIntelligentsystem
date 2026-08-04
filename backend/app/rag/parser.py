"""Repository parser."""

from pathlib import Path


class RepositoryParser:
    """
    Parses a cloned repository and extracts supported source files.
    """

    SUPPORTED_EXTENSIONS = {
        # Python
        ".py",

        # JavaScript / TypeScript
        ".js",
        ".jsx",
        ".ts",
        ".tsx",

        # Java
        ".java",

        # C / C++
        ".c",
        ".cpp",

        # C#
        ".cs",

        # Go
        ".go",

        # Rust
        ".rs",

        # Kotlin
        ".kt",

        # SQL
        ".sql",

        # Web
        ".html",
        ".css",

        # Documentation
        ".md",
    }

    LANGUAGE_MAP = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".java": "java",
        ".c": "c",
        ".cpp": "cpp",
        ".cs": "csharp",
        ".go": "go",
        ".rs": "rust",
        ".kt": "kotlin",
        ".sql": "sql",
        ".html": "html",
        ".css": "css",
        ".md": "markdown",
    }

    IGNORED_DIRECTORIES = {
        ".git",
        ".github",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".venv",
        "venv",
        "env",
        "node_modules",
        ".idea",
        ".vscode",
        "dist",
        "build",
        ".next",
        ".turbo",
        "coverage",
    }

    IGNORED_FILES = {
        "LICENSE",
        "CHANGELOG.md",
        ".pre-commit-config.yaml",
        "pyproject.toml",
        "poetry.lock",
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
    }

    def parse_repository(self, repository_path: str):
        """
        Parse every supported source file.

        Returns
        -------
        list[dict]
        """

        repository = Path(repository_path)

        if not repository.exists():
            raise FileNotFoundError(
                f"Repository not found: {repository_path}"
            )

        repository_name = repository.name

        parsed_files = []

        for file_path in repository.rglob("*"):

            if file_path.is_dir():
                continue

            if any(
                ignored in file_path.parts
                for ignored in self.IGNORED_DIRECTORIES
            ):
                continue

            if file_path.name in self.IGNORED_FILES:
                continue

            extension = file_path.suffix.lower()

            if extension not in self.SUPPORTED_EXTENSIONS:
                continue

            try:

                content = file_path.read_text(
                    encoding="utf-8",
                    errors="ignore",
                )

                if not content.strip():
                    continue

                parsed_files.append(
                    {
                        "repository": repository_name,
                        "language": self.LANGUAGE_MAP.get(
                            extension,
                            "unknown",
                        ),
                        "file_name": file_path.name,
                        "relative_path": str(
                            file_path.relative_to(repository)
                        ),
                        "extension": extension,
                        "content": content,
                    }
                )

            except Exception:
                continue

        return parsed_files