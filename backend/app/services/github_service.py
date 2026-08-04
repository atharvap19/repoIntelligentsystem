"""GitHub service integration."""
"""GitHub service for cloning repositories."""

from pathlib import Path

from git import Repo, GitCommandError


class GitHubService:
    def __init__(self):
        """
        Create the repositories directory if it doesn't exist.
        """
        self.repositories_dir = Path("repositories")
        self.repositories_dir.mkdir(exist_ok=True)

    def clone_repository(self, url: str):
        """
        Clone a GitHub repository.

        Args:
            url (str): GitHub repository URL.

        Returns:
            dict: Status of the cloning operation.
        """

        try:
            # Extract repository name
            repo_name = url.rstrip("/").split("/")[-1]

            # Remove .git if present
            if repo_name.endswith(".git"):
                repo_name = repo_name[:-4]

            repo_path = self.repositories_dir / repo_name

            # Check if repository already exists
            if repo_path.exists():
                return {
                    "status": "success",
                    "message": "Repository already exists.",
                    "repository": repo_name,
                    "path": str(repo_path.resolve())
                }

            # Clone repository
            Repo.clone_from(url, repo_path)

            return {
                "status": "success",
                "message": "Repository cloned successfully.",
                "repository": repo_name,
                "path": str(repo_path.resolve())
            }

        except GitCommandError as e:
            return {
                "status": "error",
                "message": f"Git error: {str(e)}"
            }

        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }