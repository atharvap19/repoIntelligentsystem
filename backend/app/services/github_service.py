"""GitHub service for cloning repositories."""

import re
from pathlib import Path
from urllib.parse import urlsplit

from git import GitCommandError, Repo

_SEGMENT = re.compile(r"^[A-Za-z0-9._-]+$")


class InvalidRepositoryUrl(ValueError):
    """Raised when a URL is not a plain https://github.com/owner/repo address."""


def parse_github_url(url: str) -> tuple[str, str]:
    """Validate a GitHub URL and return ``(clone_url, repository_name)``.

    The URL is handed to ``git clone``, so anything beyond a plain public
    GitHub address is refused: other hosts, other schemes (``ext::`` and
    ``file://`` run commands or read the local disk), embedded credentials,
    and path segments that could be read as options.
    """
    value = (url or "").strip()
    parts = urlsplit(value)

    if parts.scheme != "https":
        raise InvalidRepositoryUrl("Only https:// URLs are accepted.")
    if parts.hostname not in ("github.com", "www.github.com"):
        raise InvalidRepositoryUrl("Only github.com repositories are supported.")
    if parts.username or parts.password or parts.port:
        raise InvalidRepositoryUrl("The URL must not contain credentials or a port.")
    if parts.query or parts.fragment:
        raise InvalidRepositoryUrl("The URL must not contain a query or fragment.")

    segments = [s for s in parts.path.split("/") if s]
    if len(segments) != 2:
        raise InvalidRepositoryUrl("Use the repository URL: https://github.com/owner/repo")

    owner, name = segments
    if name.endswith(".git"):
        name = name[:-4]
    for segment in (owner, name):
        if not _SEGMENT.match(segment) or segment.startswith(("-", ".")):
            raise InvalidRepositoryUrl(f"{segment!r} is not a valid GitHub name.")

    return f"https://github.com/{owner}/{name}.git", name


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
            clone_url, repo_name = parse_github_url(url)

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
            Repo.clone_from(clone_url, repo_path)

            return {
                "status": "success",
                "message": "Repository cloned successfully.",
                "repository": repo_name,
                "path": str(repo_path.resolve())
            }

        except InvalidRepositoryUrl as e:
            return {
                "status": "error",
                "message": str(e)
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
