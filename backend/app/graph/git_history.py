"""Git history extraction.

Produces the temporal layer beneath the structural graph: which commits
touched which files, and therefore which *modules* were active when.

Two deliberate limits, both chosen for cost rather than completeness:

* File lists come from a diff against the first parent. Merge commits are
  recorded but their file lists are skipped, because a merge's diff against
  its first parent restates every change from the merged branch and would
  double-count module activity.
* History is walked to a configurable ceiling. FastAPI has ~5,600 commits and
  httpx ~1,500; walking every one with a full diff costs minutes, and the
  timeline only needs enough resolution to show module evolution.

Commit messages are stored but never logged — they contain emoji, and the
Windows console is cp1252, so logging one raises UnicodeEncodeError.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from git import GitCommandError, InvalidGitRepositoryError, NoSuchPathError, Repo

from app.graph.builder import module_of

logger = logging.getLogger(__name__)

#: How many commits to walk by default. Enough for a multi-year timeline
#: without paying for a full-history diff on a large repository.
DEFAULT_MAX_COMMITS = 1500

#: Commits touching more files than this are treated as bulk operations —
#: vendoring, reformatting, generated output — and their file lists are
#: skipped so they cannot swamp module activity.
BULK_CHANGE_THRESHOLD = 200


@dataclass
class HistoryStats:
    commits: int = 0
    file_changes: int = 0
    branches: int = 0
    tags: int = 0
    merges: int = 0
    skipped_bulk: int = 0
    seconds: float = 0.0
    error: str = ""

    def summary(self) -> str:
        return (
            f"{self.commits} commits, {self.file_changes} file changes, "
            f"{self.branches} branches, {self.tags} tags, {self.merges} merges "
            f"in {self.seconds:.2f}s"
        )


@dataclass
class RepositoryHistory:
    """Normalised history, ready for the store."""

    commits: list[dict] = field(default_factory=list)
    commit_files: list[dict] = field(default_factory=list)
    refs: list[dict] = field(default_factory=list)
    stats: HistoryStats = field(default_factory=HistoryStats)


class GitHistoryExtractor:
    """Reads a checkout's history into a normalised structure."""

    def __init__(
        self,
        max_commits: int = DEFAULT_MAX_COMMITS,
        bulk_threshold: int = BULK_CHANGE_THRESHOLD,
    ):
        self.max_commits = max_commits
        self.bulk_threshold = bulk_threshold

    def extract(self, repository_path: str | Path) -> RepositoryHistory:
        """Extract history. Never raises — a repo without git yields empty."""
        import time

        started = time.perf_counter()
        history = RepositoryHistory()

        try:
            repo = Repo(str(repository_path))
        except (InvalidGitRepositoryError, NoSuchPathError) as exc:
            history.stats.error = f"not a git repository: {type(exc).__name__}"
            logger.info("No git history for %s", repository_path)
            return history

        try:
            history.refs = list(self._refs(repo))
            history.stats.branches = sum(1 for r in history.refs if r["kind"] == "branch")
            history.stats.tags = sum(1 for r in history.refs if r["kind"] == "tag")
        except Exception as exc:
            logger.warning("Could not read refs: %s", type(exc).__name__)

        try:
            self._walk_commits(repo, history)
        except (GitCommandError, ValueError) as exc:
            history.stats.error = f"{type(exc).__name__}"
            logger.warning("History walk stopped early: %s", type(exc).__name__)

        history.stats.seconds = time.perf_counter() - started
        logger.info("Extracted history: %s", history.stats.summary())
        return history

    # -- internals ---------------------------------------------------------

    def _refs(self, repo: Repo) -> Iterable[dict]:
        """Branches (local and remote) and tags."""
        seen: set[tuple[str, str]] = set()

        for branch in repo.branches:
            key = ("branch", branch.name)
            if key in seen:
                continue
            seen.add(key)
            yield {
                "name": branch.name,
                "kind": "branch",
                "sha": branch.commit.hexsha,
                "created_at": int(branch.commit.committed_date),
            }

        # A fresh clone has one local branch, so remote refs are where the
        # real branch structure lives.
        try:
            for remote in repo.remotes:
                for ref in remote.refs:
                    name = ref.name.split("/", 1)[-1]
                    if name in ("HEAD",) or ("branch", name) in seen:
                        continue
                    seen.add(("branch", name))
                    yield {
                        "name": name,
                        "kind": "branch",
                        "sha": ref.commit.hexsha,
                        "created_at": int(ref.commit.committed_date),
                    }
        except Exception:
            pass  # detached or remote-less checkout

        for tag in repo.tags:
            try:
                commit = tag.commit
            except Exception:
                continue
            yield {
                "name": tag.name,
                "kind": "tag",
                "sha": commit.hexsha,
                "created_at": int(commit.committed_date),
            }

    def _walk_commits(self, repo: Repo, history: RepositoryHistory) -> None:
        """Read commits and their file lists in a single ``git log`` pass.

        Diffing each commit through GitPython costs ~77ms per commit — 31
        seconds for 400 commits, minutes for a real repository. One ``git log
        --name-status`` invocation does the same work in a single process.

        ``--name-status`` intentionally emits no file list for merge commits,
        which is exactly the behaviour wanted: merges are recorded, but their
        first-parent diff would restate the merged branch's work and double
        every module's activity.
        """
        record_separator = "\x00"
        field_separator = "\x1f"

        try:
            # The separators must be written as git's own %x00/%x1f
            # placeholders: a literal NUL inside a command-line argument is
            # rejected by the OS with "embedded null byte". Git expands them
            # when writing output, which is where they are actually needed.
            raw = repo.git.log(
                f"--max-count={self.max_commits}",
                "--name-status",
                "--no-renames",
                "--pretty=format:%x00%H%x1f%an%x1f%at%x1f%P%x1f%s",
                stdout_as_string=False,
            )
        except GitCommandError:
            logger.warning("git log failed; falling back to per-commit diffs")
            self._walk_commits_slow(repo, history)
            return

        # Commit subjects contain emoji and arbitrary encodings; decode
        # defensively rather than letting one commit abort the walk.
        text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw

        for record in text.split(record_separator):
            record = record.strip("\n")
            if not record:
                continue

            header, _, body = record.partition("\n")
            fields = header.split(field_separator)
            if len(fields) < 5:
                continue

            sha, author, timestamp, parents, summary = fields[:5]
            try:
                authored_at = int(timestamp)
            except ValueError:
                continue

            is_merge = len(parents.split()) > 1
            history.commits.append(
                {
                    "sha": sha,
                    "author": author[:120],
                    "summary": summary[:200],
                    "authored_at": authored_at,
                    "is_merge": is_merge,
                }
            )
            history.stats.commits += 1
            if is_merge:
                history.stats.merges += 1
                continue

            changes = self._parse_name_status(body)
            if len(changes) > self.bulk_threshold:
                history.stats.skipped_bulk += 1
                continue

            for path, change_type in changes:
                history.commit_files.append(
                    {
                        "sha": sha,
                        "relative_path": path,
                        "module": module_of(path),
                        "change_type": change_type,
                        "authored_at": authored_at,
                    }
                )
                history.stats.file_changes += 1

    @staticmethod
    def _parse_name_status(body: str) -> list[tuple[str, str]]:
        """Parse ``A\tpath`` / ``M\tpath`` lines from ``--name-status``."""
        changes: list[tuple[str, str]] = []
        for line in body.splitlines():
            line = line.strip()
            if not line or "\t" not in line:
                continue
            status, _, path = line.partition("\t")
            path = path.strip().replace("\\", "/")
            if path:
                changes.append((path, status.strip()[:1] or "M"))
        return changes

    def _walk_commits_slow(self, repo: Repo, history: RepositoryHistory) -> None:
        """Per-commit diff fallback, used only if ``git log`` is unavailable."""
        for commit in repo.iter_commits(max_count=self.max_commits):
            authored_at = int(commit.committed_date)
            is_merge = len(commit.parents) > 1

            history.commits.append(
                {
                    "sha": commit.hexsha,
                    "author": (commit.author.name or "")[:120],
                    "authored_at": authored_at,
                    "summary": str(commit.summary)[:200],
                    "is_merge": is_merge,
                }
            )
            history.stats.commits += 1
            if is_merge:
                history.stats.merges += 1
                continue

            for path, change_type in self._changed_paths(commit):
                history.commit_files.append(
                    {
                        "sha": commit.hexsha,
                        "relative_path": path,
                        "module": module_of(path),
                        "change_type": change_type,
                        "authored_at": authored_at,
                    }
                )
                history.stats.file_changes += 1

    def _changed_paths(self, commit) -> list[tuple[str, str]]:
        """Files touched by a commit, as ``(path, change_type)``."""
        try:
            if commit.parents:
                diffs = commit.parents[0].diff(commit)
            else:
                # Root commit: everything in the tree is an addition.
                return [
                    (blob.path.replace("\\", "/"), "A")
                    for blob in commit.tree.traverse()
                    if getattr(blob, "type", "") == "blob"
                ][: self.bulk_threshold]
        except (GitCommandError, ValueError, KeyError):
            return []

        if len(diffs) > self.bulk_threshold:
            return []

        changed: list[tuple[str, str]] = []
        for diff in diffs:
            path = diff.b_path or diff.a_path
            if not path:
                continue
            changed.append((path.replace("\\", "/"), diff.change_type or "M"))
        return changed


def extract_history(
    repository_path: str | Path, max_commits: int = DEFAULT_MAX_COMMITS
) -> RepositoryHistory:
    """Convenience wrapper used by the indexing service."""
    return GitHistoryExtractor(max_commits=max_commits).extract(repository_path)
