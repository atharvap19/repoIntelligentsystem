"""SQLite persistence for the repository graph.

Why SQLite and not a graph database: Chroma already ships it, so this adds no
dependency; adjacency lookups ("what depends on X") are indexed queries on one
table, which is all Phase 3 needs; and it is a single file that can be deleted
and rebuilt. The cost is that deep transitive traversal would need recursive
CTEs — acceptable while the product only draws one or two hops.

Everything is behind this class so a real graph database can replace it
without touching the builder, the API or the agent.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable, Sequence

from app.graph.model import Edge, Node

logger = logging.getLogger(__name__)

DEFAULT_GRAPH_PATH = "./graph_db/graph.sqlite3"

SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    id          TEXT PRIMARY KEY,
    repository  TEXT NOT NULL,
    kind        TEXT NOT NULL,
    name        TEXT NOT NULL,
    key         TEXT NOT NULL,
    parent_id   TEXT,
    level       INTEGER NOT NULL DEFAULT 0,
    data        TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_nodes_repo_kind   ON nodes(repository, kind);
CREATE INDEX IF NOT EXISTS idx_nodes_parent      ON nodes(parent_id);
CREATE INDEX IF NOT EXISTS idx_nodes_repo_key    ON nodes(repository, key);
CREATE INDEX IF NOT EXISTS idx_nodes_name        ON nodes(repository, name);

CREATE TABLE IF NOT EXISTS edges (
    id          TEXT PRIMARY KEY,
    repository  TEXT NOT NULL,
    kind        TEXT NOT NULL,
    source_id   TEXT NOT NULL,
    target_id   TEXT NOT NULL,
    data        TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id, kind);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id, kind);
CREATE INDEX IF NOT EXISTS idx_edges_repo   ON edges(repository, kind);

CREATE TABLE IF NOT EXISTS commits (
    sha         TEXT NOT NULL,
    repository  TEXT NOT NULL,
    author      TEXT,
    authored_at INTEGER NOT NULL,
    summary     TEXT,
    is_merge    INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (repository, sha)
);
CREATE INDEX IF NOT EXISTS idx_commits_time ON commits(repository, authored_at);

CREATE TABLE IF NOT EXISTS commit_files (
    repository    TEXT NOT NULL,
    sha           TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    module        TEXT,
    change_type   TEXT,
    authored_at   INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cf_path   ON commit_files(repository, relative_path, authored_at);
CREATE INDEX IF NOT EXISTS idx_cf_module ON commit_files(repository, module, authored_at);
-- The timeline joins commit_files to commits on sha for every dot. Without
-- this the join is a scan per commit: 2.6s to draw FastAPI's 1,500 commits,
-- 12ms with it. `IF NOT EXISTS` means existing databases pick it up on the
-- next open, so no migration is needed.
CREATE INDEX IF NOT EXISTS idx_cf_sha    ON commit_files(repository, sha);

CREATE TABLE IF NOT EXISTS refs (
    repository  TEXT NOT NULL,
    name        TEXT NOT NULL,
    kind        TEXT NOT NULL,          -- 'branch' | 'tag'
    sha         TEXT,
    created_at  INTEGER,
    PRIMARY KEY (repository, kind, name)
);

CREATE TABLE IF NOT EXISTS meta (
    repository  TEXT NOT NULL,
    key         TEXT NOT NULL,
    value       TEXT,
    PRIMARY KEY (repository, key)
);
"""


def _dumps(value: Any) -> str:
    return json.dumps(value or {}, default=str)


def _loads(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


class GraphStore:
    """Reads and writes the repository graph."""

    def __init__(self, path: str | None = None):
        self.path = Path(path or DEFAULT_GRAPH_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # FastAPI runs sync endpoints in a threadpool, so connections are
        # created per thread rather than shared.
        self._local = threading.local()
        with self._connect() as connection:
            connection.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = sqlite3.connect(self.path, check_same_thread=False)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            self._local.connection = connection
        return connection

    # -- writes ------------------------------------------------------------

    def replace_repository(
        self,
        repository: str,
        nodes: Sequence[Node],
        edges: Sequence[Edge],
    ) -> dict[str, int]:
        """Atomically swap in a freshly built graph.

        Delete-then-insert rather than upsert: a rebuild must not leave nodes
        for files that no longer exist, and reconciling that incrementally is
        more error-prone than rebuilding a structure this cheap to compute.
        """
        connection = self._connect()
        with connection:
            connection.execute("DELETE FROM edges WHERE repository = ?", (repository,))
            connection.execute("DELETE FROM nodes WHERE repository = ?", (repository,))

            connection.executemany(
                "INSERT OR REPLACE INTO nodes"
                " (id, repository, kind, name, key, parent_id, level, data)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (n.id, n.repository, n.kind, n.name, n.key, n.parent_id, n.level, _dumps(n.data))
                    for n in nodes
                ],
            )
            connection.executemany(
                "INSERT OR REPLACE INTO edges"
                " (id, repository, kind, source_id, target_id, data)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (e.id, e.repository, e.kind, e.source_id, e.target_id, _dumps(e.data))
                    for e in edges
                ],
            )
        logger.info(
            "Stored graph for %r: %d nodes, %d edges", repository, len(nodes), len(edges)
        )
        return {"nodes": len(nodes), "edges": len(edges)}

    def replace_history(
        self,
        repository: str,
        commits: Iterable[dict],
        commit_files: Iterable[dict],
        refs: Iterable[dict],
    ) -> dict[str, int]:
        """Swap in freshly extracted git history."""
        commits = list(commits)
        commit_files = list(commit_files)
        refs = list(refs)

        connection = self._connect()
        with connection:
            for table in ("commits", "commit_files", "refs"):
                connection.execute(f"DELETE FROM {table} WHERE repository = ?", (repository,))

            connection.executemany(
                "INSERT OR REPLACE INTO commits"
                " (sha, repository, author, authored_at, summary, is_merge)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (c["sha"], repository, c.get("author"), c["authored_at"],
                     c.get("summary"), 1 if c.get("is_merge") else 0)
                    for c in commits
                ],
            )
            connection.executemany(
                "INSERT INTO commit_files"
                " (repository, sha, relative_path, module, change_type, authored_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (repository, f["sha"], f["relative_path"], f.get("module"),
                     f.get("change_type"), f["authored_at"])
                    for f in commit_files
                ],
            )
            connection.executemany(
                "INSERT OR REPLACE INTO refs (repository, name, kind, sha, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                [
                    (repository, r["name"], r["kind"], r.get("sha"), r.get("created_at"))
                    for r in refs
                ],
            )
        return {"commits": len(commits), "commit_files": len(commit_files), "refs": len(refs)}

    def set_meta(self, repository: str, key: str, value: Any) -> None:
        connection = self._connect()
        with connection:
            connection.execute(
                "INSERT OR REPLACE INTO meta (repository, key, value) VALUES (?, ?, ?)",
                (repository, key, _dumps(value)),
            )

    def get_meta(self, repository: str, key: str) -> Any:
        row = self._connect().execute(
            "SELECT value FROM meta WHERE repository = ? AND key = ?", (repository, key)
        ).fetchone()
        return _loads(row["value"]) if row else {}

    def delete_repository(self, repository: str) -> None:
        connection = self._connect()
        with connection:
            for table in ("edges", "nodes", "commits", "commit_files", "refs", "meta"):
                connection.execute(f"DELETE FROM {table} WHERE repository = ?", (repository,))
        logger.info("Deleted graph for %r", repository)

    # -- reads -------------------------------------------------------------

    @staticmethod
    def _node(row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "repository": row["repository"],
            "kind": row["kind"],
            "name": row["name"],
            "key": row["key"],
            "parent_id": row["parent_id"],
            "level": row["level"],
            "data": _loads(row["data"]),
        }

    @staticmethod
    def _edge(row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "repository": row["repository"],
            "kind": row["kind"],
            "source": row["source_id"],
            "target": row["target_id"],
            "data": _loads(row["data"]),
        }

    def repositories(self) -> list[str]:
        rows = self._connect().execute(
            "SELECT DISTINCT repository FROM nodes ORDER BY repository"
        ).fetchall()
        return [r["repository"] for r in rows]

    def has_repository(self, repository: str) -> bool:
        row = self._connect().execute(
            "SELECT 1 FROM nodes WHERE repository = ? LIMIT 1", (repository,)
        ).fetchone()
        return row is not None

    def get_node(self, node_id: str) -> dict | None:
        row = self._connect().execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
        return self._node(row) if row else None

    def find_node(self, repository: str, key: str, kind: str | None = None) -> dict | None:
        sql = "SELECT * FROM nodes WHERE repository = ? AND key = ?"
        params: list[Any] = [repository, key]
        if kind:
            sql += " AND kind = ?"
            params.append(kind)
        row = self._connect().execute(sql + " LIMIT 1", params).fetchone()
        return self._node(row) if row else None

    def children(self, parent_id: str, kinds: Sequence[str] | None = None) -> list[dict]:
        sql = "SELECT * FROM nodes WHERE parent_id = ?"
        params: list[Any] = [parent_id]
        if kinds:
            sql += f" AND kind IN ({','.join('?' * len(kinds))})"
            params.extend(kinds)
        rows = self._connect().execute(sql + " ORDER BY kind, name", params).fetchall()
        return [self._node(r) for r in rows]

    def roots(self, repository: str, kind: str) -> list[dict]:
        rows = self._connect().execute(
            "SELECT * FROM nodes WHERE repository = ? AND kind = ? ORDER BY name",
            (repository, kind),
        ).fetchall()
        return [self._node(r) for r in rows]

    def find_by_name(
        self, repository: str, name: str, kinds: Sequence[str] | None = None, limit: int = 10
    ) -> list[dict]:
        """Nodes whose name matches exactly, case-insensitively.

        Separate from :meth:`search_nodes` because substring search is the
        wrong tool for a name the user typed in full. In FastAPI every path
        under ``fastapi/`` contains the string "fastapi", so a LIKE search for
        ``FastAPI`` matches all 956 files and the class actually called
        ``FastAPI`` never surfaces. An exact match has to be tried first.
        """
        sql = "SELECT * FROM nodes WHERE repository = ? AND name = ? COLLATE NOCASE"
        params: list[Any] = [repository, name]
        if kinds:
            sql += f" AND kind IN ({','.join('?' * len(kinds))})"
            params.extend(kinds)
        rows = self._connect().execute(
            sql + " ORDER BY LENGTH(key) LIMIT ?", [*params, limit]
        ).fetchall()
        return [self._node(r) for r in rows]

    def search_nodes(
        self, repository: str, term: str, kinds: Sequence[str] | None = None, limit: int = 25
    ) -> list[dict]:
        sql = "SELECT * FROM nodes WHERE repository = ? AND (name LIKE ? OR key LIKE ?)"
        params: list[Any] = [repository, f"%{term}%", f"%{term}%"]
        if kinds:
            sql += f" AND kind IN ({','.join('?' * len(kinds))})"
            params.extend(kinds)
        rows = self._connect().execute(
            sql + " ORDER BY LENGTH(name) LIMIT ?", [*params, limit]
        ).fetchall()
        return [self._node(r) for r in rows]

    def outgoing(self, node_id: str, kinds: Sequence[str] | None = None) -> list[dict]:
        sql = "SELECT * FROM edges WHERE source_id = ?"
        params: list[Any] = [node_id]
        if kinds:
            sql += f" AND kind IN ({','.join('?' * len(kinds))})"
            params.extend(kinds)
        return [self._edge(r) for r in self._connect().execute(sql, params).fetchall()]

    def incoming(self, node_id: str, kinds: Sequence[str] | None = None) -> list[dict]:
        sql = "SELECT * FROM edges WHERE target_id = ?"
        params: list[Any] = [node_id]
        if kinds:
            sql += f" AND kind IN ({','.join('?' * len(kinds))})"
            params.extend(kinds)
        return [self._edge(r) for r in self._connect().execute(sql, params).fetchall()]

    def nodes_by_ids(self, ids: Sequence[str]) -> list[dict]:
        if not ids:
            return []
        rows = self._connect().execute(
            f"SELECT * FROM nodes WHERE id IN ({','.join('?' * len(ids))})", list(ids)
        ).fetchall()
        return [self._node(r) for r in rows]

    def counts(self, repository: str) -> dict[str, int]:
        connection = self._connect()
        by_kind = {
            row["kind"]: row["n"]
            for row in connection.execute(
                "SELECT kind, COUNT(*) AS n FROM nodes WHERE repository = ? GROUP BY kind",
                (repository,),
            ).fetchall()
        }
        edges = connection.execute(
            "SELECT COUNT(*) AS n FROM edges WHERE repository = ?", (repository,)
        ).fetchone()["n"]
        commits = connection.execute(
            "SELECT COUNT(*) AS n FROM commits WHERE repository = ?", (repository,)
        ).fetchone()["n"]
        return {**by_kind, "edges": edges, "commits": commits}

    # -- history reads -----------------------------------------------------

    def commits_for_path(self, repository: str, relative_path: str, limit: int = 20) -> list[dict]:
        rows = self._connect().execute(
            "SELECT c.sha, c.author, c.authored_at, c.summary, f.change_type"
            " FROM commit_files f JOIN commits c"
            "   ON c.sha = f.sha AND c.repository = f.repository"
            " WHERE f.repository = ? AND f.relative_path = ?"
            " ORDER BY f.authored_at DESC LIMIT ?",
            (repository, relative_path, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def timeline_buckets(self, repository: str) -> list[dict]:
        """Commits per module per month — the timeline's raw material."""
        rows = self._connect().execute(
            "SELECT module,"
            "       strftime('%Y-%m', authored_at, 'unixepoch') AS period,"
            "       COUNT(DISTINCT sha) AS commits,"
            "       COUNT(*) AS changes,"
            "       MIN(authored_at) AS first_at,"
            "       MAX(authored_at) AS last_at"
            " FROM commit_files WHERE repository = ? AND module IS NOT NULL"
            " GROUP BY module, period ORDER BY period",
            (repository,),
        ).fetchall()
        return [dict(r) for r in rows]

    def nodes_of_kind(self, repository: str, kind: str) -> list[dict]:
        rows = self._connect().execute(
            "SELECT * FROM nodes WHERE repository = ? AND kind = ?", (repository, kind)
        ).fetchall()
        return [self._node(r) for r in rows]

    def paths_first_seen(self, repository: str) -> dict[str, int]:
        """Earliest observed commit timestamp per path."""
        rows = self._connect().execute(
            "SELECT relative_path, MIN(authored_at) AS first_at FROM commit_files"
            " WHERE repository = ? GROUP BY relative_path",
            (repository,),
        ).fetchall()
        return {r["relative_path"]: r["first_at"] for r in rows}

    def modules_first_seen(self, repository: str) -> dict[str, int]:
        rows = self._connect().execute(
            "SELECT module, MIN(authored_at) AS first_at FROM commit_files"
            " WHERE repository = ? AND module IS NOT NULL GROUP BY module",
            (repository,),
        ).fetchall()
        return {r["module"]: r["first_at"] for r in rows}

    def paths_existing_at(self, repository: str, timestamp: int) -> set[str]:
        """Paths that had been touched at or before ``timestamp``.

        An approximation of the tree at that moment — it counts a file as
        present from its first commit onwards and does not model deletions.
        Exact reconstruction would mean reading trees per commit, which is
        far more expensive; the abstraction leaves room for that later.
        """
        rows = self._connect().execute(
            "SELECT DISTINCT relative_path FROM commit_files"
            " WHERE repository = ? AND authored_at <= ?",
            (repository, timestamp),
        ).fetchall()
        return {r["relative_path"] for r in rows}

    def commits(self, repository: str, limit: int = 60, before: int | None = None) -> list[dict]:
        """Recent commits, newest first, with how many files each touched.

        The file count is joined here rather than fetched per commit: the
        timeline draws one dot per commit and labels each with its size, so
        N+1 queries would be the whole cost of rendering it.

        Merge commits report zero files by design — ``--name-status`` emits no
        file list for them (see ``git_history``), so counting would be a lie
        rather than a gap.
        """
        sql = (
            "SELECT c.sha, c.author, c.authored_at, c.summary, c.is_merge,"
            "       COUNT(f.relative_path) AS file_count"
            "  FROM commits c"
            "  LEFT JOIN commit_files f"
            "    ON f.sha = c.sha AND f.repository = c.repository"
            " WHERE c.repository = ?"
        )
        params: list[Any] = [repository]
        if before is not None:
            sql += " AND c.authored_at <= ?"
            params.append(before)
        sql += " GROUP BY c.sha ORDER BY c.authored_at DESC LIMIT ?"
        params.append(limit)

        rows = self._connect().execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def commit(self, repository: str, sha: str) -> dict | None:
        """One commit by full or abbreviated SHA."""
        row = self._connect().execute(
            "SELECT sha, author, authored_at, summary, is_merge FROM commits"
            " WHERE repository = ? AND sha LIKE ? LIMIT 1",
            (repository, f"{sha}%"),
        ).fetchone()
        return dict(row) if row else None

    def files_in_commit(self, repository: str, sha: str, limit: int = 200) -> list[dict]:
        rows = self._connect().execute(
            "SELECT relative_path, module, change_type FROM commit_files"
            " WHERE repository = ? AND sha = ? ORDER BY relative_path LIMIT ?",
            (repository, sha, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def changes_between(
        self, repository: str, start_at: int, end_at: int, limit: int = 400
    ) -> list[dict]:
        """Per-path change counts within a time window.

        The raw material for "what changed between these commits?" (Part 12).
        Aggregated in SQL because a busy window can touch thousands of paths
        and only the top of that list is ever shown.
        """
        rows = self._connect().execute(
            "SELECT relative_path, module, COUNT(DISTINCT sha) AS commits,"
            "       MAX(authored_at) AS last_at"
            "  FROM commit_files"
            " WHERE repository = ? AND authored_at > ? AND authored_at <= ?"
            " GROUP BY relative_path ORDER BY commits DESC, last_at DESC LIMIT ?",
            (repository, start_at, end_at, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def change_counts(self, repository: str) -> dict[str, int]:
        """Commits per path across the indexed window — the churn metric."""
        rows = self._connect().execute(
            "SELECT relative_path, COUNT(DISTINCT sha) AS commits FROM commit_files"
            " WHERE repository = ? GROUP BY relative_path",
            (repository,),
        ).fetchall()
        return {r["relative_path"]: r["commits"] for r in rows}

    def dependent_counts(self, repository: str, kinds: Sequence[str] | None = None) -> dict[str, int]:
        """In-degree per node — how many things import each node.

        Part 13's central metric, and the reason hotspots can be *explained*
        rather than asserted: the number is a count of real edges, so an
        answer can name them.
        """
        sql = (
            "SELECT target_id, COUNT(*) AS n FROM edges"
            " WHERE repository = ?"
        )
        params: list[Any] = [repository]
        if kinds:
            sql += f" AND kind IN ({','.join('?' * len(kinds))})"
            params.extend(kinds)
        rows = self._connect().execute(sql + " GROUP BY target_id", params).fetchall()
        return {r["target_id"]: r["n"] for r in rows}

    def commit_range(self, repository: str) -> dict:
        row = self._connect().execute(
            "SELECT MIN(authored_at) AS first_at, MAX(authored_at) AS last_at,"
            "       COUNT(*) AS total FROM commits WHERE repository = ?",
            (repository,),
        ).fetchone()
        return dict(row) if row else {"first_at": None, "last_at": None, "total": 0}

    def refs(self, repository: str, kind: str | None = None) -> list[dict]:
        sql = "SELECT * FROM refs WHERE repository = ?"
        params: list[Any] = [repository]
        if kind:
            sql += " AND kind = ?"
            params.append(kind)
        rows = self._connect().execute(sql + " ORDER BY created_at", params).fetchall()
        return [dict(r) for r in rows]
