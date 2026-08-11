"""Shared fixtures.

Tests build tiny synthetic repositories on disk. The real FastAPI checkout is
never parsed here — unit tests must stay fast and independent of what happens
to be in ``backend/repositories/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make ``app`` importable when pytest is invoked from anywhere.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


def write(root: Path, relative: str, content: str = "x = 1\n") -> Path:
    """Create a file (and its parents) under ``root``.

    ``newline=""`` disables Windows CRLF translation so fixture content round
    trips byte for byte and assertions stay platform independent.
    """
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="")
    return path


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    """A miniature repository covering every filter branch."""
    repo = tmp_path / "sample_repo"

    # --- kept ---------------------------------------------------------
    write(repo, "app/main.py", "def main():\n    return 1\n")
    write(repo, "app/rag/parser.py", "class Parser:\n    pass\n")
    write(repo, "web/index.html", "<html></html>\n")
    write(repo, "web/style.css", "body { margin: 0 }\n")
    write(repo, "web/app.ts", "export const a = 1\n")
    write(repo, "web/view.tsx", "export default () => null\n")
    write(repo, "db/schema.sql", "SELECT 1;\n")

    # --- pruned directories -------------------------------------------
    write(repo, ".git/objects/abc", "binary-ish")
    write(repo, "node_modules/left-pad/index.js", "module.exports = 1\n")
    write(repo, "__pycache__/main.cpython-311.pyc", "compiled")
    write(repo, "build/output.js", "console.log(1)\n")
    write(repo, ".venv/lib/site.py", "import sys\n")

    # --- prose / metadata ---------------------------------------------
    write(repo, "README.md", "# Sample\n")
    write(repo, "docs/guide.md", "# Guide\n")
    write(repo, "LICENSE", "MIT\n")
    write(repo, "CHANGELOG.md", "## 1.0\n")
    write(repo, "mkdocs.yml", "site_name: x\n")
    write(repo, ".pre-commit-config.yaml", "repos: []\n")

    # --- binary / generated / lock ------------------------------------
    write(repo, "assets/logo.png", "not-really-a-png")
    write(repo, "assets/icon.svg", "<svg/>")
    write(repo, "package-lock.json", "{}\n")
    write(repo, "web/vendor.min.js", "var a=1;")
    write(repo, "proto/service_pb2.py", "# generated\n")

    # --- edge cases ---------------------------------------------------
    write(repo, "app/empty.py", "   \n\n")
    (repo / "app" / "binary.py").write_bytes(b"def f():\x00 pass")

    return repo
