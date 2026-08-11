/**
 * Offline fixtures.
 *
 * The backend needs Ollama running plus an indexed repository before it can
 * answer anything, so the UI ships with a canned corpus. Demo mode engages
 * automatically when the health check fails, and can be pinned from the top bar.
 */
import type { ChatResponse, ImportResponse, IndexStats, Repository, Source } from './types'

export const DEMO_REPOSITORIES: Repository[] = [
  {
    id: 'demo-fastapi',
    name: 'fastapi',
    url: 'https://github.com/fastapi/fastapi',
    path: 'repositories/fastapi',
    state: 'ready',
    stats: { files: 842, chunks: 6431, stored_vectors: 6431 },
    languages: { python: 74, markdown: 19, html: 4, css: 3 },
    importedAt: Date.now() - 1000 * 60 * 42,
  },
  {
    id: 'demo-httpx',
    name: 'httpx',
    url: 'https://github.com/encode/httpx',
    path: 'repositories/httpx',
    state: 'unindexed',
    languages: { python: 91, markdown: 9 },
    importedAt: Date.now() - 1000 * 60 * 60 * 5,
  },
]

const DEMO_CHUNKS: Source[] = [
  {
    file: 'fastapi/routing.py',
    distance: 0.2841,
    language: 'python',
    chunk_index: 12,
    repository: 'fastapi',
    content: `class APIRouter(routing.Router):
    """
    \`APIRouter\` class, used to group *path operations*, for example to structure
    an app in multiple files. It would then be included in the \`FastAPI\` app, or
    in another \`APIRouter\` (ultimately included in the app).
    """

    def __init__(
        self,
        *,
        prefix: str = "",
        tags: Optional[List[Union[str, Enum]]] = None,
        dependencies: Optional[Sequence[params.Depends]] = None,
    ) -> None:
        super().__init__(redirect_slashes=redirect_slashes)
        if prefix:
            assert prefix.startswith("/"), "A path prefix must start with '/'"
            assert not prefix.endswith("/"), "A path prefix must not end with '/'"
        self.prefix = prefix
        self.tags: List[Union[str, Enum]] = tags or []`,
  },
  {
    file: 'fastapi/applications.py',
    distance: 0.3517,
    language: 'python',
    chunk_index: 47,
    repository: 'fastapi',
    content: `    def include_router(
        self,
        router: routing.APIRouter,
        *,
        prefix: str = "",
        tags: Optional[List[Union[str, Enum]]] = None,
    ) -> None:
        """
        Include an \`APIRouter\` in the same app.
        """
        self.router.include_router(
            router,
            prefix=prefix,
            tags=tags,
            dependencies=dependencies,
        )`,
  },
  {
    file: 'docs/en/docs/tutorial/bigger-applications.md',
    distance: 0.4193,
    language: 'markdown',
    chunk_index: 3,
    repository: 'fastapi',
    content: `# Bigger Applications - Multiple Files

If you are building an application or a web API, it's rarely the case that you
can put everything in a single file.

**FastAPI** provides a convenience tool to structure your application while
keeping all the flexibility. You can create the *path operations* for that
module using \`APIRouter\`.`,
  },
  {
    file: 'fastapi/routing.py',
    distance: 0.5024,
    language: 'python',
    chunk_index: 31,
    repository: 'fastapi',
    content: `    def add_api_route(
        self,
        path: str,
        endpoint: Callable[..., Any],
        *,
        methods: Optional[Union[Set[str], List[str]]] = None,
    ) -> None:
        route_class = route_class_override or self.route_class
        route = route_class(
            self.prefix + path,
            endpoint=endpoint,
            methods=methods,
        )
        self.routes.append(route)`,
  },
  {
    file: 'fastapi/params.py',
    distance: 0.6688,
    language: 'python',
    chunk_index: 2,
    repository: 'fastapi',
    content: `class Depends:
    def __init__(
        self, dependency: Optional[Callable[..., Any]] = None, *, use_cache: bool = True
    ):
        self.dependency = dependency
        self.use_cache = use_cache`,
  },
]

const DEMO_ANSWER = `Routing in this repository is built around two cooperating classes.

**\`APIRouter\`** in \`fastapi/routing.py\` is the grouping primitive. It subclasses Starlette's \`routing.Router\` and holds a \`prefix\`, a list of \`tags\`, and shared \`dependencies\`. The constructor asserts that a prefix starts with \`/\` and does not end with one, so prefixes compose predictably when routers nest.

Individual endpoints are registered through \`add_api_route\`, which instantiates \`route_class\` (an \`APIRoute\` by default) with \`self.prefix + path\` and appends it to \`self.routes\`. The decorator forms — \`@router.get\`, \`@router.post\` — are thin wrappers over that method.

**\`FastAPI.include_router\`** in \`fastapi/applications.py\` merges a router into the application, forwarding its own \`prefix\`, \`tags\`, and \`dependencies\` down so they stack on top of whatever the router already declared. This is what makes the multi-file layout described in \`docs/en/docs/tutorial/bigger-applications.md\` work: each module owns a router, and the app composes them at startup.

At request time the accumulated \`self.routes\` list is what Starlette matches against, so router nesting is resolved once during import rather than per request.`

const delay = (ms: number) => new Promise((r) => setTimeout(r, ms))

export async function demoChat(question: string, topK: number): Promise<ChatResponse> {
  await delay(700 + Math.random() * 900)
  return {
    question,
    answer: DEMO_ANSWER,
    sources: DEMO_CHUNKS.slice(0, Math.max(1, Math.min(topK, DEMO_CHUNKS.length))),
    timings: { embed: 34, search: 11, generate: 1180 },
  }
}

export async function demoImport(url: string): Promise<ImportResponse> {
  await delay(900)
  const name = url.replace(/\/+$/, '').split('/').pop()?.replace(/\.git$/, '') || 'repository'
  return {
    status: 'success',
    message: 'Repository cloned successfully. (demo)',
    repository: name,
    path: `repositories/${name}`,
  }
}

export async function demoIndex(): Promise<IndexStats> {
  await delay(1600)
  return { files: 128, chunks: 1042, stored_vectors: 1042 }
}
