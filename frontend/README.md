# Repoint — frontend

React + Vite + TypeScript + Tailwind UI for the Repository Intelligence Platform.

```bash
npm install
npm run dev      # http://localhost:5173
```

It needs the backend running:

```bash
cd ../backend
uvicorn app.main:app --reload      # :8000
```

Vite proxies `/api/*` → `http://localhost:8000` (see [vite.config.ts](vite.config.ts)),
which avoids CORS entirely in development. Point it elsewhere with
`VITE_BACKEND_URL` — see [.env.example](.env.example).

## What it does

One screen: a repository's **knowledge graph** and a **chat that talks to it**.

1. **Paste a GitHub URL.** The backend clones it and builds the graph in seconds,
   then embeds the code in the background. The graph is drawn as soon as it
   exists; the chat works immediately and quotes code once indexing finishes.
2. **Explore the graph.** Files, classes and functions as nodes; imports, calls,
   inheritance and containment as edges, coloured by community. Hover for the
   path, click to select a node and see what it connects to, drag to pan, scroll
   to zoom, search to jump to anything — including nodes the capped graph left out.
3. **Talk to it.** A selected node travels with the question, so *"what depends on
   this?"* needs no name. Each answer comes back with the nodes it used; the graph
   dims everything else and zooms to them. Earlier answers can be re-shown.

## Layout

```
src/
  lib/          API client, shared types, community detection, formatting
  store/        zustand — workspace (repositories, import jobs), graph, chat
  components/
    graph/      canvas renderer (d3-force) and the knowledge graph component
    workspace/  import form, graph pane, chat
    layout/     top bar, connection state, theme toggle
```

## Backend endpoints it consumes

| Endpoint | Used for |
| --- | --- |
| `GET /` | health probe |
| `GET /repositories` | repositories with a graph or an import in flight |
| `POST /repositories/import` | start clone → graph → code index |
| `GET /repositories/{name}/status` | import progress, polled |
| `GET /graph/{name}/knowledge` | the whole knowledge graph, capped at 1,500 nodes |
| `GET /graph/nodes?id=…` | load nodes an answer or search needs that the cap left out |
| `GET /graph/{name}/search` | find files and symbols |
| `POST /agent/ask` | answers, with the graph nodes to highlight |

## Scripts

| Command | What it does |
| --- | --- |
| `npm run dev` | dev server with HMR and the API proxy |
| `npm run build` | typecheck then produce `dist/` |
| `npm run typecheck` | types only |
| `npm run preview` | serve the built bundle |
