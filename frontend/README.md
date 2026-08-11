# Repoint — frontend

React + Vite + TypeScript + Tailwind UI for the Repository Intelligence Platform.

```bash
npm install
npm run dev      # http://localhost:5173
```

The app boots into **sample-data mode** if the backend is unreachable, so you can
explore the whole interface without Ollama or an index. The pill in the top bar
shows which mode you are in and switches between them.

To run against the real backend:

```bash
cd ../backend
uvicorn app.main:app --reload      # :8000
```

Vite proxies `/api/*` → `http://localhost:8000` (see [vite.config.ts](vite.config.ts)),
which avoids CORS entirely in development. Point it elsewhere with
`VITE_BACKEND_URL` — see [.env.example](.env.example).

## Design intent

Google AI Studio devotes its right panel to **model knobs** — temperature, tokens,
safety. That is the right call when the model is the product. Here the model is
the cheap part; the retrieval is what determines whether an answer is true. So the
right panel is a **Retrieval Inspector** instead:

- every chunk the model was given, ranked, with a relevance meter and raw distance
- the full chunk text, syntax-highlighted, expandable
- a coverage summary — how many distinct files the evidence came from, and whether
  it clustered in one place
- a pipeline trace for embed → search → generate

Filenames the model mentions in its answer become clickable citations wired to
those cards, so you can go from a claim to the exact bytes that produced it.

Other deliberate differences: dark-first with a fine drafting grid rather than flat
white, monospace for every code-derived string, and a left rail that is a
**repository workspace** (index state, language mix, vector counts) rather than a
prompt history.

## Layout

```
src/
  lib/          api client, shared types, formatting, offline fixtures
  store/        zustand store — repositories, conversation, run settings
  components/
    layout/     top bar, connection state, theme toggle
    repos/      repository rail + import dialog
    chat/       message list, citation linking, composer
    inspector/  retrieval evidence + run config
```

## Backend endpoints it consumes

| Endpoint | Status |
| --- | --- |
| `GET /` | used as a health probe |
| `POST /chat/` | live |
| `POST /github/import` | live |
| `POST /index/` | **not implemented yet** — the button surfaces a clear message |

Two small backend changes would light up features the UI already renders:

1. Include `content`, `language`, and `chunk_index` in the `sources` list from
   `chat_service.py` so the Inspector can show the retrieved text.
2. Return a `timings` object (`embed`, `search`, `generate`, in ms) from `/chat/`
   for the per-stage pipeline trace.

## Scripts

| Command | What it does |
| --- | --- |
| `npm run dev` | dev server with HMR and the API proxy |
| `npm run build` | typecheck then produce `dist/` |
| `npm run typecheck` | types only |
| `npm run preview` | serve the built bundle |
