import { create } from 'zustand'
import * as api from '@/lib/api'
import type { Message } from '@/lib/types'
import { useGraphStore } from '@/store/useGraphStore'

const uid = () => Math.random().toString(36).slice(2, 10)

const message = (err: unknown) => (err instanceof Error ? err.message : 'Something went wrong.')

interface ChatState {
  messages: Message[]
  asking: boolean
  /** Composer text, in the store so graph actions can prefill a question. */
  draft: string
  /**
   * Part of the backend conversation id. The agent carries "it" and "this"
   * between turns under that id, so starting over needs a fresh one.
   */
  session: string

  setDraft: (draft: string) => void
  ask: (question: string) => Promise<void>
  clear: () => void
}

export const useChatStore = create<ChatState>((set, get) => ({
  messages: [],
  asking: false,
  draft: '',
  session: uid(),

  setDraft(draft) {
    set({ draft })
  },

  /**
   * Ask about the repository on screen.
   *
   * The graph selection travels with the question, so "what depends on
   * this?" needs no name. The answer comes back with the nodes it used, and
   * those are lit up in the graph.
   */
  async ask(question) {
    const trimmed = question.trim()
    const graph = useGraphStore.getState()
    const repository = graph.repository
    if (!trimmed || get().asking || !repository) return

    const pendingId = uid()
    set((s) => ({
      asking: true,
      draft: '',
      messages: [
        ...s.messages,
        { id: uid(), role: 'user', content: trimmed },
        { id: pendingId, role: 'assistant', content: '', pending: true },
      ],
    }))

    const resolve = (fields: Partial<Message>) =>
      set((s) => ({
        asking: false,
        messages: s.messages.map((m) =>
          m.id === pendingId ? { ...m, pending: false, ...fields } : m,
        ),
      }))

    try {
      const response = await api.askAgent(
        trimmed,
        repository,
        `repoint:${repository}:${get().session}`,
        graph.selectionContext(),
      )
      // The user may have switched repositories or started over meanwhile.
      if (!get().messages.some((m) => m.id === pendingId)) return

      resolve({
        content: response.answer ?? '',
        intent: response.intent,
        focusLabel: response.focus?.label ?? null,
        highlight: response.highlight,
        sources: response.sources ?? [],
      })
      if (useGraphStore.getState().repository === repository && response.highlight) {
        await useGraphStore.getState().showAnswer(response.highlight)
      }
    } catch (err) {
      resolve({ error: message(err) })
    }
  },

  clear() {
    set({ messages: [], asking: false, draft: '', session: uid() })
  },
}))
