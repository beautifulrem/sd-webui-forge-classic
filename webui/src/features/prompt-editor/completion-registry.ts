/**
 * Completion sources for prompt editors (tag autocomplete, wildcards, LoRA
 * names, …). Registering is cheap and does not load CodeMirror: the editor
 * chunk subscribes and reconfigures its open editors. Sources receive a
 * CodeMirror `CompletionContext`; `promptKind` (see `facets.ts`) tells
 * positive from negative prompts.
 *
 *   const off = registerCompletionSource("tags", async (ctx) => {
 *     const word = ctx.matchBefore(/[\w-]+/)
 *     if (!word || (word.from === word.to && !ctx.explicit)) return null
 *     return { from: word.from, options: await lookup(word.text) }
 *   })
 */
import type { CompletionSource } from "@codemirror/autocomplete"

export type CompletionEntry = {
  id: string
  source: CompletionSource
  /** Higher runs first (CodeMirror merges all results). */
  priority: number
}

const entries = new Map<string, CompletionEntry>()
const listeners = new Set<() => void>()
let snapshot: readonly CompletionEntry[] = []

function changed() {
  snapshot = [...entries.values()].sort((a, b) => b.priority - a.priority)
  for (const listener of listeners) listener()
}

/** Registers (or replaces) the source `id`; returns the unregister function. */
export function registerCompletionSource(id: string, source: CompletionSource, options: { priority?: number } = {}): () => void {
  const entry: CompletionEntry = { id, source, priority: options.priority ?? 0 }
  entries.set(id, entry)
  changed()
  return () => {
    if (entries.get(id) !== entry) return
    entries.delete(id)
    changed()
  }
}

/** The registered sources, highest priority first (a stable array until the next change). */
export function completionSources(): readonly CompletionEntry[] {
  return snapshot
}

export function subscribeCompletionSources(listener: () => void): () => void {
  listeners.add(listener)
  return () => listeners.delete(listener)
}
