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
import type { Completion, CompletionSource } from "@codemirror/autocomplete"

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

/**
 * Optional rendering extras a source can put on its completions: a class on
 * the option row and a node appended to it (counts, translations, …).
 */
export type DecoratedCompletion = Completion & {
  className?: string
  /** Extra content after the label (a fresh node per call). */
  decorate?: () => Node | null
}

let acceptKey: "Tab" | "Enter" | null = null

/**
 * The key that accepted the completion being applied (`null`: a click),
 * readable from a completion's `apply` function (e.g. Tab completes a
 * wildcard path one folder at a time, Enter inserts it whole).
 */
export function completionAcceptKey(): "Tab" | "Enter" | null {
  return acceptKey
}

/** Set by the editor's key bindings around `acceptCompletion`. */
export function withAcceptKey<T>(key: "Tab" | "Enter" | null, run: () => T): T {
  acceptKey = key
  try {
    return run()
  } finally {
    acceptKey = null
  }
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
