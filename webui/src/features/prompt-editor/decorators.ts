/**
 * Prompt decorators: components that wrap every prompt editor (the core
 * prompts of `PromptField` and the renderer's `prompt: true` text fields),
 * e.g. the prompt tools' toolbar and tag-chips view. A decorator receives the
 * editor as `children` and the field's value; it renders them however it
 * likes (it must keep `children` mounted so the editor keeps its undo
 * history, e.g. hidden while another view is shown).
 *
 *   registerPromptDecorator("tools", PromptToolbar)
 */
import type { ComponentType, ReactNode } from "react"
import type { PromptKind } from "./facets"
import type { PromptEditorHandle } from "./PromptEditor"

export type PromptDecoratorProps = {
  /** The editor (keep it mounted). */
  children: ReactNode
  value: string
  setValue: (value: string) => void
  kind: PromptKind
  /** The editor's handle once it has loaded. */
  editor: () => PromptEditorHandle | null
  /** The field's (translated) label. */
  label: string
  disabled: boolean
  /** Id of the editor's content element. */
  fieldId: string
  /** One of the main prompts of a generation page (`PromptField`), not a script's prompt field. */
  main?: boolean
}

export type PromptDecoratorEntry = { id: string; component: ComponentType<PromptDecoratorProps>; priority: number }

const entries = new Map<string, PromptDecoratorEntry>()
const listeners = new Set<() => void>()
let snapshot: readonly PromptDecoratorEntry[] = []

/** Registers (or replaces) decorator `id`; higher priorities wrap outermost. Returns the unregister function. */
export function registerPromptDecorator(id: string, component: ComponentType<PromptDecoratorProps>, options: { priority?: number } = {}): () => void {
  const entry: PromptDecoratorEntry = { id, component, priority: options.priority ?? 0 }
  entries.set(id, entry)
  snapshot = [...entries.values()].sort((a, b) => a.priority - b.priority)
  listeners.forEach((listener) => listener())
  return () => {
    if (entries.get(id) !== entry) return
    entries.delete(id)
    snapshot = [...entries.values()].sort((a, b) => a.priority - b.priority)
    listeners.forEach((listener) => listener())
  }
}

export function subscribePromptDecorators(listener: () => void): () => void {
  listeners.add(listener)
  return () => listeners.delete(listener)
}

/** The registered decorators, innermost first (a stable array until the next change). */
export const promptDecorators = (): readonly PromptDecoratorEntry[] => snapshot
