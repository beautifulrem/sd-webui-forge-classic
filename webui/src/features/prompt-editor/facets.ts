/** Editor state facets that completion sources and commands can read. */
import { Facet } from "@codemirror/state"

export type PromptKind = "positive" | "negative"

/** Which prompt an editor edits (`state.facet(promptKind)`). */
export const promptKind = Facet.define<PromptKind, PromptKind>({
  combine: (values) => values[0] ?? "positive",
})

/**
 * Where an editor is: its form (`txt2img`, `img2img`, `page:<id>`, … or null
 * outside a form, e.g. the style editor) and whether it is a prompt field of
 * an extension (a script's or a page's `prompt: true` text field) rather than
 * one of the main prompts (`state.facet(promptScope)`).
 */
export type PromptScope = { form: string | null; extension: boolean }

export const promptScope = Facet.define<PromptScope, PromptScope>({
  combine: (values) => values[0] ?? { form: null, extension: false },
})
