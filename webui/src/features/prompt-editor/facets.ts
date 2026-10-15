/** Editor state facets that completion sources and commands can read. */
import { Facet } from "@codemirror/state"

export type PromptKind = "positive" | "negative"

/** Which prompt an editor edits (`state.facet(promptKind)`). */
export const promptKind = Facet.define<PromptKind, PromptKind>({
  combine: (values) => values[0] ?? "positive",
})
