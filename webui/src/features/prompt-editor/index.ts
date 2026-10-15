/**
 * Prompt editor public API. Importing this module does not load CodeMirror:
 * the editor itself is a lazy chunk (`LazyPromptEditor`).
 *
 * Extension point: `registerCompletionSource(id, source)` adds a CodeMirror
 * completion source to every prompt editor (open and future ones), e.g. tag
 * autocomplete. Sources can read `promptKind` from `./facets`.
 */
import { lazy } from "react"

export { registerCompletionSource, type CompletionEntry } from "./completion-registry"
export type { PromptEditorHandle, PromptEditorProps } from "./PromptEditor"
export type { PromptKind } from "./facets"
export { bracketProblems, tokenizePrompt, type BracketProblem, type PromptToken } from "./tokenize"

export const LazyPromptEditor = lazy(() => import("./PromptEditor"))
