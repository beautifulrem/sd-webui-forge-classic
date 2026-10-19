/**
 * Prompt editor public API. Importing this module does not load CodeMirror:
 * the editor itself is a lazy chunk (`LazyPromptEditor`).
 *
 * Extension points: `registerCompletionSource(id, source)` adds a CodeMirror
 * completion source to every prompt editor (open and future ones), e.g. tag
 * autocomplete; sources can read `promptKind` and `promptScope` (form, extension
 * field or main prompt) from `./facets`, decorate
 * their options (`DecoratedCompletion`) and tell Tab from Enter
 * (`completionAcceptKey`). `registerPromptDecorator(id, Component)` wraps
 * every prompt editor in a component (toolbars, alternative views).
 */
import { lazy } from "react"

export { completionAcceptKey, registerCompletionSource, type CompletionEntry, type DecoratedCompletion } from "./completion-registry"
export { registerPromptDecorator, type PromptDecoratorProps } from "./decorators"
export type { PromptEditorHandle, PromptEditorProps } from "./PromptEditor"
export type { PromptKind, PromptScope } from "./facets"
export { bracketProblems, tokenizePrompt, type BracketProblem, type PromptToken } from "./tokenize"

export const LazyPromptEditor = lazy(() => import("./PromptEditor"))
