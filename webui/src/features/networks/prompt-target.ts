/**
 * Which prompt a network card goes into: the prompt editor focused last
 * (positive or negative, read from the editor's `promptKind` facet), unless
 * the browser's target switch says otherwise. Cards change the generation
 * form's `prompt` / `negative_prompt` values, so the editors (which put
 * external changes into their history) can undo them with Ctrl+Z.
 */
import { useEffect } from "react"
import { create, useStore } from "zustand"
import type { NetworkItem } from "@/api/types"
import { getFormStore } from "@/ui/form-store"
import { isInPrompt, togglePromptText } from "./prompt-insert"
import { type PromptTarget, useNetworksUi } from "./store"

export const PROMPT_PARAMS: Record<PromptTarget, string> = { positive: "prompt", negative: "negative_prompt" }

type FocusState = { kind: PromptTarget }

/** The kind of the prompt editor focused last. */
export const useFocusedPrompt = create<FocusState>()(() => ({ kind: "positive" }))

async function editorKind(editor: HTMLElement): Promise<PromptTarget> {
  const [{ EditorView }, { promptKind }] = await Promise.all([import("@codemirror/view"), import("@/features/prompt-editor/facets")])
  const view = EditorView.findFromDOM(editor)
  return view?.state.facet(promptKind) ?? "positive"
}

function onFocusIn(event: FocusEvent) {
  const target = event.target
  if (!(target instanceof Element)) return
  const editor = target.closest<HTMLElement>(".cm-editor")
  if (!editor?.closest('[data-slot="prompt-editor"]')) return
  void editorKind(editor).then((kind) => {
    useFocusedPrompt.setState({ kind })
    // focusing a prompt hands the choice back to the focus
    useNetworksUi.getState().setTarget(null)
  })
}

let listeners = 0

/** Follows prompt-editor focus while mounted (any number of callers). */
export function useTrackPromptFocus() {
  useEffect(() => {
    if (listeners++ === 0) document.addEventListener("focusin", onFocusIn)
    return () => {
      if (--listeners === 0) document.removeEventListener("focusin", onFocusIn)
    }
  }, [])
}

/** The prompt a card without negative text goes to. */
export function usePromptTarget(): PromptTarget {
  const chosen = useNetworksUi((s) => s.target)
  const focused = useFocusedPrompt((s) => s.kind)
  return chosen ?? focused
}

export type Prompts = { positive: string; negative: string }

export type InsertPlan = {
  /** New values of the prompts that change. */
  changes: Partial<Prompts>
  /** Added (true) or removed (false). */
  added: boolean
}

/**
 * The prompts after clicking a card (old `cardClicked`): a card with negative
 * text puts its text into the prompt and the negative text into the
 * negative prompt; otherwise its text goes to `target` (the positive prompt
 * when the kind has no negative use). Clicking again removes them.
 */
export function planInsertion(
  prompts: Prompts,
  item: Pick<NetworkItem, "prompt" | "negative_prompt">,
  options: { negativeAllowed: boolean; target: PromptTarget; separator: string },
): InsertPlan | null {
  const text = item.prompt?.trim()
  if (!text) return null
  const negativeText = options.negativeAllowed ? item.negative_prompt?.trim() : undefined
  if (negativeText) {
    const positive = togglePromptText(prompts.positive, text, options.separator)
    const inNegative = isInPrompt(prompts.negative, negativeText)
    // keep the pair together: the negative text follows what the positive prompt did
    const negative = positive.added === inNegative ? prompts.negative : togglePromptText(prompts.negative, negativeText, options.separator).value
    return { changes: { positive: positive.value, ...(negative !== prompts.negative ? { negative } : {}) }, added: positive.added }
  }
  const target = options.negativeAllowed ? options.target : "positive"
  const result = togglePromptText(prompts[target], text, options.separator)
  return { changes: { [target]: result.value }, added: result.added }
}

/** The core prompts of a generation form. */
export function readPrompts(formId: string): Prompts {
  const core = getFormStore(formId).getState().values.core ?? {}
  const text = (value: unknown) => (typeof value === "string" ? value : "")
  return { positive: text(core[PROMPT_PARAMS.positive]), negative: text(core[PROMPT_PARAMS.negative]) }
}

/** Writes changed prompts into the form (also before its schema loaded: `load` keeps them). */
export function writePrompts(formId: string, changes: Partial<Prompts>) {
  const values: Record<string, string> = {}
  for (const [kind, value] of Object.entries(changes) as [PromptTarget, string][]) values[PROMPT_PARAMS[kind]] = value
  if (Object.keys(values).length) getFormStore(formId).getState().setValues("core", values)
}

/** The form's prompts, reactive. */
export function useFormPrompts(formId: string): Prompts {
  const store = getFormStore(formId)
  const positive = useStore(store, (s) => s.values.core?.[PROMPT_PARAMS.positive])
  const negative = useStore(store, (s) => s.values.core?.[PROMPT_PARAMS.negative])
  return { positive: typeof positive === "string" ? positive : "", negative: typeof negative === "string" ? negative : "" }
}
