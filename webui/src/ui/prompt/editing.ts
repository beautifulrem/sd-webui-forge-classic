import { type AttentionOptions, DEFAULT_ATTENTION_OPTIONS } from "./attention"

export type PromptKeyOptions = { attention: AttentionOptions; move: boolean }

/** Prompt-editing options from the server's settings (keyedit_*), with A1111 defaults. */
export function keyOptionsFrom(options: Record<string, unknown> | undefined): PromptKeyOptions {
  const o = options ?? {}
  const num = (key: string, fallback: number) => (typeof o[key] === "number" ? (o[key] as number) : fallback)
  return {
    attention: {
      precisionAttention: num("keyedit_precision_attention", DEFAULT_ATTENTION_OPTIONS.precisionAttention),
      precisionExtra: num("keyedit_precision_extra", DEFAULT_ATTENTION_OPTIONS.precisionExtra),
      delimiters: typeof o.keyedit_delimiters === "string" ? o.keyedit_delimiters : DEFAULT_ATTENTION_OPTIONS.delimiters,
      whitespaceDelimiters: Array.isArray(o.keyedit_delimiters_whitespace)
        ? (o.keyedit_delimiters_whitespace as string[])
        : DEFAULT_ATTENTION_OPTIONS.whitespaceDelimiters,
    },
    move: o.keyedit_move !== false,
  }
}

/** The smallest single replacement turning `before` into `after` (keeps history and marks tidy). */
export function minimalChange(before: string, after: string) {
  let start = 0
  const max = Math.min(before.length, after.length)
  while (start < max && before.charCodeAt(start) === after.charCodeAt(start)) start++
  let endBefore = before.length
  let endAfter = after.length
  while (endBefore > start && endAfter > start && before.charCodeAt(endBefore - 1) === after.charCodeAt(endAfter - 1)) {
    endBefore--
    endAfter--
  }
  return { from: start, to: endBefore, insert: after.slice(start, endAfter) }
}
