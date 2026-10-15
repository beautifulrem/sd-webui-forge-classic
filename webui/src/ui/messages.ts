/**
 * Server-made texts shown to the user (text outputs, action toasts). A plain
 * string is translated only when a catalog has it verbatim (arbitrary text
 * is never run through interpolation); a message `{ text, params }`
 * (`remi.ui.message`) is a catalog key whose `{{name}}` placeholders are
 * filled from `params` after translation.
 */
import type { i18n as I18n } from "i18next"

export type Message = { text: string; params?: Record<string, unknown> }

export function isMessage(value: unknown): value is Message {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return false
  const { text, params } = value as Record<string, unknown>
  return typeof text === "string" && (params === undefined || (typeof params === "object" && params !== null && !Array.isArray(params)))
}

/** The display text of a string or message in `i18n`'s current language. */
export function messageText(i18n: Pick<I18n, "t" | "exists">, value: string | Message): string {
  if (typeof value === "string") return value && i18n.exists(value) ? i18n.t(value) : value
  // Missing keys fall back to the English text, interpolated all the same
  return i18n.t(value.text, { ...value.params, defaultValue: value.text })
}
