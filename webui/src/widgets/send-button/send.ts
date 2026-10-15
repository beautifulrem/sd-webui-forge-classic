/** What the `send-button` widget does, without UI. */
import type { FormTab } from "@/api/types"
import { navigate } from "@/app/router"
import { runChangeEffects } from "@/ui/actions"
import { getFormStore } from "@/ui/form-store"

/**
 * `text` joined after `current` the way the old Gradio buttons did: a
 * trailing comma gets a space, anything else ", " (trailing whitespace of
 * `current` dropped); an empty `current` is replaced.
 */
export function appendText(current: string, text: string): string {
  const base = current.replace(/\s+$/, "")
  if (!base) return text
  return base.endsWith(",") ? `${base} ${text}` : `${base}, ${text}`
}

/**
 * Writes `text` into a generation form's prompt (`core.prompt`), appended or
 * replacing it, and optionally shows that form. A form not loaded yet keeps
 * it as its draft, which its first load restores. False for empty text.
 */
export function sendToPrompt(target: Extract<FormTab, "txt2img" | "img2img">, text: string, mode: "append" | "replace", show: boolean): boolean {
  const trimmed = text.trim()
  if (!trimmed) return false
  const store = getFormStore(target)
  const current = store.getState().values.core?.prompt
  const next = mode === "append" ? appendText(typeof current === "string" ? current : "", trimmed) : trimmed
  store.getState().setValues("core", { prompt: next })
  if (store.getState().revision > 0) runChangeEffects({ store, tab: target }, "core", ["prompt"])
  if (show) navigate({ name: "generate", tab: target })
  return true
}

/** Opens an http(s) URL in a new tab; false for anything else. */
export function openUrl(url: string): boolean {
  let parsed: URL
  try {
    parsed = new URL(url.trim())
  } catch {
    return false
  }
  if (parsed.protocol !== "https:" && parsed.protocol !== "http:") return false
  window.open(parsed.href, "_blank", "noopener,noreferrer")
  return true
}
