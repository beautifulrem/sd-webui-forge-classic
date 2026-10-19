/**
 * Status lines from extensions often start with an emoji marker ("✅ Saved",
 * "⚪ Model not loaded", "❌ Failed: …"), as the Gradio UI showed them. The
 * renderer turns the marker into the app's own status styling (icon and tone)
 * so every page shows status the same way, in every theme and language.
 */
export type StatusTone = "success" | "info" | "warning" | "error"

const MARKERS: readonly [string, StatusTone][] = [
  ["✅", "success"],
  ["✔️", "success"],
  ["✔", "success"],
  ["✓", "success"],
  ["🟢", "success"],
  ["⚪", "info"],
  ["⚫", "info"],
  ["🔵", "info"],
  ["ℹ️", "info"],
  ["ℹ", "info"],
  ["⏳", "info"],
  ["⚠️", "warning"],
  ["⚠", "warning"],
  ["🟡", "warning"],
  ["🟠", "warning"],
  ["❌", "error"],
  ["❗", "error"],
  ["⛔", "error"],
  ["🔴", "error"],
  ["✖", "error"],
  ["✗", "error"],
]

/** The tone a status line's leading marker stands for, and the text without it. */
export function parseStatus(text: string): { tone: StatusTone | null; text: string } {
  const trimmed = text.trimStart()
  for (const [marker, tone] of MARKERS) {
    if (trimmed.startsWith(marker)) return { tone, text: trimmed.slice(marker.length).replace(/^[\s️]+/, "") }
  }
  return { tone: null, text }
}
