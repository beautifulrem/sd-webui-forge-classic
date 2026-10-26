/**
 * Placeholders may name the generate shortcuts as `{{generate}}`, `{{skip}}`
 * and `{{interrupt}}` (the core prompt boxes do): they are filled with the
 * same platform-aware text as the Generate button and the shortcuts dialog
 * ("⌘Enter" on macOS, "Ctrl+Enter" elsewhere), in every language.
 */
import type { TFunction } from "i18next"
import { formatShortcut } from "@/app/commands"

export const SHORTCUT_PLACEHOLDERS = {
  generate: formatShortcut("Mod+Enter"),
  skip: formatShortcut("Alt+Enter"),
  interrupt: "Esc",
} as const

/** Stands for a shortcut while the lines that name one are found. */
const MARK = "\u0001"

/**
 * The translated placeholder with the shortcuts filled in. Without
 * `keyboardHints` (a touch-only device, `useKeyboardHints`) the lines that
 * name a shortcut ("(⌘Enter to Generate ; …)") are left out.
 */
export function placeholderText(t: TFunction, placeholder: string | null | undefined, keyboardHints = true): string | undefined {
  if (!placeholder) return undefined
  const full = t(placeholder, SHORTCUT_PLACEHOLDERS)
  if (keyboardHints) return full
  const lines = t(placeholder, { generate: MARK, skip: MARK, interrupt: MARK })
    .split("\n")
    .filter((line) => !line.includes(MARK))
  return lines.length > 0 ? lines.join("\n") : full
}
