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

export function placeholderText(t: TFunction, placeholder: string | null | undefined): string | undefined {
  return placeholder ? t(placeholder, SHORTCUT_PLACEHOLDERS) : undefined
}
