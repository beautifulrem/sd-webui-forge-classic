/**
 * Accessible names for inputs whose schema label is empty (the panel's title
 * already says what the box is, e.g. the "Anchor prompt" section's text box):
 * the nearest titled section or group names them, in the current language,
 * instead of the raw param name ("anchor", "extra_options").
 */
import { createContext, useContext } from "react"
import { useTranslation } from "react-i18next"

/** The translated title of the nearest titled section or group. */
export const ContainerTitleContext = createContext<string | null>(null)

/** "extra_options" → "Extra options" (the last resort, outside any titled container). */
export function readableName(name: string): string {
  const words = name.replace(/[_-]+/g, " ").trim()
  return words.charAt(0).toUpperCase() + words.slice(1)
}

export function useFallbackName(name: string): string {
  const { t } = useTranslation()
  const title = useContext(ContainerTitleContext)
  return title || t(readableName(name))
}
