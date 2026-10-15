/** Helpers and props of the prompt tools panels. */
import type { PromptKind } from "@/features/prompt-editor/facets"
import type { PromptToolsConfig, TagSettings } from "../api"
import { tagText } from "../completion/insert"

export type PanelProps = {
  value: string
  setValue: (value: string) => void
  kind: PromptKind
  label: string
  disabled: boolean
  config: PromptToolsConfig
  close: () => void
}

/** "3 minutes ago" in the UI language. */
export function timeAgo(seconds: number, locale: string): string {
  const diff = seconds - Date.now() / 1000
  const format = new Intl.RelativeTimeFormat(locale, { numeric: "auto" })
  const steps: [number, Intl.RelativeTimeFormatUnit][] = [
    [60, "second"],
    [3600, "minute"],
    [86400, "hour"],
    [604800, "day"],
    [2629800, "week"],
    [31557600, "month"],
  ]
  const abs = Math.abs(diff)
  for (const [limit, unit] of steps) {
    if (abs < limit) {
      const size = unit === "second" ? 1 : unit === "minute" ? 60 : unit === "hour" ? 3600 : unit === "day" ? 86400 : unit === "week" ? 604800 : 2629800
      return format.format(Math.round(diff / size), unit)
    }
  }
  return format.format(Math.round(diff / 31557600), "year")
}

/** Appends text to a prompt with a comma between (no doubled separators). */
export function appendText(current: string, text: string): string {
  const head = current.replace(/[\s,]+$/, "")
  const added = text.trim()
  if (!head) return added
  if (!added) return current
  return `${head}, ${added}`
}

/** A quick-group tag as it is written into the prompt, like an accepted completion (`hanten_(clothes)` → `hanten \(clothes\)`). */
export function groupTagText(tag: string, settings: TagSettings): string {
  return tagText({ text: tag, type: "tag", category: null }, settings)
}
