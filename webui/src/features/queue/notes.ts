/** Importer notes are English messages with `{{name}}` placeholders, translated like the chrome strings. */
import type { JobNote } from "@/api/types"
import { t } from "@/i18n"

export function noteText(note: JobNote | string): string {
  if (typeof note === "string") return t(note)
  return t(note.message, (note.values ?? {}) as Record<string, string | number>)
}
