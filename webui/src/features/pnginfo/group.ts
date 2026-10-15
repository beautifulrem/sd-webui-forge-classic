import type { MetadataEntry } from "@/api/types"

/** Entries grouped as the server labels them ("Image", "Metadata", "EXIF …"). */
export function groupEntries(entries: readonly MetadataEntry[]): [string, MetadataEntry[]][] {
  const groups = new Map<string, MetadataEntry[]>()
  for (const entry of entries) {
    const list = groups.get(entry.group) ?? []
    list.push(entry)
    groups.set(entry.group, list)
  }
  return [...groups]
}
