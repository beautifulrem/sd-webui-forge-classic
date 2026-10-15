/** Reads `<key>: <int>` from an infotext's parameter line ("Seed: 123, …"). */
export function infotextValue(infotext: string | null | undefined, key: string): number | null {
  if (!infotext) return null
  const escaped = key.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
  const match = new RegExp(`(?:^|[,\\n]\\s*)${escaped}:\\s*(-?\\d+)`).exec(infotext)
  return match ? Number(match[1]) : null
}
