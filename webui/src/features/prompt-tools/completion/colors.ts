import { t } from "@/i18n"

/**
 * Tag category colours per tag list family (the categories of danbooru,
 * e621, derpibooru and the merged list), as hues readable on both themes.
 */

type Hue = number | "neutral"

const DANBOORU: Record<number, Hue> = { [-1]: 25, 0: 235, 1: 20, 3: 305, 4: 145, 5: 60 }
const E621: Record<number, Hue> = { [-1]: 25, 0: 235, 1: 85, 3: 305, 4: 145, 5: 35, 6: 25, 7: "neutral", 8: 170 }
const DERPIBOORU: Record<number, Hue> = { [-1]: 25, 0: 140, 1: 100, 3: 45, 4: 330, 5: 250, 6: "neutral", 7: 210, 8: 280, 9: 20, 10: 70, 11: 345 }
const MERGED: Record<number, Hue> = { ...DANBOORU, 6: 25, 7: 235, 8: 85, 9: 85, 10: 305, 11: 145, 12: 35, 14: "neutral", 15: 170 }

const FAMILIES: Record<string, Record<number, Hue>> = { danbooru: DANBOORU, e621: E621, derpibooru: DERPIBOORU, merged: MERGED }

/** A CSS colour for a tag category, or null for an unknown one. */
export function categoryColor(source: string, category: number | null | undefined): string | null {
  if (category === null || category === undefined) return null
  const hue = (FAMILIES[source] ?? DANBOORU)[category]
  if (hue === undefined) return null
  if (hue === "neutral") return "var(--muted-foreground)"
  return `light-dark(oklch(0.52 0.15 ${hue}), oklch(0.78 0.12 ${hue}))`
}

/**
 * Option-row classes colouring the completion label (literal strings, so
 * Tailwind generates them).
 */
const LABEL_CLASSES: Record<number, string> = {
  20: "[&_.cm-completionLabel]:text-[light-dark(oklch(0.52_0.15_20),oklch(0.78_0.12_20))]",
  25: "[&_.cm-completionLabel]:text-[light-dark(oklch(0.52_0.15_25),oklch(0.78_0.12_25))]",
  35: "[&_.cm-completionLabel]:text-[light-dark(oklch(0.52_0.15_35),oklch(0.78_0.12_35))]",
  45: "[&_.cm-completionLabel]:text-[light-dark(oklch(0.52_0.15_45),oklch(0.78_0.12_45))]",
  60: "[&_.cm-completionLabel]:text-[light-dark(oklch(0.52_0.15_60),oklch(0.78_0.12_60))]",
  70: "[&_.cm-completionLabel]:text-[light-dark(oklch(0.52_0.15_70),oklch(0.78_0.12_70))]",
  85: "[&_.cm-completionLabel]:text-[light-dark(oklch(0.52_0.15_85),oklch(0.78_0.12_85))]",
  100: "[&_.cm-completionLabel]:text-[light-dark(oklch(0.52_0.15_100),oklch(0.78_0.12_100))]",
  140: "[&_.cm-completionLabel]:text-[light-dark(oklch(0.52_0.15_140),oklch(0.78_0.12_140))]",
  145: "[&_.cm-completionLabel]:text-[light-dark(oklch(0.52_0.15_145),oklch(0.78_0.12_145))]",
  170: "[&_.cm-completionLabel]:text-[light-dark(oklch(0.52_0.15_170),oklch(0.78_0.12_170))]",
  210: "[&_.cm-completionLabel]:text-[light-dark(oklch(0.52_0.15_210),oklch(0.78_0.12_210))]",
  235: "[&_.cm-completionLabel]:text-[light-dark(oklch(0.52_0.15_235),oklch(0.78_0.12_235))]",
  250: "[&_.cm-completionLabel]:text-[light-dark(oklch(0.52_0.15_250),oklch(0.78_0.12_250))]",
  280: "[&_.cm-completionLabel]:text-[light-dark(oklch(0.52_0.15_280),oklch(0.78_0.12_280))]",
  305: "[&_.cm-completionLabel]:text-[light-dark(oklch(0.52_0.15_305),oklch(0.78_0.12_305))]",
  330: "[&_.cm-completionLabel]:text-[light-dark(oklch(0.52_0.15_330),oklch(0.78_0.12_330))]",
  345: "[&_.cm-completionLabel]:text-[light-dark(oklch(0.52_0.15_345),oklch(0.78_0.12_345))]",
}

/** The class colouring a completion label by category ("" when unknown). */
export function categoryClass(source: string, category: number | null | undefined): string {
  if (category === null || category === undefined) return ""
  const hue = (FAMILIES[source] ?? DANBOORU)[category]
  if (hue === undefined) return ""
  if (hue === "neutral") return "[&_.cm-completionLabel]:text-muted-foreground"
  return LABEL_CLASSES[hue] ?? ""
}

const NAMES: Record<string, Record<number, string>> = {
  danbooru: { 0: "General", 1: "Artist", 3: "Copyright", 4: "Character", 5: "Meta" },
  e621: { 0: "General", 1: "Artist", 3: "Copyright", 4: "Character", 5: "Species", 6: "Invalid", 7: "Meta", 8: "Lore" },
  merged: { 0: "General", 1: "Artist", 3: "Copyright", 4: "Character", 5: "Meta", 7: "General", 8: "Artist", 10: "Copyright", 11: "Character", 12: "Species", 14: "Meta", 15: "Lore" },
  // derpibooru.csv: its own categories (no artist category)
  derpibooru: { 0: "General", 1: "Official content", 3: "Species", 4: "OC", 5: "Rating", 6: "Body type", 7: "Character", 8: "Origin", 9: "Error", 10: "Spoiler", 11: "Fan content" },
}

const LABELS: Record<string, () => string> = {
  General: () => t("General"),
  Artist: () => t("Artist"),
  Copyright: () => t("Copyright"),
  Character: () => t("Character"),
  Meta: () => t("Meta"),
  Species: () => t("Species"),
  Invalid: () => t("Invalid"),
  Lore: () => t("Lore"),
  "Official content": () => t("Official content"),
  OC: () => t("OC"),
  Rating: () => t("Rating"),
  "Body type": () => t("Body type"),
  Origin: () => t("Origin"),
  Error: () => t("Error"),
  Spoiler: () => t("Spoiler"),
  "Fan content": () => t("Fan content"),
}

/** The (translated) name of a category, or null. */
export function categoryName(source: string, category: number | null | undefined): string | null {
  if (category === null || category === undefined) return null
  const name = (NAMES[source] ?? NAMES.danbooru!)[category]
  return name ? (LABELS[name]?.() ?? name) : null
}

/** Post counts as "6.0M", "12k", "950". */
export function compactCount(count: number): string {
  if (count >= 1_000_000) return `${(count / 1_000_000).toFixed(count >= 10_000_000 ? 0 : 1)}M`
  if (count >= 1_000) return `${(count / 1_000).toFixed(count >= 10_000 ? 0 : 1)}k`
  return String(count)
}
