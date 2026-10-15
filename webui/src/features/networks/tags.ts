/** Training-tag helpers of the metadata dialogs (pure). */

export type WeightedTag = { tag: string; count: number }

/** Prompt-safe tag: brackets escaped, as the old editor did. */
export const escapeTag = (tag: string) => tag.replace(/[(){}[\]]/g, (char) => `\\${char}`)

/**
 * A random prompt from the dataset tags: each tag is picked with a chance of
 * its count over the top count (sd_forge_lora's "Random Prompt").
 */
export function randomPrompt(tags: readonly WeightedTag[], random: () => number = Math.random): string {
  const max = tags[0]?.count ?? 0
  if (!max) return ""
  return tags
    .filter((tag) => tag.count > random() * max)
    .map((tag) => escapeTag(tag.tag))
    .toSorted()
    .join(", ")
}

const COMMA = / *, */

/** Adds the tag to a comma-separated text, or removes it when present (old tag click). */
export function toggleTag(text: string, tag: string): string {
  const words = text.split(COMMA).filter((word) => word.trim())
  if (words.includes(tag)) return words.filter((word) => word !== tag).join(", ")
  return words.length ? `${words.join(", ")}, ${tag}` : tag
}

/** Whether `tag` is one of the comma-separated words of `text`. */
export const hasTag = (text: string, tag: string) => text.split(COMMA).includes(tag)

/** 0…1: how strongly a chip is emphasised (square root, so mid counts stay visible). */
export const tagWeight = (count: number, max: number) => (max > 0 ? Math.sqrt(Math.max(0, count) / max) : 0)
