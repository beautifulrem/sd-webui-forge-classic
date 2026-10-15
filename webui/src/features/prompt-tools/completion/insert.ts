/**
 * How an accepted completion is written into the prompt (the `tac_*`
 * insertion settings): underscores → spaces (minus the exclusion list),
 * escaped parentheses, the separator after a tag, `<lora:alias:weight>` with
 * its activation text, wildcard paths one folder at a time, Anima's `@`
 * artist prefix. Pure: returns the document changes and the cursor.
 */
import type { CompletionItem, TagSettings } from "../api"
import type { CompletionTarget } from "./context"

export type InsertSettings = Pick<
  TagSettings,
  | "replace_underscores"
  | "underscore_exclusions"
  | "escape_parentheses"
  | "append_comma"
  | "append_space"
  | "always_space_at_end"
  | "keywords"
  | "keyword_location"
  | "artist_prefix"
  | "source"
  | "network_separator"
  | "default_weight"
  | "wildcard_mode"
>

export type Change = { from: number; to: number; insert: string }
export type Insertion = {
  changes: Change[]
  /** The cursor after the insertion (in the new document). */
  cursor: number
  /** Re-open completion afterwards (a partial wildcard path, a wildcard's values). */
  reopen: boolean
}

/** The artist category of the tag lists that use one (danbooru-style). */
const ARTIST_CATEGORY = 1

export function formatWeight(weight: number): string {
  const rounded = Math.round(weight * 1000) / 1000
  return String(rounded)
}

/** Whether a tag is an artist of a list with an artist category (danbooru-style, e621, merged). */
function isArtist(item: Pick<CompletionItem, "type" | "category">, settings: Pick<InsertSettings, "source">): boolean {
  return item.type === "tag" && item.category === ARTIST_CATEGORY && settings.source !== "derpibooru"
}

/**
 * A tag as it is written into the prompt. `typedPrefix`: the user typed
 * `@` before it (kept for an artist even with the automatic prefix off).
 */
export function tagText(item: Pick<CompletionItem, "text" | "type" | "category">, settings: InsertSettings, typedPrefix?: "@"): string {
  let text = item.text
  if (settings.replace_underscores && !settings.underscore_exclusions.includes(text)) text = text.replaceAll("_", " ")
  if (settings.escape_parentheses && (item.type === "tag" || item.type === "extra")) text = text.replace(/(?<!\\)([()[\]])/g, "\\$1")
  const automatic = settings.artist_prefix && settings.source !== "e621"
  if (isArtist(item, settings) && (automatic || typedPrefix === "@") && !text.startsWith("@")) text = `@${text}`
  return text
}

/**
 * The separator written after a tag: `", "` by default; nothing in front of a
 * comma, a closing bracket, `:` or `|` (a weight or schedule continues);
 * a lone space at the very end with `always_space_at_end`.
 */
export function separatorAfter(following: string, settings: InsertSettings): string {
  const next = following.trimStart()[0]
  if (next !== undefined && /[,)\]}>|:]/.test(next)) return ""
  let separator = ""
  if (settings.append_comma) separator += ","
  if (settings.append_space) separator += " "
  else if (settings.always_space_at_end && next === undefined) separator += " "
  // a space already follows: no second one
  if (next !== undefined && /^\s/.test(following)) separator = separator.trimEnd()
  return separator
}

/** The next path level of a nested wildcard for a Tab completion (`hair/` → `hair/colors/`). */
export function nextWildcardLevel(name: string, typed: string, mode: string, common?: string): string {
  if (mode === "Always fully" || !name.includes("/")) return name
  if (mode === "To first difference") {
    return common && common.length > typed.length && common.length < name.length ? common : name
  }
  const at = name.toLowerCase().startsWith(typed.toLowerCase()) ? typed.length : 0
  let slash = name.indexOf("/", at)
  // the typed text already ends at a folder: go one level deeper
  if (slash === at - 1) slash = name.indexOf("/", at)
  return slash >= 0 && slash < name.length - 1 ? name.slice(0, slash + 1) : name
}

/** The longest common prefix of the names (case-sensitive). */
export function commonPrefix(names: readonly string[]): string {
  if (names.length === 0) return ""
  let prefix = names[0]!
  for (const name of names.slice(1)) {
    let i = 0
    while (i < prefix.length && i < name.length && prefix[i] === name[i]) i++
    prefix = prefix.slice(0, i)
  }
  return prefix
}

type InsertInput = {
  doc: string
  target: CompletionTarget
  /** The end of the replaced range (`tokenEnd`). */
  to: number
  item: CompletionItem
  settings: InsertSettings
  /** How it was accepted (Tab completes wildcard paths step by step). */
  key: "Tab" | "Enter" | null
  /** "To first difference": the common prefix of the listed wildcard names. */
  common?: string
  wrap?: string
}

export function insertion({ doc, target, to, item, settings, key, common, wrap = "__" }: InsertInput): Insertion {
  const from = target.from
  const following = doc.slice(to)
  const simple = (text: string, reopen = false): Insertion => ({ changes: [{ from, to, insert: text }], cursor: from + text.length, reopen })

  switch (item.type) {
    case "wildcard": {
      const name = key === "Tab" ? nextWildcardLevel(item.text, target.query, settings.wildcard_mode, common) : item.text
      if (name !== item.text) return simple(`${wrap}${name}`, true)
      return simple(`${wrap}${name}${wrap}`, true)
    }
    case "lora": {
      const weight = item.weight ?? settings.default_weight
      const tag = `<lora:${item.text}:${formatWeight(weight)}>`
      const separator = following.length === 0 || /^[\s,]/.test(following) ? "" : settings.network_separator || " "
      const keywords = settings.keywords && item.keywords && !doc.includes(item.keywords) ? item.keywords : null
      if (!keywords) return simple(tag + separator)
      switch (settings.keyword_location) {
        case "End of prompt": {
          // one change from the tag to the end: the prompt, then ", keywords"
          const text = tag + separator
          const base = (doc.slice(0, from) + text + following).trimEnd()
          const final = base + (base ? (base.endsWith(",") ? " " : ", ") : "") + keywords
          const changeFrom = Math.min(from, base.length)
          return { changes: [{ from: changeFrom, to: doc.length, insert: final.slice(changeFrom) }], cursor: Math.min(from + text.length, base.length), reopen: false }
        }
        case "Before LORA/LyCO": {
          const text = `${keywords}, ${tag}${separator}`
          return simple(text)
        }
        case "After LORA/LyCO": {
          const text = `${tag} ${keywords}${following.trimStart().startsWith(",") || following.length === 0 ? "" : ", "}`
          return simple(text)
        }
        default: {
          // Start of prompt
          const text = tag + separator
          if (from === 0) return { changes: [{ from: 0, to, insert: `${keywords}, ${text}` }], cursor: keywords.length + 2 + text.length, reopen: false }
          return { changes: [{ from: 0, to: 0, insert: `${keywords}, ` }, { from, to, insert: text }], cursor: from + keywords.length + 2 + text.length, reopen: false }
        }
      }
    }
    case "embedding":
    case "chant":
    case "wildcard-value":
      return simple(item.text + separatorAfter(following, settings))
    default:
      return simple(tagText(item, settings, target.prefix) + separatorAfter(following, settings))
  }
}
