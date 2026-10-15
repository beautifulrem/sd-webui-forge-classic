/**
 * Translation actions of the prompt tools (only offered when the user
 * configured a provider): write the prompt's non-English tags in English,
 * and fetch translations of English tags for the tag view.
 */
import { queryClient } from "@/api/queries"
import { promptToolsApi } from "./api"
import { displayText, parsePrompt, setChipTexts } from "./chips/parse"

/** CJK, kana, hangul: text that is not written in English. */
const FOREIGN = /[぀-ヿ㐀-䶿一-鿿가-힯豈-﫿]/

export function isForeign(text: string): boolean {
  return FOREIGN.test(text)
}

/** Brackets in translated plain text must not read as weights. */
export function escapeBrackets(text: string): string {
  return text.replace(/(?<!\\)([()[\]])/g, "\\$1")
}

/**
 * A translation as prompt text: one line, no final full stop, brackets
 * escaped (no weights), and the separators of a list (`girl, smiling`,
 * `、`, `;`) written as tag separators; a stray `AND` / `BREAK` is not a
 * prompt keyword.
 */
export function promptFromTranslation(text: string): string {
  return text
    .trim()
    .replace(/[.。]+$/, "")
    .split(/\s*[,，、;；\n]+\s*/)
    .map((part) =>
      escapeBrackets(
        part
          .replace(/\s+/g, " ")
          .replace(/\b(AND|BREAK)\b/g, (word) => word.toLowerCase())
          .trim(),
      ),
    )
    .filter(Boolean)
    .join(", ")
}

/**
 * The prompt with every non-English tag translated to English (weights,
 * networks and separators kept; every replacement is made on the same
 * parse, so a translation holding a comma does not shift the others);
 * `translated` counts the replaced tags.
 */
export async function translatePromptToEnglish(value: string): Promise<{ text: string; translated: number }> {
  const doc = parsePrompt(value)
  const targets = doc.chips.flatMap((chip, index) => (chip.kind === "tag" && isForeign(chip.text) ? [{ index, text: displayText(chip.text) }] : []))
  if (targets.length === 0) return { text: value, translated: 0 }
  const answer = await promptToolsApi.translate(
    targets.map((target) => target.text),
    "en",
  )
  const edits = targets.flatMap((target, position) => {
    const english = answer.translations[position]
    const text = english ? promptFromTranslation(english) : ""
    return text ? [{ index: target.index, text }] : []
  })
  if (edits.length === 0) return { text: value, translated: 0 }
  return { text: setChipTexts(doc, edits), translated: edits.length }
}

export const translatedKey = (language: string) => ["prompt-tools", "translated", language] as const

/** Translations fetched for the tag view (text → translation), per language. */
export function translatedTags(language: string): Record<string, string> {
  return queryClient.getQueryData<Record<string, string>>(translatedKey(language)) ?? {}
}

/** Translates English tags into `language` with the provider and keeps the result for the tag view. */
export async function translateTags(texts: string[], language: string): Promise<number> {
  const known = translatedTags(language)
  // any number: the API client sends them in chunks the server accepts
  const missing = [...new Set(texts.map(displayText).filter((text) => text.trim() && !isForeign(text) && !(text in known)))]
  if (missing.length === 0) return 0
  const answer = await promptToolsApi.translate(missing, language)
  const found: Record<string, string> = {}
  missing.forEach((text, index) => {
    const value = answer.translations[index]
    if (value) found[text] = value
  })
  queryClient.setQueryData(translatedKey(language), { ...translatedTags(language), ...found })
  return Object.keys(found).length
}
