/**
 * Chinese typesetting puts a space between Han characters and Latin letters
 * or digits ("启用 ADetailer", "第 3 步"). The catalogs follow that rule in
 * their own text, but a template cannot know what it interpolates:
 * "启用{{title}}" reads "启用高分辨率修复" (right) and "启用ADetailer"
 * (missing space). This post-processor adds the space at the edges of the
 * interpolated values only, so translated text around them is left as written.
 */
import type { PostProcessorModule, TOptions } from "i18next"

const HAN = /\p{Script=Han}/u
const LATIN = /[A-Za-z0-9]/

/** Options i18next reads itself: never interpolated. */
const RESERVED = new Set(["lng", "lngs", "fallbackLng", "ns", "defaultValue", "context", "postProcess", "interpolation", "replace", "keyPrefix", "ordinal", "returnObjects", "joinArrays"])

function needsSpace(left: string | undefined, right: string | undefined): boolean {
  if (!left || !right) return false
  return (HAN.test(left) && LATIN.test(right)) || (LATIN.test(left) && HAN.test(right))
}

/** `text` with a space wherever one of `values` meets Han text on one side and Latin text on the other. */
export function spaceInterpolations(text: string, values: readonly string[]): string {
  let result = text
  for (const value of new Set(values)) {
    if (!value.trim()) continue
    let from = 0
    for (let at = result.indexOf(value, from); at !== -1; at = result.indexOf(value, from)) {
      const end = at + value.length
      const after = needsSpace(value.at(-1), result[end]) ? " " : ""
      const before = needsSpace(result[at - 1], value[0]) ? " " : ""
      result = result.slice(0, at) + before + value + after + result.slice(end)
      from = end + before.length + after.length
    }
  }
  return result
}

/** The string and number values a `t()` call interpolates. */
function interpolated(options: TOptions): string[] {
  const values: string[] = []
  const add = (source: object) => {
    for (const [name, value] of Object.entries(source)) {
      if (RESERVED.has(name)) continue
      if (typeof value === "string" || typeof value === "number") values.push(String(value))
    }
  }
  add(options)
  if (options.replace && typeof options.replace === "object") add(options.replace)
  return values
}

/** Registered as the instance's default post-processor (`postProcess: "cjkSpacing"`); active for Chinese only. */
export const cjkSpacing: PostProcessorModule = {
  type: "postProcessor",
  name: "cjkSpacing",
  process(value, _key, options, translator: { language?: string }) {
    const lng = (typeof options.lng === "string" ? options.lng : translator.language) ?? ""
    if (typeof value !== "string" || !lng.toLowerCase().startsWith("zh")) return value
    return spaceInterpolations(value, interpolated(options))
  },
}
