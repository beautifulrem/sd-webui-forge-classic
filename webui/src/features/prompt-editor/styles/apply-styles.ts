/**
 * Style application, mirroring `modules/styles.py` (`apply_styles_to_prompt`).
 * The app asks the server (`POST /styles/apply`); mock mode answers with this.
 */
import type { PromptStyle } from "@/api/types"

/**
 * Applies style texts in order: a text containing `{prompt}` wraps the prompt
 * there; any other text is appended with ", ". Empty parts are skipped.
 */
export function applyStyleTexts(prompt: string, styleTexts: readonly (string | null | undefined)[]): string {
  let result = prompt
  for (const style of styleTexts) {
    const text = style ?? ""
    if (text.includes("{prompt}")) {
      result = text.replaceAll("{prompt}", result)
    } else {
      result = [result.trim(), text.trim()].filter(Boolean).join(", ")
    }
  }
  return result
}

/**
 * The prompts with `selected` styles written into them ("materialize"),
 * as the old 📋 button did; unknown style names contribute nothing.
 */
export function materializeStyles(
  prompts: { prompt: string; negative: string },
  selected: readonly string[],
  styles: readonly PromptStyle[],
): { prompt: string; negative: string } {
  const byName = new Map(styles.map((style) => [style.name, style]))
  const chosen = selected.map((name) => byName.get(name))
  return {
    prompt: applyStyleTexts(prompts.prompt, chosen.map((style) => style?.prompt)),
    negative: applyStyleTexts(prompts.negative, chosen.map((style) => style?.negative_prompt)),
  }
}
