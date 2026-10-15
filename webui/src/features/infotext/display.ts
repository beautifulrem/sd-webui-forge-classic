/**
 * Splits an infotext for display: prompt, negative prompt and the
 * "Key: value, …" parameter line (`infotext_utils.parse_generation_parameters`
 * layout). Display only; the server parses for pasting.
 */

export type InfotextParts = { prompt: string; negative: string; params: [key: string, value: string][] }

const PARAM = /\s*(\w[\w \-/]+):\s*("(?:\\.|[^\\"])+"|[^,]*)(?:,|$)/g

export function parseParamLine(line: string): [string, string][] {
  const out: [string, string][] = []
  for (const match of line.matchAll(PARAM)) {
    let value = match[2]!.trim()
    if (value.startsWith('"') && value.endsWith('"') && value.length >= 2) {
      try {
        value = JSON.parse(value) as string
      } catch {
        value = value.slice(1, -1)
      }
    }
    out.push([match[1]!.trim(), value])
  }
  return out
}

export function splitInfotext(text: string): InfotextParts {
  const lines = text.trim().split("\n")
  let params: [string, string][] = []
  const last = lines.at(-1) ?? ""
  const parsed = parseParamLine(last)
  if (parsed.length >= 3) {
    params = parsed
    lines.pop()
  }
  const prompt: string[] = []
  const negative: string[] = []
  let inNegative = false
  for (const line of lines) {
    if (!inNegative && line.startsWith("Negative prompt:")) {
      inNegative = true
      negative.push(line.slice("Negative prompt:".length).trimStart())
    } else if (inNegative) {
      negative.push(line)
    } else {
      prompt.push(line)
    }
  }
  return { prompt: prompt.join("\n").trim(), negative: negative.join("\n").trim(), params }
}
