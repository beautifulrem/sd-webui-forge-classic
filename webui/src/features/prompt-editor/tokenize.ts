/**
 * A small prompt scanner for highlighting and bracket diagnostics. It follows
 * the A1111 prompt grammar loosely: `(text:1.2)` attention, `[a:b:0.5]` /
 * `[a|b]` schedules, `<lora:name:0.8>` extra networks, `BREAK` / `AND`,
 * `__wildcard__`, `{a|b}` alternatives, `\(` escapes and `#` comment lines.
 * Pure and linear; the editor re-runs it on every change (prompts are short).
 */

export type PromptTokenKind =
  /** `(` `)` `[` `]` `{` `}`; `depth` = nesting level. */
  | "bracket"
  /** An unmatched or mismatched bracket. */
  | "bracket-error"
  /** `:1.2` closing an attention group, or a schedule's step number. */
  | "weight"
  /** `:` and `|` inside `[…]` / `{…}`. */
  | "schedule"
  /** `<name:args>` extra network. */
  | "extra"
  /** A numeric argument inside an extra network. */
  | "extra-weight"
  | "keyword"
  | "wildcard"
  | "comment"
  | "escape"

export type PromptToken = { from: number; to: number; kind: PromptTokenKind; depth?: number }

const OPEN: Record<string, string> = { "(": ")", "[": "]", "{": "}" }
const CLOSE: Record<string, string> = { ")": "(", "]": "[", "}": "{" }

const EXTRA = /<(\w+):([^<>\n]*)>/y
const NUMBER = /^\s*-?(?:\d+\.?\d*|\.\d+)\s*$/
const ATTENTION_WEIGHT = /:\s*-?(?:\d+\.?\d*|\.\d+)\s*(?=\))/y
const SCHEDULE_STEP = /\s*-?(?:\d+\.?\d*|\.\d+)\s*(?=\])/y
const WILDCARD = /__[\w\-/.*]+__/y
const KEYWORD = /(?:BREAK|AND)(?![\w])/y

const isWordChar = (c: string | undefined) => c !== undefined && /\w/.test(c)

function matchAt(re: RegExp, text: string, at: number): RegExpExecArray | null {
  re.lastIndex = at
  return re.exec(text)
}

export function tokenizePrompt(text: string): PromptToken[] {
  const tokens: PromptToken[] = []
  const stack: { char: string; token: PromptToken }[] = []
  let i = 0

  while (i < text.length) {
    const c = text[i]!
    const lineStart = i === 0 || text[i - 1] === "\n"

    if (lineStart && c === "#") {
      const end = text.indexOf("\n", i)
      const to = end === -1 ? text.length : end
      tokens.push({ from: i, to, kind: "comment" })
      i = to
      continue
    }
    if (c === "\\" && i + 1 < text.length) {
      tokens.push({ from: i, to: i + 2, kind: "escape" })
      i += 2
      continue
    }
    if (c === "<") {
      const m = matchAt(EXTRA, text, i)
      if (m) {
        const to = i + m[0].length
        tokens.push({ from: i, to, kind: "extra" })
        // Numeric arguments after the network name: <lora:name:0.8:0.5>.
        let at = i + 1 + m[1]!.length + 1
        m[2]!.split(":").forEach((part, index) => {
          if (index > 0 && NUMBER.test(part)) tokens.push({ from: at, to: at + part.length, kind: "extra-weight" })
          at += part.length + 1
        })
        i = to
        continue
      }
    }
    if (c in OPEN) {
      const token: PromptToken = { from: i, to: i + 1, kind: "bracket", depth: stack.length }
      tokens.push(token)
      stack.push({ char: c, token })
      i++
      continue
    }
    if (c in CLOSE) {
      const top = stack.at(-1)
      if (top && top.char === CLOSE[c]) {
        stack.pop()
        tokens.push({ from: i, to: i + 1, kind: "bracket", depth: stack.length })
      } else {
        tokens.push({ from: i, to: i + 1, kind: "bracket-error" })
      }
      i++
      continue
    }
    const inside = stack.at(-1)?.char
    if (c === ":" && inside === "(") {
      const m = matchAt(ATTENTION_WEIGHT, text, i)
      if (m) {
        tokens.push({ from: i, to: i + m[0].length, kind: "weight" })
        i += m[0].length
        continue
      }
    }
    if (c === ":" && inside === "[") {
      tokens.push({ from: i, to: i + 1, kind: "schedule" })
      const m = matchAt(SCHEDULE_STEP, text, i + 1)
      if (m) tokens.push({ from: i + 1, to: i + 1 + m[0].length, kind: "weight" })
      i += 1 + (m?.[0].length ?? 0)
      continue
    }
    if (c === "|" && (inside === "[" || inside === "{")) {
      tokens.push({ from: i, to: i + 1, kind: "schedule" })
      i++
      continue
    }
    if ((c === "B" || c === "A") && !isWordChar(text[i - 1])) {
      const m = matchAt(KEYWORD, text, i)
      if (m) {
        tokens.push({ from: i, to: i + m[0].length, kind: "keyword" })
        i += m[0].length
        continue
      }
    }
    if (c === "_" && text[i + 1] === "_" && !isWordChar(text[i - 1])) {
      const m = matchAt(WILDCARD, text, i)
      if (m) {
        tokens.push({ from: i, to: i + m[0].length, kind: "wildcard" })
        i += m[0].length
        continue
      }
    }
    i++
  }

  for (const open of stack) open.token.kind = "bracket-error"
  return tokens
}

export type BracketProblem = {
  from: number
  to: number
  char: string
  /** `unclosed`: an opening bracket never closed; `unmatched`: a closing one without its opening. */
  kind: "unclosed" | "unmatched"
}

/** Unbalanced brackets (replaces the prompt-bracket-checker extension). */
export function bracketProblems(text: string, tokens: readonly PromptToken[] = tokenizePrompt(text)): BracketProblem[] {
  return tokens
    .filter((token) => token.kind === "bracket-error")
    .map((token) => {
      const char = text.slice(token.from, token.to)
      return { from: token.from, to: token.to, char, kind: char in OPEN ? "unclosed" : "unmatched" }
    })
}
