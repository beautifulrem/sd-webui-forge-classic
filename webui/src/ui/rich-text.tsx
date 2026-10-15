/**
 * Safe rich text for schema nodes, built as React elements (never innerHTML).
 *
 * Markdown (`markdown` nodes): a small renderer that builds React
 * elements directly (never innerHTML), so raw HTML in the source is shown as
 * text. Supports the CommonMark subset extension help texts use: ATX
 * headings, paragraphs, hard breaks, emphasis, strong, strikethrough, inline
 * code, fenced code, links / autolinks, images, ordered / unordered lists
 * (one nesting level via indentation), block quotes and rules.
 */
import { createElement, Fragment, type ReactNode } from "react"

const SAFE_PROTOCOLS = new Set(["http:", "https:", "mailto:"])

/**
 * `url` when it is http(s), mailto, or relative; null otherwise. The scheme
 * is read the way the browser will read it (the URL parser drops tabs and
 * newlines and decodes nothing else), so `java\tscript:` is caught too.
 */
export function safeUrl(url: string): string | null {
  const trimmed = url.trim()
  try {
    return SAFE_PROTOCOLS.has(new URL(trimmed, "http://relative.invalid/").protocol) ? trimmed : null
  } catch {
    return null
  }
}

// ---------------------------------------------------------------- inline

/** Capture group `i` of a match ("" when it did not participate). */
const group = (m: RegExpExecArray, i: number) => m[i] ?? ""

type InlineRule = { re: RegExp; render: (m: RegExpExecArray, key: string) => ReactNode }

const INLINE: InlineRule[] = [
  { re: /`([^`]+)`/, render: (m, key) => <code key={key} className="rounded bg-muted px-1 py-0.5 font-mono text-[0.85em]">{group(m, 1)}</code> },
  {
    re: /!\[([^\]]*)\]\(([^)\s]+)(?:\s+"[^"]*")?\)/,
    render: (m, key) => {
      const src = safeUrl(group(m, 2))
      return src ? <img key={key} src={src} alt={group(m, 1)} loading="lazy" className="inline-block max-w-full rounded" /> : group(m, 0)
    },
  },
  {
    re: /\[([^\]]+)\]\(([^)\s]+)(?:\s+"[^"]*")?\)/,
    render: (m, key) => {
      const href = safeUrl(group(m, 2))
      if (!href) return group(m, 1)
      return (
        <a key={key} href={href} target="_blank" rel="noopener noreferrer" className="text-primary underline-offset-2 hover:underline">
          {inline(group(m, 1), key)}
        </a>
      )
    },
  },
  {
    re: /<(https?:\/\/[^>\s]+)>/,
    render: (m, key) => (
      <a key={key} href={group(m, 1)} target="_blank" rel="noopener noreferrer" className="text-primary underline-offset-2 hover:underline">
        {group(m, 1)}
      </a>
    ),
  },
  { re: /\*\*([^*]+)\*\*|__([^_]+)__/, render: (m, key) => <strong key={key}>{inline(m[1] ?? group(m, 2), key)}</strong> },
  { re: /~~([^~]+)~~/, render: (m, key) => <del key={key}>{inline(group(m, 1), key)}</del> },
  { re: /\*([^*\s][^*]*)\*|(?<![\w])_([^_\s][^_]*)_(?![\w])/, render: (m, key) => <em key={key}>{inline(m[1] ?? group(m, 2), key)}</em> },
  { re: / {2,}\n|\\\n/, render: (_m, key) => <br key={key} /> },
]

/** Renders inline Markdown; the earliest match wins, ties go to the rule listed first. */
export function inline(text: string, keyPrefix = "i"): ReactNode[] {
  const out: ReactNode[] = []
  let rest = text
  let n = 0
  while (rest) {
    let best: { m: RegExpExecArray; rule: InlineRule } | null = null
    for (const rule of INLINE) {
      const m = rule.re.exec(rest)
      if (m && (!best || m.index < best.m.index)) best = { m, rule }
    }
    if (!best) {
      out.push(rest)
      break
    }
    if (best.m.index > 0) out.push(rest.slice(0, best.m.index))
    out.push(best.rule.render(best.m, `${keyPrefix}.${n++}`))
    rest = rest.slice(best.m.index + best.m[0].length)
  }
  return out
}

// ---------------------------------------------------------------- blocks

type ListItem = { text: string; children: ListItem[] }

const HEADING = /^(#{1,6})\s+(.*?)\s*#*\s*$/
const FENCE = /^(```|~~~)\s*(\S*)/
const RULE = /^(?:-{3,}|\*{3,}|_{3,})\s*$/
const BULLET = /^(\s*)[-*+]\s+(.*)$/
const ORDERED = /^(\s*)\d+[.)]\s+(.*)$/
const QUOTE = /^>\s?(.*)$/

const HEADING_CLASS = [
  "text-lg font-semibold tracking-tight",
  "text-base font-semibold tracking-tight",
  "text-sm font-semibold",
  "text-sm font-medium",
  "text-sm font-medium",
  "text-sm font-medium",
]

function renderList(items: ListItem[], ordered: boolean, key: string): ReactNode {
  const Tag = ordered ? "ol" : "ul"
  return (
    <Tag key={key} className={ordered ? "list-decimal space-y-0.5 pl-5" : "list-disc space-y-0.5 pl-5"}>
      {items.map((item, i) => (
        <li key={i}>
          {inline(item.text, `${key}.${i}`)}
          {item.children.length > 0 ? renderList(item.children, ordered, `${key}.${i}.c`) : null}
        </li>
      ))}
    </Tag>
  )
}

export function renderMarkdown(source: string): ReactNode {
  const lines = source.replace(/\r\n?/g, "\n").split("\n")
  const blocks: ReactNode[] = []
  let i = 0
  let paragraph: string[] = []

  const flush = () => {
    if (paragraph.length === 0) return
    const key = `p${blocks.length}`
    blocks.push(<p key={key}>{inline(paragraph.join("\n"), key)}</p>)
    paragraph = []
  }

  while (i < lines.length) {
    const line = lines[i] ?? ""
    const key = `b${blocks.length}`

    if (line.trim() === "") {
      flush()
      i++
      continue
    }

    const fence = FENCE.exec(line)
    if (fence) {
      flush()
      const body: string[] = []
      i++
      const marker = fence[1] ?? "```"
      while (i < lines.length && !(lines[i] ?? "").startsWith(marker)) body.push(lines[i++] ?? "")
      i++
      blocks.push(
        <pre key={key} className="overflow-x-auto rounded-md bg-muted p-2.5 font-mono text-xs">
          <code>{body.join("\n")}</code>
        </pre>,
      )
      continue
    }

    const heading = HEADING.exec(line)
    if (heading) {
      flush()
      const level = (heading[1] ?? "#").length
      const Tag = `h${Math.min(level + 2, 6)}` as "h3"
      blocks.push(
        <Tag key={key} className={HEADING_CLASS[level - 1]}>
          {inline(heading[2] ?? "", key)}
        </Tag>,
      )
      i++
      continue
    }

    if (RULE.test(line)) {
      flush()
      blocks.push(<hr key={key} className="border-border" />)
      i++
      continue
    }

    if (QUOTE.test(line)) {
      flush()
      const body: string[] = []
      for (let m = QUOTE.exec(lines[i] ?? ""); i < lines.length && m; m = QUOTE.exec(lines[++i] ?? "")) body.push(m[1] ?? "")
      blocks.push(
        <blockquote key={key} className="border-l-2 border-border pl-3 text-muted-foreground">
          {renderMarkdown(body.join("\n"))}
        </blockquote>,
      )
      continue
    }

    const listMatch = BULLET.exec(line) ?? ORDERED.exec(line)
    if (listMatch) {
      flush()
      const ordered = !BULLET.test(line)
      const pattern = ordered ? ORDERED : BULLET
      const items: ListItem[] = []
      while (i < lines.length) {
        const current = lines[i] ?? ""
        const m = pattern.exec(current)
        const previous = items.at(-1)
        if (m) {
          const item: ListItem = { text: m[2] ?? "", children: [] }
          if ((m[1] ?? "").length >= 2 && previous) previous.children.push(item)
          else items.push(item)
          i++
        } else if (current.trim() !== "" && /^\s{2,}/.test(current) && previous) {
          previous.text += `\n${current.trim()}`
          i++
        } else break
      }
      blocks.push(renderList(items, ordered, key))
      continue
    }

    paragraph.push(line)
    i++
  }
  flush()
  return <Fragment>{blocks}</Fragment>
}


// ---------------------------------------------------------------- html

/*
 * Sanitised HTML for `html` nodes and `html` outputs. The markup is parsed
 * with DOMParser (inert: scripts do not run, images do not load) and rebuilt
 * as React elements from an allow-list of tags and attributes, so no string
 * ever reaches innerHTML. Event handlers, `style`, forms and embeds are dropped;
 * URLs must be http(s), mailto, or relative.
 */
const TAGS = new Set([
  "a", "abbr", "b", "blockquote", "br", "caption", "code", "dd", "del", "details", "div", "dl", "dt", "em",
  "figcaption", "figure", "h1", "h2", "h3", "h4", "h5", "h6", "hr", "i", "img", "ins", "kbd", "li", "mark",
  "ol", "p", "pre", "q", "s", "small", "span", "strong", "sub", "summary", "sup", "table", "tbody", "td",
  "tfoot", "th", "thead", "tr", "u", "ul",
])

/** Tags whose content is dropped entirely, not just unwrapped. */
const DROP = new Set(["script", "style", "iframe", "object", "embed", "template", "noscript", "svg", "math", "form", "input", "button", "select", "textarea", "link", "meta", "base"])

const ATTRS: Record<string, string> = {
  title: "title",
  alt: "alt",
  colspan: "colSpan",
  rowspan: "rowSpan",
  width: "width",
  height: "height",
  open: "open",
  lang: "lang",
  dir: "dir",
}

const BOOLEAN_ATTRS = new Set(["open"])

function convert(node: globalThis.Node, key: string): ReactNode {
  if (node.nodeType === 3) return node.textContent
  if (node.nodeType !== 1) return null
  const el = node as Element
  const tag = el.tagName.toLowerCase()
  if (DROP.has(tag)) return null
  const children = Array.from(el.childNodes, (child, i) => convert(child, `${key}.${i}`))
  // Unknown but harmless tags are unwrapped, keeping their text.
  if (!TAGS.has(tag)) return createElement(Fragment, { key }, ...children)

  const props: Record<string, unknown> = { key }
  for (const attr of Array.from(el.attributes)) {
    const name = attr.name.toLowerCase()
    const prop = ATTRS[name]
    // Boolean attributes (`<details open>`) carry "" as their value.
    if (prop) props[prop] = BOOLEAN_ATTRS.has(name) ? true : attr.value
    else if (name === "href" && tag === "a") {
      const href = safeUrl(attr.value)
      if (href) Object.assign(props, { href, target: "_blank", rel: "noopener noreferrer" })
    } else if (name === "src" && tag === "img") {
      const src = safeUrl(attr.value)
      if (src) Object.assign(props, { src, loading: "lazy" })
    }
  }
  if (tag === "img" && !props.src) return null
  if (tag === "br" || tag === "hr" || tag === "img") return createElement(tag, props)
  return createElement(tag, props, ...children)
}

export function sanitizeHtml(html: string): ReactNode {
  const doc = new DOMParser().parseFromString(`<body>${html}</body>`, "text/html")
  return Array.from(doc.body.childNodes, (node, i) => convert(node, `h${i}`))
}

