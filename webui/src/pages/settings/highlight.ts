/**
 * Search highlighting with the CSS Custom Highlight API: ranges over the
 * rendered text, so the schema renderer's markup stays untouched (and React
 * never sees extra nodes). Browsers without `CSS.highlights` just do not
 * highlight. Text inside form controls and `[data-no-highlight]` is skipped.
 */
import { type RefObject, useEffect } from "react"

export const HIGHLIGHT_NAME = "remi-settings-search"

type HighlightRegistry = { set: (name: string, highlight: unknown) => void; delete: (name: string) => void }
type HighlightCtor = new (...ranges: Range[]) => unknown

function registry(): { highlights: HighlightRegistry; Highlight: HighlightCtor } | null {
  const css = (globalThis as { CSS?: { highlights?: HighlightRegistry } }).CSS
  const Highlight = (globalThis as { Highlight?: HighlightCtor }).Highlight
  return css?.highlights && Highlight ? { highlights: css.highlights, Highlight } : null
}

/** Ranges of every (case-insensitive) occurrence of `terms` in the text under `root`. */
export function findRanges(root: Node, terms: readonly string[]): Range[] {
  const needles = terms.map((term) => term.toLowerCase()).filter(Boolean)
  if (needles.length === 0) return []
  const ranges: Range[] = []
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      const parent = node.parentElement
      if (!parent || parent.closest("input, textarea, select, script, style, [data-no-highlight], [contenteditable=true]")) return NodeFilter.FILTER_REJECT
      return node.nodeValue && node.nodeValue.trim() ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT
    },
  })
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    const text = node.nodeValue ?? ""
    const lower = text.toLowerCase()
    // Lower-casing can change the length of a few characters; match only where it did not.
    if (lower.length !== text.length) continue
    for (const needle of needles) {
      let from = 0
      for (let at = lower.indexOf(needle, from); at !== -1; at = lower.indexOf(needle, from)) {
        const range = document.createRange()
        range.setStart(node, at)
        range.setEnd(node, at + needle.length)
        ranges.push(range)
        from = at + needle.length
      }
    }
  }
  return ranges
}

/** Highlights `terms` under `container`, following DOM changes (virtualized rows, translations). */
export function useSearchHighlight(container: RefObject<HTMLElement | null>, terms: readonly string[]) {
  const signature = terms.join("\u0000")
  useEffect(() => {
    const api = registry()
    const root = container.current
    if (!api || !root) return
    const list = signature ? signature.split("\u0000") : []
    if (list.length === 0) {
      api.highlights.delete(HIGHLIGHT_NAME)
      return
    }
    let frame = 0
    const run = () => {
      frame = 0
      api.highlights.set(HIGHLIGHT_NAME, new api.Highlight(...findRanges(root, list)))
    }
    const schedule = () => {
      if (!frame) frame = requestAnimationFrame(run)
    }
    run()
    const observer = new MutationObserver(schedule)
    observer.observe(root, { subtree: true, childList: true, characterData: true })
    return () => {
      observer.disconnect()
      if (frame) cancelAnimationFrame(frame)
      api.highlights.delete(HIGHLIGHT_NAME)
    }
  }, [container, signature])
}
