import { render } from "@testing-library/react"
import { useRef } from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { findRanges, HIGHLIGHT_NAME, useSearchHighlight } from "./highlight"

const texts = (ranges: Range[]) => ranges.map((range) => range.toString())

function dom(html: string) {
  const root = document.createElement("div")
  root.innerHTML = html
  return root
}

describe("findRanges", () => {
  it("finds every case-insensitive occurrence of every term", () => {
    const root = dom("<p>Live Preview</p><span>preview every N steps; PREVIEW mode</span>")
    expect(texts(findRanges(root, ["preview"]))).toEqual(["Preview", "preview", "PREVIEW"])
    expect(texts(findRanges(root, ["live", "steps"]))).toEqual(["Live", "steps"])
  })

  it("does not match overlapping occurrences twice", () => {
    expect(texts(findRanges(dom("<p>aaaa</p>"), ["aa"]))).toEqual(["aa", "aa"])
  })

  it("finds nothing without terms", () => {
    const root = dom("<p>anything</p>")
    expect(findRanges(root, [])).toEqual([])
    expect(findRanges(root, [""])).toEqual([])
  })

  it("skips form controls, opted-out and editable text, and blank nodes", () => {
    const root = dom(
      [
        "<label>Seed</label>",
        "<textarea>seed in a textarea</textarea>",
        "<select><option>seed option</option></select>",
        '<div data-no-highlight><span>seed kept plain</span></div>',
        '<div contenteditable="true">seed being edited</div>',
        "<script>var seed = 1</script>",
        "<p>   </p>",
      ].join(""),
    )
    expect(texts(findRanges(root, ["seed"]))).toEqual(["Seed"])
  })

  it("skips text whose lower case has another length (ranges would be off)", () => {
    // "İ" lower-cases to two code units: offsets in the lower-cased copy would not match the text.
    const root = dom("<p>İstanbul cache</p><p>cache</p>")
    expect(texts(findRanges(root, ["cache"]))).toEqual(["cache"])
  })
})

class FakeHighlight {
  readonly ranges: Range[]
  constructor(...ranges: Range[]) {
    this.ranges = ranges
  }
}

function Highlighted({ terms, html }: { terms: string[]; html: string }) {
  const ref = useRef<HTMLDivElement>(null)
  useSearchHighlight(ref, terms)
  return <div ref={ref} dangerouslySetInnerHTML={{ __html: html }} />
}

describe("useSearchHighlight", () => {
  const registry = new Map<string, FakeHighlight>()
  const highlights = {
    set: (name: string, value: unknown) => registry.set(name, value as FakeHighlight),
    delete: (name: string) => registry.delete(name),
  }

  beforeEach(() => {
    registry.clear()
    vi.stubGlobal("CSS", { highlights })
    vi.stubGlobal("Highlight", FakeHighlight)
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  const current = () => texts(registry.get(HIGHLIGHT_NAME)?.ranges ?? [])

  it("registers the matches, follows DOM changes and clears on unmount", async () => {
    const view = render(<Highlighted terms={["seed"]} html="<p>Seed</p>" />)
    expect(current()).toEqual(["Seed"])

    // A virtualized row appears: the highlight is recomputed on the next frame.
    view.container.querySelector("div")!.insertAdjacentHTML("beforeend", "<p>Variation seed</p>")
    await vi.waitFor(() => expect(current()).toEqual(["Seed", "seed"]))

    view.unmount()
    expect(registry.has(HIGHLIGHT_NAME)).toBe(false)
  })

  it("removes the highlight when the search is cleared", () => {
    const view = render(<Highlighted terms={["seed"]} html="<p>Seed</p>" />)
    expect(registry.has(HIGHLIGHT_NAME)).toBe(true)
    view.rerender(<Highlighted terms={[]} html="<p>Seed</p>" />)
    expect(registry.has(HIGHLIGHT_NAME)).toBe(false)
  })

  it("does nothing in a browser without the Custom Highlight API", () => {
    vi.stubGlobal("CSS", {})
    const view = render(<Highlighted terms={["seed"]} html="<p>Seed</p>" />)
    expect(registry.size).toBe(0)
    view.unmount()
  })
})
