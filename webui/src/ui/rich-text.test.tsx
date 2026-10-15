import { render } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { Markdown, SafeHtml } from "./RichText"
import { safeUrl } from "./rich-text"

describe("safeUrl", () => {
  it("allows http(s), mailto and relative URLs", () => {
    expect(safeUrl("https://example.com")).toBe("https://example.com")
    expect(safeUrl("mailto:a@b.c")).toBe("mailto:a@b.c")
    expect(safeUrl("/docs")).toBe("/docs")
    expect(safeUrl("./a.png")).toBe("./a.png")
    expect(safeUrl("guide.html")).toBe("guide.html")
  })
  it("rejects script and data URLs", () => {
    expect(safeUrl("javascript:alert(1)")).toBeNull()
    expect(safeUrl(" JavaScript:alert(1)")).toBeNull()
    expect(safeUrl("data:text/html,<script>")).toBeNull()
    expect(safeUrl("vbscript:x")).toBeNull()
  })
  it("reads the scheme as the browser does (tabs, newlines, control characters)", () => {
    expect(safeUrl("java\tscript:alert(1)")).toBeNull()
    expect(safeUrl("java\nscript:alert(1)")).toBeNull()
    expect(safeUrl("\u0001javascript:alert(1)")).toBeNull()
    expect(safeUrl("JAVASCRIPT:alert(1)")).toBeNull()
  })
})

describe("Markdown", () => {
  it("renders the common subset", () => {
    const { container } = render(
      <Markdown text={"## Title\nSome **bold**, *em* and `code`.\n\n- one\n- two\n\n1. first\n\n> quote\n\n```\nx = 1\n```\n\n---\n[link](https://example.com)"} />,
    )
    expect(container.querySelector("h4")?.textContent).toBe("Title")
    expect(container.querySelector("strong")?.textContent).toBe("bold")
    expect(container.querySelector("em")?.textContent).toBe("em")
    expect(container.querySelector("code")?.textContent).toBe("code")
    expect(container.querySelectorAll("ul > li")).toHaveLength(2)
    expect(container.querySelectorAll("ol > li")).toHaveLength(1)
    expect(container.querySelector("blockquote")?.textContent).toBe("quote")
    expect(container.querySelector("pre")?.textContent).toBe("x = 1")
    expect(container.querySelector("hr")).not.toBeNull()
    const link = container.querySelector("a")!
    expect(link.getAttribute("href")).toBe("https://example.com")
    expect(link.getAttribute("rel")).toBe("noopener noreferrer")
  })

  it("never renders raw HTML or unsafe links", () => {
    const { container } = render(<Markdown text={'<img src=x onerror="alert(1)"> [x](javascript:alert(1))'} />)
    expect(container.querySelector("img")).toBeNull()
    expect(container.querySelector("a")).toBeNull()
    expect(container.textContent).toContain("<img src=x")
  })
})

describe("SafeHtml", () => {
  it("keeps allowed markup and drops scripts, handlers and unsafe URLs", () => {
    const { container } = render(
      <SafeHtml
        html={
          '<p class="x" onclick="evil()">Hi <b>there</b><script>alert(1)</script><a href="javascript:alert(1)">bad</a><a href="https://ok.test">ok</a><img src="x" onerror="alert(1)"><iframe src="https://x"></iframe><style>*{}</style><custom-el>kept text</custom-el></p>'
        }
      />,
    )
    const p = container.querySelector("p")!
    expect(p.getAttribute("onclick")).toBeNull()
    expect(p.getAttribute("class")).toBeNull()
    expect(container.querySelector("b")?.textContent).toBe("there")
    expect(container.querySelector("script")).toBeNull()
    expect(container.querySelector("iframe")).toBeNull()
    expect(container.querySelector("style")).toBeNull()
    const links = container.querySelectorAll("a")
    expect(links[0]!.getAttribute("href")).toBeNull()
    expect(links[1]!.getAttribute("href")).toBe("https://ok.test")
    const img = container.querySelector("img")!
    expect(img.getAttribute("onerror")).toBeNull()
    expect(img.getAttribute("src")).toBe("x")
    expect(container.textContent).toContain("kept text")
  })
})

describe("SafeHtml boolean attributes", () => {
  it("keeps <details open>", () => {
    const { container } = render(<SafeHtml html="<details open><summary>More</summary>Body</details>" />)
    expect(container.querySelector("details")).toHaveAttribute("open")
  })
})

describe("SafeHtml URL attributes", () => {
  it("drops script URLs hidden with entities or whitespace", () => {
    const { container } = render(<SafeHtml html={'<a href="java&#9;script:alert(1)">a</a><a href="&#x6A;avascript:alert(1)">b</a><img src="java&#10;script:alert(1)">'} />)
    for (const link of container.querySelectorAll("a")) expect(link.getAttribute("href")).toBeNull()
    expect(container.querySelector("img")).toBeNull()
  })
})
