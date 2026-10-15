import { describe, expect, it } from "vitest"
import { safeImageUrl, toImageRef } from "./image-ref"

describe("toImageRef", () => {
  it("keeps file ids, http(s), blob, data:image and same-site paths", () => {
    expect(toImageRef("abcdefgh12")?.src).toContain("/files/abcdefgh12")
    expect(toImageRef("https://example.com/a.png")?.src).toBe("https://example.com/a.png")
    expect(toImageRef("blob:http://127.0.0.1/x")?.src).toBe("blob:http://127.0.0.1/x")
    expect(toImageRef("data:image/png;base64,AAAA")?.src).toBe("data:image/png;base64,AAAA")
    expect(toImageRef({ url: "/api/v2/files/x", caption: "c" })).toEqual({ src: "/api/v2/files/x", caption: "c" })
  })

  it("drops script and other URLs an output could carry (they also become links)", () => {
    expect(toImageRef("javascript:alert(1)")).toBeNull()
    expect(toImageRef({ url: "javascript:alert(1)" })).toBeNull()
    expect(toImageRef({ url: "java\tscript:alert(1)" })).toBeNull()
    expect(toImageRef({ url: "data:text/html,<script>alert(1)</script>" })).toBeNull()
    expect(safeImageUrl("vbscript:x")).toBeNull()
    expect(safeImageUrl("file:///etc/passwd")).toBeNull()
  })
})
