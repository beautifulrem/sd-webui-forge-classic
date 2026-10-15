import { describe, expect, it } from "vitest"
import { adoptFromUrl, filtersToUrl, INITIAL_FILTERS, withFilters } from "./settings-filters"

describe("settings search and filter in the route", () => {
  it("writes only what is set", () => {
    expect(filtersToUrl("  ", "all")).toEqual({ q: null, filter: null })
    expect(filtersToUrl("jpeg", "unsaved")).toEqual({ q: "jpeg", filter: "unsaved" })
    expect(withFilters(new URLSearchParams("view=server&q=old"), { q: null, filter: "changed" }).toString()).toBe("view=server&filter=changed")
  })

  it("adopts values written elsewhere and keeps the remembered ones when the route has none", () => {
    const remembered = { query: "sampler", mode: "changed" as const, written: { q: "sampler", filter: "changed" } }
    // Back on the page through a plain link: nothing to adopt, the page writes its values again.
    expect(adoptFromUrl(remembered, { q: null, filter: null })).toEqual({ written: { q: null, filter: null } })
    // A "Review" link.
    expect(adoptFromUrl(remembered, { q: null, filter: "unsaved" })).toEqual({ written: { q: null, filter: "unsaved" }, mode: "unsaved" })
    // The page's own write coming back.
    expect(adoptFromUrl(remembered, { q: "sampler", filter: "changed" })).toBeNull()
    expect(adoptFromUrl(INITIAL_FILTERS, { q: "vae", filter: "bogus" })).toEqual({ written: { q: "vae", filter: "bogus" }, query: "vae", mode: "all" })
  })
})
