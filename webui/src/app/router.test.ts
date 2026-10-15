import { describe, expect, it } from "vitest"
import { buildNav } from "./routes"
import { DEFAULT_ROUTE, parseHash, routeHref, routeKey } from "./router"

describe("router", () => {
  it("parses every route shape", () => {
    expect(parseHash("")).toEqual(DEFAULT_ROUTE)
    expect(parseHash("#/")).toEqual(DEFAULT_ROUTE)
    expect(parseHash("#/img2img")).toMatchObject({ name: "generate", tab: "img2img" })
    expect(parseHash("#/queue")).toMatchObject({ name: "queue" })
    expect(parseHash("#/page/agent%20scheduler")).toMatchObject({ name: "page", id: "agent scheduler" })
    expect(parseHash("#/nope/x")).toEqual({ name: "not-found", path: "nope/x" })
  })

  it("keeps the query", () => {
    const route = parseHash("#/settings?q=sampler")
    expect(route.query?.get("q")).toBe("sampler")
    expect(routeHref(route)).toBe("#/settings?q=sampler")
    expect(routeKey(route)).toBe("#/settings")
  })

  it("treats a malformed escape as not found instead of throwing", () => {
    expect(parseHash("#/page/%E0%A4")).toEqual({ name: "not-found", path: "page/%E0%A4" })
    expect(() => parseHash("#/%")).not.toThrow()
  })

  it("round-trips hrefs", () => {
    for (const hash of ["#/txt2img", "#/png-info", "#/page/a%2Fb"]) expect(routeHref(parseHash(hash))).toBe(hash)
  })
})

describe("buildNav", () => {
  it("orders and filters built-in tabs by the server list, then pages, then system", () => {
    const nav = buildNav(["img2img", "txt2img", "gallery"], [{ id: "tools", title: "Tools", icon: "wrench" }])
    expect(nav.map((item) => item.id)).toEqual(["img2img", "txt2img", "gallery", "queue", "page:tools", "extensions", "settings"])
  })

  it("shows everything when the server sends no tab list", () => {
    expect(buildNav(undefined, undefined)).toHaveLength(9)
  })
})
