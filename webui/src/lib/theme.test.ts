import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

/** theme.ts reads the URL, the server's --theme and the stored pick when it loads. */
async function load({ stored, server, search = "" }: { stored?: object; server?: string | null; search?: string }) {
  vi.resetModules()
  localStorage.clear()
  if (stored) localStorage.setItem("remi-appearance", JSON.stringify(stored))
  window.__REMI_THEME__ = server
  window.history.replaceState(null, "", `/${search}`)
  return import("./theme")
}

describe("theme sources", () => {
  beforeEach(() => localStorage.clear())
  afterEach(() => {
    delete window.__REMI_THEME__
    window.history.replaceState(null, "", "/")
  })

  it("defaults to dark, or the server's --theme while the user has not picked one", async () => {
    expect((await load({})).useAppearance.getState().theme).toBe("dark")
    const { useAppearance } = await load({ server: "light" })
    expect(useAppearance.getState()).toMatchObject({ theme: "light", themeChosen: false })
    // changing the density stores the state, but the theme still follows the server
    useAppearance.getState().setDensity("compact")
    const again = await load({ server: "dark", stored: JSON.parse(localStorage.getItem("remi-appearance")!) })
    expect(again.useAppearance.getState()).toMatchObject({ theme: "dark", density: "compact", themeChosen: false })
  })

  it("keeps the user's pick over --theme (also a pick stored by version 1)", async () => {
    const picked = await load({ server: "light", stored: { state: { theme: "dark", themeChosen: true, density: "comfortable", accent: "iris" }, version: 2 } })
    expect(picked.useAppearance.getState().theme).toBe("dark")
    const legacy = await load({ server: "dark", stored: { state: { theme: "light", density: "comfortable", accent: "iris" }, version: 1 } })
    expect(legacy.useAppearance.getState()).toMatchObject({ theme: "light", themeChosen: true })
  })

  it("?__theme= picks the theme and remembers it; unknown values are ignored", async () => {
    const { useAppearance, urlTheme } = await load({ server: "dark", search: "?__theme=light" })
    expect(useAppearance.getState()).toMatchObject({ theme: "light", themeChosen: true })
    expect(JSON.parse(localStorage.getItem("remi-appearance")!).state.theme).toBe("light")
    expect(urlTheme("?__theme=system")).toBe("system")
    expect(urlTheme("?__theme=purple")).toBeNull()
    expect((await load({ search: "?__theme=purple" })).useAppearance.getState().theme).toBe("dark")
  })
})
