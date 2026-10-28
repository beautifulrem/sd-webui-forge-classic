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

  it("moves version 2's default iris accent to blue, and keeps other accents", async () => {
    const old = await load({ server: "dark", stored: { state: { theme: "dark", themeChosen: false, density: "comfortable", accent: "iris" }, version: 2 } })
    expect(old.useAppearance.getState().accent).toBe("blue")
    const teal = await load({ server: "dark", stored: { state: { theme: "dark", themeChosen: false, density: "comfortable", accent: "teal" }, version: 2 } })
    expect(teal.useAppearance.getState().accent).toBe("teal")
    const picked = await load({ server: "dark", stored: { state: { theme: "dark", themeChosen: false, density: "comfortable", accent: "iris" }, version: 3 } })
    expect(picked.useAppearance.getState().accent).toBe("iris")
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

describe("theme in effect", () => {
  /** A `prefers-color-scheme: dark` query whose answer the test changes. */
  function systemScheme(dark: boolean) {
    const listeners = new Set<() => void>()
    const media = {
      matches: dark,
      addEventListener: (_type: string, listener: () => void) => listeners.add(listener),
      removeEventListener: (_type: string, listener: () => void) => listeners.delete(listener),
    }
    vi.spyOn(window, "matchMedia").mockImplementation((query) => {
      expect(query).toBe("(prefers-color-scheme: dark)")
      return media as unknown as MediaQueryList
    })
    return {
      listeners,
      set(value: boolean) {
        media.matches = value
        for (const listener of listeners) listener()
      },
    }
  }

  afterEach(() => {
    vi.restoreAllMocks()
    document.documentElement.className = ""
    delete document.documentElement.dataset.density
    delete document.documentElement.dataset.accent
  })

  it("resolves 'system' from the OS colour scheme and follows its changes", async () => {
    const { renderHook, act } = await import("@testing-library/react")
    const scheme = systemScheme(false)
    const { useAppearance, useTheme } = await load({})
    act(() => useAppearance.getState().setTheme("system"))
    const { result, unmount } = renderHook(() => useTheme())
    expect(result.current.theme).toBe("light")
    act(() => scheme.set(true))
    expect(result.current.theme).toBe("dark")
    // a picked theme wins over the OS
    act(() => result.current.setTheme("light"))
    expect(result.current.theme).toBe("light")
    expect(useAppearance.getState().themeChosen).toBe(true)
    unmount()
    expect(scheme.listeners.size).toBe(0)
  })

  it("mirrors theme, density and accent onto <html>", async () => {
    const { renderHook, act } = await import("@testing-library/react")
    systemScheme(true)
    const { useAppearance, useApplyAppearance } = await load({})
    const root = document.documentElement
    renderHook(() => useApplyAppearance())
    expect(root.classList.contains("dark")).toBe(true)
    expect(root.dataset).toMatchObject({ density: "comfortable", accent: "blue" })
    act(() => {
      useAppearance.getState().setTheme("light")
      useAppearance.getState().setDensity("compact")
      useAppearance.getState().setAccent("teal")
    })
    expect(root.classList.contains("dark")).toBe(false)
    expect(root.dataset).toMatchObject({ density: "compact", accent: "teal" })
    act(() => useAppearance.getState().setTheme("system"))
    expect(root.classList.contains("dark")).toBe(true)
  })

  it("ignores an unknown server theme and an unreadable stored state", async () => {
    expect((await load({ server: "sepia" })).serverTheme()).toBeNull()
    localStorage.setItem("remi-appearance", "{not json")
    vi.resetModules()
    const { useAppearance } = await import("./theme")
    expect(useAppearance.getState()).toMatchObject({ theme: "dark", themeChosen: false, density: "comfortable", accent: "blue" })
  })
})
