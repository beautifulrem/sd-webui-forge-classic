import { useEffect, useSyncExternalStore } from "react"
import { create } from "zustand"
import { persist } from "zustand/middleware"

export type Theme = "light" | "dark" | "system"
export type Density = "comfortable" | "compact"
export const ACCENTS = ["blue", "iris", "teal", "green", "amber", "coral", "rose"] as const
export type Accent = (typeof ACCENTS)[number]

/** Storage key; index.html reads it before first paint to avoid a theme flash. */
export const APPEARANCE_KEY = "remi-appearance"

declare global {
  interface Window {
    /** `--theme`, set by the server in the shell (remi/server/spa.py). */
    __REMI_THEME__?: string | null
  }
}

const isTheme = (value: unknown): value is Theme => value === "light" || value === "dark" || value === "system"

/** The server's `--theme`: the theme of a browser that has not picked one. */
export function serverTheme(): Theme | null {
  const value = typeof window === "undefined" ? null : window.__REMI_THEME__
  return isTheme(value) ? value : null
}

/** `?__theme=light|dark|system` in the page URL (A1111's theme link): picks that theme. */
export function urlTheme(search: string = typeof window === "undefined" ? "" : window.location.search): Theme | null {
  const value = new URLSearchParams(search).get("__theme")
  return isTheme(value) ? value : null
}

/** What the browser keeps (`partialize`). */
type StoredAppearance = Pick<AppearanceState, "theme" | "themeChosen" | "density" | "accent">

type AppearanceState = {
  theme: Theme
  /** The user picked `theme` (Appearance menu, command, URL); otherwise it follows `--theme`. */
  themeChosen: boolean
  density: Density
  accent: Accent
  setTheme: (theme: Theme) => void
  setDensity: (density: Density) => void
  setAccent: (accent: Accent) => void
}

export const useAppearance = create<AppearanceState>()(
  persist(
    (set) => ({
      theme: serverTheme() ?? "dark",
      themeChosen: false,
      density: "comfortable",
      accent: "blue",
      setTheme: (theme) => set({ theme, themeChosen: true }),
      setDensity: (density) => set({ density }),
      setAccent: (accent) => set({ accent }),
    }),
    {
      name: APPEARANCE_KEY,
      version: 3,
      partialize: ({ theme, themeChosen, density, accent }) => ({ theme, themeChosen, density, accent }),
      // Version 1 stored the theme in effect without telling whether it was picked: keep it as picked.
      // Version 2's default accent was iris; the blue pencil is the default since, and a stored iris is
      // almost always that old default, not a choice. (missing keys are filled by `merge`)
      migrate: (persisted, version) => {
        let state = (persisted ?? {}) as Partial<StoredAppearance>
        if (version < 2) state = { ...state, themeChosen: isTheme(state.theme) }
        if (version < 3 && state.accent === "iris") state = { ...state, accent: "blue" }
        return state as StoredAppearance
      },
      // A theme never picked follows the server's `--theme` (it may have changed since).
      merge: (persisted, current) => {
        const state = (persisted ?? {}) as Partial<AppearanceState>
        const chosen = state.themeChosen === true && isTheme(state.theme)
        return { ...current, ...state, theme: chosen ? state.theme! : current.theme, themeChosen: chosen }
      },
    },
  ),
)

/** Opening the UI with `?__theme=…` picks that theme, as the Appearance menu would. */
export function applyUrlTheme(search?: string) {
  const theme = urlTheme(search)
  if (theme) useAppearance.getState().setTheme(theme)
}
applyUrlTheme()

const darkQuery = () => window.matchMedia("(prefers-color-scheme: dark)")

function subscribeSystem(onChange: () => void) {
  const media = darkQuery()
  media.addEventListener("change", onChange)
  return () => media.removeEventListener("change", onChange)
}

/** The OS colour scheme, live. */
function useSystemDark() {
  return useSyncExternalStore(
    subscribeSystem,
    () => darkQuery().matches,
    () => true,
  )
}

/** The theme in effect (system resolved), for components such as the toaster. */
export function useTheme() {
  const theme = useAppearance((s) => s.theme)
  const systemDark = useSystemDark()
  const setTheme = useAppearance((s) => s.setTheme)
  const resolved: "light" | "dark" = theme === "system" ? (systemDark ? "dark" : "light") : theme
  return { theme: resolved, setTheme }
}

/** Mirrors the appearance preferences onto <html> (class + data attributes). */
export function useApplyAppearance() {
  const { theme } = useTheme()
  const density = useAppearance((s) => s.density)
  const accent = useAppearance((s) => s.accent)
  useEffect(() => {
    const root = document.documentElement
    root.classList.toggle("dark", theme === "dark")
    root.dataset.density = density
    root.dataset.accent = accent
  }, [theme, density, accent])
}
