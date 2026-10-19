import { useEffect, useSyncExternalStore } from "react"
import { create } from "zustand"
import { persist } from "zustand/middleware"

export type Theme = "light" | "dark" | "system"
export type Density = "comfortable" | "compact"
export const ACCENTS = ["iris", "blue", "teal", "green", "amber", "coral", "rose"] as const
export type Accent = (typeof ACCENTS)[number]

/** Storage key; index.html reads it before first paint to avoid a theme flash. */
export const APPEARANCE_KEY = "remi-appearance"

type AppearanceState = {
  theme: Theme
  density: Density
  accent: Accent
  setTheme: (theme: Theme) => void
  setDensity: (density: Density) => void
  setAccent: (accent: Accent) => void
}

export const useAppearance = create<AppearanceState>()(
  persist(
    (set) => ({
      theme: "dark",
      density: "comfortable",
      accent: "iris",
      setTheme: (theme) => set({ theme }),
      setDensity: (density) => set({ density }),
      setAccent: (accent) => set({ accent }),
    }),
    {
      name: APPEARANCE_KEY,
      version: 1,
      partialize: ({ theme, density, accent }) => ({ theme, density, accent }),
    },
  ),
)

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
