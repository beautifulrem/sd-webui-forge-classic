/**
 * Command registry for the palette. The shell registers navigation and
 * appearance commands; pages and widgets add their own with
 * `useRegisterCommands` while mounted.
 */
import { useEffect } from "react"
import type { LucideIcon } from "lucide-react"
import { create } from "zustand"

export type Command = {
  id: string
  /** English label; translated when rendered. */
  label: string
  /** English group heading. */
  group: string
  icon?: LucideIcon
  /** Extra search terms (not shown). */
  keywords?: string[]
  /** Display-only shortcut hint, e.g. "Mod+Enter". */
  shortcut?: string
  run: () => void
}

type CommandState = {
  sources: Record<string, readonly Command[]>
  palette: boolean
  /** True once the palette has been opened (its chunk stays mounted afterwards). */
  paletteLoaded: boolean
  setPalette: (open: boolean) => void
  register: (source: string, commands: readonly Command[]) => void
  unregister: (source: string) => void
}

export const useCommands = create<CommandState>()((set) => ({
  sources: {},
  palette: false,
  paletteLoaded: false,
  setPalette: (palette) => set((s) => ({ palette, paletteLoaded: s.paletteLoaded || palette })),
  register: (source, commands) => set((s) => ({ sources: { ...s.sources, [source]: commands } })),
  unregister: (source) =>
    set((s) => {
      const { [source]: _removed, ...rest } = s.sources
      return { sources: rest }
    }),
}))

/** Registers `commands` under `source` for the lifetime of the calling component. */
export function useRegisterCommands(source: string, commands: readonly Command[]) {
  useEffect(() => {
    const { register, unregister } = useCommands.getState()
    register(source, commands)
    return () => unregister(source)
  }, [source, commands])
}

export const isMac = typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform)

/** "Mod+K" → "⌘K" on macOS, "Ctrl+K" elsewhere. */
export function formatShortcut(shortcut: string): string {
  if (!isMac) return shortcut.replace("Mod", "Ctrl")
  return shortcut
    .replace("Mod+", "⌘")
    .replace("Alt+", "⌥")
    .replace("Shift+", "⇧")
    .replace("Ctrl+", "⌃")
}
