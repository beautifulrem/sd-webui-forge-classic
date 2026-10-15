/**
 * Unsaved-changes registry: views with pending edits (the settings page)
 * report how many they hold. The shell then asks before the page unloads,
 * and server commands warn that a reload drops them.
 */
import { create } from "zustand"

type UnsavedState = {
  /** source → number of unsaved edits */
  sources: Record<string, number>
  set: (source: string, count: number) => void
}

export const useUnsaved = create<UnsavedState>()((set) => ({
  sources: {},
  set: (source, count) =>
    set((s) => {
      if ((s.sources[source] ?? 0) === count) return s
      const sources = { ...s.sources }
      if (count > 0) sources[source] = count
      else delete sources[source]
      return { sources }
    }),
}))

export function unsavedTotal(sources: Record<string, number> = useUnsaved.getState().sources): number {
  return Object.values(sources).reduce((sum, count) => sum + count, 0)
}

let installed = false
let bypass = false

/** The next unload is intended: call right before reloading the page (the server came back, or "Reload page now"). */
export function leaveWithoutAsking() {
  bypass = true
}

/** Ask again before unloading with unsaved edits (a server command was dismissed or failed). */
export function askBeforeLeaving() {
  bypass = false
}

/** Asks the browser to confirm leaving while edits are unsaved (installed once by the shell). */
export function installBeforeUnloadGuard() {
  if (installed || typeof window === "undefined") return
  installed = true
  window.addEventListener("beforeunload", (event) => {
    if (!bypass && unsavedTotal() > 0) {
      event.preventDefault()
      // Legacy browsers need returnValue set.
      event.returnValue = ""
    }
  })
}
