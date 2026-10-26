/**
 * A module fetched on first intent instead of with the page: the popup
 * controls (select, search select, chips select, popover) render a light
 * stand-in of their closed state and load the real implementation when the
 * pointer enters them, they get focus or are pressed
 * (docs/research/ui-performance.md, "Bundle"). Once loaded, every
 * subscriber re-renders with the module.
 */
import { useSyncExternalStore } from "react"

export type LazyModule<M> = {
  /** Starts the fetch (once; again after a failed one); resolves with the module. */
  load: () => Promise<M>
  /** The module if it has loaded. */
  get: () => M | null
  /** The module once loaded (re-renders the caller when it arrives): a hook. */
  useModule: () => M | null
}

export function lazyModule<M>(importer: () => Promise<M>): LazyModule<M> {
  let loaded: M | null = null
  let loading: Promise<M> | null = null
  const listeners = new Set<() => void>()
  const subscribe = (listener: () => void) => {
    listeners.add(listener)
    return () => {
      listeners.delete(listener)
    }
  }
  const get = () => loaded
  const load = () => {
    loading ??= importer().then(
      (module) => {
        loaded = module
        for (const listener of listeners) listener()
        return module
      },
      (error: unknown) => {
        // a failed fetch (offline, a replaced build) is retried on the next intent
        loading = null
        throw error
      },
    )
    return loading
  }
  return { load, get, useModule: () => useSyncExternalStore(subscribe, get, get) }
}

/** Starts a fetch on intent; a failure is retried on the next intent instead of surfacing here. */
export function prefetch(module: LazyModule<unknown>): void {
  module.load().catch(() => {})
}

/**
 * Hands the focus from a stand-in to the control that replaces it: the
 * stand-in records itself under `key` while focused; the real control, when
 * it mounts, takes the focus if the stand-in had it (the stand-in's element
 * is gone, so the focus fell to the body).
 */
const focusedStandIns = new Set<string>()

export const standInFocus = {
  focused(key: string) {
    focusedStandIns.add(key)
  },
  blurred(key: string) {
    focusedStandIns.delete(key)
  },
  /** Whether the stand-in under `key` had the focus when it was replaced (clears the record). */
  take(key: string): boolean {
    if (!focusedStandIns.delete(key)) return false
    const active = document.activeElement
    return !active || active === document.body || !active.isConnected
  },
}
