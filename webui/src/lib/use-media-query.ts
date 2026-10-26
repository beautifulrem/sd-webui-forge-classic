import { useSyncExternalStore } from "react"

/** Live `matchMedia` result. */
export function useMediaQuery(query: string): boolean {
  return useSyncExternalStore(
    (onChange) => {
      const media = window.matchMedia(query)
      media.addEventListener("change", onChange)
      return () => media.removeEventListener("change", onChange)
    },
    () => window.matchMedia(query).matches,
    () => false,
  )
}

/** No mouse or trackpad (a phone or tablet): the CSS variant `touch-only` (index.css) is the same test. */
export const TOUCH_ONLY_QUERY = "not (any-pointer: fine)"

/** Whether keyboard shortcut hints are worth showing (false on touch-only devices). */
export const useKeyboardHints = () => !useMediaQuery(TOUCH_ONLY_QUERY)

/** Tailwind's `md` breakpoint and up. */
export const useIsDesktop = () => useMediaQuery("(min-width: 48rem)")
