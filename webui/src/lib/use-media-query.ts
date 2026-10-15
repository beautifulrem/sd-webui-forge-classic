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

/** Tailwind's `md` breakpoint and up. */
export const useIsDesktop = () => useMediaQuery("(min-width: 48rem)")
