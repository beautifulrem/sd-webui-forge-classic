import { cleanup } from "@testing-library/react"
import { afterEach, beforeAll } from "vitest"
import "@testing-library/jest-dom/vitest"
import "@/i18n"

// jsdom lacks these browser APIs used by the shell and Base UI.
if (!window.matchMedia) {
  window.matchMedia = (query: string) =>
    ({
      matches: false,
      media: query,
      onchange: null,
      addEventListener: () => {},
      removeEventListener: () => {},
      addListener: () => {},
      removeListener: () => {},
      dispatchEvent: () => false,
    }) as MediaQueryList
}

if (!globalThis.ResizeObserver) {
  globalThis.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
}

// jsdom has no 2D canvas (it logs "Not implemented" and returns null): return null quietly.
// Tests that draw spy on getContext themselves.
HTMLCanvasElement.prototype.getContext = (() => null) as unknown as HTMLCanvasElement["getContext"]

if (!URL.createObjectURL) {
  let n = 0
  URL.createObjectURL = () => `blob:mock/${++n}`
  URL.revokeObjectURL = () => {}
}

// Dropdown menus load their implementation when one opens (components/ui/dropdown-menu.tsx).
// Transform it once up front: opening a menu then still goes through the on-demand load, but a
// test's `findByRole("menu")` does not also wait for the first transform on a busy machine.
beforeAll(async () => {
  await import("@/components/ui/dropdown-menu-impl")
})

afterEach(() => {
  cleanup()
  localStorage.clear()
})
