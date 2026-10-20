/**
 * Node views that the default views' first screen does not show: image,
 * file, color and JSON inputs, tabs and output views. txt2img has them only
 * in collapsed sections, settings not at all, so their code (with the image
 * editor's form glue and the tabs primitives) is fetched the first time one
 * renders instead of with every form (docs/research/ui-performance.md,
 * "Bundle"). A small skeleton stands in while one loads; `preloadViews()`
 * fetches them all (tests, or a view that is about to show them).
 */
// oxlint-disable react/only-export-components -- the exports are components made by deferred(), plus their preload
import { type ComponentType, lazy, Suspense } from "react"
import { Skeleton } from "@/components/ui/skeleton"

// oxlint-disable-next-line typescript/no-explicit-any -- each view receives its own node variant
type View = ComponentType<{ node: any }>

const loaders: (() => Promise<unknown>)[] = []

function deferred(load: () => Promise<View>, fallback: string): View {
  let loaded: View | null = null
  const fetch = () =>
    load().then((component) => {
      loaded = component
      return component
    })
  loaders.push(fetch)
  // Once fetched (by an earlier render or preloadViews), React.lazy resolves synchronously: a
  // thenable that calls back at once renders without suspending.
  const Lazy = lazy(() => (loaded ? ({ then: (resolve: (module: { default: View }) => void) => resolve({ default: loaded as View }) } as Promise<{ default: View }>) : fetch().then((component) => ({ default: component }))))
  function DeferredView(props: { node: unknown }) {
    return (
      <Suspense fallback={<Skeleton className={fallback} />}>
        <Lazy {...props} />
      </Suspense>
    )
  }
  return DeferredView
}

export const ImageInput = deferred(() => import("./inputs/Image").then((m) => m.ImageInput), "h-40 w-full rounded-lg")
export const ImageListInput = deferred(() => import("./inputs/Image").then((m) => m.ImageListInput), "h-24 w-full rounded-lg")
export const Color = deferred(() => import("./inputs/Color").then((m) => m.Color), "h-9 w-full rounded-md")
export const FileInput = deferred(() => import("./inputs/File").then((m) => m.FileInput), "h-9 w-full rounded-md")
export const JsonInput = deferred(() => import("./inputs/Json").then((m) => m.JsonInput), "h-20 w-full rounded-md")
export const TabsView = deferred(() => import("./layout/Tabs").then((m) => m.TabsView), "h-24 w-full rounded-lg")
export const GalleryView = deferred(() => import("./layout/Outputs").then((m) => m.GalleryView), "h-40 w-full rounded-lg")
export const OutputView = deferred(() => import("./layout/Outputs").then((m) => m.OutputView), "h-9 w-full rounded-md")

/** Fetches every deferred view; afterwards they render without a placeholder. */
export function preloadViews(): Promise<unknown> {
  return Promise.all(loaders.map((load) => load()))
}
