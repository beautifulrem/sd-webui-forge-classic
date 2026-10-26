/**
 * A confirmation dialog ("Clear the history?") whose code (the alert dialog,
 * focus trap and scroll lock) loads the first time it opens, so the views
 * that offer one do not carry it in their first load
 * (docs/research/ui-performance.md, "Bundle"). It renders nothing until then
 * and stays mounted afterwards (close animation). `preloadConfirmDialog()`
 * fetches it ahead, e.g. when the pointer enters the button that opens it.
 */
// oxlint-disable react/only-export-components -- the component and its preload
import { lazy, Suspense, useState } from "react"
import type { ConfirmDialogProps } from "./ConfirmDialogView"

const load = () => import("./ConfirmDialogView")
const View = lazy(() => load().then((module) => ({ default: module.ConfirmDialogView })))

/** Fetches the dialog's code (button intent); module loads are cached. */
export function preloadConfirmDialog() {
  void load().catch(() => {})
}

export function ConfirmDialog(props: ConfirmDialogProps) {
  const [used, setUsed] = useState(props.open)
  if (props.open && !used) setUsed(true)
  if (!used) return null
  return (
    <Suspense fallback={null}>
      <View {...props} />
    </Suspense>
  )
}

export type { ConfirmDialogProps }
