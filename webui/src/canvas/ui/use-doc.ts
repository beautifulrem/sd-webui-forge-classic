/** React bindings of an editor document (selective: a stroke must not re-render the chrome per frame). */
import { useSyncExternalStore } from "react"
import type { EditorDocument } from "../engine/document"

/** A primitive derived from the document; re-renders only when it changes. */
export function useDocValue<T extends string | number | boolean | null>(doc: EditorDocument, select: (doc: EditorDocument) => T): T {
  return useSyncExternalStore(
    (listener) => doc.subscribe(listener),
    () => select(doc),
  )
}

/**
 * Whether a layer's edits are still being uploaded, following the layer's
 * sync (layers appear lazily, on the first edit).
 */
export function useLayerPending(doc: EditorDocument, key: string | null): boolean {
  return useSyncExternalStore(
    (listener) => {
      let unsync = () => {}
      let bound: unknown = null
      const rebind = () => {
        const sync = key ? doc.layer(key)?.sync : undefined
        if ((sync ?? null) === bound) return
        unsync()
        bound = sync ?? null
        unsync = sync ? sync.subscribe(listener) : () => {}
      }
      rebind()
      const undoc = doc.subscribe(() => {
        rebind()
        listener()
      })
      return () => {
        undoc()
        unsync()
      }
    },
    () => Boolean(key && doc.layer(key)?.sync.pending),
  )
}
