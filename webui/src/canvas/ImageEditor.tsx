/**
 * `<ImageEditor>`: the lazily loaded image editor (Konva lives in its own
 * chunk, fetched the first time an editor renders). Use it in widgets, e.g.
 * a ControlNet unit's input image and mask (height: `height`, or the "Canvas
 * Height" setting; `fill` makes it fill a sized container instead):
 *
 * ```tsx
 * <ImageEditor docKey={`cn-${unit}`} pendingKey={formId} image={value.image}
 *   onImageChange={(image) => set({ image: image?.id ?? null })}
 *   layers={{ mask: { kind: "mask", value: value.mask } }} activeLayer="mask"
 *   onLayerChange={(_, id) => set({ mask: id })} height={320} />
 * ```
 */
import { lazy, Suspense } from "react"
import { clientOption } from "@/api/app-options"
import { Skeleton } from "@/components/ui/skeleton"
import { cn } from "@/lib/utils"
import type { ImageEditorProps } from "./types"

const Editor = lazy(() => import("./ui/Editor"))

export function ImageEditor(props: ImageEditorProps) {
  return (
    <Suspense fallback={<Skeleton className={cn("rounded-xl", props.className)} style={props.fill ? undefined : { height: props.height ?? clientOption("forge_canvas_height") }} />}>
      <Editor {...props} />
    </Suspense>
  )
}
