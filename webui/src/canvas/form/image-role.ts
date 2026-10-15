/** How the renderer shows an image input: a layered editor, nothing (a layer), or an upload field. */
import type { FormState } from "@/ui/form-store"
import { useFormState, useOwner } from "@/ui/hooks"
import type { ImageNode, InputNode } from "@/ui/schema"
import type { LayerKind } from "../types"

const isLayer = (node: InputNode | undefined): node is ImageNode & { kind: LayerKind } => node?.type === "image" && (node.kind === "sketch" || node.kind === "mask")

/** The layer inputs painted over `background` in an owner. */
export function layersOf(state: FormState, owner: string, background: string): (ImageNode & { kind: LayerKind })[] {
  const out: (ImageNode & { kind: LayerKind })[] = []
  for (const node of state.index.inputs.get(owner)?.values() ?? []) if (isLayer(node) && node.source === background) out.push(node)
  return out
}

/**
 * How an image input renders: `"editor"` (a background with painted
 * layers), `"layer"` (painted on its source's editor: nothing here) or
 * `"upload"` (a plain upload field).
 */
export function useImageRole(node: ImageNode): "editor" | "layer" | "upload" {
  const owner = useOwner()
  return useFormState((s) => {
    const kind = node.kind ?? "image"
    if (kind === "image") return layersOf(s, owner, node.name).length > 0 ? "editor" : "upload"
    if (!node.source) return "upload"
    const source = s.index.inputs.get(owner)?.get(node.source)
    return source?.type === "image" && (source.kind ?? "image") === "image" ? "layer" : "upload"
  })
}

