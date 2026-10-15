/**
 * Which canvas layers a form uses, read from its schema (pure): an `image`
 * input is the background; `sketch` / `mask` inputs whose `source` names it
 * are painted over it; a `mask` without `source` is an uploaded mask the
 * editor shows (and lets the user touch up) over the same background. In
 * each mode exactly the layers whose `visible_if` holds are in use, so the
 * img2img modes map onto the wave-1 core params without a table here:
 *
 * | mode           | background | layer edited          | mask the server reads |
 * |----------------|------------|-----------------------|-----------------------|
 * | img2img        | init_img   | –                     | –                     |
 * | Sketch         | init_img   | sketch (colour)       | –                     |
 * | Inpaint        | init_img   | mask                  | mask                  |
 * | Inpaint sketch | init_img   | sketch (colour)       | the sketch's alpha    |
 * | Inpaint upload | init_img   | mask_upload (uploaded) | mask_upload          |
 * | Batch          | –          | –                     | –                     |
 */
import { evaluate } from "@/ui/conditions"
import { childrenOf, type Condition, type ImageNode, type InputNode, isInput, type Node } from "@/ui/schema"

export type LayerKind = "sketch" | "mask"

export type LayerBinding = {
  /** The param holding the layer's upload id. */
  param: string
  kind: LayerKind
  /** Painted over the background (false: an uploaded mask image). */
  painted: boolean
}

export type EditorBinding = {
  /** The background image param. */
  background: string
  layers: LayerBinding[]
  /** Every input's `visible_if` and its containers', root first (`hidden` inputs: `false`). */
  conditions: ReadonlyMap<string, readonly Condition[]>
}

const isImage = (node: InputNode | undefined): node is ImageNode => node?.type === "image"

/** Input name → the visibility conditions on its path from the root. */
function visibilityPaths(layout: Node | null | undefined): Map<string, Condition[]> {
  const out = new Map<string, Condition[]>()
  const visit = (node: Node, path: Condition[]) => {
    const here = node.visible_if === undefined ? path : [...path, node.visible_if]
    if (isInput(node)) {
      out.set(node.name, node.hidden ? [...here, false] : here)
      return
    }
    if (node.type === "section" && node.enable) visit(node.enable, here)
    // A bound tab's content shows only while its choice is selected.
    if (node.type === "tabs" && node.bind) {
      for (const tab of node.children) {
        const shown: Condition = tab.value === undefined || tab.value === null ? true : { param: node.bind, eq: tab.value }
        visit(tab, [...here, shown])
      }
      return
    }
    for (const child of childrenOf(node)) visit(child, here)
  }
  if (layout) visit(layout, [])
  return out
}

function inputsOf(layout: Node | null | undefined): InputNode[] {
  const found: InputNode[] = []
  const visit = (node: Node) => {
    if (isInput(node)) found.push(node)
    for (const child of childrenOf(node)) visit(child)
  }
  if (layout) visit(layout)
  return found
}

/** The editor's params in a form owner's layout; null when it has no image input. */
export function editorBinding(layout: Node | null | undefined): EditorBinding | null {
  const images = inputsOf(layout).filter(isImage)
  const sources = new Set(images.map((node) => node.source).filter((source): source is string => Boolean(source)))
  const backgrounds = images.filter((node) => (node.kind ?? "image") === "image")
  const background = backgrounds.find((node) => sources.has(node.name)) ?? backgrounds[0]
  if (!background) return null
  const layers: LayerBinding[] = []
  for (const node of images) {
    if (node.kind === "sketch" && node.source === background.name) layers.push({ param: node.name, kind: "sketch", painted: true })
    else if (node.kind === "mask" && node.source === background.name) layers.push({ param: node.name, kind: "mask", painted: true })
    else if (node.kind === "mask" && !node.source) layers.push({ param: node.name, kind: "mask", painted: false })
  }
  return { background: background.name, layers, conditions: visibilityPaths(layout) }
}

export type ModeLayers = {
  /** Whether the background image is in use (not in batch modes). */
  canvas: boolean
  /** The layer the user edits in this mode. */
  active: LayerBinding | null
  /** Where the processing takes the inpaint mask from in this mode. */
  mask: "painted" | "sketch" | "uploaded" | null
}

type Lookup = (owner: string, name: string) => unknown

/**
 * The layers in use for the current values. `inpaintParam` is a param shown
 * only in inpaint modes (the core form's `mask_blur`): when it is visible
 * and the edited layer is a sketch, the sketch's alpha is the mask.
 */
export function modeLayers(binding: EditorBinding, get: Lookup, owner = "core", inpaintParam = "mask_blur"): ModeLayers {
  const visible = (name: string) => {
    const path = binding.conditions.get(name)
    return path !== undefined && path.every((cond) => evaluate(cond, owner, get))
  }
  if (!visible(binding.background)) return { canvas: false, active: null, mask: null }
  const active = binding.layers.find((layer) => visible(layer.param)) ?? null
  const inpaint = visible(inpaintParam)
  let mask: ModeLayers["mask"] = null
  if (active?.kind === "mask") mask = active.painted ? "painted" : "uploaded"
  else if (active?.kind === "sketch" && inpaint) mask = "sketch"
  return { canvas: true, active, mask }
}

/**
 * The image params a submit reads in a mode (what must be set, in order):
 * the background and the edited layer.
 */
export function modeParams(layers: ModeLayers, binding: EditorBinding): string[] {
  if (!layers.canvas) return []
  return [binding.background, ...(layers.active ? [layers.active.param] : [])]
}
