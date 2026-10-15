/**
 * The generate page's arrangement of a core form: the prompts, the styles
 * and the script choice are taken out (the page renders them itself) and the
 * remaining top-level blocks are classified by what they hold (source image,
 * sampler, size, guidance, seed, sections, the rest) so the page can frame
 * them. Blocks stay in schema order — the user's category order
 * (`ui_reorder_list`) decides the layout — and keep their schema subtree, so
 * server-side layout changes still render.
 */
import { DEFAULT_SLOT, walk } from "@/ui/form-index"
import type { ChoiceNode, MultiChoiceNode, Node, TextNode } from "@/ui/schema"
import { isInput } from "@/ui/schema"

export type BlockKind = "source" | "sampler" | "size" | "guidance" | "seed" | "sections" | "other"

export type CuratedForm = {
  prompt: TextNode | null
  negative: TextNode | null
  styles: MultiChoiceNode | null
  /** The selectable-script choice (`core.script`). */
  script: ChoiceNode | null
  blocks: { kind: BlockKind; node: Node }[]
}

const TAKEN = new Set(["prompt", "negative_prompt", "styles", "script"])
const SOURCE_INPUTS = new Set(["init_img", "sketch", "mask", "mask_upload", "batch_upload", "mask_blur"])

function prune(node: Node): Node | null {
  if (isInput(node)) return TAKEN.has(node.name) ? null : node
  if (node.type === "slot") return node.section === DEFAULT_SLOT ? null : node
  if (!("children" in node) || node.type === "tab" || node.type === "tabs") return node
  if (node.type === "section" && node.enable) {
    return { ...node, children: node.children.map(prune).filter((child): child is Node => child !== null) }
  }
  const children = node.children.map(prune).filter((child): child is Node => child !== null)
  if (children.length === 0 && node.children.length > 0) return null
  return { ...node, children } as Node
}

function classify(node: Node): BlockKind {
  const inputs = new Set<string>()
  const slots = new Set<string>()
  let section = false
  let image = false
  walk(node, (n) => {
    if (isInput(n)) {
      inputs.add(n.name)
      if (n.type === "image" || n.type === "image_list") image = true
    } else if (n.type === "slot") slots.add(n.section)
    else if (n.type === "section") section = true
  })
  if (image || [...inputs].some((name) => SOURCE_INPUTS.has(name))) return "source"
  if (slots.has("sampler")) return "sampler"
  if (inputs.has("width") || inputs.has("height")) return "size"
  // Before guidance: the Hires. fix section holds a denoising strength too.
  if (section || slots.has("accordions")) return "sections"
  if (inputs.has("cfg_scale") || inputs.has("distilled_cfg_scale") || inputs.has("denoising_strength")) return "guidance"
  if (slots.has("seed")) return "seed"
  return "other"
}

function find<T extends Node>(layout: Node, name: string): T | null {
  const found: Node[] = []
  walk(layout, (n) => {
    if (isInput(n) && n.name === name) found.push(n)
  })
  return (found[0] as T | undefined) ?? null
}

export function curateCore(layout: Node): CuratedForm {
  const pruned = prune(layout)
  const top = !pruned ? [] : pruned.type === "column" || (pruned.type === "group" && !pruned.title) ? pruned.children : [pruned]
  const blocks = top.map((node) => ({ kind: classify(node), node }))
  return {
    prompt: find<TextNode>(layout, "prompt"),
    negative: find<TextNode>(layout, "negative_prompt"),
    styles: find<MultiChoiceNode>(layout, "styles"),
    script: find<ChoiceNode>(layout, "script"),
    blocks,
  }
}
