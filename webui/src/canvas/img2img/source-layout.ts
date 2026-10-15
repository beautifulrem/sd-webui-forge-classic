/** The img2img Source card's arrangement of the schema's source blocks (pure). */
import { childrenOf, isInput, type Node, type TabsNode } from "@/ui/schema"

/**
 * The source blocks without what the mode picker and the stage editor show,
 * as one flat list: the card frames them, so untitled groups lose their own
 * box (a group with conditions becomes a plain column keeping them), and
 * hidden inputs and containers left empty go (no empty frames). The mode
 * tabs (bound to `modeParam`) are taken out into `found.tabs`.
 */
export function pruneSource(nodes: readonly Node[], handled: ReadonlySet<string>, modeParam: string, found: { tabs: TabsNode | null }): Node[] {
  const out: Node[] = []
  for (const node of nodes) {
    if (isInput(node)) {
      if (!handled.has(node.name) && !node.hidden && node.name !== modeParam) out.push(node)
      continue
    }
    if (node.type === "tabs" && node.bind === modeParam) {
      found.tabs = node
      continue
    }
    if (node.type === "slot" || !("children" in node) || node.type === "tab" || node.type === "tabs" || node.type === "section") {
      out.push(node)
      continue
    }
    const children = pruneSource(childrenOf(node), handled, modeParam, found)
    if (children.length === 0) continue
    const plain = node.type === "column" || (node.type === "group" && !node.title)
    if (!plain) out.push({ ...node, children } as Node)
    else if (node.visible_if === undefined && node.enabled_if === undefined) out.push(...children)
    else out.push({ type: "column", visible_if: node.visible_if, enabled_if: node.enabled_if, children })
  }
  return out
}
