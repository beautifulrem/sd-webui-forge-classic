/** Static facts derived from layouts: input nodes per owner, slots, subtree params. */
import {
  childrenOf,
  type GroupNode,
  type InputNode,
  isInput,
  type Node,
  type OwnerDefaults,
  type ScriptSchema,
  type TabsNode,
  type ToggleNode,
} from "./schema"

export type OwnerSpec = { owner: string; layout: Node | null; defaults: OwnerDefaults }

/** What to open so an input becomes visible: collapsibles and unbound tab selections. */
export type Reveal = { open: string[]; tabs: [key: string, value: string][] }

export type FormIndex = {
  /** owner → name → input node (including section `enable` toggles). */
  inputs: Map<string, Map<string, InputNode>>
  /** Slot sections present in the core layout. */
  slots: Set<string>
  /** owner → name → the containers to open to show the input (validation errors). */
  reveal: Map<string, Map<string, Reveal>>
}

/** Visits every node of a layout, `enable` toggles included (before the section's children). */
export function walk(node: Node | null | undefined, visit: (node: Node) => void) {
  if (!node) return
  visit(node)
  if (node.type === "section" && node.enable) visit(node.enable)
  for (const child of childrenOf(node)) walk(child, visit)
}

/** Input nodes of a subtree, in declaration order. */
export function inputsIn(node: Node | null | undefined): InputNode[] {
  const found: InputNode[] = []
  walk(node, (n) => {
    if (isInput(n)) found.push(n)
  })
  return found
}

/** Records, for every input under `node`, the collapsibles and unbound tabs around it. */
function collectReveal(owner: string, node: Node | null | undefined, path: Reveal, out: Map<string, Reveal>) {
  if (!node) return
  if (isInput(node)) {
    out.set(node.name, path)
    return
  }
  if (node.type === "section") {
    const inside = { ...path, open: [...path.open, sectionKey(owner, node)] }
    if (node.enable) out.set(node.enable.name, path)
    for (const child of node.children) collectReveal(owner, child, inside, out)
  } else if (node.type === "group" && node.title) {
    const inside = { ...path, open: [...path.open, groupKey(owner, node)] }
    for (const child of node.children) collectReveal(owner, child, inside, out)
  } else if (node.type === "tabs" && !node.bind) {
    const key = tabsKey(owner, node)
    node.children.forEach((tab, i) => {
      for (const child of tab.children) collectReveal(owner, child, { ...path, tabs: [...path.tabs, [key, String(i)]] }, out)
    })
  } else {
    for (const child of childrenOf(node)) collectReveal(owner, child, path, out)
  }
}

export function buildIndex(owners: readonly OwnerSpec[]): FormIndex {
  const inputs = new Map<string, Map<string, InputNode>>()
  const slots = new Set<string>()
  const reveal = new Map<string, Map<string, Reveal>>()
  for (const { owner, layout } of owners) {
    const byName = new Map<string, InputNode>()
    walk(layout, (node) => {
      if (isInput(node)) byName.set(node.name, node)
      else if (node.type === "slot" && owner === "core") slots.add(node.section)
    })
    inputs.set(owner, byName)
    const paths = new Map<string, Reveal>()
    collectReveal(owner, layout, { open: [], tabs: [] }, paths)
    reveal.set(owner, paths)
  }
  return { inputs, slots, reveal }
}

/** The default for a param: the owner's `defaults` map first, then the node's own `default`. */
export function defaultOf(spec: OwnerSpec, node: InputNode): unknown {
  return Object.hasOwn(spec.defaults, node.name) ? spec.defaults[node.name] : (node.default ?? null)
}

/** The default slot for scripts without a (known) section. */
export const DEFAULT_SLOT = "scripts"

/** Scripts for `section`, ordered; unknown/absent sections fall back to the default slot. */
export function scriptsForSlot(scripts: readonly ScriptSchema[], section: string, slots: ReadonlySet<string>): ScriptSchema[] {
  return scripts
    .filter((script) => {
      const target = script.section && slots.has(script.section) ? script.section : DEFAULT_SLOT
      return target === section
    })
    .sort((a, b) => (a.order ?? 0) - (b.order ?? 0))
}

/** Key of a section's open state in the form store. */
export function sectionKey(owner: string, node: { title?: string | null; enable?: ToggleNode | null }) {
  return `${owner}::${node.enable?.name ?? node.title ?? ""}`
}

/** Key of a titled group's open state in the form store. */
export function groupKey(owner: string, node: GroupNode) {
  return `${sectionKey(owner, node)}::group`
}

/** Key of an unbound tab strip's selection (the tab index as a string) in the form store. */
export function tabsKey(owner: string, node: TabsNode) {
  return `${owner}::tabs::${node.children.map((tab) => tab.title).join("|")}`
}
