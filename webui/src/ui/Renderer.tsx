/**
 * Schema renderer: dispatches every node `type` to its component and applies
 * `visible_if` / `enabled_if`, patch overrides and `width` sizing inside rows.
 */
import type { ComponentType } from "react"
import { useTranslation } from "react-i18next"
import { Separator } from "@/components/ui/separator"
import { EnabledProvider } from "./context"
import { useCondition, useFormState, useOwner } from "./hooks"
import { Choice } from "./inputs/Choice"
import { Color } from "./inputs/Color"
import { Custom } from "./inputs/Custom"
import { FileInput } from "./inputs/File"
import { ImageInput, ImageListInput } from "./inputs/Image"
import { JsonInput } from "./inputs/Json"
import { MultiChoice } from "./inputs/MultiChoice"
import { NumberInput } from "./inputs/Number"
import { Seed } from "./inputs/Seed"
import { Slider } from "./inputs/Slider"
import { Text } from "./inputs/Text"
import { Toggle } from "./inputs/Toggle"
import { ActionButton } from "./layout/ActionButton"
import { Group } from "./layout/Group"
import { GalleryView, OutputView } from "./layout/Outputs"
import { Section } from "./layout/Section"
import { cellProps } from "./layout/sizing"
import { Slot } from "./layout/Slot"
import { TabsView } from "./layout/Tabs"
import { Markdown, SafeHtml } from "./RichText"
import { type InputNode, type InputType, isInput, type Node } from "./schema"

// oxlint-disable-next-line typescript/no-explicit-any -- each entry receives its own node variant
const INPUTS: Record<Exclude<InputType, "state">, ComponentType<{ node: any }>> = {
  toggle: Toggle,
  slider: Slider,
  number: NumberInput,
  text: Text,
  choice: Choice,
  multichoice: MultiChoice,
  seed: Seed,
  color: Color,
  image: ImageInput,
  image_list: ImageListInput,
  file: FileInput,
  json: JsonInput,
  custom: Custom,
}

function InputView({ node }: { node: Exclude<InputNode, { type: "state" }> }) {
  // An image list with a widget hint keeps its upload semantics (drafts, validation) but the widget's UI.
  if (node.type === "image_list" && node.widget) return <Custom node={{ ...node, widget: node.widget }} />
  const Component = INPUTS[node.type]
  return Component ? <Component node={node} /> : null
}

function Children({ nodes, inRow = false }: { nodes: readonly Node[]; inRow?: boolean }) {
  return nodes.map((child, index) => <RenderNode key={index} node={child} inRow={inRow} />)
}

function LayoutView({ node }: { node: Exclude<Node, InputNode> }) {
  const { t } = useTranslation()
  switch (node.type) {
    case "section":
      return <Section node={node} />
    case "group":
      return <Group node={node} />
    case "row":
      return (
        <div className="remi-row">
          <Children nodes={node.children} inRow />
        </div>
      )
    case "column":
      return (
        <div className="flex min-w-0 flex-col gap-3">
          <Children nodes={node.children} />
        </div>
      )
    case "tabs":
      return <TabsView node={node} />
    case "tab":
      // A stray tab outside `tabs`: render its content.
      return (
        <div className="flex flex-col gap-3">
          <Children nodes={node.children} />
        </div>
      )
    case "markdown":
      return <Markdown text={t(node.text)} />
    case "html":
      return <SafeHtml html={node.text} />
    case "separator":
      return <Separator className="my-1" />
    case "button":
      return <ActionButton node={node} />
    case "gallery":
      return <GalleryView node={node} />
    case "output":
      return <OutputView node={node} />
    case "slot":
      return <Slot section={node.section} />
    default:
      if (import.meta.env.DEV) console.warn("[ui] unknown node type", node)
      return null
  }
}

/** Renders one node (and its subtree) for the current owner. */
export function RenderNode({ node, inRow = false }: { node: Node; inRow?: boolean }) {
  const owner = useOwner()
  const visible = useCondition(node.visible_if)
  const enabled = useCondition(node.enabled_if)
  // Action patches can hide an input (`visible: false`), ANDed with visible_if.
  const hiddenByPatch = useFormState((s) => isInput(node) && s.overrides[owner]?.[node.name]?.visible === false)
  if (!visible || hiddenByPatch) return null
  if (isInput(node) && (node.type === "state" || node.hidden)) return null

  const content = isInput(node) ? (
    <InputView node={node as Exclude<InputNode, { type: "state" }>} />
  ) : (
    <EnabledProvider enabled={enabled}>
      <LayoutView node={node} />
    </EnabledProvider>
  )
  return inRow ? <div {...cellProps(node)}>{content}</div> : content
}

/** A layout tree for one owner (no-op for a null layout). */
export function RenderLayout({ layout }: { layout: Node | null | undefined }) {
  if (!layout) return null
  return <RenderNode node={layout} />
}

export { Children as RenderChildren }
