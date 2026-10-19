import { useTranslation } from "react-i18next"
import { FitTabsList } from "@/components/FitTabsList"
import { Tabs, TabsContent, TabsTrigger } from "@/components/ui/tabs"
import { evaluate } from "../conditions"
import { EnabledProvider } from "../context"
import { tabsKey } from "../form-index"
import { useCondition, useForm, useFormState, useOwner, useParam } from "../hooks"
import { RenderChildren } from "../Renderer"
import type { InputNode, TabNode, TabsNode } from "../schema"

/** Tab ids: the bound choice value, else the tab's index. */
function tabId(tab: TabNode, index: number, bound: boolean): string | number {
  return bound && tab.value !== undefined && tab.value !== null ? tab.value : String(index)
}

/** Which tabs pass their `visible_if`, as a stable string of 0/1 per tab. */
function useVisibleTabs(node: TabsNode): boolean[] {
  const owner = useOwner()
  const mask = useFormState((s) =>
    node.children.map((tab) => (evaluate(tab.visible_if, owner, (o, n) => s.values[o]?.[n]) ? "1" : "0")).join(""),
  )
  return [...mask].map((bit) => bit === "1")
}

/** Which tabs pass their `enabled_if` (same encoding as `useVisibleTabs`). */
function useEnabledTabs(node: TabsNode): boolean[] {
  const owner = useOwner()
  const mask = useFormState((s) =>
    node.children.map((tab) => (evaluate(tab.enabled_if, owner, (o, n) => s.values[o]?.[n]) ? "1" : "0")).join(""),
  )
  return [...mask].map((bit) => bit === "1")
}

function TabTrigger({ tab, value }: { tab: TabNode; value: string | number }) {
  const { t } = useTranslation()
  const enabled = useCondition(tab.enabled_if)
  return (
    <TabsTrigger value={value} disabled={!enabled} className="flex-none px-2.5">
      {t(tab.title)}
    </TabsTrigger>
  )
}

/** A tab's content; `enabled_if` disables everything inside, like on any layout node. */
function TabPanel({ tab, value }: { tab: TabNode; value: string | number }) {
  const enabled = useCondition(tab.enabled_if)
  return (
    <TabsContent value={value} className="flex flex-col gap-3">
      <EnabledProvider enabled={enabled}>
        <RenderChildren nodes={tab.children} />
      </EnabledProvider>
    </TabsContent>
  )
}

/** Hidden tabs (`visible_if` false) have neither a trigger nor content. */
function TabStrip(props: {
  node: TabsNode
  value: string | number | null
  onValueChange: (value: string | number) => void
  bound: boolean
  /** Names the strip (and its compact select); already translated. */
  label: string
}) {
  const { t } = useTranslation()
  const { node, bound } = props
  const visible = useVisibleTabs(node)
  const enabled = useEnabledTabs(node)
  const shown = node.children.map((tab, i) => ({ tab, id: tabId(tab, i, bound), enabled: enabled[i] })).filter((_, i) => visible[i])
  if (shown.length === 0) return null
  return (
    <Tabs value={props.value} onValueChange={(v) => props.onValueChange(v as string | number)} className="gap-3">
      {/* a strip wider than its pane becomes a select (never scrolls, wraps or cuts a label) */}
      <FitTabsList
        aria-label={props.label}
        options={shown.map(({ tab, id, enabled }) => ({ value: id, label: t(tab.title), disabled: !enabled }))}
        value={props.value}
        onValueChange={props.onValueChange}
        className="w-fit max-w-full"
      >
        {shown.map(({ tab, id }) => (
          <TabTrigger key={String(id)} tab={tab} value={id} />
        ))}
      </FitTabsList>
      {shown.map(({ tab, id }) => (
        <TabPanel key={String(id)} tab={tab} value={id} />
      ))}
    </Tabs>
  )
}

function BoundTabs({ node, param }: { node: TabsNode; param: InputNode }) {
  const { t } = useTranslation()
  const binding = useParam<string | number>(param)
  return <TabStrip node={node} bound label={t(binding.label) || t("Tabs")} value={binding.value ?? null} onValueChange={binding.setValue} />
}

/** An unbound strip keeps its selection in the form store; a hidden selection falls back to the first visible tab. */
function UnboundTabs({ node }: { node: TabsNode }) {
  const { store } = useForm()
  const owner = useOwner()
  const key = tabsKey(owner, node)
  const selected = useFormState((s) => s.tabs[key] ?? "0")
  const visible = useVisibleTabs(node)
  const value = visible[Number(selected)] ? selected : String(Math.max(0, visible.indexOf(true)))
  const { t } = useTranslation()
  return <TabStrip node={node} bound={false} label={t("Tabs")} value={value} onValueChange={(v) => store.getState().setTab(key, String(v))} />
}

/** `tabs` node; with `bind`, the selected tab is the value of that choice param. */
export function TabsView({ node }: { node: TabsNode }) {
  const owner = useOwner()
  const bound = useFormState((s) => (node.bind ? s.index.inputs.get(owner)?.get(node.bind) : undefined))
  return bound ? <BoundTabs node={node} param={bound} /> : <UnboundTabs node={node} />
}
