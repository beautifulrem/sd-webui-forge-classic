/**
 * The scripts that have no slot of their own (always-on extensions) as a
 * searchable, collapsible list with an "only modified" filter and per-script
 * reset, followed by the selectable-script choice and the chosen script.
 */
import { Collapsible } from "@base-ui/react/collapsible"
import { ChevronRight, ListFilter, RotateCcw, Search } from "lucide-react"
import { useId, useState } from "react"
import { useTranslation } from "react-i18next"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Toggle } from "@/components/ui/toggle"
import { cn } from "@/lib/utils"
import { resetOwners, resetParams } from "@/ui/actions"
import { evaluate } from "@/ui/conditions"
import { OwnerProvider } from "@/ui/context"
import { DEFAULT_SLOT, inputsIn, scriptsForSlot } from "@/ui/form-index"
import { anyModified, type FormState } from "@/ui/form-store"
import { useForm, useFormState } from "@/ui/hooks"
import { RenderLayout, RenderNode } from "@/ui/Renderer"
import type { ChoiceNode, Node, ScriptSchema, TabSchema } from "@/ui/schema"

function scriptNames(script: ScriptSchema): string[] {
  return inputsIn(script.layout).map((input) => input.name)
}

/** Enabled = the section's enable toggle is on (scripts without one count as enabled when modified). */
function isActive(s: FormState, script: ScriptSchema): boolean {
  const layout = script.layout
  if (layout?.type === "section" && layout.enable) return Boolean(s.values[script.id]?.[layout.enable.name])
  return false
}

function ScriptCard({ script }: { script: ScriptSchema }) {
  const { t } = useTranslation()
  const { store, formId } = useForm()
  const names = scriptNames(script)
  const modified = useFormState((s) => anyModified(s, script.id, names))
  const visible = useFormState((s) => evaluate(script.visible_if, script.id, (o, n) => s.values[o]?.[n]))
  if (!visible || !script.layout) return null
  const layout = script.layout
  const titled = layout.type === "section" || (layout.type === "group" && Boolean(layout.title))
  const body: Node = layout.type === "group" && !titled ? { ...layout, type: "column" } : layout
  // Listed extensions start closed (legacy ones are long flat lists); the chosen selectable script stays open.
  const collapsed: Node | null = !titled && script.kind !== "selectable" ? { type: "section", title: script.title, open: false, children: [body] } : null
  return (
    <OwnerProvider owner={script.id}>
      <div data-script={script.id} className="min-w-0">
        {titled ? (
          <RenderLayout layout={layout} />
        ) : collapsed ? (
          <RenderLayout layout={collapsed} />
        ) : (
          <div role="group" aria-label={t(script.title)} className="flex flex-col gap-3 rounded-xl border bg-card p-3">
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium">{t(script.title)}</span>
              {modified ? (
                <Button
                  variant="ghost"
                  size="icon-xs"
                  className="ml-auto"
                  aria-label={t("Reset {{label}} to default", { label: t(script.title) })}
                  onClick={() => resetParams({ store, tab: formId }, script.id, names)}
                >
                  <RotateCcw />
                </Button>
              ) : null}
            </div>
            <RenderLayout layout={body} />
          </div>
        )}
      </div>
    </OwnerProvider>
  )
}

function matches(script: ScriptSchema, query: string, translate: (s: string) => string): boolean {
  if (!query) return true
  const haystack = [script.title, translate(script.title), ...inputsIn(script.layout).flatMap((input) => [input.label, translate(input.label)])]
  return haystack.some((text) => text.toLowerCase().includes(query))
}

export function ScriptsPanel({ schema, scriptChoice }: { schema: TabSchema; scriptChoice: ChoiceNode | null }) {
  const { t } = useTranslation()
  const id = useId()
  const { store, formId } = useForm()
  const slots = useFormState((s) => s.index.slots)
  const [query, setQuery] = useState("")
  const [onlyModified, setOnlyModified] = useState(false)
  const [open, setOpen] = useState(true)

  const listed = scriptsForSlot(schema.scripts, DEFAULT_SLOT, slots).filter((script) => script.kind !== "selectable" && script.layout)
  const selectable = schema.scripts.filter((script) => script.kind === "selectable")
  const modifiedIds = useFormState((s) => listed.filter((script) => anyModified(s, script.id, scriptNames(script))).map((script) => script.id).join(" "))
  const activeCount = useFormState((s) => listed.filter((script) => isActive(s, script)).length)
  const modifiedSet = new Set(modifiedIds.split(" ").filter(Boolean))
  const q = query.trim().toLowerCase()
  const shown = listed.filter((script) => (!onlyModified || modifiedSet.has(script.id)) && matches(script, q, t))

  if (listed.length === 0 && !scriptChoice) return null

  return (
    <div className="flex flex-col gap-3">
      {listed.length > 0 ? (
        <Collapsible.Root open={open} onOpenChange={setOpen} render={<section aria-labelledby={`${id}-title`} />} className="flex flex-col gap-2">
          <div className="flex items-center gap-2">
            <Collapsible.Trigger className="group/trigger flex min-w-0 items-center gap-1.5 rounded-md text-sm font-semibold outline-none focus-visible:ring-3 focus-visible:ring-ring/50">
              <ChevronRight aria-hidden className="size-4 text-muted-foreground transition-transform group-data-[panel-open]/trigger:rotate-90" />
              <span id={`${id}-title`}>{t("Extensions")}</span>
            </Collapsible.Trigger>
            <Badge variant="secondary" className="tabular-nums">
              {listed.length}
            </Badge>
            {activeCount > 0 ? <Badge className="tabular-nums">{t("{{count}} on", { count: activeCount })}</Badge> : null}
            {modifiedSet.size > 0 ? (
              <Button
                variant="ghost"
                size="xs"
                className="ml-auto text-muted-foreground"
                onClick={() =>
                  resetOwners(
                    { store, tab: formId },
                    listed.filter((script) => modifiedSet.has(script.id)).map((script) => script.id),
                  )
                }
              >
                <RotateCcw />
                {t("Reset all")}
              </Button>
            ) : null}
          </div>
          <Collapsible.Panel className="flex flex-col gap-2">
            <div className="flex items-center gap-1.5">
              <div className="relative min-w-0 flex-1">
                <Search aria-hidden className="pointer-events-none absolute top-1/2 left-2 size-3.5 -translate-y-1/2 text-muted-foreground" />
                <Input
                  type="search"
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder={t("Filter extensions…")}
                  aria-label={t("Filter extensions")}
                  className="h-8 pl-7"
                />
              </div>
              <Toggle
                variant="outline"
                pressed={onlyModified}
                onPressedChange={setOnlyModified}
                aria-label={t("Only modified")}
                title={t("Only modified")}
                className="gap-1"
              >
                <ListFilter />
                <span className="hidden sm:inline">{t("Modified")}</span>
              </Toggle>
            </div>
            <div className="flex flex-col gap-2">
              {shown.map((script) => (
                <ScriptCard key={script.id} script={script} />
              ))}
              {shown.length === 0 ? (
                <p className={cn("rounded-lg border border-dashed p-4 text-center text-xs text-muted-foreground")}>{t("No extension matches")}</p>
              ) : null}
            </div>
          </Collapsible.Panel>
        </Collapsible.Root>
      ) : null}

      {scriptChoice ? (
        <section aria-label={t("Script")} className="flex flex-col gap-3 rounded-xl border bg-card p-3">
          <OwnerProvider owner="core">
            <RenderNode node={scriptChoice} />
          </OwnerProvider>
          {selectable.map((script) => (
            <ScriptCard key={script.id} script={script} />
          ))}
        </section>
      ) : null}
    </div>
  )
}
