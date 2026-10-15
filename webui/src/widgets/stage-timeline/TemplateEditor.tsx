/**
 * The stage-timeline widget's template editor (a dialog): combine base
 * presets into an auto-timing template (previewed on the timeline at a
 * chosen Shift), save the panel as a template, rename or delete custom
 * templates. Every change runs a panel action (`template_combo`,
 * `template_save`, `template_rename`, `template_delete`), which reports in a
 * toast and refreshes both panels' template choices.
 */
import { useState } from "react"
import { useTranslation } from "react-i18next"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { NumberField } from "@/components/ui/number-field"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import type { ActionExtra } from "@/api/types"
import { useRunAction } from "@/ui/hooks"
import { fmt, stageWindow, type StageKind } from "../_anima/schedule"
import { Timeline } from "./Timeline"
import { type Curve, factorAt, kindOf, type Stage, type Template, resolve } from "./model"

const NS = "widget:stage-timeline"

/** The arguments of a panel action (ActionExtra types the core actions' only). */
const args = (extra: Record<string, unknown>) => extra as unknown as ActionExtra

type BuilderPreset = { value: string; label: string; stage: string; mode: string }

type Props = {
  open: boolean
  onOpenChange: (open: boolean) => void
  panel: string
  templateName: string
  templates: Record<string, Template>
  builder: readonly BuilderPreset[]
  shift: number
  stages: readonly Stage[]
  curves: readonly Curve[]
}

export function TemplateEditor(props: Props) {
  const { t: tw } = useTranslation(NS)
  return (
    <Dialog open={props.open} onOpenChange={props.onOpenChange}>
      <DialogContent className="sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>{tw("Stage templates")}</DialogTitle>
          <DialogDescription>{tw("Templates are shared by the base and Hires. fix panels and saved in the extension's data folder.")}</DialogDescription>
        </DialogHeader>
        <Tabs defaultValue="combine">
          <TabsList>
            <TabsTrigger value="combine">{tw("Combine presets")}</TabsTrigger>
            <TabsTrigger value="manage">{tw("Manage templates")}</TabsTrigger>
          </TabsList>
          <TabsContent value="combine">
            <Combine {...props} />
          </TabsContent>
          <TabsContent value="manage">
            <Manage {...props} />
          </TabsContent>
        </Tabs>
      </DialogContent>
    </Dialog>
  )
}

function Combine(props: Props) {
  const { t: tw } = useTranslation(NS)
  const run = useRunAction()
  const current = props.templates[props.templateName]
  const initial = props.builder.filter((preset) => current?.parts.some((part) => part.stage === preset.stage && part.mode === preset.mode)).map((preset) => preset.value)
  const [selected, setSelected] = useState<string[]>(initial.length ? initial : props.builder.slice(2, 3).map((preset) => preset.value))
  const [name, setName] = useState(current && !current.builtin ? props.templateName : "")
  const [shift, setShift] = useState(props.shift)
  const [busy, setBusy] = useState(false)

  const chosen = props.builder.filter((preset) => selected.includes(preset.value))
  const conflict = chosen.find((preset) => chosen.some((other) => other.stage === preset.stage && other.mode !== preset.mode))
  const template: Template = { label: name, builtin: false, auto: true, stage: chosen.find((preset) => preset.mode === "allow")?.stage ?? chosen[0]?.stage ?? "", curve: null, parts: chosen.map((preset) => ({ stage: preset.stage, mode: preset.mode, curve: "", strength: 1 })) }
  const resolved = resolve({ auto: true, start: 0, end: 1, peak: 0.5, strength: 1, curve: "" }, template, shift, props.stages, props.curves)
  const stageLabel = (kind: StageKind) => tw(props.stages.find((stage) => stage.kind === kind)?.label ?? kind)
  const stageName = (stage: string) => tw(props.stages.find((item) => item.value === stage)?.label ?? stage)

  const save = async () => {
    setBusy(true)
    await run("template_combo", args({ name: name.trim(), presets: selected }))
    setBusy(false)
  }

  return (
    <div className="flex flex-col gap-3 pt-2">
      <fieldset className="grid gap-1.5 sm:grid-cols-2">
        <legend className="mb-1 text-xs text-muted-foreground">{tw("Base presets")}</legend>
        {props.builder.map((preset) => (
          <label key={preset.value} className="flex items-center gap-2 text-sm">
            <Checkbox
              checked={selected.includes(preset.value)}
              onCheckedChange={(checked) => setSelected((list) => (checked ? [...list, preset.value] : list.filter((item) => item !== preset.value)))}
            />
            {tw(preset.label)}
          </label>
        ))}
      </fieldset>
      <div className="flex flex-wrap items-end gap-3">
        <label className="flex flex-col gap-1 text-xs text-muted-foreground">
          {tw("Preview Shift")}
          <NumberField aria-label={tw("Preview Shift")} size="sm" steppers={false} className="w-20" value={shift} min={1} max={24} step={0.5} onValueCommitted={(next) => next !== null && setShift(next)} />
        </label>
      </div>
      <Timeline
        shift={shift}
        stageLabels={{ composition: stageLabel("composition"), character: stageLabel("character"), style: stageLabel("style") }}
        factor={(x) => factorAt(resolved, x, props.curves)}
        bans={resolved.parts.filter((part) => part.mode === "ban").map((part) => part.window)}
        window={resolved.window}
        editable={false}
        tick={(at) => fmt(at)}
        summary={tw("Preview of the combined template at Shift {{shift}}", { shift: fmt(shift) })}
      />
      <ul className="text-xs text-muted-foreground">
        {chosen.map((preset) => {
          const window = stageWindow(kindOf(preset.stage, props.stages), shift)
          return (
            <li key={preset.value}>
              {tw("{{stage}}: {{action}}", { stage: stageName(preset.stage), action: preset.mode === "ban" ? tw("Block") : tw("Apply") })} · {fmt(window.start)}–{fmt(window.end)}
            </li>
          )
        })}
      </ul>
      {conflict ? (
        <p role="alert" className="text-xs text-destructive">
          {tw("{{stage}} cannot be both applied and blocked.", { stage: stageName(conflict.stage) })}
        </p>
      ) : null}
      <div className="flex flex-wrap items-end gap-2">
        <label className="flex min-w-48 flex-1 flex-col gap-1 text-xs text-muted-foreground">
          {tw("Template name")}
          <Input value={name} onChange={(event) => setName(event.target.value)} />
        </label>
        <Button disabled={busy || !name.trim() || !chosen.length || Boolean(conflict)} onClick={() => void save()}>
          {tw("Save combo template")}
        </Button>
      </div>
    </div>
  )
}

function Manage(props: Props) {
  const { t: tw } = useTranslation(NS)
  const run = useRunAction()
  const custom = Object.entries(props.templates).filter(([, template]) => !template.builtin).map(([name]) => name)
  const [chosen, setChosen] = useState<string>("")
  const target = custom.includes(chosen) ? chosen : (custom.includes(props.templateName) ? props.templateName : custom[0]) ?? ""
  const [renameTo, setRenameTo] = useState("")
  const [saveName, setSaveName] = useState("")
  const [auto, setAuto] = useState(true)
  const [busy, setBusy] = useState(false)

  const act = async (action: string, extra: Record<string, unknown>, after?: () => void) => {
    setBusy(true)
    const response = await run(action, args(extra))
    setBusy(false)
    if (response?.toast?.level === "success") after?.()
  }

  return (
    <div className="flex flex-col gap-4 pt-2">
      <section className="flex flex-col gap-2">
        <h3 className="text-sm font-medium">{tw("Save this panel as a template")}</h3>
        <div className="flex flex-wrap items-end gap-2">
          <label className="flex min-w-48 flex-1 flex-col gap-1 text-xs text-muted-foreground">
            {tw("New template name")}
            <Input value={saveName} onChange={(event) => setSaveName(event.target.value)} />
          </label>
          <label className="flex items-center gap-2 text-sm">
            <Checkbox checked={auto} onCheckedChange={(checked) => setAuto(Boolean(checked))} />
            {tw("Save as auto-timing template")}
          </label>
          <Button disabled={busy || !saveName.trim()} onClick={() => void act("template_save", { name: saveName.trim(), auto, panel: props.panel }, () => setSaveName(""))}>
            {tw("Save template")}
          </Button>
        </div>
      </section>
      <section className="flex flex-col gap-2">
        <h3 className="text-sm font-medium">{tw("Custom templates")}</h3>
        {custom.length ? (
          <div className="flex flex-wrap items-end gap-2">
            <label className="flex flex-col gap-1 text-xs text-muted-foreground">
              {tw("Existing template")}
              <select
                className="h-8 min-w-40 rounded-lg border border-input bg-transparent px-2 text-sm text-foreground outline-none focus-visible:ring-3 focus-visible:ring-ring/50 dark:bg-input/30"
                value={target}
                onChange={(event) => setChosen(event.target.value)}
              >
                {custom.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex min-w-40 flex-1 flex-col gap-1 text-xs text-muted-foreground">
              {tw("Rename to")}
              <Input value={renameTo} onChange={(event) => setRenameTo(event.target.value)} />
            </label>
            <Button variant="outline" disabled={busy || !renameTo.trim()} onClick={() => void act("template_rename", { old: target, new: renameTo.trim() }, () => setRenameTo(""))}>
              {tw("Rename template")}
            </Button>
            <Button variant="destructive" disabled={busy} onClick={() => void act("template_delete", { name: target })}>
              {tw("Delete template")}
            </Button>
          </div>
        ) : (
          <p className="text-xs text-muted-foreground">{tw("No custom templates yet: built-in templates cannot be renamed or deleted.")}</p>
        )}
      </section>
    </div>
  )
}
