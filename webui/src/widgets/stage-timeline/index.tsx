/**
 * `stage-timeline`: when a stage-scheduled LoRA applies, on the sampling
 * timeline. The three stage windows follow the generation form's Shift; the
 * panel's own window has draggable start / peak / end handles (dragging an
 * auto-timed window switches auto timing off, keeping the window shown). A
 * template's parts are drawn as the run applies them, blocked windows
 * hatched. The template bar copies the panel's template name (for per-LoRA
 * template lines), explains its parts and opens the template editor (combine
 * base presets, save the panel, rename, delete) through the panel's actions.
 *
 * props: { panel: "base" | "hr", template_param, shift: string[], steps: string[],
 *          strength_max, curves: {value,label,shape}[], stages: {value,label,kind}[],
 *          builder: {value,label,stage,mode}[], templates: { [name]: Template } }
 *   `shift` / `steps`: param references, the first set one wins; the
 *   templates are refreshed by the actions' `__outputs__.templates`.
 * value: { auto, start, end, peak, strength, curve }
 */
import { Copy, HelpCircle, Library } from "@/components/icons"
import { useState } from "react"
import { useTranslation } from "react-i18next"
import { toast } from "sonner"
import { Segmented } from "@/components/Segmented"
import { Button } from "@/components/ui/button"
import { NumberField } from "@/components/ui/number-field"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { Switch } from "@/components/ui/switch"
import { labelId } from "@/ui/a11y"
import type { WidgetProps } from "@/ui/widgets"
import { registerWidgetCatalogs } from "../_lib/i18n"
import { useFirstNumber, useShift, useOwnerOutput, useSibling } from "../_anima/params"
import { flowSigma, fmt, type StageKind, type Window } from "../_anima/schedule"
import { TemplateEditor } from "./TemplateEditor"
import { Timeline } from "./Timeline"
import { type Curve, factorAt, resolve, type Stage, type Template, type Timing, timingOf } from "./model"

const NS = registerWidgetCatalogs("stage-timeline", import.meta.glob("./locales/*.json", { eager: true, import: "default" }))

type BuilderPreset = { value: string; label: string; stage: string; mode: string }
type Props = {
  panel?: string
  template_param?: string
  shift?: string[]
  steps?: string[]
  strength_max?: number
  curves?: Curve[]
  stages?: Stage[]
  builder?: BuilderPreset[]
  templates?: Record<string, Template>
}

type Axis = "progress" | "steps" | "sigma"
const round4 = (value: number) => Number(value.toFixed(4))

export default function StageTimeline({ value, onChange, props, disabled, id, label }: WidgetProps<Timing | null, Props>) {
  const { t: tw } = useTranslation(NS)
  const curves = props.curves ?? []
  const stages = props.stages ?? []
  const timing = timingOf(value, curves[0]?.value ?? "")
  const shift = useShift(props.shift)
  const steps = Math.round(useFirstNumber(props.steps, 20, (v) => v >= 1))
  const templateName = useSibling<string>(props.template_param) ?? ""
  const templates = useOwnerOutput<Record<string, Template>>("templates") ?? props.templates ?? {}
  const template = templates[templateName]
  const resolved = resolve(timing, template, shift, stages, curves)
  const [axis, setAxis] = useState<Axis>("progress")
  const [editor, setEditor] = useState(false)

  const stageLabel = (kind: StageKind) => tw(stages.find((stage) => stage.kind === kind)?.label ?? kind)
  const stageLabels = { composition: stageLabel("composition"), character: stageLabel("character"), style: stageLabel("style") }
  const tick = (at: number) => (axis === "steps" ? tw("step {{step}}", { step: Math.round(at * Math.max(steps - 1, 0)) + 1 }) : axis === "sigma" ? `σ ${fmt(flowSigma(at, shift))}` : fmt(at))
  const partsRun = resolved.parts.length > 0
  const bans = partsRun ? resolved.parts.filter((part) => part.mode === "ban").map((part) => part.window) : resolved.banWindow ? [resolved.window] : []

  const set = (patch: Partial<Timing>) => onChange({ ...timing, ...patch })
  /** A window edit: switches auto timing off from the window shown. */
  const setWindow = (next: Window) => onChange({ ...timing, auto: false, start: round4(next.start), end: round4(next.end), peak: round4(next.peak) })
  const toggleAuto = (auto: boolean) =>
    auto ? set({ auto }) : onChange({ ...timing, auto, start: round4(resolved.window.start), end: round4(resolved.window.end), peak: round4(resolved.window.peak) })

  const summary = tw("Strength over sampling: window {{start}}-{{end}}, peak {{peak}}, strength {{strength}}, Shift {{shift}}", {
    start: fmt(resolved.window.start),
    end: fmt(resolved.window.end),
    peak: fmt(resolved.window.peak),
    strength: fmt(resolved.strength),
    shift: fmt(shift),
  })

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(templateName)
      toast.success(tw("Copied {{name}}", { name: templateName }))
    } catch {
      toast.error(tw("Could not copy to the clipboard"))
    }
  }

  return (
    <div id={id} tabIndex={-1} className="flex flex-col gap-2 outline-none" role="group" aria-labelledby={label ? labelId(id) : undefined} aria-label={label ? undefined : tw("Stage timing")}>
      <div className="flex flex-wrap items-center gap-1.5 text-xs">
        <span className="text-muted-foreground">{tw("Template")}</span>
        <span className="max-w-60 truncate font-medium" title={templateName}>
          {template ? tw(template.label) : templateName || "—"}
        </span>
        <Button variant="ghost" size="icon-xs" aria-label={tw("Copy the template name")} title={tw("Copy the template name")} disabled={!templateName} onClick={() => void copy()}>
          <Copy />
        </Button>
        <TemplateHelp template={template} name={templateName} resolved={resolved} shift={shift} stages={stages} />
        <Button variant="outline" size="xs" disabled={disabled} onClick={() => setEditor(true)}>
          <Library />
          {tw("Templates…")}
        </Button>
        <span className="ml-auto text-muted-foreground tabular-nums">{tw("Shift {{shift}}", { shift: fmt(shift) })}</span>
        <Segmented
          size="sm"
          aria-label={tw("Timeline axis")}
          options={[
            { value: "progress", label: tw("Progress") },
            { value: "steps", label: tw("Steps") },
            { value: "sigma", label: tw("Sigma") },
          ]}
          value={axis}
          onValueChange={setAxis}
        />
      </div>
      <Timeline
        shift={shift}
        stageLabels={stageLabels}
        factor={(x) => factorAt(resolved, x, curves)}
        bans={bans}
        window={resolved.window}
        editable={!partsRun}
        disabled={disabled}
        handleLabels={{ start: tw("Start ratio"), peak: tw("Peak ratio"), end: tw("End ratio") }}
        onWindow={setWindow}
        tick={tick}
        summary={summary}
      />
      {partsRun ? <p className="text-xs text-muted-foreground">{tw("Auto timing runs the template's parts, each on its stage window and with its own curve.")}</p> : null}
      {resolved.banWindow ? <p className="text-xs text-muted-foreground">{tw("This template only blocks stages: the window blocks the LoRA.")}</p> : null}
      <div className="flex flex-wrap items-end gap-3">
        <label className="flex items-center gap-2 text-sm">
          <Switch checked={timing.auto} disabled={disabled} onCheckedChange={toggleAuto} />
          {tw("Auto timing from Shift")}
        </label>
        {(["start", "end", "peak"] as const).map((key) => (
          <Ratio
            key={key}
            label={key === "start" ? tw("Start ratio") : key === "end" ? tw("End ratio") : tw("Peak ratio")}
            value={resolved.window[key]}
            disabled={disabled || timing.auto}
            onCommit={(next) => setWindow({ ...resolved.window, [key]: next })}
          />
        ))}
        <Ratio label={tw("Stage strength multiplier")} value={timing.strength} max={props.strength_max ?? 2} disabled={disabled} onCommit={(strength) => set({ strength })} />
        <label className="flex flex-col gap-1 text-xs text-muted-foreground">
          {tw("Strength curve")}
          <select
            className="h-7 rounded-lg border border-input bg-transparent px-2 text-sm text-foreground outline-none focus-visible:ring-3 focus-visible:ring-ring/50 disabled:opacity-50 dark:bg-input/30"
            value={timing.curve}
            disabled={disabled || partsRun}
            onChange={(event) => set({ curve: event.target.value })}
          >
            {curves.map((curve) => (
              <option key={curve.value} value={curve.value}>
                {tw(curve.label)}
              </option>
            ))}
          </select>
        </label>
      </div>
      <TemplateEditor
        open={editor}
        onOpenChange={setEditor}
        panel={props.panel ?? "base"}
        templateName={templateName}
        templates={templates}
        builder={props.builder ?? []}
        shift={shift}
        stages={stages}
        curves={curves}
      />
    </div>
  )
}

function Ratio(props: { label: string; value: number; max?: number; disabled?: boolean; onCommit: (value: number) => void }) {
  const max = props.max ?? 1
  return (
    <label className="flex flex-col gap-1 text-xs text-muted-foreground">
      {props.label}
      <NumberField
        aria-label={props.label}
        size="sm"
        steppers={false}
        className="w-20"
        inputClassName="text-right"
        value={round4(props.value)}
        min={0}
        max={max}
        step={0.01}
        format={{ maximumFractionDigits: 4, useGrouping: false }}
        disabled={props.disabled}
        onValueCommitted={(next) => next !== null && Number.isFinite(next) && props.onCommit(Math.min(Math.max(next, 0), max))}
      />
    </label>
  )
}

function TemplateHelp(props: { template: Template | undefined; name: string; resolved: ReturnType<typeof resolve>; shift: number; stages: readonly Stage[] }) {
  const { t: tw } = useTranslation(NS)
  const parts = props.template?.parts ?? []
  const stageName = (stage: string) => tw(props.stages.find((item) => item.value === stage)?.label ?? stage)
  return (
    <Popover>
      <PopoverTrigger render={<Button variant="ghost" size="icon-xs" aria-label={tw("About this template")} title={tw("About this template")} />}>
        <HelpCircle />
      </PopoverTrigger>
      <PopoverContent aria-label={tw("About this template")} className="w-80 text-xs">
        <p className="font-medium">{props.template ? tw(props.template.label) : props.name}</p>
        {parts.length ? (
          <ul className="mt-1 flex flex-col gap-1">
            {parts.map((part, i) => {
              const window = props.resolved.parts[i]?.window
              return (
                <li key={i}>
                  {tw("{{stage}}: {{action}}", { stage: stageName(part.stage), action: part.mode === "ban" ? tw("Block") : tw("Apply") })}
                  {window ? ` · ${fmt(window.start)}–${fmt(window.end)}` : null}
                  {part.mode === "ban" ? ` · ${tw("0 in window")}` : ` · ${tw("strength {{strength}}", { strength: fmt(part.strength) })}`}
                </li>
              )
            })}
          </ul>
        ) : (
          <p className="mt-1">{tw("No parts: the panel's window applies, on the {{stage}} window with auto timing.", { stage: stageName(props.template?.stage ?? "") })}</p>
        )}
        <ul className="mt-2 list-disc pl-4 text-muted-foreground">
          <li>{tw("Apply: the LoRA is active only in that stage. Block: it is 0 in that stage.")}</li>
          <li>{tw("Auto timing moves every window with the Shift ({{shift}} now).", { shift: fmt(props.shift) })}</li>
          <li>{tw("Start and end are sampling progress (0-1); the peak is where the curve is strongest; the stage strength multiplies the LoRA strength.")}</li>
          <li>{tw("Per-LoRA template lines name templates exactly: copy a name with the copy button.")}</li>
        </ul>
      </PopoverContent>
    </Popover>
  )
}

