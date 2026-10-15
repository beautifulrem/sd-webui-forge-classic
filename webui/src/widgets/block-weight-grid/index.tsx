/**
 * `block-weight-grid`: an Anima LoRA weight text edited as a grid. Blocks
 * (28/40/52-block layouts, depth bands, presets) and Qwen text-encoder
 * layers are a bar chart you paint with the pointer or adjust per bar from
 * the keyboard; module weights are one slider per module. "Text" shows the
 * weight text itself, which is the value: the run, the infotext and the
 * presets read exactly what is written there.
 *
 * props: { mode: "blocks" | "layers" | "modules",
 *          layouts?: number[], mappings?: { [layout]: number[] }   // blocks: source 2B block per block
 *          bands?: { label, start, end }[], presets?: { label, value }[],
 *          count?: number (layers), modules?: string[], placeholder?: string,
 *          minimum?: number, maximum?: number, step?: number }
 * value: string (the weight text)
 */
import { useState } from "react"
import { useTranslation } from "react-i18next"
import { Segmented } from "@/components/Segmented"
import { Button } from "@/components/ui/button"
import { NumberField } from "@/components/ui/number-field"
import { Slider } from "@/components/ui/slider"
import { Textarea } from "@/components/ui/textarea"
import { labelId } from "@/ui/a11y"
import type { WidgetProps } from "@/ui/widgets"
import { registerWidgetCatalogs } from "../_lib/i18n"
import { WeightChart } from "./WeightChart"
import { formatWeight, gridOf, type LayerWeights, parseLayerWeights, parseModuleWeights, serializeLayout, serializeModuleWeights } from "./weights"

const NS = registerWidgetCatalogs("block-weight-grid", import.meta.glob("./locales/*.json", { eager: true, import: "default" }))

type Band = { label: string; start: number; end: number }
type Preset = { label: string; value: string }
type Props = {
  mode?: "blocks" | "layers" | "modules"
  layouts?: number[]
  mappings?: Record<string, number[]>
  bands?: Band[]
  presets?: Preset[]
  count?: number
  modules?: string[]
  placeholder?: string
  minimum?: number
  maximum?: number
  step?: number
}

const MODULES = ["self_attn", "cross_attn", "mlp", "adaln", "norm"]

/** The smallest layout holding every written index (a 40-block LoRA's 39 needs 40). */
function layoutFor(indexes: Iterable<number>, layouts: readonly number[]): number {
  const deepest = Math.max(-1, ...indexes)
  return layouts.find((size) => deepest < size) ?? layouts.at(-1) ?? 28
}

export default function BlockWeightGrid({ value, onChange, props, disabled, id, label }: WidgetProps<string | null, Props>) {
  const { t: tw } = useTranslation(NS)
  const [view, setView] = useState<"grid" | "text">("grid")
  const text = typeof value === "string" ? value : ""
  const mode = props.mode ?? "blocks"

  return (
    <div className="flex flex-col gap-2" role="group" aria-labelledby={label ? labelId(id) : undefined} aria-label={label ? undefined : tw("Weights")}>
      <div className="flex flex-wrap items-center gap-2">
        <Segmented
          size="sm"
          aria-label={tw("Edit as")}
          options={[
            { value: "grid", label: tw("Grid") },
            { value: "text", label: tw("Text") },
          ]}
          value={view}
          onValueChange={setView}
        />
        {props.presets?.length && view === "grid" ? (
          <div className="flex flex-wrap items-center gap-1" role="group" aria-label={tw("Presets")}>
            {props.presets.map((preset) => (
              <Button key={preset.label} variant="outline" size="xs" disabled={disabled} onClick={() => onChange(preset.value)} title={preset.value}>
                {tw(preset.label)}
              </Button>
            ))}
          </div>
        ) : null}
      </div>
      {view === "text" ? (
        <Textarea
          id={id}
          value={text}
          rows={mode === "modules" ? 4 : 3}
          spellCheck={false}
          className="font-mono text-xs"
          placeholder={props.placeholder ? tw(props.placeholder) : undefined}
          disabled={disabled}
          onChange={(event) => onChange(event.target.value)}
        />
      ) : mode === "modules" ? (
        <ModuleGrid id={id} text={text} onChange={onChange} props={props} disabled={disabled} />
      ) : (
        <LayerGrid id={id} text={text} onChange={onChange} props={props} disabled={disabled} />
      )}
    </div>
  )
}

type GridProps = { id: string; text: string; onChange: (text: string) => void; props: Props; disabled: boolean }

function Ignored({ entries }: { entries: string[] }) {
  const { t: tw } = useTranslation(NS)
  if (!entries.length) return null
  return (
    <p className="text-xs text-warning" role="note">
      {tw("Not valid weights (a run ignores them; editing the grid drops them): {{entries}}", { entries: entries.join(", ") })}
    </p>
  )
}

function Kept({ entries }: { entries: string[] }) {
  const { t: tw } = useTranslation(NS)
  if (!entries.length) return null
  return (
    <p className="text-xs text-muted-foreground" role="note">
      {tw("Not shown in the grid (a run applies them; editing the grid keeps them): {{entries}}", { entries: entries.join(", ") })}
    </p>
  )
}

/** The entries the text has from `start` on, as runs ("28-31=0.5"). */
function entriesFrom(parsed: LayerWeights, start: number): string[] {
  const indexes = [...parsed.values.keys()].filter((index) => index >= start).sort((a, b) => a - b)
  const runs: string[] = []
  for (let i = 0; i < indexes.length; ) {
    const value = parsed.values.get(indexes[i]!)!
    let j = i
    while (j + 1 < indexes.length && indexes[j + 1] === indexes[j]! + 1 && parsed.values.get(indexes[j + 1]!) === value) j++
    runs.push(`${i === j ? indexes[i] : `${indexes[i]}-${indexes[j]}`}=${formatWeight(value)}`)
    i = j + 1
  }
  return runs
}

/** The value range a grid shows: the props' (default 0-2), widened to every weight so no edit clamps one. */
function rangeOf(props: Props, values: readonly number[]): [number, number] {
  return [Math.min(props.minimum ?? 0, ...values), Math.max(props.maximum ?? 2, ...values)]
}

function LayerGrid({ id, text, onChange, props, disabled }: GridProps) {
  const { t: tw } = useTranslation(NS)
  const blocks = (props.mode ?? "blocks") === "blocks"
  const parsed = parseLayerWeights(text)
  const layouts = blocks ? (props.layouts?.length ? props.layouts : [28]) : [props.count ?? 28]
  const [chosen, setChosen] = useState<number | null>(null)
  const layout = chosen ?? layoutFor(parsed.values.keys(), layouts)
  const weights = gridOf(parsed, layout)
  const [selected, setSelected] = useState(0)
  const at = Math.min(selected, layout - 1)
  const mapping = blocks && layout !== 28 ? props.mappings?.[String(layout)] : undefined
  const sourceOf = (index: number) => mapping?.[index] ?? index
  const bands = blocks ? (props.bands ?? []) : []
  const bandOf = (index: number) => bands.findIndex((band) => sourceOf(index) >= band.start && sourceOf(index) <= band.end)
  const [min, max] = rangeOf(props, [...weights, parsed.fallback])
  const step = props.step ?? 0.05

  const commit = (next: readonly number[], fallback = parsed.fallback) => onChange(serializeLayout(parsed, next, fallback, !blocks))
  const setOne = (index: number, next: number) => commit(weights.map((weight, i) => (i === index ? next : weight)))
  const itemLabel = (index: number) => (blocks ? tw("Block {{index}}", { index }) : tw("Layer {{index}}", { index }))

  return (
    <div className="flex flex-col gap-2">
      {blocks && layouts.length > 1 ? (
        <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
          <span id={`${id}-layout`}>{tw("LoRA depth")}</span>
          <Segmented
            size="sm"
            aria-labelledby={`${id}-layout`}
            options={layouts.map((size) => ({ value: size, label: tw("{{size}} blocks", { size }) }))}
            value={layout}
            onValueChange={setChosen}
          />
          {mapping ? <span>{tw("Inserted copies are marked; a 28-block LoRA applies each weight to its copies too.")}</span> : null}
        </div>
      ) : null}
      <WeightChart
        id={id}
        weights={weights}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        selected={at}
        onSelect={setSelected}
        onCommit={commit}
        bandOf={bandOf}
        copyOf={(index) => (mapping && index > 0 && mapping[index] === mapping[index - 1] ? mapping[index]! : null)}
        itemLabel={itemLabel}
      />
      <div className="flex flex-wrap items-end gap-3">
        <WeightInput label={itemLabel(at)} value={weights[at] ?? parsed.fallback} step={step} disabled={disabled} onCommit={(next) => setOne(at, next)} />
        <WeightInput label={tw("All others")} value={parsed.fallback} step={step} disabled={disabled} onCommit={(next) => commit(weights.map((weight) => (weight === parsed.fallback ? next : weight)), next)} />
        {bands.map((band, b) => {
          const members = weights.flatMap((_, i) => (bandOf(i) === b ? [i] : []))
          const shared = members.every((i) => weights[i] === weights[members[0]!]) ? weights[members[0]!] : undefined
          return (
            <WeightInput
              key={band.label}
              label={tw("{{band}} ({{start}}-{{end}})", { band: tw(band.label), start: band.start, end: band.end })}
              value={shared ?? null}
              step={step}
              disabled={disabled || !members.length}
              onCommit={(next) => commit(weights.map((weight, i) => (bandOf(i) === b ? next : weight)))}
              band={b}
            />
          )
        })}
      </div>
      <Kept entries={entriesFrom(parsed, layout)} />
      <Ignored entries={parsed.ignored} />
    </div>
  )
}

const BAND_SWATCH = ["bg-sky-500", "bg-violet-500", "bg-amber-500", "bg-emerald-500"]

function WeightInput(props: { label: string; value: number | null; step: number; disabled: boolean; onCommit: (value: number) => void; band?: number }) {
  return (
    <label className="flex flex-col gap-1 text-xs text-muted-foreground">
      <span className="inline-flex items-center gap-1.5">
        {props.band !== undefined ? <span aria-hidden className={`size-2 rounded-full ${BAND_SWATCH[props.band % BAND_SWATCH.length]}`} /> : null}
        {props.label}
      </span>
      <NumberField
        aria-label={props.label}
        size="sm"
        steppers={false}
        className="w-20"
        inputClassName="text-right"
        value={props.value}
        step={props.step}
        format={{ maximumFractionDigits: 4, useGrouping: false }}
        disabled={props.disabled}
        onValueCommitted={(next) => next !== null && Number.isFinite(next) && props.onCommit(next)}
      />
    </label>
  )
}

function ModuleGrid({ id, text, onChange, props, disabled }: GridProps) {
  const { t: tw } = useTranslation(NS)
  const modules = props.modules?.length ? props.modules : MODULES
  const parsed = parseModuleWeights(text, modules)
  const [min, max] = rangeOf(props, Object.values(parsed.values))
  const step = props.step ?? 0.05
  const set = (module: string, next: number) => onChange(serializeModuleWeights({ values: { ...parsed.values, [module]: next }, written: [...parsed.written, module], kept: parsed.kept }, modules))

  return (
    <div id={id} className="flex flex-col gap-2">
      <div className="grid grid-cols-[minmax(6rem,auto)_1fr_auto] items-center gap-x-3 gap-y-2">
        {modules.map((module) => (
          <ModuleRow key={module} module={module} value={parsed.values[module] ?? 1} min={min} max={max} step={step} disabled={disabled} onChange={(next) => set(module, next)} />
        ))}
      </div>
      <Kept entries={parsed.kept} />
      <Ignored entries={parsed.ignored} />
      <p className="text-xs text-muted-foreground">{tw("A module weight multiplies the block weight of every block.")}</p>
    </div>
  )
}

function ModuleRow(props: { module: string; value: number; min: number; max: number; step: number; disabled: boolean; onChange: (value: number) => void }) {
  return (
    <>
      <code className="text-xs">{props.module}</code>
      <Slider
        value={props.value}
        min={props.min}
        max={props.max}
        step={props.step}
        disabled={props.disabled}
        getAriaLabel={() => props.module}
        onValueChange={(next) => props.onChange(Array.isArray(next) ? next[0]! : next)}
      />
      <NumberField
        aria-label={props.module}
        size="sm"
        steppers={false}
        className="w-20"
        inputClassName="text-right"
        value={props.value}
        step={props.step}
        format={{ maximumFractionDigits: 4, useGrouping: false }}
        disabled={props.disabled}
        onValueCommitted={(next) => next !== null && Number.isFinite(next) && props.onChange(next)}
      />
    </>
  )
}
