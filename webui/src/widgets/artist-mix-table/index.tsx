/**
 * `artist-mix-table`: one chain (base or Hires. fix) of the Artist Scheduled
 * Mixer as an editable, virtualized table, with a schedule overview: one lane
 * per artist row showing its strength over the sampling steps. Rows can be
 * added (up to `max_rows`), duplicated, reordered and removed. A row on a
 * stage with Auto Shift follows the generation form's Shift live (its window
 * is shown, not stored; the run re-times it from its own Shift).
 *
 * props: { chain, shift: string[], steps: string[], max_rows, optimization_param,
 *          default_blocks: { [optimization]: string }, max_artists: { [optimization]: number },
 *          curves: { value, label, shape }[], curve_floor, stages: { value, label, kind? }[] }
 * value: Row[] ({ enabled, artist, weight, blocks, start, end, peak, curve, stage, auto_shift })
 */
import { useVirtualizer } from "@tanstack/react-virtual"
import { ArrowDown, ArrowUp, CopyPlus, Plus, Trash2 } from "lucide-react"
import { useRef } from "react"
import { useTranslation } from "react-i18next"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { NumberField } from "@/components/ui/number-field"
import { Switch } from "@/components/ui/switch"
import { cn } from "@/lib/utils"
import { labelId } from "@/ui/a11y"
import type { WidgetProps } from "@/ui/widgets"
import { registerWidgetCatalogs } from "../_lib/i18n"
import { useFirstNumber, useShift, useSibling } from "../_anima/params"
import { curvePath, fmt, windowCurveFactor } from "../_anima/schedule"
import { artistNames, type Choices, isAuto, newRow, type Row, rowOf, windowOf, withAuto, withStage } from "./rows"

const NS = registerWidgetCatalogs("artist-mix-table", import.meta.glob("./locales/*.json", { eager: true, import: "default" }))

type Props = {
  chain?: string
  shift?: string[]
  steps?: string[]
  max_rows?: number
  optimization_param?: string
  default_blocks?: Record<string, string>
  max_artists?: Record<string, number>
  curves?: { value: string; label: string; shape?: string }[]
  curve_floor?: number
  stages?: { value: string; label: string; kind?: "composition" | "character" | "style" }[]
}

const ROW_HEIGHT = 92
const SELECT =
  "h-7 min-w-0 rounded-md border border-input bg-transparent px-1.5 text-xs text-foreground outline-none focus-visible:ring-3 focus-visible:ring-ring/50 disabled:opacity-50 dark:bg-input/30"

export default function ArtistMixTable({ value, onChange, props, disabled, id, label }: WidgetProps<unknown[] | null, Props>) {
  // The virtualizer returns functions the React Compiler cannot memoise
  "use no memo"
  const { t: tw } = useTranslation(NS)
  const choices: Choices = { curves: props.curves ?? [], stages: props.stages ?? [] }
  const optimization = useSibling<string>(props.optimization_param) ?? ""
  const defaultBlocks = props.default_blocks?.[optimization] ?? "0-51"
  const rows = (Array.isArray(value) ? value : []).map((raw) => rowOf(raw, defaultBlocks, choices))
  const shift = useShift(props.shift)
  const steps = Math.round(useFirstNumber(props.steps, 20, (v) => v >= 1))
  const maxRows = props.max_rows ?? 32
  const cap = props.max_artists?.[optimization]
  const names = artistNames(rows)
  const shapeOf = (curve: string) => props.curves?.find((item) => item.value === curve)?.shape ?? "smooth"
  const floor = props.curve_floor ?? 0.25

  const commit = (next: Row[]) => onChange(next)
  const update = (index: number, row: Row) => commit(rows.map((current, i) => (i === index ? row : current)))
  const patch = (index: number, change: Partial<Row>) => update(index, { ...rows[index]!, ...change })
  const remove = (index: number) => commit(rows.length > 1 ? rows.filter((_, i) => i !== index) : [newRow(defaultBlocks)])
  const move = (index: number, by: number) => {
    const to = index + by
    if (to < 0 || to >= rows.length) return
    const next = [...rows]
    ;[next[index], next[to]] = [next[to]!, next[index]!]
    commit(next)
  }
  const factorOf = (row: Row) => {
    const window = windowOf(row, shift, choices)
    return (x: number) => windowCurveFactor(x, window.start, window.end, window.peak, shapeOf(row.curve), floor)
  }

  const scroller = useRef<HTMLDivElement>(null)
  // oxlint-disable-next-line react/incompatible-library -- opted out of memoisation above
  const virtualizer = useVirtualizer({ count: rows.length, getScrollElement: () => scroller.current, estimateSize: () => ROW_HEIGHT, overscan: 4 })

  return (
    <div id={id} tabIndex={-1} className="flex flex-col gap-2 outline-none" role="group" aria-labelledby={label ? labelId(id) : undefined} aria-label={label ? undefined : tw("Artist rows")}>
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <Button size="xs" variant="outline" disabled={disabled || rows.length >= maxRows} onClick={() => commit([...rows, newRow(defaultBlocks)])}>
          <Plus />
          {tw("Add row")}
        </Button>
        <span className="text-muted-foreground tabular-nums">{tw("{{rows}} of {{max}} rows", { rows: rows.length, max: maxRows })}</span>
        <span className={cn("tabular-nums", cap !== undefined && names.length > cap ? "text-warning" : "text-muted-foreground")} role="status">
          {cap !== undefined && names.length > cap
            ? tw("{{artists}} artists: the preset mixes the first {{cap}}", { artists: names.length, cap })
            : tw("{{artists}} artists mix", { artists: names.length })}
        </span>
        <span className="ml-auto text-muted-foreground tabular-nums">{tw("Shift {{shift}}", { shift: fmt(shift) })}</span>
      </div>
      <Overview rows={rows} factorOf={factorOf} steps={steps} />
      <div ref={scroller} className="max-h-[28rem] overflow-auto rounded-lg border" data-testid="artist-rows">
        <div className="relative w-full" style={{ height: virtualizer.getTotalSize() }}>
          {virtualizer.getVirtualItems().map((item) => (
            <div key={item.key} className="absolute inset-x-0 border-b last:border-b-0" style={{ height: item.size, transform: `translateY(${item.start}px)` }}>
              <ArtistRow
                index={item.index}
                count={rows.length}
                row={rows[item.index]!}
                shift={shift}
                choices={choices}
                disabled={disabled}
                factor={factorOf(rows[item.index]!)}
                onPatch={(change) => patch(item.index, change)}
                onReplace={(row) => update(item.index, row)}
                onMove={(by) => move(item.index, by)}
                onDuplicate={() => rows.length < maxRows && commit([...rows.slice(0, item.index + 1), { ...rows[item.index]! }, ...rows.slice(item.index + 1)])}
                onRemove={() => remove(item.index)}
                canAdd={rows.length < maxRows}
              />
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

/** Each row's strength over the steps, one lane per row (rows that mix nothing are dimmed). */
function Overview(props: { rows: readonly Row[]; factorOf: (row: Row) => (x: number) => number; steps: number }) {
  const { t: tw } = useTranslation(NS)
  const lane = 12
  const height = Math.max(props.rows.length, 1) * lane
  const ticks = [0, 0.25, 0.5, 0.75, 1]
  return (
    <figure className="flex flex-col gap-0.5" aria-label={tw("Schedule overview")}>
      <svg viewBox={`0 0 1000 ${height}`} preserveAspectRatio="none" className="h-auto max-h-48 w-full rounded-md bg-muted/30" style={{ aspectRatio: `1000 / ${Math.max(height, 36)}` }} aria-hidden>
        {props.rows.map((row, i) => {
          const live = row.enabled && row.artist.trim() !== ""
          return (
            <g key={i} transform={`translate(0 ${i * lane})`}>
              <path d={`${curvePath(props.factorOf(row), 1000, lane - 2, 64)}L1000,${lane - 2}L0,${lane - 2}Z`} className={live ? "fill-primary/45" : "fill-muted-foreground/15"} />
            </g>
          )
        })}
      </svg>
      <figcaption className="relative h-3.5 text-[0.625rem] text-muted-foreground tabular-nums">
        {ticks.map((at) => (
          <span key={at} className={cn("absolute", at === 0 ? "left-0" : at === 1 ? "right-0" : "-translate-x-1/2")} style={at > 0 && at < 1 ? { left: `${at * 100}%` } : undefined}>
            {tw("step {{step}}", { step: Math.round(at * Math.max(props.steps - 1, 0)) + 1 })}
          </span>
        ))}
      </figcaption>
    </figure>
  )
}

type RowProps = {
  index: number
  count: number
  row: Row
  shift: number
  choices: Choices
  disabled: boolean
  factor: (x: number) => number
  onPatch: (change: Partial<Row>) => void
  onReplace: (row: Row) => void
  onMove: (by: number) => void
  onDuplicate: () => void
  onRemove: () => void
  canAdd: boolean
}

function ArtistRow({ index, count, row, shift, choices, disabled, factor, onPatch, onReplace, onMove, onDuplicate, onRemove, canAdd }: RowProps) {
  const { t: tw } = useTranslation(NS)
  const auto = isAuto(row, choices)
  const window = windowOf(row, shift, choices)
  const n = index + 1
  const off = disabled || !row.enabled
  return (
    <div className={cn("flex h-full flex-col justify-center gap-1.5 px-2 py-1.5", !row.enabled && "bg-muted/40")} role="group" aria-label={tw("Row {{n}}", { n })}>
      <div className="flex items-center gap-2">
        <Switch size="sm" checked={row.enabled} disabled={disabled} aria-label={tw("Row {{n}} enabled", { n })} onCheckedChange={(enabled) => onPatch({ enabled })} />
        <span className="w-5 text-right text-xs text-muted-foreground tabular-nums">{n}</span>
        <Input
          className="h-7 min-w-24 flex-1 text-sm"
          aria-label={tw("Row {{n}} artist", { n })}
          placeholder={tw("Artist, e.g. (wlop:1.2)")}
          value={row.artist}
          disabled={off}
          onChange={(event) => onPatch({ artist: event.target.value })}
        />
        <Num label={tw("Row {{n}} weight", { n })} title={tw("Weight")} value={row.weight} min={-4} max={4} step={0.05} disabled={off} onCommit={(weight) => onPatch({ weight })} />
        <Input
          className="h-7 w-24 font-mono text-xs"
          aria-label={tw("Row {{n}} blocks", { n })}
          title={tw("Blocks (28-block layout; 0-51 is every block)")}
          value={row.blocks}
          disabled={off}
          onChange={(event) => onPatch({ blocks: event.target.value })}
        />
        <div className="flex items-center">
          <Button variant="ghost" size="icon-xs" aria-label={tw("Move row {{n}} up", { n })} disabled={disabled || index === 0} onClick={() => onMove(-1)}>
            <ArrowUp />
          </Button>
          <Button variant="ghost" size="icon-xs" aria-label={tw("Move row {{n}} down", { n })} disabled={disabled || index === count - 1} onClick={() => onMove(1)}>
            <ArrowDown />
          </Button>
          <Button variant="ghost" size="icon-xs" aria-label={tw("Duplicate row {{n}}", { n })} disabled={disabled || !canAdd} onClick={onDuplicate}>
            <CopyPlus />
          </Button>
          <Button variant="ghost" size="icon-xs" aria-label={tw("Remove row {{n}}", { n })} disabled={disabled} onClick={onRemove}>
            <Trash2 />
          </Button>
        </div>
      </div>
      <div className="flex items-center gap-2 pl-14">
        <select className={SELECT} aria-label={tw("Row {{n}} stage", { n })} value={row.stage} disabled={off} onChange={(event) => onReplace(withStage(row, event.target.value, shift, choices))}>
          {choices.stages.map((stage) => (
            <option key={stage.value} value={stage.value}>
              {tw(stage.label)}
            </option>
          ))}
        </select>
        <label className="flex items-center gap-1 text-xs text-muted-foreground" title={tw("Auto Shift: the stage's window follows the Shift")}>
          <Switch size="sm" checked={row.auto_shift} disabled={off} aria-label={tw("Row {{n}} auto shift", { n })} onCheckedChange={(on) => onReplace(withAuto(row, on, shift, choices))} />
          {tw("Auto")}
        </label>
        <select className={SELECT} aria-label={tw("Row {{n}} curve", { n })} value={row.curve} disabled={off} onChange={(event) => onPatch({ curve: event.target.value })}>
          {choices.curves.map((curve) => (
            <option key={curve.value} value={curve.value}>
              {tw(curve.label)}
            </option>
          ))}
        </select>
        {(["start", "end", "peak"] as const).map((key) => (
          <Num
            key={key}
            label={key === "start" ? tw("Row {{n}} start", { n }) : key === "end" ? tw("Row {{n}} end", { n }) : tw("Row {{n}} peak", { n })}
            title={auto ? tw("Timed by Auto Shift; turn it off to edit") : key === "start" ? tw("Start") : key === "end" ? tw("End") : tw("Peak")}
            value={window[key]}
            min={0}
            max={1}
            step={0.01}
            disabled={off || auto}
            onCommit={(next) => onPatch({ ...window, [key]: next })}
          />
        ))}
        <svg viewBox="0 0 100 20" preserveAspectRatio="none" className="h-5 min-w-12 flex-1 rounded bg-muted/40" role="img" aria-label={tw("Row {{n}} window {{start}}-{{end}}", { n, start: fmt(window.start), end: fmt(window.end) })}>
          <path d={`${curvePath(factor, 100, 20, 48)}L100,20L0,20Z`} className={row.enabled ? "fill-primary/50" : "fill-muted-foreground/20"} />
        </svg>
      </div>
    </div>
  )
}

function Num(props: { label: string; title: string; value: number; min: number; max: number; step: number; disabled: boolean; onCommit: (value: number) => void }) {
  return (
    <NumberField
      aria-label={props.label}
      title={props.title}
      size="sm"
      steppers={false}
      className="w-14"
      inputClassName="px-1 text-right text-xs"
      value={Number(props.value.toFixed(4))}
      min={props.min}
      max={props.max}
      step={props.step}
      format={{ maximumFractionDigits: 4, useGrouping: false }}
      disabled={props.disabled}
      onValueCommitted={(next) => next !== null && Number.isFinite(next) && props.onCommit(Math.min(Math.max(next, props.min), props.max))}
    />
  )
}
