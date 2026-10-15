/**
 * `sigma-curve`: Detail Daemon's schedule preview, drawn in the browser from
 * the daemons' params (no server round trip; the old UI rendered matplotlib
 * figures on every slider release). The selected daemon's curve is large;
 * every daemon has a thumbnail that selects its tab.
 *
 * props: { daemons: { prefix: string, title: string, value: string }[], tab: string, enable?: string }
 *   - each daemon's params are `<prefix>active`, `<prefix>start`, …;
 *   - `tab` is the hidden choice the daemon tabs are bound to.
 * value: unused (null)
 */
import { useTranslation } from "react-i18next"
import { cn } from "@/lib/utils"
import { useForm, useFormState, useOwner } from "@/ui/hooks"
import type { WidgetProps } from "@/ui/widgets"
import { registerWidgetCatalogs } from "../_lib/i18n"
import { curvePath, type DaemonState, daemonState } from "./curve"
import { makeSchedule, summarize } from "./schedule"

const NS = registerWidgetCatalogs("sigma-curve", import.meta.glob("./locales/*.json", { eager: true, import: "default" }))

/** Steps of the preview (the old figures used 50). */
const STEPS = 50

type Daemon = { prefix: string; title: string; value: string }
type Props = { daemons?: Daemon[]; tab?: string; enable?: string }
/** Colour class: grey when off, green for added detail (mean ≥ 0), red for smoothing. */
function tone(state: DaemonState, on: boolean, mean: number): string {
  if (!on || !state.active) return "text-muted-foreground"
  return mean >= 0 ? "text-success" : "text-destructive"
}

function Curve(props: { state: DaemonState; on: boolean; width: number; height: number; detailed: boolean; label: string }) {
  const { state, width, height, detailed } = props
  const values = makeSchedule(STEPS, state)
  const { mean, mid } = summarize(values, state)
  // The old figures' formula: the curve fades with the Fade param.
  const opacity = 0.1 + (1 - state.fade) * 0.7
  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      role={props.label ? "img" : undefined}
      aria-label={props.label || undefined}
      aria-hidden={props.label ? undefined : true}
      className={cn("block h-full w-full overflow-visible", tone(state, props.on, mean))}
      preserveAspectRatio="none"
    >
      {detailed ? (
        <g className="text-muted-foreground" stroke="currentColor" strokeWidth={0.6} strokeDasharray="2 2" opacity={0.6}>
          <line x1={0} x2={width} y1={height / 2} y2={height / 2} vectorEffect="non-scaling-stroke" />
          <line x1={mid * width} x2={mid * width} y1={0} y2={height} vectorEffect="non-scaling-stroke" />
        </g>
      ) : (
        <line x1={0} x2={width} y1={height / 2} y2={height / 2} stroke="currentColor" strokeWidth={0.5} opacity={0.3} vectorEffect="non-scaling-stroke" />
      )}
      <path
        d={curvePath(values, width, height)}
        fill="none"
        stroke="currentColor"
        strokeWidth={detailed ? 2 : 1.5}
        strokeDasharray={state.hires ? "4 3" : undefined}
        strokeLinejoin="round"
        opacity={opacity}
        vectorEffect="non-scaling-stroke"
      />
    </svg>
  )
}

export default function SigmaCurve({ props, id, disabled }: WidgetProps<null, Props>) {
  const { t: tw } = useTranslation(NS)
  const { store } = useForm()
  const owner = useOwner()
  const daemons = props.daemons ?? []
  const values = useFormState((s) => s.values[owner])
  const selectedValue = props.tab ? String(values?.[props.tab] ?? daemons[0]?.value ?? "") : (daemons[0]?.value ?? "")
  const on = props.enable ? values?.[props.enable] !== false : true
  const selected = daemons.find((d) => d.value === selectedValue) ?? daemons[0]
  if (!selected) return null

  const describe = (daemon: Daemon, state: DaemonState) => {
    const values = makeSchedule(STEPS, state)
    const { peak } = summarize(values, state)
    return tw("Daemon {{name}}: {{status}}, amount {{amount}}, peak {{peak}}", {
      name: daemon.title,
      status: state.active ? tw("active") : tw("inactive"),
      amount: state.amount,
      peak: peak.toFixed(2),
    })
  }
  const selectedState = daemonState(values, selected.prefix)

  return (
    <div id={id} className="flex flex-col gap-2 sm:flex-row sm:items-stretch">
      <figure className="relative m-0 h-28 min-w-0 flex-1 rounded-lg border bg-muted/30 px-2 py-2">
        <Curve state={selectedState} on={on} width={200} height={100} detailed label={describe(selected, selectedState)} />
        <figcaption className="pointer-events-none absolute top-1 left-2 text-xs font-medium text-muted-foreground tabular-nums">
          {selected.title}
          {selectedState.mode !== "both" ? ` · ${selectedState.mode.toUpperCase()}` : ""}
        </figcaption>
        <span className="pointer-events-none absolute top-1 right-2 text-[0.65rem] text-muted-foreground">1</span>
        <span className="pointer-events-none absolute right-2 bottom-1 text-[0.65rem] text-muted-foreground">−1</span>
      </figure>
      {daemons.length > 1 ? (
        <div role="group" aria-label={tw("Daemons")} className="grid grid-cols-5 gap-1 sm:w-48 sm:grid-cols-3">
          {daemons.map((daemon) => {
            const state = daemonState(values, daemon.prefix)
            const current = daemon.value === selected.value
            return (
              <button
                key={daemon.value}
                type="button"
                disabled={disabled || !props.tab}
                aria-pressed={current}
                aria-label={describe(daemon, state)}
                title={describe(daemon, state)}
                onClick={() => props.tab && store.getState().setValue(owner, props.tab, daemon.value)}
                className={cn(
                  "relative flex h-12 flex-col rounded-md border bg-card px-1 pt-0.5 pb-1 text-left outline-none",
                  "focus-visible:ring-3 focus-visible:ring-ring/50",
                  current ? "border-primary/60 ring-1 ring-primary/30" : "hover:bg-accent",
                )}
              >
                <span className="text-[0.65rem] leading-none font-semibold text-muted-foreground">
                  {daemon.title}
                  {state.mode !== "both" ? ` ${state.mode.toUpperCase()}` : ""}
                </span>
                <span className="min-h-0 flex-1">
                  <Curve state={state} on={on} width={100} height={60} detailed={false} label="" />
                </span>
              </button>
            )
          })}
        </div>
      ) : null}
    </div>
  )
}
