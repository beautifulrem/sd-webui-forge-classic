/**
 * `shift-curve`: Dynamic Shift's curve choice with a live plot of the sigma
 * schedule it gives, against the two constant-shift schedules at the first-
 * and last-step shift. The plot reads the shifts from sibling params and the
 * step count from `props.steps`; it is computed in the browser.
 *
 * props: { choices: string[], start: string, end: string, steps?: string }
 *   `start` / `end` / `steps` name a param of this owner or "owner.param".
 * value: string (one of `choices`)
 */
import { useTranslation } from "react-i18next"
import { Segmented } from "@/components/Segmented"
import { labelId } from "@/ui/a11y"
import { splitRef } from "@/ui/conditions"
import { useFormState, useOwner } from "@/ui/hooks"
import type { WidgetProps } from "@/ui/widgets"
import { registerWidgetCatalogs } from "../_lib/i18n"
import { CURVES, computeSigmas, sigmaPath } from "./schedule"

const NS = registerWidgetCatalogs("shift-curve", import.meta.glob("./locales/*.json", { eager: true, import: "default" }))

type Props = { choices?: string[]; start?: string; end?: string; steps?: string }

const WIDTH = 360
const HEIGHT = 140
const PAD = 8
const PREVIEW_STEPS = 30

function useNumber(ref: string | undefined, fallback: number): number {
  const owner = useOwner()
  const at = ref ? splitRef(ref, owner) : null
  const value = useFormState((s) => (at ? s.values[at[0]]?.[at[1]] : undefined))
  return typeof value === "number" && Number.isFinite(value) ? value : fallback
}

const format = (value: number) => String(Number(value.toFixed(2)))

export default function ShiftCurve({ value, onChange, props, disabled, id, label }: WidgetProps<string | null, Props>) {
  const { t: tw } = useTranslation(NS)
  const choices = props.choices?.length ? props.choices : [...CURVES]
  const curve = typeof value === "string" ? value : choices[0]!
  const start = useNumber(props.start, 5)
  const end = useNumber(props.end, 1.5)
  const steps = Math.min(Math.max(Math.round(useNumber(props.steps, PREVIEW_STEPS)), 1), 1000)

  const path = (s0: number, s1: number, kind: string) => sigmaPath(computeSigmas(steps, s0, s1, kind).slice(0, -1), WIDTH, HEIGHT, PAD)
  const summary = tw("Sigma schedule over {{steps}} steps: shift {{start}} to {{end}}, {{curve}}", {
    steps,
    start: format(start),
    end: format(end),
    curve: tw(curve),
  })

  return (
    <div className="flex flex-col gap-2">
      <Segmented
        id={id}
        size="sm"
        options={choices.map((choice) => ({ value: choice, label: tw(choice) }))}
        value={curve}
        onValueChange={onChange}
        disabled={disabled}
        aria-labelledby={label ? labelId(id) : undefined}
        aria-label={label ? undefined : tw("Curve")}
      />
      <figure className="flex flex-col gap-1.5 rounded-lg border bg-muted/30 p-2">
        <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label={summary} className="h-auto w-full" preserveAspectRatio="none">
          <path d={path(start, start, "Linear")} fill="none" strokeWidth={1.5} strokeDasharray="4 4" className="stroke-muted-foreground/70" />
          <path d={path(end, end, "Linear")} fill="none" strokeWidth={1.5} strokeDasharray="1 3" className="stroke-muted-foreground/70" />
          <path d={path(start, end, curve)} fill="none" strokeWidth={2} strokeLinejoin="round" className="stroke-primary" data-testid="dynamic-schedule" />
        </svg>
        <figcaption className="flex flex-wrap gap-x-3 gap-y-1 text-xs text-muted-foreground">
          <span className="inline-flex items-center gap-1.5">
            <span aria-hidden className="h-0.5 w-4 rounded bg-primary" />
            {tw("Dynamic")}
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span aria-hidden className="w-4 border-t-2 border-dashed border-muted-foreground/70" />
            {tw("Constant {{value}}", { value: format(start) })}
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span aria-hidden className="w-4 border-t-2 border-dotted border-muted-foreground/70" />
            {tw("Constant {{value}}", { value: format(end) })}
          </span>
        </figcaption>
      </figure>
    </div>
  )
}
