/**
 * Sampling progress drawn as a row of an exposure sheet (タイムシート): one
 * cell per step, filled as steps are shot, with a wider gap every six frames
 * like the sheet's ruling. Long schedules fold several steps into a cell.
 * Static on purpose: it changes once per step and must not animate between
 * steps (docs/research/ui-performance.md).
 */
import { Progress as ProgressPrimitive } from "@base-ui/react/progress"
import { cn } from "@/lib/utils"

/** More cells than this and they get too thin to read. */
const MAX_CELLS = 36

export function ExposureSheet({
  step,
  steps,
  fraction,
  className,
  "aria-label": ariaLabel,
}: {
  step: number
  steps: number
  fraction: number
  className?: string
  "aria-label": string
}) {
  const cells = Math.max(1, Math.min(steps || 1, MAX_CELLS))
  // Without a step count (a job's own text), fall back to the fraction.
  const done = steps ? Math.floor((Math.min(step, steps) / steps) * cells) : Math.floor(fraction * cells)
  return (
    <ProgressPrimitive.Root value={fraction * 100} aria-label={ariaLabel} data-slot="exposure-sheet" className={cn("flex h-2 w-full", className)}>
      {Array.from({ length: cells }, (_, index) => (
        <span
          key={index}
          aria-hidden
          className={cn(
            "h-full min-w-px flex-1",
            index > 0 && (index % 6 === 0 ? "ml-[3px]" : "ml-px"),
            index < done ? "bg-glow" : index === done ? "bg-glow/45" : "bg-foreground/10",
          )}
        />
      ))}
    </ProgressPrimitive.Root>
  )
}
