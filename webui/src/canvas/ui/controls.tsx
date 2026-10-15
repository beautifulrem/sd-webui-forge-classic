/** Small building blocks of the editor chrome. */
import type { ReactElement, ReactNode } from "react"
import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Kbd } from "@/components/ui/kbd"
import { Slider } from "@/components/ui/slider"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"
import { ariaKeyShortcuts } from "./a11y"

/** Floating toolbar surface over the canvas. */
export const PANEL = "pointer-events-auto rounded-xl border bg-popover/90 text-popover-foreground shadow-md ring-1 ring-black/5 backdrop-blur-md supports-backdrop-filter:bg-popover/75 dark:ring-white/5"

/**
 * A tooltip that opens on hover and on keyboard focus only: focus coming
 * back to a trigger after its menu closes does not reopen it.
 */
export function Hint({ label, keys, side = "top", disabled, children }: { label: string; keys?: string[]; side?: "top" | "bottom" | "left" | "right"; disabled?: boolean; children: ReactElement }) {
  const [open, setOpen] = useState(false)
  return (
    <Tooltip
      open={open && !disabled}
      disabled={disabled}
      onOpenChange={(next, details) => {
        if (next && details.reason === "trigger-focus") {
          const target = details.event?.target
          if (!(target instanceof Element) || !target.matches(":focus-visible")) return
        }
        setOpen(next)
      }}
    >
      <TooltipTrigger render={children} />
      <TooltipContent side={side}>
        {label}
        {keys?.map((key) => (
          <Kbd key={key}>{key}</Kbd>
        ))}
      </TooltipContent>
    </Tooltip>
  )
}

type IconButtonProps = {
  label: string
  keys?: string[]
  onClick?: () => void
  pressed?: boolean
  disabled?: boolean
  side?: "top" | "bottom" | "left" | "right"
  className?: string
  children: ReactNode
}

/** An icon button with a tooltip; `pressed` makes it a toggle (tools). */
export function IconButton({ label, keys, onClick, pressed, disabled, side, className, children }: IconButtonProps) {
  return (
    <Hint label={label} keys={keys} side={side}>
      <Button
        variant="ghost"
        size="icon-sm"
        aria-label={label}
        aria-pressed={pressed}
        aria-keyshortcuts={ariaKeyShortcuts(keys)}
        onClick={onClick}
        disabled={disabled}
        className={cn("aria-pressed:bg-primary/15 aria-pressed:text-primary dark:aria-pressed:bg-primary/20", className)}
      >
        {children}
      </Button>
    </Hint>
  )
}

type ValueSliderProps = {
  label: string
  /** Shown instead of the label in narrow editors (the label stays the slider's name and tooltip). */
  icon?: ReactNode
  value: number
  min: number
  max: number
  step?: number
  onChange: (value: number) => void
  format: (value: number) => string
  /** What screen readers announce (default: `format`); e.g. the brush size in px where the slider position is a log scale. */
  valueText?: (value: number) => string
  className?: string
}

/** A compact labelled slider with its value; it keeps its size (a narrow toolbar scrolls instead). */
export function ValueSlider({ label, icon, value, min, max, step = 1, onChange, format, valueText, className }: ValueSliderProps) {
  return (
    <div className={cn("flex shrink-0 items-center gap-1.5 @md:gap-2", className)}>
      {icon ? (
        <span aria-hidden title={label} className="text-muted-foreground @md:hidden [&_svg]:size-3.5">
          {icon}
        </span>
      ) : null}
      <span className={cn("shrink-0 text-xs text-muted-foreground", icon && "hidden @md:inline")}>{label}</span>
      <Slider
        value={value}
        min={min}
        max={max}
        step={step}
        onValueChange={(next) => onChange(Array.isArray(next) ? (next[0] ?? value) : next)}
        getAriaLabel={() => label}
        getAriaValueText={(_, current) => (valueText ?? format)(current)}
        className="min-w-12 @md:min-w-20 @lg:min-w-24"
      />
      <span className="w-8 shrink-0 text-right text-xs tabular-nums @md:w-9">{format(value)}</span>
    </div>
  )
}

export function Divider({ vertical = true }: { vertical?: boolean }) {
  return <span aria-hidden className={vertical ? "mx-0.5 h-5 w-px shrink-0 bg-border" : "my-0.5 h-px w-5 shrink-0 bg-border"} />
}
