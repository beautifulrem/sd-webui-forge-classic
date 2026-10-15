/** Segmented control: radio-group semantics (arrow keys, one tab stop), pill styling. */
import { Radio } from "@base-ui/react/radio"
import { RadioGroup } from "@base-ui/react/radio-group"
import { cn } from "@/lib/utils"

type SegmentedProps<V extends string | number> = {
  options: readonly { value: V; label: string }[]
  value: V | null
  onValueChange: (value: V) => void
  disabled?: boolean
  id?: string
  "aria-labelledby"?: string
  "aria-label"?: string
  "aria-describedby"?: string
  "aria-invalid"?: boolean
  className?: string
  size?: "sm" | "default"
}

export function Segmented<V extends string | number>(props: SegmentedProps<V>) {
  return (
    <RadioGroup
      id={props.id}
      value={props.value}
      onValueChange={(value) => props.onValueChange(value as V)}
      disabled={props.disabled}
      aria-labelledby={props["aria-labelledby"]}
      aria-label={props["aria-label"]}
      aria-describedby={props["aria-describedby"]}
      aria-invalid={props["aria-invalid"] || undefined}
      className={cn(
        // equal columns as wide as the longest label (stretched when the group is full width), one row
        "inline-grid max-w-full auto-cols-fr grid-flow-col items-center gap-0.5 rounded-lg bg-muted p-0.5 data-disabled:opacity-60",
        props.className,
      )}
    >
      {props.options.map((option) => (
        <Radio.Root
          key={String(option.value)}
          value={option.value}
          nativeButton
          render={<button type="button" />}
          className={cn(
            "min-w-0 truncate rounded-md px-2.5 text-[0.8rem] font-medium text-muted-foreground transition-colors outline-none select-none",
            "hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring/60 data-disabled:pointer-events-none",
            "data-checked:bg-background data-checked:text-foreground data-checked:shadow-sm dark:data-checked:bg-accent",
            props.size === "sm" ? "h-6" : "h-7",
          )}
        >
          {option.label}
        </Radio.Root>
      ))}
    </RadioGroup>
  )
}
