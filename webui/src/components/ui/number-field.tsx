import { NumberField as NumberFieldPrimitive } from "@base-ui/react/number-field"
import { cn } from "cn"
import { ChevronDownIcon, ChevronUpIcon } from "lucide-react"

/** Numeric input in the base-nova style; steppers are optional (`steppers={false}` for tight spots). */
function NumberField({
  className,
  inputClassName,
  steppers = true,
  size = "default",
  id,
  "aria-label": ariaLabel,
  "aria-labelledby": ariaLabelledBy,
  "aria-describedby": ariaDescribedBy,
  "aria-invalid": ariaInvalid,
  ...props
}: NumberFieldPrimitive.Root.Props & {
  inputClassName?: string
  steppers?: boolean
  size?: "sm" | "default"
  "aria-label"?: string
  "aria-labelledby"?: string
  "aria-describedby"?: string
  "aria-invalid"?: boolean
}) {
  return (
    <NumberFieldPrimitive.Root data-slot="number-field" className={cn("min-w-0", className)} {...props}>
      <NumberFieldPrimitive.Group
        data-size={size}
        data-invalid={ariaInvalid || undefined}
        className="flex w-full items-stretch overflow-hidden rounded-lg border border-input bg-transparent transition-colors focus-within:border-ring focus-within:ring-3 focus-within:ring-ring/50 data-disabled:cursor-not-allowed data-disabled:opacity-50 data-invalid:border-destructive data-invalid:ring-3 data-invalid:ring-destructive/20 data-[size=default]:h-8 data-[size=sm]:h-7 dark:bg-input/30"
      >
        <NumberFieldPrimitive.Input
          id={id}
          aria-label={ariaLabel}
          aria-labelledby={ariaLabelledBy}
          aria-describedby={ariaDescribedBy}
          aria-invalid={ariaInvalid || undefined}
          className={cn(
            "w-full min-w-0 flex-1 bg-transparent px-2 text-sm tabular-nums outline-none placeholder:text-muted-foreground",
            inputClassName,
          )}
        />
        {steppers ? (
          <div className="flex w-5 shrink-0 flex-col border-l border-input">
            <NumberFieldPrimitive.Increment className="grid flex-1 place-items-center text-muted-foreground transition-colors outline-none hover:bg-muted hover:text-foreground disabled:opacity-40">
              <ChevronUpIcon className="size-3" />
            </NumberFieldPrimitive.Increment>
            <NumberFieldPrimitive.Decrement className="grid flex-1 place-items-center border-t border-input text-muted-foreground transition-colors outline-none hover:bg-muted hover:text-foreground disabled:opacity-40">
              <ChevronDownIcon className="size-3" />
            </NumberFieldPrimitive.Decrement>
          </div>
        ) : null}
      </NumberFieldPrimitive.Group>
    </NumberFieldPrimitive.Root>
  )
}

export { NumberField }
