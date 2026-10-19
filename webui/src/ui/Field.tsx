/** Shared chrome for inputs: label (with its hint tooltip), modified dot (click = reset), trailing slot, info text. */
import { CircleAlert, RotateCcw } from "lucide-react"
import type { ReactElement, ReactNode } from "react"
import { useTranslation } from "react-i18next"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"
import { errorTextId, hintTextId, infoTextId, labelId as labelIdOf } from "./a11y"

/** Dotted underline: the label has a tooltip. */
export const HINTED = "cursor-help underline decoration-muted-foreground/50 decoration-dotted underline-offset-[3px]"

/**
 * A field label with the param's `hint` (A1111's tooltips): shown on hover
 * of the label, and given to assistive technology as the hidden text
 * `hintTextId(id)` the control lists in its aria-describedby (`hintRef`).
 * `hint` is untranslated; without one the label renders as is.
 */
export function HintedLabel({ id, hint, label }: { id: string; hint?: string | null; label: ReactElement }) {
  const { t } = useTranslation()
  if (!hint) return label
  const text = t(hint)
  return (
    <>
      <Tooltip>
        <TooltipTrigger render={label} />
        <TooltipContent className="max-w-sm">{text}</TooltipContent>
      </Tooltip>
      <span id={hintTextId(id)} className="sr-only">
        {text}
      </span>
    </>
  )
}

function formatDefault(value: unknown): string {
  if (value === null || value === undefined || value === "") return "—"
  if (typeof value === "string") return value.length > 40 ? `${value.slice(0, 40)}…` : value
  const text = JSON.stringify(value)
  return text.length > 40 ? `${text.slice(0, 40)}…` : text
}

export function ModifiedDot(props: { onReset: () => void; defaultValue?: unknown; label?: string; disabled?: boolean }) {
  const { t } = useTranslation()
  const what = props.label ? t("Reset {{label}} to default", { label: props.label }) : t("Reset to default")
  return (
    <Tooltip>
      <TooltipTrigger
        render={
          <button
            type="button"
            aria-label={what}
            disabled={props.disabled}
            onClick={props.onReset}
            className="group/dot relative grid size-4 shrink-0 place-items-center rounded-full outline-none focus-visible:ring-2 focus-visible:ring-ring/60 disabled:pointer-events-none"
          />
        }
      >
        <span aria-hidden className="size-1.5 rounded-full bg-primary transition-opacity group-hover/dot:opacity-0 group-focus-visible/dot:opacity-0" />
        <RotateCcw
          aria-hidden
          className="absolute size-3 text-primary opacity-0 transition-opacity group-hover/dot:opacity-100 group-focus-visible/dot:opacity-100"
        />
      </TooltipTrigger>
      <TooltipContent>
        {what}
        {"defaultValue" in props ? <span className="text-muted-foreground/80"> · {formatDefault(props.defaultValue)}</span> : null}
      </TooltipContent>
    </Tooltip>
  )
}

type FieldProps = {
  id: string
  /** Already translated; empty hides the label row (unless `trailing`). */
  label: string
  info?: string | null
  /** Untranslated tooltip of the label; the control lists `hintRef(id, hint)` in its aria-describedby. */
  hint?: string | null
  /** Server validation message; the control must reference it (`describedBy`) and set aria-invalid. */
  error?: string | null
  modified?: boolean
  defaultValue?: unknown
  onReset?: () => void
  disabled?: boolean
  /** Rendered at the right end of the label row (e.g. a slider's number box). */
  trailing?: ReactNode
  /** `label` points at the control via htmlFor (false for groups: uses aria-labelledby). */
  labelFor?: boolean
  className?: string
  children: ReactNode
}

export function Field(props: FieldProps) {
  const { t } = useTranslation()
  const showHeader = props.label !== "" || props.trailing !== undefined || props.modified
  const labelId = labelIdOf(props.id)
  const infoId = infoTextId(props.id)
  return (
    <div
      className={cn("group/field flex min-w-0 flex-col gap-1.5", props.disabled && "opacity-60", props.className)}
      data-disabled={props.disabled || undefined}
      data-invalid={props.error ? true : undefined}
    >
      {showHeader ? (
        <div className="flex min-h-5 items-center gap-1">
          {props.label ? (
            <HintedLabel
              id={props.id}
              hint={props.hint}
              label={
                <label
                  id={labelId}
                  htmlFor={props.labelFor === false ? undefined : props.id}
                  title={props.hint ? undefined : props.label}
                  // wraps (two lines at most) rather than hiding the end of long or translated labels
                  className={cn(
                    "line-clamp-2 min-w-0 text-[0.8rem] leading-5 font-medium break-words hyphens-auto text-foreground/85 select-none",
                    props.hint && HINTED,
                  )}
                >
                  {props.label}
                </label>
              }
            />
          ) : null}
          {props.modified && props.onReset ? (
            <ModifiedDot onReset={props.onReset} defaultValue={props.defaultValue} label={props.label} disabled={props.disabled} />
          ) : null}
          {props.trailing ? <div className="ml-auto flex shrink-0 items-center">{props.trailing}</div> : null}
        </div>
      ) : null}
      {props.children}
      {props.info ? (
        <p id={infoId} className="text-xs leading-snug text-muted-foreground">
          {t(props.info)}
        </p>
      ) : null}
      {props.error ? <FieldError id={props.id} message={props.error} /> : null}
    </div>
  )
}

/** A server validation message under a control (server messages are not translated). */
export function FieldError({ id, message, className }: { id: string; message: string; className?: string }) {
  return (
    <p id={errorTextId(id)} className={cn("flex items-start gap-1 text-xs leading-snug font-medium text-destructive", className)}>
      <CircleAlert aria-hidden className="mt-px size-3.5 shrink-0" />
      <span className="min-w-0 break-words">{message}</span>
    </p>
  )
}
