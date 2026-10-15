import { useId } from "react"
import { useTranslation } from "react-i18next"
import { Switch } from "@/components/ui/switch"
import { cn } from "@/lib/utils"
import { useParam } from "../hooks"
import { describedBy } from "../a11y"
import { FieldError, ModifiedDot } from "../Field"
import type { ToggleNode } from "../schema"

/**
 * Switch with an inline label. Conflict locks (`exclusive` groups and the
 * schema's `conflict_params`) run from `useParam`, like for every input.
 */
export function Toggle({ node }: { node: ToggleNode }) {
  const { t } = useTranslation()
  const id = useId()
  const param = useParam<boolean>(node)
  const label = t(param.label)
  const info = param.info

  return (
    <div className={cn("flex min-w-0 flex-col gap-1", !param.enabled && "opacity-60")} data-invalid={param.error ? true : undefined}>
      <div className="flex min-h-7 items-center gap-2.5">
        <Switch
          id={id}
          checked={Boolean(param.value)}
          onCheckedChange={param.setValue}
          disabled={!param.enabled}
          aria-describedby={describedBy(id, info, param.error)}
          aria-invalid={param.error ? true : undefined}
          aria-label={label ? undefined : node.name}
        />
        {label ? (
          <label htmlFor={id} className="min-w-0 truncate text-sm leading-5 select-none">
            {label}
          </label>
        ) : null}
        {param.modified ? <ModifiedDot onReset={param.reset} defaultValue={param.defaultValue} label={label} /> : null}
      </div>
      {info ? (
        <p id={`${id}-info`} className="pl-11 text-xs leading-snug text-muted-foreground">
          {t(info)}
        </p>
      ) : null}
      {param.error ? <FieldError id={id} message={param.error} className="pl-11" /> : null}
    </div>
  )
}
