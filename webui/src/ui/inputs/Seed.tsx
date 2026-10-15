import { Dices, Recycle } from "lucide-react"
import { useId } from "react"
import { useTranslation } from "react-i18next"
import { lastResult, useProgress } from "@/api/progress"
import { Button } from "@/components/ui/button"
import { NumberField } from "@/components/ui/number-field"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { useForm, useParam } from "../hooks"
import { describedBy } from "../a11y"
import { Field } from "../Field"
import { infotextValue } from "../infotext"
import type { SeedNode } from "../schema"

function IconButton(props: { label: string; onClick: () => void; disabled?: boolean; children: React.ReactNode }) {
  return (
    <Tooltip>
      <TooltipTrigger
        render={
          <Button
            variant="outline"
            size="icon"
            aria-label={props.label}
            onClick={props.onClick}
            disabled={props.disabled}
            className="shrink-0"
          />
        }
      >
        {props.children}
      </TooltipTrigger>
      <TooltipContent>{props.label}</TooltipContent>
    </Tooltip>
  )
}

/** Integer seed with "random" (dice → -1) and "reuse last" (recycle) buttons. */
export function Seed({ node }: { node: SeedNode }) {
  const { t } = useTranslation()
  const id = useId()
  const param = useParam<number>(node)
  const { formId } = useForm()
  const label = t(param.label)
  const info = param.info
  const lastSeed = useProgress((s) => infotextValue(lastResult(s, formId)?.infotext, node.infotext || "Seed"))
  const disabled = !param.enabled

  return (
    <Field
      id={id}
      label={label}
      info={info}
      error={param.error}
      modified={param.modified}
      defaultValue={param.defaultValue}
      onReset={param.reset}
      disabled={disabled}
    >
      <div className="flex items-center gap-1.5">
        <NumberField
          id={id}
          aria-label={label ? undefined : node.name}
          aria-describedby={describedBy(id, info, param.error)}
          aria-invalid={Boolean(param.error)}
          className="flex-1"
          value={typeof param.value === "number" ? param.value : -1}
          min={-1}
          step={1}
          format={{ maximumFractionDigits: 0, useGrouping: false }}
          onValueChange={(next) => next !== null && param.setValue(Math.round(next))}
          disabled={disabled}
        />
        <IconButton label={t("Random seed")} onClick={() => param.setValue(-1)} disabled={disabled}>
          <Dices />
        </IconButton>
        <IconButton
          label={lastSeed === null ? t("Reuse seed from the last image") : t("Reuse seed {{seed}}", { seed: lastSeed })}
          onClick={() => lastSeed !== null && param.setValue(lastSeed)}
          disabled={disabled || lastSeed === null}
        >
          <Recycle />
        </IconButton>
      </div>
    </Field>
  )
}
