/**
 * "12/75" badge: tokens of one prompt for the loaded model, counted by the
 * server after typing pauses. Requests are cancelled when superseded and
 * never block typing; the last count stays visible while the next loads.
 */
import { useTranslation } from "react-i18next"
import { useClientOption } from "@/api/app-options"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"
import { type TokenCountInput, useTokenCount } from "./use-token-count"

type TokenCounterProps = TokenCountInput & { className?: string }

export function TokenCounter(props: TokenCounterProps) {
  const { t } = useTranslation()
  const disabled = useClientOption("disable_token_counters")
  const { data, isFetching, isError } = useTokenCount(props)
  if (disabled) return null
  const tokens = data?.tokens
  const max = data?.max_length
  const known = typeof tokens === "number" && typeof max === "number"
  const label = known ? `${tokens}/${max}` : "–/–"
  const description = known
    ? t("{{tokens}} tokens; the prompt is split into chunks of 75", { tokens })
    : isError
      ? t("Could not count tokens")
      : t("Tokens are counted once a model is loaded")
  return (
    <Tooltip>
      <span className="sr-only">{description}</span>
      <TooltipTrigger
        render={
          <span
            aria-hidden
            className={cn(
              "inline-flex h-5 min-w-12 items-center justify-center rounded-md bg-muted px-1.5 font-mono text-[0.7rem] text-muted-foreground tabular-nums transition-opacity",
              isFetching && "opacity-60",
              props.className,
            )}
          />
        }
      >
        {label}
      </TooltipTrigger>
      <TooltipContent>{description}</TooltipContent>
    </Tooltip>
  )
}
