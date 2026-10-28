/**
 * "12/75" badge: tokens of one prompt for the loaded model, counted by the
 * server after typing pauses. Requests are cancelled when superseded and
 * never block typing; the last count stays visible while the next loads.
 */
import NumberFlow from "@number-flow/react"
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
              "inline-flex h-5 min-w-12 items-baseline justify-end gap-px px-1 text-xs text-muted-foreground tabular-nums",
              known && tokens > max && "text-foreground",
              isFetching && "opacity-70",
              props.className,
            )}
          />
        }
      >
        {known ? (
          <>
            <NumberFlow value={tokens} className="font-medium text-foreground" />
            <span>/{max}</span>
          </>
        ) : (
          "–/–"
        )}
      </TooltipTrigger>
      <TooltipContent>{description}</TooltipContent>
    </Tooltip>
  )
}
