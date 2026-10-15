/** Debounced, cancellable token count of one prompt (`POST /prompt/tokens`). */
import { keepPreviousData, type Query, useQuery } from "@tanstack/react-query"
import { useMemo } from "react"
import { useClientOption } from "@/api/app-options"
import { api } from "@/api/endpoints"
import { queryClient } from "@/api/queries"
import type { TokenCount } from "@/api/types"
import { onWsEvent } from "@/api/ws"
import { useDebouncedValue } from "./use-debounced-value"

export type TokenCountInput = {
  text: string
  steps?: number
  styles?: readonly string[]
  negative?: boolean
}

export const TOKEN_DEBOUNCE_MS = 700

const TOKENS = ["tokens"] as const

/** Counts taken while no model was loaded (`tokens: null`). */
const uncounted = (query: Query) => (query.state.data as TokenCount | undefined)?.tokens == null

let wired = false

/**
 * Counts depend on the loaded model, which can change under the same
 * checkpoint name (the first load, a reload): recount after `model.loaded`,
 * and after a finished job while a count is still unknown (that job loaded
 * the model).
 */
export function wireTokenRecount() {
  if (wired) return
  wired = true
  onWsEvent("model.loaded", () => void queryClient.invalidateQueries({ queryKey: TOKENS }))
  onWsEvent("job.completed", () => void queryClient.invalidateQueries({ queryKey: TOKENS, predicate: uncounted }))
}

export function useTokenCount({ text, steps, styles = [], negative = false }: TokenCountInput) {
  wireTokenRecount()
  const disabled = useClientOption("disable_token_counters")
  const includeStyles = useClientOption("include_styles_into_token_counters")
  const checkpoint = useClientOption("sd_model_checkpoint")
  // Text and steps settle together: dragging the steps slider must not count per tick.
  const input = useMemo(() => ({ text, steps: steps ?? null }), [text, steps])
  const debounced = useDebouncedValue(input, TOKEN_DEBOUNCE_MS)
  const styleList = includeStyles ? [...styles] : []
  return useQuery({
    queryKey: [...TOKENS, checkpoint, negative, debounced.text, debounced.steps, styleList],
    queryFn: ({ signal }) => api.countTokens({ text: debounced.text, steps: debounced.steps ?? undefined, styles: styleList, negative }, signal),
    enabled: !disabled,
    placeholderData: keepPreviousData,
    staleTime: Infinity,
    gcTime: 60_000,
    retry: false,
  })
}
