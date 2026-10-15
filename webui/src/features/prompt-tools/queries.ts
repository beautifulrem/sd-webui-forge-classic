/** Query keys and options of the prompt tools (kept fresh by `register.ts`). */
import { queryOptions, useQuery } from "@tanstack/react-query"
import { queryClient } from "@/api/queries"
import { type PromptToolsConfig, promptToolsApi } from "./api"

export const toolKeys = {
  all: ["prompt-tools"] as const,
  config: ["prompt-tools", "config"] as const,
  groups: (locale: string) => ["prompt-tools", "groups", locale] as const,
  lookup: (tags: readonly string[]) => ["prompt-tools", "lookup", ...tags] as const,
  history: (field: string, q: string) => ["prompt-tools", "history", field, q] as const,
  collections: ["prompt-tools", "collections"] as const,
  favorites: (filter: string, kind: string, q: string) => ["prompt-tools", "favorites", filter, kind, q] as const,
}

export const configQuery = queryOptions({
  queryKey: toolKeys.config,
  queryFn: ({ signal }) => promptToolsApi.config(signal),
  staleTime: Infinity,
  // An older server or mock mode has no prompt tools: they stay off.
  retry: false,
  retryOnMount: false,
})

export function useToolsConfig(): PromptToolsConfig | undefined {
  return useQuery(configQuery).data
}

const RETRY_AFTER_MS = 30_000
let failedAt = 0

/** The config from the cache, or fetched (null when the server has none; asked again after 30 s). */
export async function loadConfig(): Promise<PromptToolsConfig | null> {
  const cached = queryClient.getQueryData<PromptToolsConfig>(toolKeys.config)
  if (cached) return cached
  if (Date.now() - failedAt < RETRY_AFTER_MS) return null
  try {
    return await queryClient.fetchQuery(configQuery)
  } catch {
    failedAt = Date.now()
    return null
  }
}
