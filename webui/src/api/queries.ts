/** TanStack Query setup: one client, query-key factory and `queryOptions` per resource. */
import { infiniteQueryOptions, QueryClient, queryOptions, useInfiniteQuery, useQuery, useSuspenseQuery } from "@tanstack/react-query"
import { ApiError } from "./http"
import { api } from "./endpoints"
import type { FormTab } from "./types"

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000,
      refetchOnWindowFocus: false,
      // Client errors will not fix themselves; retry only transient failures.
      retry: (count, error) => !(error instanceof ApiError && error.status >= 400 && error.status < 500) && count < 2,
    },
    mutations: { retry: false },
  },
})

export const keys = {
  app: ["app"] as const,
  schema: (tab: FormTab) => ["schema", tab] as const,
  page: (id: string) => ["page", id] as const,
  jobs: ["jobs"] as const,
  outputs: (q = "") => ["outputs", q] as const,
  networks: (kind: string, q = "") => ["networks", kind, q] as const,
  options: ["options"] as const,
  models: ["models"] as const,
}

export const bootstrapQuery = queryOptions({
  queryKey: keys.app,
  queryFn: ({ signal }) => api.app(signal),
  staleTime: Infinity,
})

export const schemaQuery = (tab: FormTab) =>
  queryOptions({ queryKey: keys.schema(tab), queryFn: ({ signal }) => api.schema(tab, signal), staleTime: Infinity })

export const pageSchemaQuery = (id: string) =>
  queryOptions({ queryKey: keys.page(id), queryFn: ({ signal }) => api.pageSchema(id, signal), staleTime: Infinity })

/** Queue + history, newest first, paged by the server's cursor. */
export const jobsQuery = infiniteQueryOptions({
  queryKey: keys.jobs,
  queryFn: ({ pageParam, signal }) => api.jobs.list({ cursor: pageParam, limit: 50 }, signal),
  initialPageParam: null as string | null,
  getNextPageParam: (page) => page.next_cursor ?? undefined,
  staleTime: 5_000,
})

export const modelsQuery = queryOptions({ queryKey: keys.models, queryFn: ({ signal }) => api.models(signal) })

export const optionsQuery = queryOptions({ queryKey: keys.options, queryFn: ({ signal }) => api.options(signal) })

export const useBootstrap = () => useQuery(bootstrapQuery)
export const useSchema = (tab: FormTab) => useSuspenseQuery(schemaQuery(tab))
export const useModels = () => useQuery(modelsQuery)
export const useJobs = () => useInfiniteQuery(jobsQuery)
