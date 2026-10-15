/** Query options shared by the settings views. */
import { queryOptions } from "@tanstack/react-query"
import { api } from "@/api/endpoints"

/** What the server allows (restart, extension access, frozen settings) and the loaded model. */
export const serverStatusQuery = queryOptions({ queryKey: ["server", "status"], queryFn: ({ signal }) => api.system.status(signal), staleTime: 10_000 })

/** The checkpoint and the weights the memory manager holds (fetched on demand). */
export const loadedModelsQuery = queryOptions({ queryKey: ["server", "loaded-models"], queryFn: ({ signal }) => api.system.loadedModels(signal), staleTime: 0 })

/** Progress of hashing every checkpoint (polled while it runs). */
export const hashProgressQuery = queryOptions({
  queryKey: ["server", "hashes"],
  queryFn: ({ signal }) => api.system.hashStatus(signal),
  refetchInterval: (query) => (query.state.data?.running ? 1000 : false),
})
