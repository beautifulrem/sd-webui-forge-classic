/** Queries of the extensions page and cache helpers. */
import { queryOptions } from "@tanstack/react-query"
import { api } from "@/api/endpoints"
import { queryClient } from "@/api/queries"
import type { ExtensionInfo, ExtensionList } from "./extensions-model"

export const extensionKeys = {
  installed: ["extensions", "installed"] as const,
  index: (url: string) => ["extensions", "index", url] as const,
  states: ["extensions", "states"] as const,
  state: (id: string) => ["extensions", "states", id] as const,
}

export const installedQuery = queryOptions({ queryKey: extensionKeys.installed, queryFn: ({ signal }) => api.extensions.list(signal), staleTime: 30_000 })

export const statesQuery = queryOptions({ queryKey: extensionKeys.states, queryFn: ({ signal }) => api.extensions.states.list(signal) })

export const stateQuery = (id: string) =>
  queryOptions({ queryKey: extensionKeys.state(id), queryFn: ({ signal }) => api.extensions.states.get(id, signal) })

/** Replaces one extension in the cached list (after a per-extension check). */
export function putExtension(info: ExtensionInfo) {
  queryClient.setQueryData<ExtensionList>(extensionKeys.installed, (list) =>
    list ? { ...list, items: list.items.map((item) => (item.name === info.name ? info : item)) } : list,
  )
}
