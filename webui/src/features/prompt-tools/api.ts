/**
 * `/api/v2/prompt-tools` endpoints (remi/server/routes/prompt_tools.py),
 * typed by the generated client like `@/api/endpoints`.
 */
import { apiUrl, client, unwrap } from "@/api/http"
import type { components } from "@/api/gen/schema"

type Schemas = components["schemas"]

export type PromptToolsConfig = Schemas["PromptToolsConfig"]
export type TagSettings = Schemas["PromptToolsTagSettings"]
export type ToolSettings = Schemas["PromptToolsToolSettings"]
export type CompletionItem = Schemas["PromptCompletionItem"]
export type CompletionMode = "tag" | "lora" | "embedding" | "chant" | "network" | "wildcard" | "wildcard-value"
export type CompletionType = CompletionItem["type"]
export type TagInfo = Schemas["PromptTagInfo"]
export type TagCategory = Schemas["PromptTagCategory"]
export type HistoryItem = Schemas["PromptHistoryItem"]
export type HistoryPage = Schemas["Page_PromptHistoryItem_"]
export type Collection = Schemas["PromptCollection"]
export type Favorite = Schemas["PromptFavorite"]
export type FavoritesPage = Schemas["Page_PromptFavorite_"]
export type FavoriteKind = Favorite["kind"]
export type FavoritePatch = Schemas["PromptFavoritePatch"]
export type FavoritesExport = Schemas["PromptFavoritesExport"]

export type CompletionQuery = { mode: CompletionMode; q: string; negative?: boolean; limit?: number; wildcard?: string }
export type HistoryField = "prompt" | "negative"

/** Most tags one `/lookup` call takes (the server's limit). */
export const LOOKUP_CHUNK = 500
/** Most texts one `/translate` call takes (the server's limit). */
export const TRANSLATE_CHUNK = 200

/** Splits a list into chunks of at most `size`. */
export function chunks<T>(items: readonly T[], size: number): T[][] {
  const out: T[][] = []
  for (let start = 0; start < items.length; start += size) out.push(items.slice(start, start + size))
  return out
}

/** A query string with repeated keys for arrays (`kind=a&kind=b`); unset values dropped. */
function repeatedQuery(query: Record<string, unknown>): string {
  const params = new URLSearchParams()
  for (const [key, value] of Object.entries(query)) {
    for (const item of Array.isArray(value) ? value : [value]) {
      if (item !== undefined && item !== null && item !== "") params.append(key, String(item))
    }
  }
  return params.toString()
}

async function lookupAll(tags: string[], signal?: AbortSignal): Promise<Record<string, TagInfo>> {
  const answers = await Promise.all(
    chunks(tags, LOOKUP_CHUNK).map((part) => unwrap(client.POST("/api/v2/prompt-tools/lookup", { body: { tags: part }, signal })).then((answer) => answer.tags)),
  )
  return Object.assign({}, ...answers) as Record<string, TagInfo>
}

async function translateAll(texts: string[], target: string): Promise<{ provider: string; translations: (string | null)[] }> {
  const answers = await Promise.all(chunks(texts, TRANSLATE_CHUNK).map((part) => unwrap(client.POST("/api/v2/prompt-tools/translate", { body: { texts: part, target } }))))
  return { provider: answers[0]?.provider ?? "", translations: answers.flatMap((answer) => answer.translations) }
}

export const promptToolsApi = {
  config: (signal?: AbortSignal) => unwrap(client.GET("/api/v2/prompt-tools/config", { signal })),

  complete: (query: CompletionQuery, signal?: AbortSignal) =>
    unwrap(client.GET("/api/v2/prompt-tools/complete", { params: { query: { negative: false, ...query } }, signal })).then((list) => list.items),

  /** Counts an accepted completion for the frequency ranking. */
  recordUse: (body: { name: string; type: CompletionType; negative: boolean }) => unwrap(client.POST("/api/v2/prompt-tools/usage", { body })),

  /** Category, posts and translation of known tags (any number: sent in chunks). */
  lookup: lookupAll,

  /** Translations in the order of `texts` (any number: sent in chunks). */
  translate: translateAll,

  /** Stores the translation API key (`""` forgets it); it is never sent back. */
  setTranslationKey: (key: string) => unwrap(client.PUT("/api/v2/prompt-tools/translation-key", { body: { key } })),

  groups: (locale: string, signal?: AbortSignal) =>
    unwrap(client.GET("/api/v2/prompt-tools/groups", { params: { query: { locale } }, signal })).then((answer) => answer.categories),

  history: {
    list: (query: { q?: string; field?: HistoryField; cursor?: string | null; limit?: number }, signal?: AbortSignal) =>
      unwrap(client.GET("/api/v2/prompt-tools/history", { params: { query }, signal })),
    add: (body: { prompt?: string; negative_prompt?: string; tab?: string }) =>
      unwrap(client.POST("/api/v2/prompt-tools/history", { body: { prompt: "", negative_prompt: "", tab: "", ...body } })),
    /** `field`: the list is grouped by that text, so every entry of the group is named. */
    rename: (id: string, name: string, field?: HistoryField) =>
      unwrap(client.PATCH("/api/v2/prompt-tools/history/{entry_id}", { params: { path: { entry_id: id } }, body: { name, field: field ?? null } })),
    /** `field`: removes every entry of the group (a grouped list item). */
    remove: (id: string, field?: HistoryField) =>
      unwrap(client.DELETE("/api/v2/prompt-tools/history/{entry_id}", { params: { path: { entry_id: id }, query: { field } } })),
    clear: () => unwrap(client.DELETE("/api/v2/prompt-tools/history", { params: { query: { keep_named: true } } })),
  },

  collections: {
    list: (signal?: AbortSignal) => unwrap(client.GET("/api/v2/prompt-tools/collections", { signal })),
    create: (name: string) => unwrap(client.POST("/api/v2/prompt-tools/collections", { body: { name } })),
    rename: (id: string, name: string) =>
      unwrap(client.PATCH("/api/v2/prompt-tools/collections/{collection_id}", { params: { path: { collection_id: id } }, body: { name } })),
    remove: (id: string) => unwrap(client.DELETE("/api/v2/prompt-tools/collections/{collection_id}", { params: { path: { collection_id: id } } })),
  },

  favorites: {
    /** A page; `collection`: an id, `""` for the unsorted favorites, absent for all; `kind`: any of these. */
    list: (query: { collection?: string; kind?: FavoriteKind[]; q?: string; cursor?: string | null; limit?: number }, signal?: AbortSignal) =>
      unwrap(client.GET("/api/v2/prompt-tools/favorites", { params: { query }, querySerializer: repeatedQuery, signal })),
    add: (body: { text: string; kind: FavoriteKind; name?: string; collection_id?: string | null }) =>
      unwrap(client.POST("/api/v2/prompt-tools/favorites", { body: { name: "", collection_id: null, ...body } })),
    update: (id: string, body: FavoritePatch) =>
      unwrap(client.PATCH("/api/v2/prompt-tools/favorites/{favorite_id}", { params: { path: { favorite_id: id } }, body })),
    remove: (id: string) => unwrap(client.DELETE("/api/v2/prompt-tools/favorites/{favorite_id}", { params: { path: { favorite_id: id } } })),
    exportUrl: () => apiUrl("/prompt-tools/favorites-export"),
    import: (data: FavoritesExport) => unwrap(client.POST("/api/v2/prompt-tools/favorites-import", { body: data })),
  },
}
