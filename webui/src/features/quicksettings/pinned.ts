/**
 * Quick settings: the options the user pinned (`quicksettings_list`), shown
 * in the header. The model bar already covers the checkpoint, VAE / text
 * encoders, low-bits dtype and UI preset, so those keys never show twice.
 * Pinning writes `quicksettings_list` at once (`PATCH /options`).
 */
import { useQuery } from "@tanstack/react-query"
import { toast } from "sonner"
import { api } from "@/api/endpoints"
import { bootstrapQuery, keys, queryClient } from "@/api/queries"
import type { Bootstrap, OptionsResponse } from "@/api/types"
import { t } from "@/i18n"

export const QUICKSETTINGS_KEY = "quicksettings_list"

/** Covered by the header's model bar. */
export const MODEL_BAR_KEYS: ReadonlySet<string> = new Set(["sd_model_checkpoint", "forge_additional_modules", "forge_unet_storage_dtype", "forge_preset", "sd_vae", "sd_modules"])

/** Pinned keys in the user's order, without the model bar's and duplicates. */
export function quickKeys(list: unknown): string[] {
  if (!Array.isArray(list)) return []
  return [...new Set(list.filter((key): key is string => typeof key === "string" && !MODEL_BAR_KEYS.has(key)))]
}

function currentList(): string[] {
  const options = queryClient.getQueryData<OptionsResponse>(keys.options)
  const fromOptions = options?.values[QUICKSETTINGS_KEY]
  if (Array.isArray(fromOptions)) return fromOptions.filter((key): key is string => typeof key === "string")
  const fromApp = queryClient.getQueryData<Bootstrap>(keys.app)?.options[QUICKSETTINGS_KEY]
  return Array.isArray(fromApp) ? fromApp.filter((key): key is string => typeof key === "string") : []
}

/** The full list with `key` added (at the end) or removed. */
export function withPin(list: readonly string[], key: string, pinned: boolean): string[] {
  const without = list.filter((item) => item !== key)
  return pinned ? [...without, key] : without
}

function writeList(next: string[]) {
  queryClient.setQueryData<Bootstrap>(keys.app, (data) => (data ? { ...data, options: { ...data.options, [QUICKSETTINGS_KEY]: next } } : data))
  queryClient.setQueryData<OptionsResponse>(keys.options, (data) => (data ? { ...data, quicksettings: next, values: { ...data.values, [QUICKSETTINGS_KEY]: next } } : data))
}

/** Pins or unpins a setting (optimistic; rolled back when the server refuses). */
export async function setPinned(key: string, pinned: boolean): Promise<void> {
  const before = currentList()
  const next = withPin(before, key, pinned)
  writeList(next)
  try {
    await api.patchOptions({ [QUICKSETTINGS_KEY]: next })
  } catch (error) {
    writeList(before)
    toast.error(t("Could not change the quick settings"), { description: error instanceof Error ? error.message : String(error) })
  } finally {
    void queryClient.invalidateQueries({ queryKey: keys.app })
    void queryClient.invalidateQueries({ queryKey: keys.options })
  }
}

/** The pinned keys shown in the header (reactive). */
export function useQuickKeys(): string[] {
  const { data } = useQuery(bootstrapQuery)
  return quickKeys(data?.options[QUICKSETTINGS_KEY])
}

/** Whether `key` is pinned (reactive). */
export function useIsPinned(key: string): boolean {
  const { data } = useQuery(bootstrapQuery)
  const list = data?.options[QUICKSETTINGS_KEY]
  return Array.isArray(list) && list.includes(key)
}
