/** Model-bar wiring: option writes, the UI preset action, and display helpers. */
import { useMutation } from "@tanstack/react-query"
import { toast } from "sonner"
import { clientOption } from "@/api/app-options"
import { keys, queryClient } from "@/api/queries"
import { api } from "@/api/endpoints"
import type { Bootstrap, FormTab } from "@/api/types"
import { currentRoute } from "@/app/router"
import { t } from "@/i18n"
import { runAction } from "@/ui/actions"
import { getFormStore } from "@/ui/form-store"

export const MODEL_OPTION_KEYS = {
  checkpoint: "sd_model_checkpoint",
  modules: "forge_additional_modules",
  dtype: "forge_unet_storage_dtype",
  preset: "forge_preset",
} as const

/** Display names of the UI presets (`modules_forge.presets.PresetArch`). */
export const PRESET_LABELS: Record<string, string> = {
  sd: "SD 1.5",
  xl: "SDXL",
  flux: "Flux.1",
  klein: "Flux.2",
  qwen: "Qwen-Image",
  lumina: "Lumina 2",
  zit: "Z-Image Turbo",
  wan: "Wan 2.2",
  anima: "Anima",
  ernie: "ERNIE-Image",
  pid: "PiD",
  krea: "Krea 2",
  qwen21: "Qwen-Image 2.1",
  moss: "Moss-Image",
}

export const presetLabel = (id: string) => PRESET_LABELS[id] ?? id

/** "model.safetensors [a1b2c3d4]" → "model.safetensors" (the hash is noise in a picker). */
export const checkpointLabel = (title: string) => title.replace(/\s*\[[0-9a-f]{6,}\]$/i, "")

/** The file name of a path ("C:\\x\\vae.safetensors", "/m/vae.safetensors" → "vae.safetensors"). */
export const fileName = (path: string) => path.replace(/\\/g, "/").split("/").pop() ?? path

export function formatBytes(bytes: number | null | undefined): string | null {
  if (!bytes || bytes <= 0) return null
  const units = ["B", "KB", "MB", "GB", "TB"]
  const exponent = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)))
  const value = bytes / 1024 ** exponent
  return `${value >= 100 || exponent === 0 ? Math.round(value) : value.toFixed(1)} ${units[exponent]}`
}

/** Writes one option with `PATCH /options`, showing the new value at once. */
export function useOptionWrite<T>(key: string) {
  return useMutation({
    mutationKey: ["option", key],
    mutationFn: (value: T) => api.patchOptions({ [key]: value }),
    onMutate: (value: T) => {
      queryClient.setQueryData<Bootstrap>(keys.app, (data) => (data ? { ...data, options: { ...data.options, [key]: value } } : data))
    },
    onSettled: () => void queryClient.invalidateQueries({ queryKey: keys.app }),
    onError: (error) => toast.error(t("Could not change the model settings"), { description: error.message }),
  })
}

/**
 * Writes `forge_additional_modules` when `next` (file names) differs from the
 * setting. Not tied to a component: a picker that unmounts with a pending
 * selection (its parent popover closed) still saves it.
 */
export function writeModules(next: readonly string[]) {
  const current = clientOption("forge_additional_modules").map(fileName)
  if (next.length === current.length && next.every((name, i) => name === current[i])) return
  const key = MODEL_OPTION_KEYS.modules
  queryClient.setQueryData<Bootstrap>(keys.app, (data) => (data ? { ...data, options: { ...data.options, [key]: [...next] } } : data))
  void api
    .patchOptions({ [key]: next })
    .catch((error: unknown) => toast.error(t("Could not change the model settings"), { description: error instanceof Error ? error.message : String(error) }))
    .finally(() => void queryClient.invalidateQueries({ queryKey: keys.app }))
}

/** The generation form the preset's form patch applies to: the one on screen, else txt2img. */
function presetForm(): FormTab {
  const route = currentRoute()
  return route.name === "generate" ? route.tab : "txt2img"
}

/**
 * Selects a UI preset with the core `__forge_preset__` action (it switches
 * checkpoint, modules and dtype on the server and returns the form patch);
 * the other forms follow through their schema reload.
 */
export function useApplyPreset() {
  return useMutation({
    mutationKey: ["option", MODEL_OPTION_KEYS.preset],
    mutationFn: async (preset: string) => {
      const tab = presetForm()
      const result = await runAction({ store: getFormStore(tab), tab }, "core", "__forge_preset__", { preset })
      if (!result) throw new Error(t("The preset was not applied"))
      return result
    },
    onSettled: () => {
      void queryClient.invalidateQueries({ queryKey: keys.app })
      void queryClient.invalidateQueries({ queryKey: ["schema"] })
    },
  })
}
