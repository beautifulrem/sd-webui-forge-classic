/**
 * What a card click does, per kind action: `prompt` toggles the card's text in
 * the generation form's prompts, `checkpoint` loads the model, `module`
 * toggles a VAE / text encoder. Plus the per-item helpers (copy path, open
 * folder) and the settings the browser reads.
 */
import { useQuery } from "@tanstack/react-query"
import { toast } from "sonner"
import { clientOption, useClientOption } from "@/api/app-options"
import { api } from "@/api/endpoints"
import { bootstrapQuery, keys, queryClient } from "@/api/queries"
import type { Bootstrap, NetworkItem, NetworkKind } from "@/api/types"
import { checkpointLabel, fileName, MODEL_OPTION_KEYS, writeModules } from "@/features/model-bar/model-options"
import { t } from "@/i18n"
import { isInPrompt } from "./prompt-insert"
import { planInsertion, readPrompts, useFormPrompts, writePrompts } from "./prompt-target"
import type { SortKey, SortOrder } from "./queries"
import type { PromptTarget } from "./store"

/** A bootstrap option the shared client-option table does not list. */
function bootstrapOption(options: Record<string, unknown> | undefined, key: string): unknown {
  return options?.[key]
}

export function separatorOf(options: Record<string, unknown> | undefined): string {
  const value = bootstrapOption(options, "extra_networks_add_text_separator")
  return typeof value === "string" ? value : " "
}

const SORT_FIELDS: Record<string, SortKey> = { Path: "path", Name: "name", "Date Created": "created", "Date Modified": "date" }

/** The default sort (`extra_networks_card_order_field` / `extra_networks_card_order`; upstream: path, ascending). */
export function defaultSortOf(options: Record<string, unknown> | undefined): { sort: SortKey; order: SortOrder } {
  const field = bootstrapOption(options, "extra_networks_card_order_field")
  return {
    sort: (typeof field === "string" && SORT_FIELDS[field]) || "path",
    order: bootstrapOption(options, "extra_networks_card_order") === "Descending" ? "desc" : "asc",
  }
}

/** Network-related settings, reactive. */
export function useNetworkSettings() {
  const { data } = useQuery(bootstrapQuery)
  const options = data?.options as Record<string, unknown> | undefined
  return {
    separator: separatorOf(options),
    /** `lora_preset_filter`: the LoRA browser starts with "compatible only". */
    presetFilter: bootstrapOption(options, "lora_preset_filter") === true,
    defaultSort: defaultSortOf(options),
    /** "Open folder" works (not `--hide-ui-dir-config`). */
    canOpenFolders: Boolean(data?.features.includes("folders.open")),
  }
}

function currentSeparator(): string {
  return separatorOf(queryClient.getQueryData<Bootstrap>(keys.app)?.options as Record<string, unknown> | undefined)
}

/** Loads a checkpoint (optimistic, like the model bar). */
export function selectCheckpoint(title: string) {
  const key = MODEL_OPTION_KEYS.checkpoint
  queryClient.setQueryData<Bootstrap>(keys.app, (data) => (data ? { ...data, options: { ...data.options, [key]: title } } : data))
  void api
    .patchOptions({ [key]: title })
    .catch((error: unknown) => toast.error(t("Could not change the model settings"), { description: error instanceof Error ? error.message : String(error) }))
    .finally(() => void queryClient.invalidateQueries({ queryKey: keys.app }))
}

/** Adds or removes a VAE / text encoder module. */
export function toggleModule(name: string): boolean {
  const current = clientOption("forge_additional_modules").map(fileName)
  const added = !current.includes(name)
  writeModules(added ? [...current, name] : current.filter((module) => module !== name))
  return added
}

export type ApplyOutcome = { added: boolean; target?: PromptTarget | "both" } | null

/**
 * Applies a card. `form` is the generation form whose prompts change;
 * `target` the prompt a card without negative text goes to.
 */
export function applyItem(item: NetworkItem, kind: Pick<NetworkKind, "action" | "negative">, form: string, target: PromptTarget): ApplyOutcome {
  if (kind.action === "checkpoint") {
    if (!item.value) return null
    selectCheckpoint(item.value)
    return { added: true }
  }
  if (kind.action === "module") {
    if (!item.value) return null
    return { added: toggleModule(item.value) }
  }
  if (kind.action !== "prompt") return null
  const plan = planInsertion(readPrompts(form), item, { negativeAllowed: kind.negative, target, separator: currentSeparator() })
  if (!plan) return null
  writePrompts(form, plan.changes)
  const changed = Object.keys(plan.changes)
  return { added: plan.added, target: changed.length > 1 ? "both" : (changed[0] as PromptTarget) }
}

/** Whether a card is "on": in a prompt, the loaded checkpoint, a chosen module. */
export function useActiveTest(kind: Pick<NetworkKind, "action" | "negative"> | undefined, form: string): (item: NetworkItem) => boolean {
  const prompts = useFormPrompts(form)
  const checkpoint = useClientOption("sd_model_checkpoint")
  const modules = useClientOption("forge_additional_modules")
  if (!kind) return () => false
  if (kind.action === "checkpoint") {
    const current = checkpoint ? checkpointLabel(checkpoint) : null
    return (item) => Boolean(item.value && current && (item.value === checkpoint || checkpointLabel(item.value) === current))
  }
  if (kind.action === "module") {
    const names = new Set(modules.map(fileName))
    return (item) => Boolean(item.value && names.has(item.value))
  }
  if (kind.action !== "prompt") return () => false
  return (item) => isInPrompt(prompts.positive, item.prompt) || (kind.negative && isInPrompt(prompts.negative, item.prompt))
}

export async function copyPath(item: NetworkItem) {
  try {
    await navigator.clipboard.writeText(item.path)
    toast.success(t("Path copied"), { description: item.path })
  } catch {
    toast.error(t("Could not copy to the clipboard"))
  }
}

export async function openFolder(item: NetworkItem) {
  try {
    await api.networks.openFolder(item.kind, item.name)
  } catch (error) {
    toast.error(t("Could not open the folder"), { description: error instanceof Error ? error.message : String(error) })
  }
}
