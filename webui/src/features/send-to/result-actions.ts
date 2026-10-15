/** What the result actions do, without UI: reuse seed, use parameters, send to. */
import { toast } from "sonner"
import { api } from "@/api/endpoints"
import type { FormTab } from "@/api/types"
import { downloadFile } from "@/features/gallery/files"
import { t } from "@/i18n"
import { runChangeEffects } from "@/ui/actions"
import type { FormIndex } from "@/ui/form-index"
import { getFormStore } from "@/ui/form-store"
import { infotextValue } from "@/ui/infotext"
import { type SendItem, type SendTarget, sendTo } from "./send-to"

/** The `[owner, name]` of the input that reads infotext key `key` ("Seed" → the seed script's `seed`). */
export function findByInfotext(index: FormIndex, key: string): [owner: string, name: string] | null {
  for (const [owner, nodes] of index.inputs) {
    for (const [name, node] of nodes) if (node.infotext === key) return [owner, name]
  }
  return null
}

/** Sets the form's seed to the one recorded in `infotext`; false when either is missing. */
export function reuseSeed(tab: FormTab, infotext: string | null | undefined): boolean {
  const seed = infotextValue(infotext, "Seed")
  const store = getFormStore(tab)
  const target = findByInfotext(store.getState().index, "Seed")
  if (seed === null || !target) return false
  const [owner, name] = target
  store.getState().setValue(owner, name, seed)
  runChangeEffects({ store, tab }, owner, [name])
  return true
}

const TARGET_LABEL: Record<SendTarget, string> = {
  txt2img: "txt2img",
  img2img: "img2img",
  inpaint: "Inpaint",
  extras: "Extras",
}

/** Pastes a result's parameters (all of them, seed and size included) into form `tab`. */
export function applyParameters(tab: SendTarget, infotext: string) {
  const id = toast.loading(t("Reading parameters…"))
  sendTo(tab, { infotext }, {})
    .then((count) => toast.success(t("Parameters applied: {{count}}", { count }), { id }))
    .catch((error: unknown) =>
      toast.error(t("Could not read the parameters"), { id, description: error instanceof Error ? error.message : String(error) }),
    )
}

/** Runs a send with progress and result toasts. */
export function sendWithFeedback(target: SendTarget, item: SendItem) {
  const where = t(TARGET_LABEL[target])
  const id = toast.loading(t("Sending to {{target}}…", { target: where }))
  sendTo(target, item)
    .then(() => toast.success(t("Sent to {{target}}", { target: where }), { id }))
    .catch((error: unknown) =>
      toast.error(t("Could not send to {{target}}", { target: where }), { id, description: error instanceof Error ? error.message : String(error) }),
    )
}

/** Copies outputs to the Save folder (`outdir_save`); with `zip`, also downloads them packed. */
export async function saveResults(fileIds: readonly string[], zip = false) {
  const id = toast.loading(t("Saving…"))
  try {
    const result = await api.saveOutputs({ ids: [...fileIds], zip })
    if (zip && result.zip) downloadFile(result.zip)
    toast.success(t("Saved: {{count}}", { count: result.saved.length }), { id })
  } catch (error) {
    toast.error(t("Could not save"), { id, description: error instanceof Error ? error.message : String(error) })
  }
}
