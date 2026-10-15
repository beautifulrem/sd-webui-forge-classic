/**
 * Replacing (or removing) the image a form's layers are painted over: the
 * painted layers go with it. Their last strokes are uploaded first, so the
 * change can be undone from a toast (the old image and its layers come back
 * from their uploads).
 */
import { toast } from "sonner"
import { t } from "@/i18n"
import { runChangeEffects } from "@/ui/actions"
import type { FormStore } from "@/ui/form-store"
import { flushPendingWrites } from "@/ui/pending-writes"
import { releaseUpload } from "@/ui/uploads"

export type BackgroundTarget = {
  store: FormStore
  /** Form id (its pending writes are flushed first, its change effects run). */
  formId: string
  owner: string
  /** The background image param. */
  background: string
  /** The params painted over it (cleared with it). */
  painted: readonly string[]
}

/** Writes params of `owner` as a user edit (change effects run). */
export function writeValues({ store, formId, owner }: Pick<BackgroundTarget, "store" | "formId" | "owner">, values: Record<string, unknown>) {
  store.getState().setValues(owner, values)
  runChangeEffects({ store, tab: formId }, owner, Object.keys(values))
}

/** Sets the background to `id` (null: removes it), clearing the painted layers; undoable when they had paint. */
export async function replaceBackground(target: BackgroundTarget, id: string | null): Promise<void> {
  const { store, formId, owner, background, painted } = target
  await flushPendingWrites(formId).catch(() => {})
  const current = store.getState().values[owner] ?? {}
  const old = current[background] ?? null
  if (old === id) return
  const values: Record<string, unknown> = { [background]: id }
  const previous: Record<string, unknown> = { [background]: old }
  let hadPaint = false
  for (const param of painted) {
    values[param] = null
    previous[param] = current[param] ?? null
    if (previous[param]) hadPaint = true
  }
  writeValues(target, values)
  const release = () => {
    if (typeof old === "string" && store.getState().values[owner]?.[background] !== old) releaseUpload(old)
  }
  if (!hadPaint) {
    release()
    return
  }
  // The old image stays referenced by the undo until the toast goes.
  toast(id ? t("Image replaced") : t("Image removed"), {
    description: t("The painted layers were cleared with it."),
    action: { label: t("Undo"), onClick: () => writeValues(target, previous) },
    onAutoClose: release,
    onDismiss: release,
  })
}
