/** Top-level renderers: a generation tab (core form + scripts) and a single-owner page. */
import { useLayoutEffect, useState } from "react"
import { CONFLICTS_ACTION, runAction } from "./actions"
import { splitRef } from "./conditions"
import { FormProvider, OwnerProvider } from "./context"
import type { OwnerSpec } from "./form-index"
import { type FormStore, getFormStore, isModified } from "./form-store"
import { DEFAULT_SLOT } from "./form-index"
import { ScriptsProvider, Slot } from "./layout/Slot"
import { RenderLayout } from "./Renderer"
import type { PageSchema, TabSchema } from "./schema"
import { tabOwners } from "./tab"

/**
 * Loads `owners` into the store before the first paint; renders nothing until
 * then. `afterLoad` runs once per load (e.g. to re-derive server state).
 */
function useLoaded(store: FormStore, source: object, owners: () => OwnerSpec[], conflictParams?: readonly string[], afterLoad?: () => void): boolean {
  const [loaded, setLoaded] = useState<object | null>(null)
  useLayoutEffect(() => {
    store.getState().load(owners(), { conflictParams })
    afterLoad?.()
    setLoaded(source)
    // `owners` derives from `source`.
    // oxlint-disable-next-line react-hooks/exhaustive-deps
  }, [store, source])
  return loaded === source
}

type TabFormProps = {
  schema: TabSchema
  /** Defaults to the schema's tab. */
  formId?: string
  /** Override the store (tests). */
  store?: FormStore
}

export function TabForm({ schema, formId = schema.tab, store = getFormStore(formId) }: TabFormProps) {
  const ready = useLoaded(store, schema, () => tabOwners(schema), schema.conflict_params, () => {
    // Patch locks are not persisted: re-derive them when a draft restored a conflict param.
    const state = store.getState()
    const restored = (schema.conflict_params ?? []).some((ref) => isModified(state, ...splitRef(ref, "core")))
    if (restored) void runAction({ store, tab: formId }, "core", CONFLICTS_ACTION)
  })
  if (!ready) return null
  const hasDefaultSlot = store.getState().index.slots.has(DEFAULT_SLOT)
  return (
    <FormProvider formId={formId} store={store}>
      <ScriptsProvider scripts={schema.scripts}>
        <div className="flex flex-col gap-3">
          <OwnerProvider owner="core">
            <RenderLayout layout={schema.core.layout} />
            {hasDefaultSlot ? null : <Slot section={DEFAULT_SLOT} />}
          </OwnerProvider>
        </div>
      </ScriptsProvider>
    </FormProvider>
  )
}

export function PageForm({ schema, store }: { schema: PageSchema; store?: FormStore }) {
  const formId = `page:${schema.id}`
  const formStore = store ?? getFormStore(formId)
  const owner = schema.owner || formId
  const ready = useLoaded(formStore, schema, () => [{ owner, layout: schema.layout, defaults: schema.defaults ?? {} }])
  if (!ready) return null
  return (
    <FormProvider formId={formId} store={formStore}>
      <OwnerProvider owner={owner}>
        <div className="flex flex-col gap-3">
          <RenderLayout layout={schema.layout} />
        </div>
      </OwnerProvider>
    </FormProvider>
  )
}
