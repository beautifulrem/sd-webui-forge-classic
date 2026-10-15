/**
 * A schema-rendered extension page: its `page:<id>` form, kept across visits
 * (values, uploads and outputs survive leaving the page). The page's
 * `on_load` action runs each time it is shown.
 */
import { useEffect, useLayoutEffect } from "react"
import { useStore } from "zustand"
import { runAction } from "@/ui/actions"
import { FormProvider, OwnerProvider } from "@/ui/context"
import { RenderLayout } from "@/ui/Renderer"
import type { PageSchema } from "@/ui/schema"
import { flushPending, loadedSchema, loadPage, pageOwner, pageStore } from "./deliver"

/** schema-format.md §8 keys the shared `PageSchema` type does not list. */
type PageExtras = { on_load?: string | null }

export function PageView({ schema }: { schema: PageSchema }) {
  const store = pageStore(schema.id)
  const owner = schema.owner || pageOwner(schema.id)
  const onLoad = (schema as PageSchema & PageExtras).on_load
  // A load bumps the revision, which renders this view again with the schema in
  // place (the revision is part of the expression, so it is not memoized away).
  const revision = useStore(store, (s) => s.revision)
  const ready = revision > 0 && loadedSchema(store) === schema

  useLayoutEffect(() => {
    loadPage(schema)
    flushPending(schema.id)
  }, [schema])

  useEffect(() => {
    if (ready && onLoad) void runAction({ store, tab: owner }, owner, onLoad)
  }, [ready, onLoad, store, owner])

  if (!ready) return null
  return (
    <FormProvider formId={owner} store={store}>
      <OwnerProvider owner={owner}>
        <div className="flex flex-col gap-3">
          <RenderLayout layout={schema.layout} />
        </div>
      </OwnerProvider>
    </FormProvider>
  )
}
