/** Providers for the renderer: form, owner, inherited enablement (hooks live in hooks.ts). */
import { type ReactNode, use } from "react"
import type { FormStore } from "./form-store"
import { EnabledContext, FormContext, OwnerContext } from "./hooks"

export function FormProvider({ formId, store, children }: { formId: string; store: FormStore; children: ReactNode }) {
  return <FormContext value={{ formId, store }}>{children}</FormContext>
}

export function OwnerProvider({ owner, children }: { owner: string; children: ReactNode }) {
  return <OwnerContext value={owner}>{children}</OwnerContext>
}

/** Disables everything below when `enabled` is false (inherits ancestors). */
export function EnabledProvider({ enabled, children }: { enabled: boolean; children: ReactNode }) {
  const parent = use(EnabledContext)
  return <EnabledContext value={parent && enabled}>{children}</EnabledContext>
}
