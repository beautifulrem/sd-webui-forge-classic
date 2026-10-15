/** Renderer contexts and hooks: which form/owner a node belongs to, and inherited enablement. */
import { createContext, use, useEffect, useRef } from "react"
import { useStore } from "zustand"
import type { ActionExtra } from "@/api/types"
import { runAction, runChangeEffects } from "./actions"
import { evaluate } from "./conditions"
import { type FormState, type FormStore, isModified, type Override } from "./form-store"
import type { Condition, InputNode } from "./schema"

export type FormContextValue = {
  /** Form id: txt2img / img2img / extras / page:<id>. */
  formId: string
  store: FormStore
}

export const FormContext = createContext<FormContextValue | null>(null)
export const OwnerContext = createContext<string>("core")
export const EnabledContext = createContext<boolean>(true)

export function useForm(): FormContextValue {
  const ctx = use(FormContext)
  if (!ctx) throw new Error("renderer nodes must be inside <FormProvider>")
  return ctx
}

export const useOwner = () => use(OwnerContext)
export const useInheritedEnabled = () => use(EnabledContext)

export function useFormState<T>(selector: (s: FormState) => T): T {
  return useStore(useForm().store, selector)
}

/** Live result of a condition for the current owner. */
export function useCondition(cond: Condition | null | undefined): boolean {
  const owner = useOwner()
  return useFormState((s) => evaluate(cond, owner, (o, n) => s.values[o]?.[n]))
}

/** Runs an action of the current owner (or `owner`) and applies the patch. */
export function useRunAction() {
  const { store, formId } = useForm()
  const owner = useOwner()
  return (action: string, extra?: ActionExtra, target = owner) => runAction({ store, tab: formId }, target, action, extra)
}

const CONTINUOUS = new Set(["slider", "number", "text", "color", "json"])
const EMPTY_OVERRIDE: Override = {}

/** A patch override wins over the node's static text; `null` in the patch clears it. */
function displayText(override: Override, key: "label" | "info", fallback: string | null | undefined): string {
  if (Object.hasOwn(override, key)) return override[key] ?? ""
  return fallback ?? ""
}

export type ParamBinding<T> = {
  owner: string
  value: T
  defaultValue: unknown
  modified: boolean
  override: Override
  /** Untranslated label and info, after patch overrides ("" = none). */
  label: string
  info: string
  /** Server validation message from the last rejected submit. */
  error: string | undefined
  /** Node enablement: inherited × enabled_if × patch override. */
  enabled: boolean
  /** User edit: stores the value, then `on_change` and conflict locks (debounced for continuous kinds). */
  setValue: (value: T) => void
  reset: () => void
}

/** Binds an input node to the form store. */
export function useParam<T = unknown>(node: InputNode): ParamBinding<T> {
  const { store, formId } = useForm()
  const owner = useOwner()
  const inherited = useInheritedEnabled()
  const value = useStore(store, (s) => s.values[owner]?.[node.name]) as T
  const defaultValue = useStore(store, (s) => s.defaults[owner]?.[node.name])
  const modified = useStore(store, (s) => isModified(s, owner, node.name))
  const override = useStore(store, (s) => s.overrides[owner]?.[node.name]) ?? EMPTY_OVERRIDE
  const error = useStore(store, (s) => s.errors[owner]?.[node.name])
  const conditionEnabled = useCondition(node.enabled_if)
  const enabled = inherited && conditionEnabled && override.enabled !== false

  const timer = useRef<ReturnType<typeof setTimeout>>(undefined)
  useEffect(() => () => clearTimeout(timer.current), [])

  const changed = () => {
    clearTimeout(timer.current)
    const run = () => runChangeEffects({ store, tab: formId }, owner, [node.name])
    if (CONTINUOUS.has(node.type)) timer.current = setTimeout(run, 300)
    else run()
  }

  return {
    owner,
    value,
    defaultValue,
    modified,
    override,
    label: displayText(override, "label", node.label),
    info: displayText(override, "info", node.info),
    error,
    enabled,
    setValue: (next) => {
      store.getState().setValue(owner, node.name, next)
      changed()
    },
    reset: () => {
      store.getState().reset(owner, [node.name])
      changed()
    },
  }
}
