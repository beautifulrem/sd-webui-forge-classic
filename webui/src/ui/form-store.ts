/**
 * Per-form state (one store per tab or page): values per owner, defaults,
 * action overrides (choices / visible / enabled / label / info / bounds),
 * outputs, and UI state (section open, selected tabs).
 *
 * Values, open sections and selected tabs persist as drafts in localStorage
 * (`remi-form:<formId>`), written at most every 300 ms and on page hide.
 * Uploads, hidden server state and `persist: false` inputs are not drafted.
 *
 * Programmatic writes (`setValues`, `applyPatches`) skip params locked by a
 * toggle's `locks` (see locks.ts); `setValue` and `reset` are user edits.
 */
import { createStore, type StoreApi } from "zustand/vanilla"
import { createJSONStorage, persist, type StateStorage } from "zustand/middleware"
import type { ParamPatch } from "@/api/types"
import { jsonEqual } from "./conditions"
import { buildIndex, defaultOf, type FormIndex, type OwnerSpec } from "./form-index"
import { collectLocks, lockedNames, type ParamLock } from "./locks"
import type { ChoiceOption, ValidationError } from "./schema"
import { isPersisted, restoreValue } from "./values"

export type OwnerValues = Record<string, unknown>

/** Display-time overrides set by action patches; combined with the node's static keys. */
export type Override = {
  visible?: boolean
  enabled?: boolean
  /** `null` clears the static text. */
  label?: string | null
  info?: string | null
  choices?: ChoiceOption[]
  minimum?: number
  maximum?: number
  step?: number
}

export type FormState = {
  values: Record<string, OwnerValues>
  defaults: Record<string, OwnerValues>
  overrides: Record<string, Record<string, Override>>
  outputs: Record<string, Record<string, unknown>>
  /** Section / group open state by key (see `sectionKey`). */
  open: Record<string, boolean>
  /** Selected tab of unbound `tabs` nodes, by key. */
  tabs: Record<string, string>
  /** Server validation messages of the last rejected submit, by owner and param. */
  errors: Record<string, Record<string, string>>
  /** `"owner.param"` refs whose changes run `core.__conflicts__` (tab schemas). */
  conflictParams: ReadonlySet<string>
  index: FormIndex
  /** Toggles that protect another param from programmatic writes while on. */
  locks: readonly ParamLock[]
  /** Bumped by `load`; lets consumers know the schema changed. */
  revision: number
}

export type FormActions = {
  /** (Re)initialises from schema owners, keeping compatible draft values. */
  load: (owners: readonly OwnerSpec[], options?: { conflictParams?: readonly string[] }) => void
  setValue: (owner: string, name: string, value: unknown) => void
  /** A programmatic write (paste, send-to, widgets): locked params are skipped. */
  setValues: (owner: string, values: OwnerValues) => void
  /** Resets the given names (all of the owner's when omitted) to their defaults. */
  reset: (owner: string, names?: readonly string[]) => void
  /** Applies param patches `[owner, name, patch]` from an action (locked params keep their value). */
  applyPatches: (patches: readonly [string, string, ParamPatch][]) => void
  setOutputs: (owner: string, outputs: Record<string, unknown>) => void
  /** Shows server validation errors on their fields and opens what hides them. */
  setErrors: (errors: readonly ValidationError[]) => void
  clearErrors: () => void
  setOpen: (key: string, open: boolean) => void
  setTab: (key: string, value: string) => void
}

export type FormStore = StoreApi<FormState & FormActions>

type Persisted = Pick<FormState, "values" | "open" | "tabs">

const emptyIndex = (): FormIndex => ({ inputs: new Map(), slots: new Set(), reveal: new Map() })

/** `errors` minus the given names of `owner` (the same object when nothing changes). */
function withoutErrors(errors: FormState["errors"], owner: string, names: Iterable<string>): FormState["errors"] {
  const current = errors[owner]
  if (!current) return errors
  let next: Record<string, string> | undefined
  for (const name of names) {
    if (!Object.hasOwn(current, name)) continue
    next ??= { ...current }
    delete next[name]
  }
  return next ? { ...errors, [owner]: next } : errors
}

/** localStorage with coalesced writes: a slider drag must not serialise the form per frame. */
function debouncedStorage(delay = 300): StateStorage {
  const pending = new Map<string, string>()
  let timer: ReturnType<typeof setTimeout> | undefined
  const flush = () => {
    clearTimeout(timer)
    timer = undefined
    for (const [key, value] of pending) {
      try {
        localStorage.setItem(key, value)
      } catch {
        // Quota exceeded or storage disabled: drafts are best-effort.
      }
    }
    pending.clear()
  }
  if (typeof window !== "undefined") {
    window.addEventListener("pagehide", flush)
    document.addEventListener("visibilitychange", () => document.visibilityState === "hidden" && flush())
  }
  return {
    getItem: (key) => pending.get(key) ?? localStorage.getItem(key),
    setItem: (key, value) => {
      pending.set(key, value)
      timer ??= setTimeout(flush, delay)
    },
    removeItem: (key) => {
      pending.delete(key)
      localStorage.removeItem(key)
    },
  }
}

const storage = createJSONStorage<Persisted>(() => debouncedStorage())

export const draftKey = (formId: string) => `remi-form:${formId}`

/** Every store created in this session (upload previews check them for references). */
const liveStores = new Set<FormStore>()

export function allFormStores(): Iterable<FormStore> {
  return liveStores
}

export function createFormStore(formId: string, { persistDrafts = true } = {}): FormStore {
  const initializer = (
    set: StoreApi<FormState & FormActions>["setState"],
    get: StoreApi<FormState & FormActions>["getState"],
  ): FormState & FormActions => ({
    values: {},
    defaults: {},
    overrides: {},
    outputs: {},
    open: {},
    tabs: {},
    errors: {},
    conflictParams: new Set(),
    index: emptyIndex(),
    locks: [],
    revision: 0,

    load(owners, options = {}) {
      const index = buildIndex(owners)
      // Before the first load the values are the drafts hydrated from localStorage
      const hydrated = get().revision === 0
      const drafts = get().values
      const values: Record<string, OwnerValues> = {}
      const defaults: Record<string, OwnerValues> = {}
      for (const spec of owners) {
        const ownerValues: OwnerValues = {}
        const ownerDefaults: OwnerValues = {}
        // Defaults for names without a node (e.g. legacy positional extras) are kept verbatim.
        for (const [name, value] of Object.entries(spec.defaults)) {
          ownerDefaults[name] = value
          ownerValues[name] = value
        }
        for (const [name, node] of index.inputs.get(spec.owner) ?? []) {
          const fallback = defaultOf(spec, node)
          ownerDefaults[name] = fallback
          const draft = hydrated && node.persist === false ? undefined : drafts[spec.owner]?.[name]
          ownerValues[name] = restoreValue(node, draft, fallback)
        }
        values[spec.owner] = ownerValues
        defaults[spec.owner] = ownerDefaults
      }
      set((s) => ({
        values,
        defaults,
        index,
        locks: collectLocks(index),
        overrides: {},
        outputs: {},
        errors: {},
        conflictParams: new Set(options.conflictParams ?? []),
        revision: s.revision + 1,
      }))
    },

    setValue(owner, name, value) {
      set((s) => ({
        values: { ...s.values, [owner]: { ...s.values[owner], [name]: value } },
        errors: withoutErrors(s.errors, owner, [name]),
      }))
    },

    setValues(owner, partial) {
      set((s) => {
        const locked = lockedNames(s.locks, s.values, owner)
        const kept = locked.size === 0 ? partial : Object.fromEntries(Object.entries(partial).filter(([name]) => !locked.has(name)))
        return {
          values: { ...s.values, [owner]: { ...s.values[owner], ...kept } },
          errors: withoutErrors(s.errors, owner, Object.keys(kept)),
        }
      })
    },

    reset(owner, names) {
      set((s) => {
        const defaults = s.defaults[owner] ?? {}
        const reset = (names ?? Object.keys(defaults)).filter((name) => Object.hasOwn(defaults, name))
        const next = { ...s.values[owner] }
        for (const name of reset) next[name] = defaults[name]
        return { values: { ...s.values, [owner]: next }, errors: withoutErrors(s.errors, owner, reset) }
      })
    },

    applyPatches(patches) {
      set((s) => {
        const values = { ...s.values }
        const overrides = { ...s.overrides }
        let errors = s.errors
        const locked = new Map<string, Set<string>>()
        const isLocked = (owner: string, name: string) => {
          if (!locked.has(owner)) locked.set(owner, lockedNames(s.locks, s.values, owner))
          return locked.get(owner)!.has(name)
        }
        for (const [owner, name, patch] of patches) {
          const { value, ...display } = patch
          if ("value" in patch && !isLocked(owner, name)) {
            values[owner] = { ...values[owner], [name]: value }
            errors = withoutErrors(errors, owner, [name])
          }
          if (Object.keys(display).length > 0) {
            overrides[owner] = { ...overrides[owner], [name]: { ...overrides[owner]?.[name], ...display } }
          }
        }
        return { values, overrides, errors }
      })
    },

    setErrors(list) {
      set((s) => {
        const errors: FormState["errors"] = {}
        const open = { ...s.open }
        const tabs = { ...s.tabs }
        for (const { owner, param, message } of list) {
          // Several messages for one param are joined.
          const previous = errors[owner]?.[param]
          errors[owner] = { ...errors[owner], [param]: previous ? `${previous}; ${message}` : message }
          const path = s.index.reveal.get(owner)?.get(param)
          for (const key of path?.open ?? []) open[key] = true
          for (const [key, value] of path?.tabs ?? []) tabs[key] = value
        }
        return { errors, open, tabs }
      })
    },

    clearErrors() {
      set({ errors: {} })
    },

    setOutputs(owner, outputs) {
      set((s) => ({ outputs: { ...s.outputs, [owner]: { ...s.outputs[owner], ...outputs } } }))
    },

    setOpen(key, open) {
      set((s) => ({ open: { ...s.open, [key]: open } }))
    },

    setTab(key, value) {
      set((s) => ({ tabs: { ...s.tabs, [key]: value } }))
    },
  })

  const store = persistDrafts
    ? createStore<FormState & FormActions>()(
        persist(initializer, {
      name: draftKey(formId),
      version: 1,
      storage,
      partialize: (s): Persisted => ({ values: persistableValues(s), open: s.open, tabs: s.tabs }),
          // Drafts are merged in `load` against the schema; hydration only restores the raw data.
          merge: (persisted, current) => ({ ...current, ...(persisted as Persisted | undefined) }),
        }),
      )
    : createStore<FormState & FormActions>()(initializer)
  liveStores.add(store)
  return store
}

/** Values minus what must not be restored: uploads, hidden server state, `persist: false` inputs. */
function persistableValues(s: FormState): Record<string, OwnerValues> {
  const out: Record<string, OwnerValues> = {}
  for (const [owner, values] of Object.entries(s.values)) {
    const nodes = s.index.inputs.get(owner)
    const kept: OwnerValues = {}
    for (const [name, value] of Object.entries(values)) {
      const node = nodes?.get(name)
      if (!node || (isPersisted(node) && node.persist !== false)) kept[name] = value
    }
    out[owner] = kept
  }
  return out
}

const stores = new Map<string, FormStore>()

/** The (cached) store of a form: `txt2img`, `img2img`, `extras`, `page:<id>`. */
export function getFormStore(formId: string): FormStore {
  let store = stores.get(formId)
  if (!store) {
    store = createFormStore(formId)
    stores.set(formId, store)
  }
  return store
}

// ---------------------------------------------------------------- selectors

export function isModified(s: FormState, owner: string, name: string): boolean {
  const defaults = s.defaults[owner]
  if (!defaults || !Object.hasOwn(defaults, name)) return false
  return !jsonEqual(s.values[owner]?.[name], defaults[name])
}

export function anyModified(s: FormState, owner: string, names: readonly string[]): boolean {
  return names.some((name) => isModified(s, owner, name))
}
