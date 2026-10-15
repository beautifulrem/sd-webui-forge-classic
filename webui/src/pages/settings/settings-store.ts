/**
 * Settings editor state: one form store (owner `settings`, not persisted)
 * whose defaults are the built-in defaults, so the renderer's modified dot
 * resets a setting to its default; the server's stored values are kept next
 * to it, and the difference between the two is the unsaved edits.
 *
 * Edits survive leaving the page (the store is module-level) and an options
 * reload (`options.changed`): a schema update keeps every unsaved edit.
 * Applying sends the edits with `PATCH /options`, shows them as saved at
 * once, and rolls back what the server rejected.
 */
import { create } from "zustand"
import { api } from "@/api/endpoints"
import { validationErrors } from "@/api/http"
import { keys, queryClient } from "@/api/queries"
import { jsonEqual } from "@/ui/conditions"
import { createFormStore, type FormStore } from "@/ui/form-store"
import type { GroupNode, SettingNode, SettingsSchema, ValidationError } from "@/ui/schema"
import { settingInputs, unsavedKeys } from "./settings-model"

export const SETTINGS_OWNER = "settings"
export const SETTINGS_FORM = "settings"

export const settingsForm: FormStore = createFormStore(SETTINGS_FORM, { persistDrafts: false })

type Notice = { restart: string[]; reload: string[] }

type SettingsEditorState = {
  /** The schema the store was last loaded from. */
  schema: SettingsSchema | null
  /** Values stored on the server (as last known). */
  saved: Record<string, unknown>
  /** Keys of the schema's inputs (the editable set). */
  inputs: ReadonlyMap<string, SettingNode>
  /** An apply in flight. */
  applying: boolean
  /** Applied settings that take effect after a restart / UI reload. */
  notice: Notice
  dismissNotice: () => void
}

export const useSettingsEditor = create<SettingsEditorState>()((set) => ({
  schema: null,
  saved: {},
  inputs: new Map(),
  applying: false,
  notice: { restart: [], reload: [] },
  dismissNotice: () => set({ notice: { restart: [], reload: [] } }),
}))

export function editorValues(): Record<string, unknown> {
  return settingsForm.getState().values[SETTINGS_OWNER] ?? {}
}

/** The unsaved edits `{key: value}`. */
export function pendingChanges(): Record<string, unknown> {
  const { saved, inputs } = useSettingsEditor.getState()
  const values = editorValues()
  const known = new Set(inputs.keys())
  return Object.fromEntries(unsavedKeys(values, saved, known).map((key) => [key, values[key]]))
}

/**
 * Loads `schema` into the editor, keeping unsaved edits (an edit equal to the
 * new stored value is no longer unsaved) and their validation messages.
 */
export function syncSchema(schema: SettingsSchema) {
  const editor = useSettingsEditor.getState()
  if (editor.schema === schema) return
  const previousValues = editorValues()
  const keep = editor.schema ? unsavedKeys(previousValues, editor.saved) : []
  const nodes = settingInputs(schema)
  const layout: GroupNode = { type: "group", children: nodes }
  const defaults = { ...schema.values, ...schema.defaults }
  const errors = settingsForm.getState().errors[SETTINGS_OWNER] ?? {}
  settingsForm.getState().load([{ owner: SETTINGS_OWNER, layout, defaults }])
  const values: Record<string, unknown> = {}
  for (const node of nodes) values[node.name] = schema.values[node.name] ?? node.default ?? null
  for (const key of keep) if (Object.hasOwn(values, key)) values[key] = previousValues[key]
  settingsForm.setState((s) => ({
    values: { ...s.values, [SETTINGS_OWNER]: { ...s.values[SETTINGS_OWNER], ...values } },
    errors: { [SETTINGS_OWNER]: Object.fromEntries(Object.entries(errors).filter(([key]) => keep.includes(key))) },
  }))
  useSettingsEditor.setState({ schema, saved: { ...schema.values }, inputs: new Map(nodes.map((node) => [node.name, node])) })
}

/** Drops every unsaved edit (or the given keys). */
export function discardChanges(only?: readonly string[]) {
  const { saved } = useSettingsEditor.getState()
  const values = editorValues()
  const names = only ?? Object.keys(values)
  const next: Record<string, unknown> = {}
  for (const key of names) if (Object.hasOwn(saved, key)) next[key] = saved[key]
  settingsForm.getState().setValues(SETTINGS_OWNER, next)
  settingsForm.getState().clearErrors()
}

/** Stages values as unsaved edits (settings import). */
export function stageValues(values: Record<string, unknown>) {
  settingsForm.getState().setValues(SETTINGS_OWNER, values)
}

function setSaved(values: Record<string, unknown>) {
  useSettingsEditor.setState((s) => ({ saved: { ...s.saved, ...values } }))
  queryClient.setQueryData<SettingsSchema>(keys.options, (data) => (data ? { ...data, values: { ...data.values, ...(values as SettingsSchema["values"]) } } : data))
}

export type ApplyResult = { applied: string[]; errors: ValidationError[]; message?: string }

/**
 * Applies the unsaved edits. Optimistic: they show as saved at once; when the
 * server refuses, they are edits again (rejected ones with their messages)
 * until its current values arrive.
 */
export async function applyChanges(): Promise<ApplyResult> {
  const changes = pendingChanges()
  const keysToApply = Object.keys(changes)
  if (keysToApply.length === 0) return { applied: [], errors: [] }
  const { saved: before, inputs } = useSettingsEditor.getState()
  const previous = Object.fromEntries(keysToApply.map((key) => [key, before[key]]))
  useSettingsEditor.setState({ applying: true })
  settingsForm.getState().clearErrors()
  setSaved(changes)
  try {
    const schema = await api.patchOptions(changes)
    queryClient.setQueryData(keys.options, schema)
    noteEffects(keysToApply, inputs)
    return { applied: keysToApply, errors: [] }
  } catch (error) {
    const errors = (validationErrors(error) ?? []).filter((e) => e.owner === SETTINGS_OWNER)
    // Nothing is known to be applied: a 422 for invalid values applies nothing,
    // one for a failing `onchange` applies the others. Every edit stays an
    // edit; the refetch below (syncSchema) then drops those the server stored.
    useSettingsEditor.setState((s) => ({ saved: { ...s.saved, ...previous } }))
    if (errors.length > 0) settingsForm.getState().setErrors(errors.map((e) => ({ ...e, owner: SETTINGS_OWNER })))
    return { applied: [], errors, message: error instanceof Error ? error.message : String(error) }
  } finally {
    useSettingsEditor.setState({ applying: false })
    void queryClient.invalidateQueries({ queryKey: keys.options })
    void queryClient.invalidateQueries({ queryKey: keys.app })
  }
}

function noteEffects(applied: readonly string[], inputs: ReadonlyMap<string, SettingNode>) {
  const restart = applied.filter((key) => inputs.get(key)?.needs_restart)
  const reload = applied.filter((key) => inputs.get(key)?.needs_reload_ui && !inputs.get(key)?.needs_restart)
  if (restart.length === 0 && reload.length === 0) return
  useSettingsEditor.setState((s) => ({
    notice: { restart: [...new Set([...s.notice.restart, ...restart])], reload: [...new Set([...s.notice.reload, ...reload])] },
  }))
}

/** Whether `key`'s editor value equals the stored one. */
export function isSaved(key: string): boolean {
  return jsonEqual(editorValues()[key], useSettingsEditor.getState().saved[key])
}
