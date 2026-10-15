/**
 * Hand-written API models for /api/v2. They mirror the server's Pydantic
 * models; when `webui/src/api/gen` is generated from /openapi.json these become
 * re-exports of the generated types.
 */
import type { ChoiceOption, PageSchema, SettingsSchema, TabSchema } from "@/ui/schema"

export type { PageSchema, SettingsSchema, TabSchema }

/** An extension page (former `on_ui_tabs`). */
export type PageInfo = {
  id: string
  title: string
  /** lucide icon name in kebab-case, e.g. "calendar-clock". */
  icon?: string | null
  /** A dedicated frontend view in src/pages; otherwise the page is schema-rendered. */
  widget?: string | null
}

export type LocaleInfo = { id: string; name: string }

export type ModelRef = { name: string; title?: string; hash?: string | null }

export type ModelsSummary = {
  checkpoint?: string | null
  vae?: string | null
  text_encoders?: string[]
  preset?: string | null
}

export type Bootstrap = {
  version: string
  features: string[]
  /** Built-in tab ids in display order; hidden tabs are omitted. */
  tabs: string[]
  /** Extension pages, in display order. */
  pages: PageInfo[]
  locales: LocaleInfo[]
  /** Server default locale (e.g. "zh_CN"); the user's choice wins. */
  locale?: string | null
  /** The subset of options the shell needs at startup. */
  options: Record<string, unknown>
  models: ModelsSummary
}

export type Models = {
  checkpoints: ModelRef[]
  vae: ModelRef[]
  text_encoders: ModelRef[]
  samplers: { name: string; aliases?: string[] }[]
  schedulers: { name: string; label?: string }[]
  upscalers: { name: string }[]
}

/** A form submitted as a job (the generation tabs and extras). */
export type FormTab = "txt2img" | "img2img" | "extras"

/** Param values for one owner: `{param: value}`. */
export type ParamValues = Record<string, unknown>

export type JobRequest = {
  tab: FormTab
  core: ParamValues
  scripts: Record<string, ParamValues>
  priority?: number
  repeat?: number
}

export type JobStatus = "queued" | "running" | "completed" | "failed" | "cancelled" | "interrupted"

export type JobOutput = {
  /** File id for /files/{id} and /thumbs/{id}. */
  id: string
  kind?: "image" | "text" | "json"
  width?: number
  height?: number
  infotext?: string | null
}

export type Job = {
  id: string
  seq?: number
  tab: FormTab
  status: JobStatus
  created_at: string
  started_at?: string | null
  finished_at?: string | null
  progress?: { step: number; steps: number; eta?: number | null } | null
  outputs: JobOutput[]
  error?: string | null
  /** Short description for lists (usually the prompt). */
  summary?: string | null
}

export type Page<T> = { items: T[]; next_cursor?: string | null; total?: number | null }

export type Upload = { id: string; width?: number | null; height?: number | null; name?: string; mime?: string }

export type OutputItem = JobOutput & { created_at?: string; job_id?: string | null }

export type NetworkItem = {
  name: string
  title?: string
  kind: string
  path?: string
  preview?: string | null
  description?: string | null
  prompt?: string | null
  negative_prompt?: string | null
  search_terms?: string[]
}

/** GET/PATCH /options: settings values and layout (schema-format.md §6). */
export type OptionsResponse = SettingsSchema

export type InfotextParsed = {
  tab: FormTab
  core: ParamValues
  scripts: Record<string, ParamValues>
}

/** Per-param patch returned by an action; `null` label / info clears the text. */
export type ParamPatch = {
  value?: unknown
  choices?: ChoiceOption[]
  visible?: boolean
  enabled?: boolean
  label?: string | null
  info?: string | null
  minimum?: number
  maximum?: number
  step?: number
}

export type ToastLevel = "info" | "success" | "warning" | "error"

export type ActionToast = { level?: ToastLevel; message: string }

/** Reserved keys of an action response (schema-format.md §5). */
export type ActionReserved = {
  __toast__?: ActionToast
  /** Values for the owner's `output` / `gallery` nodes, by name. */
  __outputs__?: Record<string, unknown>
  /** Settings to change; the client applies them with `PATCH /options`. */
  __options__?: Record<string, unknown>
}

/**
 * An action's patch: `{param | "owner.param": ParamPatch}` plus the reserved
 * keys. `ui/actions.ts` validates and splits it (unknown `__keys__` ignored).
 */
export type ActionResponse = ActionReserved & { [param: string]: ParamPatch | ActionReserved[keyof ActionReserved] }

/** Arguments of the built-in core actions (schema-format.md §5). */
export type ActionExtra =
  /** `__conflicts__`: the `"owner.param"` the user just changed (absent after a bulk reset). */
  | { selected?: string }
  /** `__preset__`: the Anima preset label. */
  | { label: string }
  /** `__forge_preset__`: the UI preset. */
  | { preset: string }

export type ActionRequest = {
  /** The form that invoked the action: txt2img / img2img / extras, or `page:<id>`. */
  tab?: string
  /** The calling owner's current values. */
  values: ParamValues
  /** The generation form's values (when the owner is not `core`). */
  core?: ParamValues
  /** Every other non-core owner's values in the form, by owner id (cross-owner actions). */
  scripts?: Record<string, ParamValues>
} & ActionExtra
