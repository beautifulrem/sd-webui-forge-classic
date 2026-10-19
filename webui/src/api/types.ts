/**
 * API models for /api/v2. Server models are the types generated from the
 * server's /openapi.json (`gen/schema.d.ts`, regenerate with `pnpm gen:api`
 * after `tools/export_openapi.py`); only the free-form JSON the server types
 * as `dict` (schemas, param values, action patches) is refined here.
 */
import type { ChoiceOption, PageSchema, SettingsSchema, TabSchema } from "@/ui/schema"
import type { components, paths } from "./gen/schema"

export type { PageSchema, SettingsSchema, TabSchema }

type Schemas = components["schemas"]

/** A request model: the fields the server defaults may be left out. */
type Defaulted<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>

/** An extension page (former `on_ui_tabs`); `icon` is a kebab-case lucide name, `widget` a dedicated view. */
export type PageInfo = Schemas["PageInfo"]
export type LocaleInfo = Schemas["LocaleInfo"]
export type ModelRef = Schemas["ModelRef"]
export type ModelsSummary = Schemas["ModelsSummary"]
/** A script file, script or Extras operation that could not be loaded (bootstrap `not_ported`). */
export type NotPorted = Schemas["NotPorted"]
export type QueueStatus = Schemas["QueueStatus"]
export type AuthStatus = Schemas["AuthStatus"]

/** `GET /app`: what the shell needs at startup (`options` is the subset it reads). */
export type Bootstrap = Schemas["Bootstrap"]

export type Models = Schemas["Models"]

/** A prompt style (styles.csv row). */
export type PromptStyle = Schemas["PromptStyle"]

/** `POST /prompt/tokens`: count the tokens of one prompt the way the loaded model would. */
export type TokenCountRequest = Defaulted<Schemas["TokenCountRequest"], "steps" | "styles" | "negative">

/** Token count and the chunked limit ("12/75"); `null` while no model is loaded. */
export type TokenCount = Schemas["TokenCount"]

/** `POST /styles/apply`: the prompts with the styles written into them. */
export type StylesApplied = Schemas["StylesApplied"]
export type StylesApplyRequest = Schemas["StylesApplyRequest"]

/** `POST /jobs` answer: the first job, every job (`repeat` > 1) and the first job's preview sequence. */
export type JobCreated = Schemas["JobCreated"]

/** A form submitted as a job (the generation tabs and extras). */
export type FormTab = Schemas["Job"]["tab"]

/** Param values for one owner: `{param: value}`. */
export type ParamValues = Record<string, unknown>

export type JobRequest = Defaulted<Omit<Schemas["JobRequest"], "core" | "scripts">, "priority" | "repeat" | "forever"> & {
  core: ParamValues
  scripts: Record<string, ParamValues>
}

export type JobStatus = Schemas["Job"]["status"]

/** A job output: file id for /files/{id} and /thumbs/{id}. */
export type JobOutput = Schemas["JobOutput"]

/** A running job's progress snapshot (`progress` is 0..1 when the server knows it). */
export type JobProgress = Schemas["JobProgress"]

export type Job = Schemas["Job"]
export type JobDetail = Schemas["JobDetail"]

export type Page<T> = { items: T[]; next_cursor?: string | null; total?: number | null }

/** `GET /jobs` query: cursor, statuses (comma-separated), search, stars, form. */
export type JobsQuery = NonNullable<paths["/api/v2/jobs"]["get"]["parameters"]["query"]>
/** Why an imported job cannot run as it is: the arguments that do not map to the current parameters. */
export type OpaqueArgs = Schemas["OpaqueArgs"]

export type ImportSource = Schemas["ImportSourceInfo"]["id"]
export type ImportSourceInfo = Schemas["ImportSourceInfo"]
/** What an import did, per record (`outcome`) and in total. */
export type ImportReport = Schemas["ImportReport"]
export type ImportItem = Schemas["ImportItem"]
/** A translatable importer note: an English message with `{{name}}` placeholders and their values. */
export type JobNote = Schemas["Note"]

/** One output in the Gallery (`category`: the form or folder it came from). */
export type GalleryItem = Schemas["GalleryItem"]
export type GalleryCategory = GalleryItem["category"]
export type GalleryFacets = Schemas["GalleryFacets"]
/** `GET /gallery` query: search, filters (dates as ISO strings), sort and cursor. */
export type GalleryQuery = NonNullable<paths["/api/v2/gallery"]["get"]["parameters"]["query"]>
export type GallerySort = NonNullable<GalleryQuery["sort"]>

/** `POST /pnginfo`: an image's generation parameters, metadata entries and PNG chunks. */
export type ImageMetadata = Schemas["ImageMetadata"]
export type MetadataEntry = Schemas["MetadataEntry"]
export type PngChunk = Schemas["PngChunk"]

export type Upload = Schemas["Upload"]

/** `POST /outputs/save`: file names written to the Save folder, and the zip's file id when asked for. */
export type SaveResult = Schemas["SaveResult"]

export type OutputItem = Schemas["OutputItem"]

export type NetworkItem = Schemas["NetworkItem"]
export type NetworkMeta = Schemas["NetworkMeta"]
export type NetworkKind = Schemas["NetworkKind"]
export type NetworkFacets = Schemas["NetworkFacets"]
export type NetworkFolder = Schemas["NetworkFolder"]
/** `GET /networks/{kind}` query: search, filters, sort and cursor. */
export type NetworkQuery = NonNullable<paths["/api/v2/networks/{kind}"]["get"]["parameters"]["query"]>

/** GET/PATCH /options: settings values and layout (schema-format.md §6). */
export type OptionsResponse = SettingsSchema

/** A parsed value that did not fit its param (schema-format.md §7). */
export type InfotextProblem = Schemas["ErrorItem"]

/** `POST /infotext/image`: what PNG Info shows for an uploaded or generated image. */
export type ImageInfo = Schemas["ImageInfo"]

export type InfotextParsed = Defaulted<Omit<Schemas["InfotextParsed"], "scripts">, "params" | "problems"> & {
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
  hint?: string | null
  minimum?: number
  maximum?: number
  step?: number
}

export type ToastLevel = "info" | "success" | "warning" | "error"

/** `message`: a text, or a `{ text, params }` message (`remi.ui.message`) translated with its placeholders filled. */
export type ActionToast = { level?: ToastLevel; message: string | { text: string; params?: Record<string, unknown> } }

/** Reserved keys of an action response (schema-format.md §5). */
export type ActionReserved = {
  __toast__?: ActionToast
  /** Values for the owner's `output` / `gallery` nodes, by name. */
  __outputs__?: Record<string, unknown>
  /** Settings the action already stored on the server; the client only updates its cache. */
  __settings__?: Record<string, unknown>
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
