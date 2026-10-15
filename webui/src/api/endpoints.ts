/**
 * Typed endpoint functions for /api/v2, built on the client generated from
 * the server's OpenAPI schema: paths, parameters and bodies are checked
 * against the server at compile time. Callers import `api` and never build
 * URLs themselves.
 */
import { apiUrl, client, unwrap } from "./http"
import type {
  ActionRequest,
  ActionResponse,
  Bootstrap,
  FormTab,
  InfotextParsed,
  JobRequest,
  OptionsResponse,
  PageSchema,
  PromptStyle,
  TabSchema,
  TokenCountRequest,
} from "./types"

const enc = encodeURIComponent

export const api = {
  app: (signal?: AbortSignal) => unwrap(client.GET("/api/v2/app", { signal })) as Promise<Bootstrap>,

  schema: (tab: FormTab, signal?: AbortSignal) =>
    unwrap(client.GET("/api/v2/schema/{tab}", { params: { path: { tab } }, signal })) as Promise<TabSchema>,

  /** Extension page layout (schema-rendered pages). */
  pageSchema: (id: string, signal?: AbortSignal) =>
    unwrap(client.GET("/api/v2/pages/{page_id}", { params: { path: { page_id: id } }, signal })) as Promise<PageSchema>,

  /** `owner` is a script id, `page:<id>`, `core` or `settings`. */
  action: (owner: string, action: string, body: ActionRequest) =>
    unwrap(client.POST("/api/v2/actions/{owner}/{action}", { params: { path: { owner, action } }, body })) as Promise<ActionResponse>,

  jobs: {
    create: (body: JobRequest) => unwrap(client.POST("/api/v2/jobs", { body: { priority: 0, repeat: 1, forever: false, ...body } })),
    list: (query: { cursor?: string | null; status?: string; limit?: number } = {}, signal?: AbortSignal) =>
      unwrap(client.GET("/api/v2/jobs", { params: { query }, signal })),
    /** One job (404 once it was removed from the history). */
    get: (id: string, signal?: AbortSignal) => unwrap(client.GET("/api/v2/jobs/{job_id}", { params: { path: { job_id: id } }, signal })),
    remove: (id: string) => unwrap(client.DELETE("/api/v2/jobs/{job_id}", { params: { path: { job_id: id } } })),
    interrupt: (id: string) => unwrap(client.POST("/api/v2/jobs/{job_id}/interrupt", { params: { path: { job_id: id } } })),
    skip: (id: string) => unwrap(client.POST("/api/v2/jobs/{job_id}/skip", { params: { path: { job_id: id } } })),
    retry: (id: string) => unwrap(client.POST("/api/v2/jobs/{job_id}/retry", { params: { path: { job_id: id } } })),
  },

  queue: {
    /** Queued jobs in run order (priority, then the user's arrangement). */
    items: (signal?: AbortSignal) => unwrap(client.GET("/api/v2/queue/items", { signal })),
  },

  upload: (file: Blob, name = file instanceof File ? file.name : "upload") => {
    const form = new FormData()
    form.append("file", file, name)
    // The generated body type is the multipart schema; the client sends FormData as is.
    return unwrap(client.POST("/api/v2/uploads", { body: form as unknown as { file: string } }))
  },

  fileUrl: (id: string) => apiUrl(`/files/${enc(id)}`),
  /** The file as an attachment (the server names it). */
  downloadUrl: (id: string) => apiUrl(`/files/${enc(id)}`, { download: 1 }),
  thumbUrl: (id: string, width?: number) => apiUrl(`/thumbs/${enc(id)}`, { w: width }),
  /** A file the user added to the data folder (bootstrap `user_assets`). */
  userAssetUrl: (name: string) => apiUrl(`/user-assets/${enc(name)}`),

  outputs: (query: { cursor?: string | null; q?: string } = {}, signal?: AbortSignal) =>
    unwrap(client.GET("/api/v2/outputs", { params: { query }, signal })),
  /** Copies outputs to the "Save" folder; `zip` also packs them (its file id is returned). */
  saveOutputs: (body: { ids: string[]; zip?: boolean }) => unwrap(client.POST("/api/v2/outputs/save", { body: { zip: false, ...body } })),

  networks: (kind: string, query: { q?: string; cursor?: string | null } = {}, signal?: AbortSignal) =>
    unwrap(client.GET("/api/v2/networks/{kind}", { params: { path: { kind }, query }, signal })),
  networkMeta: (kind: string, name: string, signal?: AbortSignal) =>
    unwrap(client.GET("/api/v2/networks/{kind}/{name}/meta", { params: { path: { kind, name } }, signal })),

  options: (signal?: AbortSignal) => unwrap(client.GET("/api/v2/options", { signal })) as Promise<OptionsResponse>,
  patchOptions: (values: Record<string, unknown>) =>
    unwrap(client.PATCH("/api/v2/options", { body: values })) as Promise<OptionsResponse>,

  /** Without `tab` the server detects it; an empty text reads the last generation's parameters. */
  parseInfotext: (text: string, tab?: FormTab | null) =>
    unwrap(client.POST("/api/v2/infotext/parse", { body: tab ? { text, tab } : { text } })) as Promise<InfotextParsed>,

  models: (signal?: AbortSignal) => unwrap(client.GET("/api/v2/models", { signal })),

  /** Token count of one prompt; never queued behind generation. */
  countTokens: (body: TokenCountRequest, signal?: AbortSignal) =>
    unwrap(client.POST("/api/v2/prompt/tokens", { body: { steps: 20, styles: [], negative: false, ...body }, signal })),

  /** PNG Info of an uploaded or generated image, parsed for `tab`. */
  imageInfo: (fileId: string, tab?: FormTab | null) =>
    unwrap(client.POST("/api/v2/infotext/image", { body: tab ? { file_id: fileId, tab } : { file_id: fileId } })),

  /** The merged locale catalog (`zh-CN` or `zh_CN`). */
  i18n: (locale: string) =>
    unwrap(client.GET("/api/v2/i18n/{locale}", { params: { path: { locale } } })) as Promise<Record<string, string>>,

  styles: {
    list: (signal?: AbortSignal) => unwrap(client.GET("/api/v2/styles", { signal })),
    /** Writes `styles` into the prompts the way generation would. */
    apply: (body: { prompt: string; negative_prompt: string; styles: string[] }) => unwrap(client.POST("/api/v2/styles/apply", { body })),
    /** Creates or replaces a style. */
    save: (style: PromptStyle) =>
      unwrap(
        client.PUT("/api/v2/styles/{name}", {
          params: { path: { name: style.name } },
          body: { prompt: style.prompt, negative_prompt: style.negative_prompt },
        }),
      ),
    remove: (name: string) => unwrap(client.DELETE("/api/v2/styles/{name}", { params: { path: { name } } })),
  },
}

export type Api = typeof api
