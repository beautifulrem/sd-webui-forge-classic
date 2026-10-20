/**
 * Typed endpoint functions for /api/v2, built on the client generated from
 * the server's OpenAPI schema: paths, parameters and bodies are checked
 * against the server at compile time. Callers import `api` and never build
 * URLs themselves.
 */
import { apiUrl, client, ROOT, unwrap } from "./http"
import type {
  ActionRequest,
  ActionResponse,
  Bootstrap,
  FormTab,
  GalleryQuery,
  ImportSource,
  InfotextParsed,
  JobsQuery,
  JobRequest,
  NetworkQuery,
  OptionsResponse,
  PageSchema,
  ParamValues,
  PromptStyle,
  TabSchema,
  TokenCountRequest,
} from "./types"

const enc = encodeURIComponent

export const api = {
  app: (signal?: AbortSignal) => unwrap(client.GET("/api/v2/app", { signal })) as Promise<Bootstrap>,

  /** Login (`--auth`); these never count as a refused request (`session.ts`). */
  auth: {
    status: (signal?: AbortSignal) => unwrap(client.GET("/api/v2/auth/status", { signal })),
    /** Sets the session cookie; 401 for wrong credentials, 429 after too many attempts. */
    login: (username: string, password: string) => unwrap(client.POST("/api/v2/auth/login", { body: { username, password } })),
    /** Revokes and clears the session cookie. */
    logout: () => unwrap(client.POST("/api/v2/auth/logout")),
  },

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
    list: (query: JobsQuery = {}, signal?: AbortSignal) => unwrap(client.GET("/api/v2/jobs", { params: { query }, signal })),
    /** One job (404 once it was removed from the history). */
    get: (id: string, signal?: AbortSignal) => unwrap(client.GET("/api/v2/jobs/{job_id}", { params: { path: { job_id: id } }, signal })),
    remove: (id: string) => unwrap(client.DELETE("/api/v2/jobs/{job_id}", { params: { path: { job_id: id } } })),
    interrupt: (id: string) => unwrap(client.POST("/api/v2/jobs/{job_id}/interrupt", { params: { path: { job_id: id } } })),
    skip: (id: string) => unwrap(client.POST("/api/v2/jobs/{job_id}/skip", { params: { path: { job_id: id } } })),
    retry: (id: string) => unwrap(client.POST("/api/v2/jobs/{job_id}/retry", { params: { path: { job_id: id } } })),
    /** Queues a copy: right after the job while it is queued, else at the end. */
    duplicate: (id: string) => unwrap(client.POST("/api/v2/jobs/{job_id}/duplicate", { params: { path: { job_id: id } } })),
    /** Priority (higher runs first) or the generate-forever flag. */
    update: (id: string, body: { priority?: number | null; forever?: boolean | null }) =>
      unwrap(client.PATCH("/api/v2/jobs/{job_id}", { params: { path: { job_id: id } }, body })),
    /** Replaces a queued job's parameters (validated; 409 once it started). */
    updateParams: (id: string, body: { core: ParamValues; scripts: Record<string, ParamValues> }) =>
      unwrap(client.PUT("/api/v2/jobs/{job_id}/params", { params: { path: { job_id: id } }, body })),
    star: (ids: string[], starred: boolean) => unwrap(client.POST("/api/v2/jobs/star", { body: { ids, starred } })),
    /** Deletes finished jobs (starred ones too unless `keepStarred`). */
    clearHistory: (keepStarred = true) => unwrap(client.POST("/api/v2/jobs/clear-history", { body: { keep_starred: keepStarred } })),
  },

  queue: {
    /** Queued jobs in run order (priority, then the user's arrangement). */
    items: (signal?: AbortSignal) => unwrap(client.GET("/api/v2/queue/items", { signal })),
    status: (signal?: AbortSignal) => unwrap(client.GET("/api/v2/queue", { signal })),
    pause: () => unwrap(client.POST("/api/v2/queue/pause")),
    resume: () => unwrap(client.POST("/api/v2/queue/resume")),
    /** Puts queued jobs in this order (within each priority). */
    reorder: (order: string[]) => unwrap(client.POST("/api/v2/queue/reorder", { body: { order } })),
    /** Moves a queued job next to a neighbour of its (new) priority group: after `after`, else before `before`. */
    move: (body: { id: string; after?: string | null; before?: string | null; priority?: number | null }) => unwrap(client.POST("/api/v2/queue/move", { body })),
    stopForever: () => unwrap(client.POST("/api/v2/queue/stop-forever")),
    /** Interrupts whatever runs; `clearQueue` also cancels every queued job. */
    interruptAll: (clearQueue = false) => unwrap(client.POST("/api/v2/interrupt", { body: { clear_queue: clearQueue } })),
    /** Queues of replaced extensions (Agent Scheduler, Prompt Queue) found on the server. */
    importSources: (signal?: AbortSignal) => unwrap(client.GET("/api/v2/queue/import", { signal })),
    importFrom: (source: ImportSource) => unwrap(client.POST("/api/v2/queue/import/{source}", { params: { path: { source } } })),
  },

  /** The Gallery page: every output, filtered and sorted (`features/library`). */
  gallery: {
    list: (query: GalleryQuery = {}, signal?: AbortSignal) => unwrap(client.GET("/api/v2/gallery", { params: { query }, signal })),
    facets: (signal?: AbortSignal) => unwrap(client.GET("/api/v2/gallery/facets", { signal })),
    get: (id: string, signal?: AbortSignal) => unwrap(client.GET("/api/v2/gallery/{file_id}", { params: { path: { file_id: id } }, signal })),
    star: (ids: string[], starred: boolean) => unwrap(client.POST("/api/v2/gallery/star", { body: { ids, starred } })),
    /** Removes outputs from the gallery; `deleteFiles` also deletes them on disk. */
    remove: (ids: string[], deleteFiles = false) => unwrap(client.POST("/api/v2/gallery/delete", { body: { ids, delete_files: deleteFiles } })),
    /** Packs outputs into a zip (its file id; download with `downloadUrl`). */
    zip: (ids: string[]) => unwrap(client.POST("/api/v2/gallery/zip", { body: { ids } })),
    /** Shows the file in the WebUI machine's file manager (403 for remote browsers). */
    reveal: (id: string) => unwrap(client.POST("/api/v2/gallery/{file_id}/reveal", { params: { path: { file_id: id } } })),
  },
  /** Indexes images in the output folders that the gallery does not know yet. */
  scanOutputs: () => unwrap(client.POST("/api/v2/outputs/refresh")),

  /** PNG Info: everything stored in an uploaded or generated image. */
  pngInfo: (fileId: string, signal?: AbortSignal) => unwrap(client.POST("/api/v2/pnginfo", { body: { file_id: fileId }, signal })),

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

  /** Extra networks and model files (`features/networks`); `name` may contain "/". */
  networks: {
    kinds: (signal?: AbortSignal) => unwrap(client.GET("/api/v2/networks", { signal })),
    list: (kind: string, query: NetworkQuery = {}, signal?: AbortSignal) =>
      unwrap(client.GET("/api/v2/networks/{kind}", { params: { path: { kind }, query }, signal })),
    facets: (kind: string, query: Omit<NetworkQuery, "cursor" | "limit" | "sort" | "order"> = {}, signal?: AbortSignal) =>
      unwrap(client.GET("/api/v2/networks/{kind}/facets", { params: { path: { kind }, query }, signal })),
    meta: (kind: string, name: string, signal?: AbortSignal) =>
      unwrap(client.GET("/api/v2/networks/{kind}/{name}/meta", { params: { path: { kind, name } }, signal })),
    /** Sets keys of the `<model>.json` sidecar (`null` removes one). */
    editMeta: (kind: string, name: string, userMetadata: Record<string, unknown>, replace = false) =>
      unwrap(client.PATCH("/api/v2/networks/{kind}/{name}/meta", { params: { path: { kind, name } }, body: { user_metadata: userMetadata, replace } })),
    /** Uses an uploaded or generated image (file id) as the preview. */
    setPreview: (kind: string, name: string, fileId: string) =>
      unwrap(client.PUT("/api/v2/networks/{kind}/{name}/preview", { params: { path: { kind, name } }, body: { file_id: fileId } })),
    openFolder: (kind: string, name: string) => unwrap(client.POST("/api/v2/networks/{kind}/{name}/open-folder", { params: { path: { kind, name } } })),
    /** Re-scans one kind, or every kind. */
    refresh: (kind?: string) =>
      kind
        ? unwrap(client.POST("/api/v2/networks/{kind}/refresh", { params: { path: { kind } } }))
        : unwrap(client.POST("/api/v2/networks/refresh")),
  },

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

  /** Validates setting values without applying them (422 lists the rejected ones). */
  validateOptions: (values: Record<string, unknown>) => unwrap(client.POST("/api/v2/options/validate", { body: values })),

  /** Extensions management; writes answer 403 when the server disallows extension access. */
  extensions: {
    list: (signal?: AbortSignal) => unwrap(client.GET("/api/v2/extensions", { signal })),
    configure: (body: { disabled?: string[]; disable_all?: "none" | "extra" | "all" }) => unwrap(client.PATCH("/api/v2/extensions", { body })),
    /** One extension's update check (clients check one at a time to show progress). */
    check: (name: string) => unwrap(client.POST("/api/v2/extensions/{name}/check", { params: { path: { name } } })),
    update: (names: string[]) => unwrap(client.POST("/api/v2/extensions/update", { body: { names } })),
    install: (body: { url: string; branch?: string | null; dirname?: string | null }) => unwrap(client.POST("/api/v2/extensions/install", { body })),
    uninstall: (name: string) => unwrap(client.DELETE("/api/v2/extensions/{name}", { params: { path: { name } } })),
    index: (query: { url?: string; refresh?: boolean } = {}, signal?: AbortSignal) =>
      unwrap(client.GET("/api/v2/extensions/index", { params: { query }, signal })),
    states: {
      list: (signal?: AbortSignal) => unwrap(client.GET("/api/v2/extensions/config-states", { signal })),
      get: (id: string, signal?: AbortSignal) =>
        unwrap(client.GET("/api/v2/extensions/config-states/{state_id}", { params: { path: { state_id: id } }, signal })),
      save: (name: string) => unwrap(client.POST("/api/v2/extensions/config-states", { body: { name } })),
      remove: (id: string) => unwrap(client.DELETE("/api/v2/extensions/config-states/{state_id}", { params: { path: { state_id: id } } })),
      /** Restores the backup's extensions (applied by the next Reload UI); the WebUI checkout is never reset. */
      restore: (id: string) =>
        unwrap(client.POST("/api/v2/extensions/config-states/{state_id}/restore", { params: { path: { state_id: id } }, body: { what: "extensions" } })),
    },
  },

  /** Server control and information (Settings → System). */
  system: {
    status: (signal?: AbortSignal) => unwrap(client.GET("/api/v2/server/status", { signal })),
    /** "Reload UI": rebuild the app in-process (after the running job). */
    reload: () => unwrap(client.POST("/api/v2/server/reload")),
    /** Restart the process (501 when no launcher can restart it). */
    restart: () => unwrap(client.POST("/api/v2/server/restart")),
    /** Answers once the server is up; `instance` changes with every reload or restart. */
    ping: (signal?: AbortSignal) => unwrap(client.GET("/internal/ping", { signal })) as Promise<{ instance?: string }>,
    sysinfo: (signal?: AbortSignal) => unwrap(client.GET("/api/v2/sysinfo", { signal })) as Promise<Record<string, unknown>>,
    sysinfoDownloadUrl: () => `${ROOT}/internal/sysinfo-download`,
    checkSysinfo: (text: string) => unwrap(client.POST("/api/v2/sysinfo/check", { body: { text } })),
    licenses: (signal?: AbortSignal) => unwrap(client.GET("/api/v2/licenses", { signal })),
    unloadModel: () => unwrap(client.POST("/api/v2/models/unload")),
    /** Loads the selected model (`force`: again, even when it is loaded). */
    reloadModel: (force = false) => unwrap(client.POST("/api/v2/models/reload", { body: { force } })),
    /** The checkpoint and the weights the memory manager holds. */
    loadedModels: (signal?: AbortSignal) => unwrap(client.GET("/api/v2/models/loaded", { signal })),
    /** Progress of hashing every checkpoint. */
    hashStatus: (signal?: AbortSignal) => unwrap(client.GET("/api/v2/models/hashes", { signal })),
    /** Starts hashing every checkpoint in the background (409 while it runs). */
    hashCheckpoints: (threads: number) => unwrap(client.POST("/api/v2/models/hashes", { body: { threads } })),
  },
}

export type Api = typeof api
