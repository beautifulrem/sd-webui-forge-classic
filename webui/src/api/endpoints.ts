/**
 * Typed endpoint functions for /api/v2. Callers import `api` and never build
 * URLs themselves, so this module is the only thing an OpenAPI-generated
 * client has to replace.
 */
import { apiUrl, http } from "./http"
import type {
  ActionRequest,
  ActionResponse,
  Bootstrap,
  FormTab,
  InfotextParsed,
  Job,
  JobRequest,
  Models,
  NetworkItem,
  OptionsResponse,
  OutputItem,
  Page,
  PageSchema,
  TabSchema,
  Upload,
} from "./types"

const enc = encodeURIComponent

export const api = {
  app: (signal?: AbortSignal) => http.get<Bootstrap>("/app", { signal }),

  schema: (tab: FormTab, signal?: AbortSignal) => http.get<TabSchema>(`/schema/${tab}`, { signal }),

  /** Extension page layout (schema-rendered pages). */
  pageSchema: (id: string, signal?: AbortSignal) => http.get<PageSchema>(`/pages/${enc(id)}`, { signal }),

  /** `owner` is a script id, `page:<id>` or `core`. */
  action: (owner: string, action: string, body: ActionRequest) =>
    http.post<ActionResponse>(`/actions/${enc(owner)}/${enc(action)}`, body),

  jobs: {
    create: (body: JobRequest) => http.post<{ job_id: string }>("/jobs", body),
    list: (params: { cursor?: string | null; status?: string; limit?: number } = {}, signal?: AbortSignal) =>
      http.get<Page<Job>>("/jobs", { query: params, signal }),
    remove: (id: string) => http.delete<void>(`/jobs/${enc(id)}`),
    interrupt: (id: string) => http.post<void>(`/jobs/${enc(id)}/interrupt`),
    skip: (id: string) => http.post<void>(`/jobs/${enc(id)}/skip`),
    retry: (id: string) => http.post<{ job_id: string }>(`/jobs/${enc(id)}/retry`),
  },

  upload: (file: Blob, name = file instanceof File ? file.name : "upload") => {
    const form = new FormData()
    form.append("file", file, name)
    return http.post<Upload>("/uploads", form)
  },

  fileUrl: (id: string) => apiUrl(`/files/${enc(id)}`),
  thumbUrl: (id: string, width?: number) => apiUrl(`/thumbs/${enc(id)}`, { w: width }),

  outputs: (params: { cursor?: string | null; q?: string } = {}, signal?: AbortSignal) =>
    http.get<Page<OutputItem>>("/outputs", { query: params, signal }),

  networks: (kind: string, params: { q?: string; cursor?: string | null } = {}, signal?: AbortSignal) =>
    http.get<Page<NetworkItem>>(`/networks/${enc(kind)}`, { query: params, signal }),
  networkMeta: (kind: string, name: string, signal?: AbortSignal) =>
    http.get<Record<string, unknown>>(`/networks/${enc(kind)}/${enc(name)}/meta`, { signal }),

  options: (signal?: AbortSignal) => http.get<OptionsResponse>("/options", { signal }),
  patchOptions: (values: Record<string, unknown>) => http.patch<OptionsResponse>("/options", values),

  parseInfotext: (text: string) => http.post<InfotextParsed>("/infotext/parse", { text }),

  models: (signal?: AbortSignal) => http.get<Models>("/models", { signal }),
}

export type Api = typeof api
