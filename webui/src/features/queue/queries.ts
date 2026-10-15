/** Queue page data: the queue state, the paged history per filter set, and job actions with feedback. */
import { infiniteQueryOptions, queryOptions } from "@tanstack/react-query"
import { toast } from "sonner"
import { api } from "@/api/endpoints"
import { useProgress } from "@/api/progress"
import { keys, queryClient } from "@/api/queries"
import type { Job, JobsQuery } from "@/api/types"
import { errorText, type StatusTexts } from "@/features/errors/error-text"
import { t } from "@/i18n"

export type HistoryFilter = "all" | "completed" | "failed" | "stopped"

export const HISTORY_STATUSES: Record<HistoryFilter, string> = {
  all: "completed,failed,cancelled,interrupted",
  completed: "completed",
  failed: "failed",
  stopped: "cancelled,interrupted",
}

export const queueKeys = {
  status: ["queue", "status"] as const,
  sources: ["queue", "import-sources"] as const,
  /** Under `keys.jobs`, so job events refresh it. */
  history: (query: JobsQuery) => [...keys.jobs, "history", query] as const,
}

export const queueStatusQuery = queryOptions({ queryKey: queueKeys.status, queryFn: ({ signal }) => api.queue.status(signal), staleTime: 5_000 })

export const historyQuery = (query: JobsQuery) =>
  infiniteQueryOptions({
    queryKey: queueKeys.history(query),
    queryFn: ({ pageParam, signal }) => api.jobs.list({ ...query, limit: 50, ...(pageParam ? { cursor: pageParam } : {}) }, signal),
    initialPageParam: null as string | null,
    getNextPageParam: (page) => page.next_cursor ?? undefined,
    staleTime: 5_000,
  })

export const importSourcesQuery = queryOptions({ queryKey: queueKeys.sources, queryFn: ({ signal }) => api.queue.importSources(signal), staleTime: 60_000 })

export function refreshQueue() {
  void queryClient.invalidateQueries({ queryKey: keys.jobs })
  void queryClient.invalidateQueries({ queryKey: keys.queueItems })
  void queryClient.invalidateQueries({ queryKey: queueKeys.status })
}

/** A job that cannot run as it is (imported with arguments that do not map), refused with 409. */
const notReplayable = () => ({ 409: t("This job has arguments that do not map to the current parameters; open it in its form to queue it again.") })

/** Runs a queue action, toasting its failure (and `success` when given); refreshes the queue views. */
export async function act<T>(run: () => Promise<T>, failure: string, success?: string, texts?: StatusTexts): Promise<T | undefined> {
  try {
    const result = await run()
    if (success) toast.success(success)
    return result
  } catch (error) {
    toast.error(failure, { description: errorText(error, { 409: t("The job changed meanwhile (it started or finished)."), ...texts }) })
    return undefined
  } finally {
    refreshQueue()
  }
}

/** Queues a finished job again; its result shows on its form like the user's own runs. */
export function requeue(job: Job) {
  return act(async () => {
    const { job_id } = await api.jobs.retry(job.id)
    useProgress.getState().track(job_id, job.tab)
  }, t("Could not queue the job again"), t("Queued again"), notReplayable())
}

export function duplicate(job: Job) {
  return act(() => api.jobs.duplicate(job.id), t("Could not duplicate the job"), t("Duplicated"), notReplayable())
}

export function removeJob(job: Job) {
  return act(() => api.jobs.remove(job.id), job.status === "queued" ? t("Could not cancel the job") : t("Could not remove the job"))
}

export function setPriority(job: Job, priority: number) {
  return act(() => api.jobs.update(job.id, { priority }), t("Could not change the priority"))
}

export function starJob(job: Job, starred: boolean) {
  queryClient.setQueriesData<{ pages: { items: Job[] }[]; pageParams: unknown[] }>({ queryKey: [...keys.jobs, "history"] }, (data) =>
    data ? { ...data, pages: data.pages.map((page) => ({ ...page, items: page.items.map((item) => (item.id === job.id ? { ...item, starred } : item)) })) } : data,
  )
  return act(() => api.jobs.star([job.id], starred), t("Could not update the star"))
}

/** Seconds between two ISO instants (null when either is missing). */
export function durationOf(job: Pick<Job, "started_at" | "finished_at">): number | null {
  if (!job.started_at || !job.finished_at) return null
  const ms = Date.parse(job.finished_at) - Date.parse(job.started_at)
  return Number.isFinite(ms) && ms >= 0 ? ms / 1000 : null
}

/** "42 s", "3 min 05 s", "1 h 02 min". */
export function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds < 10 ? seconds.toFixed(1) : Math.round(seconds)} s`
  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) return `${minutes} min ${String(Math.round(seconds % 60)).padStart(2, "0")} s`
  return `${Math.floor(minutes / 60)} h ${String(minutes % 60).padStart(2, "0")} min`
}

/** The form a job belongs to, named as in the navigation. */
export function formLabel(tab: string): string {
  if (tab === "img2img") return t("img2img")
  if (tab === "extras") return t("Extras")
  return t("txt2img")
}
