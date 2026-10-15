/**
 * Opening a job's parameters in its form (txt2img, img2img or Extras). For a
 * queued job this starts an *edit*: the form shows a banner, and "Update job"
 * replaces the queued job's parameters with the form's (`PUT
 * /jobs/{id}/params`) instead of queueing a new one. Finished jobs (and
 * imported ones that cannot be replayed) simply load their parameters.
 */
import { toast } from "sonner"
import { create } from "zustand"
import { api } from "@/api/endpoints"
import { ApiError, validationErrors } from "@/api/http"
import { keys, queryClient } from "@/api/queries"
import type { FormTab, JobDetail, ParamValues, TabSchema } from "@/api/types"
import { navigate } from "@/app/router"
import type { PasteEntry } from "@/features/infotext/paste"
import { deliver } from "@/features/send-to/send-to"
import { errorText } from "@/features/errors/error-text"
import { t } from "@/i18n"
import { getFormStore } from "@/ui/form-store"
import { jobRequest } from "@/ui/tab"

export type EditSession = { jobId: string; seq: number; tab: FormTab; title: string }

type EditState = {
  sessions: Partial<Record<FormTab, EditSession>>
  saving: Partial<Record<FormTab, boolean>>
}

export const useQueueEdit = create<EditState>()(() => ({ sessions: {}, saving: {} }))

/** One paste entry per owner: the job's core values, then each script's. */
export function jobEntries(job: Pick<JobDetail, "core" | "scripts">): PasteEntry[] {
  const scripts = Object.entries(job.scripts ?? {}).filter((entry): entry is [string, ParamValues] => entry[1] !== null && typeof entry[1] === "object" && !Array.isArray(entry[1]))
  return [{ owner: "core", values: job.core ?? {} }, ...scripts.map(([owner, values]) => ({ owner, values }))]
}

/** The failure as text in the UI language; rejected values are marked on the form itself. */
const message = (error: unknown) => errorText(error, { 422: t("Check the marked fields.") })

/** Loads a job's parameters into its form (and, while it is queued, edits it). */
export async function openJob(jobId: string): Promise<void> {
  const id = toast.loading(t("Opening the job…"))
  try {
    const job = await api.jobs.get(jobId)
    const editing = job.status === "queued"
    useQueueEdit.setState((s) => {
      const sessions = { ...s.sessions }
      if (editing) sessions[job.tab] = { jobId: job.id, seq: job.seq, tab: job.tab, title: job.name || job.summary || "" }
      else delete sessions[job.tab]
      return { sessions }
    })
    const count = await deliver(job.tab, jobEntries(job))
    toast.success(editing ? t("Editing queued job #{{seq}}", { seq: job.seq }) : t("Parameters applied: {{count}}", { count }), { id })
  } catch (error) {
    toast.error(t("Could not open the job"), { id, description: message(error) })
  }
}

export function cancelEdit(tab: FormTab) {
  useQueueEdit.setState((s) => {
    const sessions = { ...s.sessions }
    delete sessions[tab]
    return { sessions }
  })
}

/** Writes the form's values into the edited queued job; true when it was updated. */
export async function saveEdit(tab: FormTab): Promise<boolean> {
  const session = useQueueEdit.getState().sessions[tab]
  const schema = queryClient.getQueryData<TabSchema>(keys.schema(tab))
  if (!session || !schema) return false
  const store = getFormStore(tab)
  const { core, scripts } = jobRequest(store, schema)
  useQueueEdit.setState((s) => ({ saving: { ...s.saving, [tab]: true } }))
  try {
    await api.jobs.updateParams(session.jobId, { core, scripts })
    store.getState().clearErrors()
    cancelEdit(tab)
    void queryClient.invalidateQueries({ queryKey: keys.jobs })
    void queryClient.invalidateQueries({ queryKey: keys.queueItems })
    toast.success(t("Queued job #{{seq}} updated", { seq: session.seq }), { action: { label: t("Open queue"), onClick: () => navigate({ name: "queue" }) } })
    return true
  } catch (error) {
    const errors = validationErrors(error)
    if (errors) {
      store.getState().setErrors(errors)
      toast.error(t("Some parameters are invalid"), { description: message(error) })
    } else if (error instanceof ApiError && (error.status === 409 || error.status === 404)) {
      cancelEdit(tab)
      toast.error(t("Job #{{seq}} already started or was removed", { seq: session.seq }), {
        description: t("Your changes are still in the form: generate or add them to the queue as a new job."),
      })
    } else {
      toast.error(t("Could not update the job"), { description: message(error) })
    }
    return false
  } finally {
    useQueueEdit.setState((s) => ({ saving: { ...s.saving, [tab]: false } }))
  }
}

/** Queues the form as it is, without tracking it in the form's run (the "Add to queue" button). */
export async function enqueueForm(tab: FormTab): Promise<string | null> {
  const schema = queryClient.getQueryData<TabSchema>(keys.schema(tab))
  if (!schema) return null
  const store = getFormStore(tab)
  try {
    const created = await api.jobs.create(jobRequest(store, schema))
    store.getState().clearErrors()
    const queued = queryClient.getQueryData<unknown[]>(keys.queueItems)?.length
    toast.success(t("Added to the queue as #{{seq}}", { seq: created.seq }), {
      description: queued ? t("{{count}} jobs were waiting", { count: queued }) : undefined,
      action: { label: t("Open queue"), onClick: () => navigate({ name: "queue" }) },
    })
    void queryClient.invalidateQueries({ queryKey: keys.queueItems })
    return created.job_id
  } catch (error) {
    const errors = validationErrors(error)
    if (errors) store.getState().setErrors(errors)
    toast.error(errors ? t("Some parameters are invalid") : t("Could not queue the job"), { description: message(error) })
    return null
  }
}
