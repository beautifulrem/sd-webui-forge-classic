/** Connects the WebSocket to the rest of the app: query cache, progress store, toasts. */
import { useEffect } from "react"
import { toast } from "sonner"
import { useProgress } from "@/api/progress"
import { keys, queryClient } from "@/api/queries"
import { onWsEvent, useWs } from "@/api/ws"
import { onNewOutputs, onOutputsChanged } from "@/features/library/cache"
import { networksFollowOptions } from "@/features/networks/queries"
import { i18n, reloadServerCatalog, t } from "@/i18n"
import { noteText } from "@/features/queue/notes"
import { showToast } from "@/lib/toast"
import { messageText } from "@/ui/messages"

/** `model.loading` stages that end a load (besides the `model.loaded` event). */
const TERMINAL_STAGES = new Set(["done", "loaded", "error", "failed"])
/** Without any model event for this long, the quick-bar is released anyway. */
export const MODEL_LOADING_TIMEOUT_MS = 180_000

export function useWsBridge() {
  useEffect(() => {
    const ws = useWs.getState()
    const progress = useProgress.getState()
    let greeted = false
    let loadingTimer: ReturnType<typeof setTimeout> | undefined
    const modelIdle = () => {
      clearTimeout(loadingTimer)
      progress.setModelLoading(null)
    }
    ws.connect()

    const offs = [
      onWsEvent("job.queued", () => {
        void queryClient.invalidateQueries({ queryKey: keys.jobs })
        void queryClient.invalidateQueries({ queryKey: keys.queueItems })
      }),
      onWsEvent("job.started", (e) => {
        // A new job: the previous job's last frame must not show.
        useWs.getState().clearPreview()
        progress.started(e)
        void queryClient.invalidateQueries({ queryKey: keys.jobs })
      }),
      onWsEvent("job.progress", progress.progress),
      onWsEvent("job.completed", (e) => {
        if (e.job_id in useProgress.getState().owned && e.warnings?.length)
          toast.warning(t("Some extension steps failed; the job went on without them"), { description: e.warnings.map(noteText).join("\n") })
        progress.completed(e)
        useWs.getState().clearPreview()
        void queryClient.invalidateQueries({ queryKey: keys.jobs })
        void queryClient.invalidateQueries({ queryKey: ["outputs"] })
        // The gallery merges the new outputs in (refetching every loaded page would be slow).
        if (e.outputs?.length) void onNewOutputs()
      }),
      onWsEvent("outputs.changed", (e) => void onOutputsChanged(e.ids)),
      onWsEvent("job.failed", (e) => {
        const mine = e.job_id in useProgress.getState().owned
        progress.failed(e)
        useWs.getState().clearPreview()
        void queryClient.invalidateQueries({ queryKey: keys.jobs })
        // Cancelled / interrupted jobs (e.g. the queue was cleared) are not failures.
        const stopped = e.status === "cancelled" || e.status === "interrupted"
        if (mine && !stopped) toast.error(t("Generation failed"), { description: e.error })
      }),
      onWsEvent("queue.changed", (e) => {
        // Also sent right after a (re)connect: drops the progress of a job that ended while the socket was down.
        if (e.running !== undefined) progress.syncRunning(e.running)
        void queryClient.invalidateQueries({ queryKey: keys.jobs })
        void queryClient.invalidateQueries({ queryKey: keys.queueItems })
      }),
      onWsEvent("options.changed", (e) => {
        void queryClient.invalidateQueries({ queryKey: keys.options })
        // Network lists follow the preset (compatibility), LoRA weight / name and hidden-folder options.
        if (networksFollowOptions(e.keys)) void queryClient.invalidateQueries({ queryKey: ["networks"] })
        void queryClient.invalidateQueries({ queryKey: keys.app })
        // The checkpoint / module lists and the styles follow some options (e.g. a model switch).
        void queryClient.invalidateQueries({ queryKey: keys.models })
        void queryClient.invalidateQueries({ queryKey: keys.styles })
        // Core forms follow the options (category order, choice lists); the
        // form stores keep every value that still fits the new schema.
        void queryClient.invalidateQueries({ queryKey: ["schema"] })
        void queryClient.invalidateQueries({ queryKey: ["page"] })
      }),
      onWsEvent("model.loading", (e) => {
        if (TERMINAL_STAGES.has(e.stage.toLowerCase())) return modelIdle()
        progress.setModelLoading(e)
        clearTimeout(loadingTimer)
        loadingTimer = setTimeout(modelIdle, MODEL_LOADING_TIMEOUT_MS)
      }),
      onWsEvent("model.loaded", (e) => {
        modelIdle()
        if (e.error) toast.error(t("Could not load the model"), { description: e.error })
      }),
      onWsEvent("networks.changed", () => {
        void queryClient.invalidateQueries({ queryKey: ["networks"] })
        void queryClient.invalidateQueries({ queryKey: keys.models })
      }),
      // Events were lost: the server follows with a snapshot; refetch everything shown.
      onWsEvent("resync", () => void queryClient.invalidateQueries()),
      // translated like action toasts: a catalog key (or its English text when no catalog has it)
      onWsEvent("toast", (e) => showToast(e.level, messageText(i18n, e.message))),
      // A reconnect may follow a server restart: refresh everything the server owns.
      onWsEvent("hello", () => {
        if (greeted) {
          modelIdle()
          void queryClient.invalidateQueries()
          void reloadServerCatalog()
        }
        greeted = true
      }),
    ]

    // Reconnect promptly when the tab comes back or the network returns.
    const wake = () => {
      if (document.visibilityState === "visible" && useWs.getState().status === "reconnecting") useWs.getState().connect()
    }
    document.addEventListener("visibilitychange", wake)
    window.addEventListener("online", wake)

    return () => {
      offs.forEach((off) => off())
      clearTimeout(loadingTimer)
      document.removeEventListener("visibilitychange", wake)
      window.removeEventListener("online", wake)
      useWs.getState().disconnect()
    }
  }, [])
}
