/** Connects the WebSocket to the rest of the app: query cache, progress store, toasts. */
import { useEffect } from "react"
import { toast } from "sonner"
import { useProgress } from "@/api/progress"
import { keys, queryClient } from "@/api/queries"
import { onWsEvent, useWs } from "@/api/ws"
import { reloadServerCatalog, t } from "@/i18n"
import { showToast } from "@/lib/toast"

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
      onWsEvent("job.queued", () => void queryClient.invalidateQueries({ queryKey: keys.jobs })),
      onWsEvent("job.started", (e) => {
        // A new job: the previous job's last frame must not show.
        useWs.getState().clearPreview()
        progress.started(e)
        void queryClient.invalidateQueries({ queryKey: keys.jobs })
      }),
      onWsEvent("job.progress", progress.progress),
      onWsEvent("job.completed", (e) => {
        progress.completed(e)
        useWs.getState().clearPreview()
        void queryClient.invalidateQueries({ queryKey: keys.jobs })
        void queryClient.invalidateQueries({ queryKey: ["outputs"] })
      }),
      onWsEvent("job.failed", (e) => {
        const mine = e.job_id in useProgress.getState().owned
        progress.failed(e)
        useWs.getState().clearPreview()
        void queryClient.invalidateQueries({ queryKey: keys.jobs })
        if (mine) toast.error(t("Generation failed"), { description: e.error })
      }),
      onWsEvent("queue.changed", () => void queryClient.invalidateQueries({ queryKey: keys.jobs })),
      onWsEvent("options.changed", () => {
        void queryClient.invalidateQueries({ queryKey: keys.options })
        void queryClient.invalidateQueries({ queryKey: keys.app })
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
      onWsEvent("toast", (e) => showToast(e.level, e.message)),
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
