/**
 * Side effects while generating: the screen wake lock, progress in the
 * document title, and a sound / browser notification when results arrive
 * while the page is not in front.
 */
import { useEffect } from "react"
import { clientOption } from "@/api/app-options"
import { api } from "@/api/endpoints"
import { keys, queryClient } from "@/api/queries"
import type { Bootstrap } from "@/api/types"
import type { RunningJob } from "@/api/progress"
import { progressFraction } from "@/api/progress"
import { t } from "@/i18n"

type WakeLockSentinelLike = { release: () => Promise<void>; released?: boolean }
type WakeLockLike = { request: (type: "screen") => Promise<WakeLockSentinelLike> }

/** Keeps the screen awake while `active` (re-acquired when the tab becomes visible again). */
export function useWakeLock(active: boolean) {
  useEffect(() => {
    const wakeLock = (navigator as Navigator & { wakeLock?: WakeLockLike }).wakeLock
    if (!active || !wakeLock) return
    let sentinel: WakeLockSentinelLike | null = null
    let disposed = false
    const acquire = () => {
      if (document.visibilityState !== "visible" || (sentinel && !sentinel.released)) return
      wakeLock
        .request("screen")
        .then((lock) => {
          if (disposed) void lock.release()
          else sentinel = lock
        })
        .catch(() => {})
    }
    acquire()
    document.addEventListener("visibilitychange", acquire)
    return () => {
      disposed = true
      document.removeEventListener("visibilitychange", acquire)
      void sentinel?.release().catch(() => {})
    }
  }, [active])
}

export function formatEta(seconds: number | null | undefined): string | null {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return null
  const s = Math.max(0, Math.round(seconds))
  if (s >= 3600) return `${Math.floor(s / 3600)}:${String(Math.floor((s % 3600) / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`
  return s >= 60 ? `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}` : `${s}s`
}

/** The status of a running job: the server's text, else the step (before the first step, while the model loads: "Starting…"). */
export function stepText(job: Pick<RunningJob, "text" | "step" | "steps">): string {
  if (job.text) return t(job.text)
  return job.steps ? t("Step {{step}} of {{steps}}", { step: job.step, steps: job.steps }) : t("Starting…")
}

/** "[42%] ETA 12s" while `job` runs; the title the app set is restored afterwards. */
export function useProgressTitle(job: RunningJob | null, enabled: boolean) {
  const fraction = progressFraction(job)
  const percent = fraction === null ? null : Math.round(fraction * 100)
  const eta = formatEta(job?.eta)
  useEffect(() => {
    if (!enabled || percent === null) return
    const base = document.title.replace(/^\[\d+%\][^·]*· /, "")
    document.title = `[${percent}%]${eta ? ` ${t("ETA {{eta}}", { eta })}` : ""} · ${base}`
    return () => {
      document.title = base
    }
  }, [enabled, percent, eta])
}

let sound: HTMLAudioElement | null = null

/**
 * Plays the user's `notification.mp3` (WebUI folder, served from
 * /user-assets) when they added one, as the setting describes; silent otherwise.
 * `volume` 0–100.
 */
export function playNotificationSound(volume: number) {
  const assets = queryClient.getQueryData<Bootstrap>(keys.app)?.user_assets ?? []
  if (volume <= 0 || !assets.includes("notification.mp3") || typeof Audio === "undefined") return
  sound ??= new Audio(api.userAssetUrl("notification.mp3"))
  sound.volume = Math.min(1, volume / 100)
  sound.currentTime = 0
  // Autoplay can be blocked until the user interacted with the page.
  void sound.play().catch(() => {})
}

/** Generated images among `images` outputs: a batch's grid (`return_grid`) is not counted. */
export function resultCount(images: number): number {
  return images > 1 && clientOption("return_grid") ? images - 1 : images
}

/** Sound and, when allowed, a browser notification, if the page is not in front. */
export function notifyResults(images: number) {
  const count = resultCount(images)
  if (typeof document === "undefined" || (document.visibilityState === "visible" && document.hasFocus())) return
  if (clientOption("notification_audio")) playNotificationSound(clientOption("notification_volume"))
  if (typeof Notification === "undefined" || Notification.permission !== "granted" || count <= 0) return
  const notification = new Notification("Remi", {
    body: t("Images generated: {{count}}", { count }),
    tag: "remi-generation",
  })
  notification.onclick = () => {
    window.focus()
    notification.close()
  }
}

export function notificationsAvailable(): boolean {
  return typeof Notification !== "undefined"
}
