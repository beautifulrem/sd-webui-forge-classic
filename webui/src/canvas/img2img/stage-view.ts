/**
 * What the img2img stage shows: the source editor or the results. It turns
 * to the results when the first job of a run of this client's img2img jobs
 * starts (not for every job of "Generate forever", a repeat or a queue, and
 * never while the user paints: the next mask is being drawn), and back to
 * the source when a new source image arrives from elsewhere (send to
 * img2img / inpaint, pasted parameters) or a submit was rejected because of
 * the source layers.
 */
import { useEffect } from "react"
import { create } from "zustand"
import { runningFor, useProgress } from "@/api/progress"
import { useRuns } from "@/features/progress/run-controller"
import { getFormStore } from "@/ui/form-store"
import { isPainting } from "../state/activity"

export type StageView = "source" | "result"

type StageViewState = {
  view: StageView
  setView: (view: StageView) => void
}

export const useStageView = create<StageViewState>()((set) => ({
  view: "source",
  setView: (view) => set({ view }),
}))

/** Source image ids already seen by the stage (a new one shows the editor). */
let seenImage: unknown = undefined

/** The current img2img run already had its say (the first job's start); reset once the run is over. */
let runSeen = false

/** No img2img job queued, submitting, repeating or generating forever. */
function runIdle(): boolean {
  const run = useRuns.getState().runs.img2img
  return run.jobs.length === 0 && run.pending === 0 && !run.forever && run.repeatLeft === 0
}

/** Shows the results when a run's first job starts, unless the user is painting then. */
function followRun() {
  const running = runningFor(useProgress.getState(), "img2img")?.jobId ?? null
  if (running) {
    if (runSeen) return
    runSeen = true
    if (!isPainting()) useStageView.getState().setView("result")
  } else if (runIdle()) runSeen = false
}

/**
 * The view, with the automatic switches; `params` are the editor's image
 * params (background first) whose errors bring the source back.
 */
export function useImg2imgStageView(params: readonly string[] = ["init_img"]): [StageView, (view: StageView) => void] {
  const view = useStageView((s) => s.view)
  const setView = useStageView((s) => s.setView)
  const key = params.join("|")

  useEffect(() => {
    followRun()
    const offProgress = useProgress.subscribe(followRun)
    const offRuns = useRuns.subscribe(followRun)
    return () => {
      offProgress()
      offRuns()
    }
  }, [])

  useEffect(() => {
    const store = getFormStore("img2img")
    const names = key.split("|")
    const check = (image: unknown, errors: Record<string, string> | undefined) => {
      if (seenImage !== undefined && image !== seenImage && image) useStageView.getState().setView("source")
      seenImage = image
      if (errors && names.some((name) => errors[name])) useStageView.getState().setView("source")
    }
    const state = store.getState()
    check(state.values.core?.[names[0]!], undefined)
    return store.subscribe((s, prev) => {
      const name = names[0]!
      const image = s.values.core?.[name]
      if (image !== prev.values.core?.[name] || s.errors !== prev.errors) check(image, s.errors !== prev.errors ? s.errors.core : undefined)
    })
  }, [key])

  return [view, setView]
}
