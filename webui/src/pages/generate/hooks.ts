/** Generate-page hooks shared by the stage, the viewer and the window effects. */
import { type RunningJob, useProgress } from "@/api/progress"
import { GENERATE_FORMS, type GenerateTab } from "@/features/progress/run-controller"

/** The running job when this client queued it from a generation form. */
export function useOwnRunningJob(): RunningJob | null {
  return useProgress((s) => (s.running && GENERATE_FORMS.includes(s.owned[s.running.jobId] as GenerateTab) ? s.running : null))
}
