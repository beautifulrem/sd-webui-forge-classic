/**
 * This browser session's generation results (txt2img and img2img), newest
 * first, with the selection the stage and lightbox show. Kept in
 * sessionStorage, so a reload keeps them and a new tab starts empty. Fed by
 * `useProgress` (the jobs this client queued) and by `restoreRuns`.
 */
import { create } from "zustand"
import { createJSONStorage, persist } from "zustand/middleware"
import { useProgress } from "@/api/progress"
import type { FormTab, JobOutput } from "@/api/types"

export type ResultItem = {
  /** `${jobId}:${index}` */
  key: string
  fileId: string
  jobId: string
  tab: FormTab
  /** Position in the job's outputs. */
  index: number
  infotext: string | null
  width?: number
  height?: number
  createdAt: number
}

/** How many results are remembered. */
export const SESSION_LIMIT = 400

type SessionState = {
  items: ResultItem[]
  /** Selected result key; null = the newest. */
  selected: string | null
  /** The stage shows the live preview while a job runs (off after picking an older result). */
  follow: boolean
  addJob: (job: { jobId: string; tab: FormTab; outputs: readonly JobOutput[]; infotext?: string | null }) => void
  select: (key: string | null) => void
  setFollow: (follow: boolean) => void
  remove: (key: string) => void
  clear: () => void
}

export const resultKey = (jobId: string, index: number) => `${jobId}:${index}`

export const useSession = create<SessionState>()(
  persist(
    (set) => ({
      items: [],
      selected: null,
      follow: true,
      addJob: ({ jobId, tab, outputs, infotext }) =>
        set((s) => {
          if (s.items.some((item) => item.jobId === jobId)) return s
          const now = Date.now()
          const added = outputs
            .filter((output) => (output.kind ?? "image") === "image")
            .map<ResultItem>((output, index) => ({
              key: resultKey(jobId, index),
              fileId: output.id,
              jobId,
              tab,
              index,
              infotext: output.infotext ?? infotext ?? null,
              width: output.width ?? undefined,
              height: output.height ?? undefined,
              createdAt: now,
            }))
          if (added.length === 0) return s
          return { items: [...added, ...s.items].slice(0, SESSION_LIMIT), selected: added[0]!.key, follow: true }
        }),
      select: (selected) => set({ selected }),
      setFollow: (follow) => set({ follow }),
      remove: (key) =>
        set((s) => {
          const index = s.items.findIndex((item) => item.key === key)
          if (index === -1) return s
          const items = s.items.filter((item) => item.key !== key)
          const selected = s.selected === key ? (items[Math.min(index, items.length - 1)]?.key ?? null) : s.selected
          return { items, selected }
        }),
      clear: () => set({ items: [], selected: null, follow: true }),
    }),
    {
      name: "remi-session-results",
      version: 1,
      storage: createJSONStorage(() => sessionStorage),
      partialize: (s) => ({ items: s.items, selected: s.selected }),
    },
  ),
)

/** The selected result, or the newest one. */
export function selectedItem(s: Pick<SessionState, "items" | "selected">): ResultItem | null {
  return (s.selected ? s.items.find((item) => item.key === s.selected) : undefined) ?? s.items[0] ?? null
}

let wired = false

/** Adds every finished job this client queued from a generation form. */
export function wireSessionResults(tabs: readonly FormTab[] = ["txt2img", "img2img"]) {
  if (wired) return
  wired = true
  useProgress.subscribe((s, prev) => {
    if (s.last === prev.last) return
    for (const tab of tabs) {
      const job = s.last[tab]
      if (job && job !== prev.last[tab] && !job.error) useSession.getState().addJob({ ...job, tab })
    }
  })
}
