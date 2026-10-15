/** Page-level wiring of the generate view: controllers, session results, restore, commands. */
import { Ban, GalleryVerticalEnd, Images, Infinity as InfinityIcon, RotateCcw, SkipForward, Sparkles, WandSparkles } from "lucide-react"
import { useEffect, useMemo } from "react"
import { useClientOption } from "@/api/app-options"
import { queryClient, schemaQuery } from "@/api/queries"
import { type Command, useRegisterCommands } from "@/app/commands"
import { navigate } from "@/app/router"
import { wireSessionResults } from "@/features/gallery/session-store"
import { useProgressTitle, useWakeLock } from "@/features/progress/effects"
import { restoreRuns, wireRunSync } from "@/features/progress/restore"
import { dispatchRun, type GenerateTab, useRuns, wireRunControllers } from "@/features/progress/run-controller"
import { useOwnRunningJob } from "./hooks"
import { requestReset } from "./reset-request"

/** Wires the module-level services once and keeps the window state (title, wake lock) in sync. */
export function useGenerateRuntime(tab: GenerateTab) {
  useEffect(() => {
    wireRunControllers()
    wireRunSync()
    wireSessionResults()
    void restoreRuns()
  }, [])

  // Switching modes should not wait for the other form's schema.
  useEffect(() => {
    const other: GenerateTab = tab === "txt2img" ? "img2img" : "txt2img"
    void queryClient.prefetchQuery(schemaQuery(other))
  }, [tab])

  const job = useOwnRunningJob()
  const keepAwake = useClientOption("prevent_screen_sleep_during_generation")
  const titleProgress = useClientOption("show_progress_in_title")
  useWakeLock(job !== null && keepAwake)
  useProgressTitle(job, titleProgress)
}

export function useGenerateCommands(tab: GenerateTab, openViewer: () => void) {
  const forever = useRuns((s) => s.runs[tab].forever)
  const commands = useMemo<Command[]>(
    () => [
      { id: "generate", label: "Generate", group: "Generation", icon: Sparkles, shortcut: "Mod+Enter", run: () => dispatchRun(tab, { type: "generate" }) },
      { id: "interrupt", label: "Interrupt generation", group: "Generation", icon: Ban, run: () => dispatchRun(tab, { type: "interrupt" }) },
      { id: "skip", label: "Skip current image", group: "Generation", icon: SkipForward, shortcut: "Alt+Enter", run: () => dispatchRun(tab, { type: "skip" }) },
      {
        id: "forever",
        label: forever ? "Stop generating forever" : "Generate forever",
        group: "Generation",
        icon: InfinityIcon,
        run: () => dispatchRun(tab, { type: "forever", on: !forever }),
      },
      { id: "reset-form", label: "Reset all parameters", group: "Generation", icon: RotateCcw, run: () => requestReset(tab) },
      { id: "viewer", label: "Open image viewer", group: "Generation", icon: GalleryVerticalEnd, run: openViewer },
      { id: "mode:txt2img", label: "Switch to txt2img", group: "Generation", icon: WandSparkles, run: () => navigate({ name: "generate", tab: "txt2img" }) },
      { id: "mode:img2img", label: "Switch to img2img", group: "Generation", icon: Images, run: () => navigate({ name: "generate", tab: "img2img" }) },
    ],
    [tab, forever, openViewer],
  )
  useRegisterCommands(`generate:${tab}`, commands)
}
