import { QueryClientProvider } from "@tanstack/react-query"
import { render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { api } from "@/api/endpoints"
import { bootstrap } from "@/api/mock/fixtures"
import { keys, queryClient } from "@/api/queries"
import { IDLE, type RunEvent, reduce, type RunState } from "@/features/progress/run-machine"
import ShortcutsDialog from "./ShortcutsDialog"
import { useShortcutsHelp } from "./shortcuts-help"

function openWith(ctrlEnterInterrupt: boolean) {
  const app = structuredClone(bootstrap)
  app.options = { ...app.options, ctrl_enter_interrupt: ctrlEnterInterrupt }
  queryClient.setQueryData(keys.app, app)
  vi.spyOn(api, "app").mockResolvedValue(app)
  useShortcutsHelp.setState({ open: true })
  render(
    <QueryClientProvider client={queryClient}>
      <ShortcutsDialog />
    </QueryClientProvider>,
  )
}

/** What Ctrl/Cmd+Enter does to a running job (the run machine the shortcut feeds). */
function shortcutWhileRunning(interruptOnly: boolean) {
  const events: RunEvent[] = [{ type: "generate" }, { type: "submitted", jobIds: ["a"] }, { type: "started", jobId: "a" }]
  const running = events.reduce<RunState>((state, event) => reduce(state, event, { interruptOnly })[0], IDLE)
  const [after, effects] = reduce(running, { type: "shortcut" }, { interruptOnly })
  return { restart: after.restart, effects: effects.map((effect) => effect.type) }
}

describe("keyboard shortcuts dialog", () => {
  afterEach(() => {
    useShortcutsHelp.setState({ open: false })
    vi.restoreAllMocks()
    queryClient.clear()
  })

  it("describes Ctrl/Cmd+Enter during a run as the run machine handles it", () => {
    expect(shortcutWhileRunning(false)).toEqual({ restart: true, effects: ["interrupt"] })
    openWith(false)
    expect(screen.getByText("Generate (while running: interrupt and generate again)")).toBeTruthy()
  })

  it("follows the setting that makes Ctrl/Cmd+Enter only interrupt", () => {
    expect(shortcutWhileRunning(true)).toEqual({ restart: false, effects: ["interrupt"] })
    openWith(true)
    expect(screen.getByText("Generate (while running: interrupt)")).toBeTruthy()
    expect(screen.queryByText(/queue another/)).toBeNull()
  })
})
