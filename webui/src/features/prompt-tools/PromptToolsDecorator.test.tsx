import { QueryClientProvider } from "@tanstack/react-query"
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { type ReactNode, useState } from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { queryClient } from "@/api/queries"
import { TEST_CONFIG } from "./testing"
import { type PromptToolsConfig, promptToolsApi } from "./api"
import { PromptToolsDecorator } from "./PromptToolsDecorator"
import { toolKeys } from "./queries"
import { usePromptViews } from "./view-store"

const CONFIG = TEST_CONFIG

/** Lazy chunks load slowly on a busy machine. */
const SLOW = { timeout: 8000 }

const wrapper = ({ children }: { children: ReactNode }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>

function Field({ initial, onChange }: { initial: string; onChange?: (value: string) => void }) {
  const [value, setValue] = useState(initial)
  const set = (next: string) => {
    setValue(next)
    onChange?.(next)
  }
  return (
    <PromptToolsDecorator value={value} setValue={set} kind="positive" editor={() => null} label="Prompt" disabled={false} fieldId="prompt">
      <textarea aria-label="Prompt editor" value={value} onChange={(event) => set(event.target.value)} />
    </PromptToolsDecorator>
  )
}

function setup(initial: string, config: PromptToolsConfig = CONFIG) {
  queryClient.setQueryData(toolKeys.config, config)
  const onChange = vi.fn()
  render(<Field initial={initial} onChange={onChange} />, { wrapper })
  return onChange
}

describe("prompt tools around a prompt editor", () => {
  beforeEach(() => {
    usePromptViews.setState({ views: {} })
    vi.spyOn(promptToolsApi, "lookup").mockResolvedValue({})
  })
  afterEach(() => {
    vi.restoreAllMocks()
    queryClient.clear()
  })

  it("shows the tools bar, without translation when no provider is configured", async () => {
    setup("a, b")
    const bar = await screen.findByRole("group", { name: "Prompt tools" }, SLOW)
    for (const name of ["Show as tags", "Quick tags", "Prompt history", "Favorites"]) expect(within(bar).getByRole("button", { name })).toBeInTheDocument()
    expect(within(bar).queryByRole("button", { name: "Translate" })).toBeNull()
  }, 20_000)

  it("adds nothing when the prompt tools are off, and keeps the editor", async () => {
    setup("a", { ...CONFIG, tools: { ...CONFIG.tools, enabled: false } })
    expect(screen.getByRole("textbox", { name: "Prompt editor" })).toBeVisible()
    await act(() => new Promise((resolve) => setTimeout(resolve, 50)))
    expect(screen.queryByRole("group", { name: "Prompt tools" })).toBeNull()
  })

  it("switches between the text editor and the tag view (kept mounted, remembered)", async () => {
    setup("a, b")
    fireEvent.click(await screen.findByRole("button", { name: "Show as tags" }, SLOW))
    expect(await screen.findByRole("list", { name: "Tags of Prompt" }, SLOW)).toBeInTheDocument()
    expect(screen.getByRole("textbox", { name: "Prompt editor", hidden: true })).not.toBeVisible()
    expect(usePromptViews.getState().views["positive:Prompt"]).toBe("chips")
    fireEvent.click(screen.getByRole("button", { name: "Edit as text" }))
    await waitFor(() => expect(screen.getByRole("textbox", { name: "Prompt editor" })).toBeVisible(), SLOW)
  }, 20_000)

  it("translates non-English tags into English with the configured provider", async () => {
    const translate = vi.spyOn(promptToolsApi, "translate").mockResolvedValue({ provider: "dictionary", translations: ["long hair"] })
    const onChange = setup("1girl, (长发:1.2)", { ...CONFIG, tools: { ...CONFIG.tools, translation: "dictionary" } })
    fireEvent.click(await screen.findByRole("button", { name: "Translate" }, SLOW))
    fireEvent.click(await screen.findByRole("menuitem", { name: "Write the prompt in English" }, SLOW))
    await waitFor(() => expect(onChange).toHaveBeenLastCalledWith("1girl, (long hair:1.2)"), SLOW)
    expect(translate).toHaveBeenCalledWith(["长发"], "en")
  }, 20_000)

  it("toggles quick tags in the prompt", async () => {
    vi.spyOn(promptToolsApi, "groups").mockResolvedValue([{ name: "人物", groups: [{ name: "对象", color: null, tags: [{ tag: "1girl", label: "1女孩" }, { tag: "smile", label: "微笑" }] }] }])
    const onChange = setup("1girl")
    fireEvent.click(await screen.findByRole("button", { name: "Quick tags" }, SLOW))
    const girl = await screen.findByRole("button", { name: /1女孩/ }, SLOW)
    expect(girl).toHaveAttribute("aria-pressed", "true")
    fireEvent.click(screen.getByRole("button", { name: /微笑/ }))
    expect(onChange).toHaveBeenLastCalledWith("1girl, smile")
  }, 20_000)

  it("lists the history and uses an entry", async () => {
    vi.spyOn(promptToolsApi.history, "list").mockResolvedValue({
      items: [{ id: "h1", tab: "txt2img", prompt: "a cat, sunset", negative_prompt: "blurry", name: "", created: 1, last_used: Date.now() / 1000 - 60, uses: 3 }],
      next_cursor: null,
      total: 1,
    })
    const onChange = setup("")
    fireEvent.click(await screen.findByRole("button", { name: "Prompt history" }, SLOW))
    const entry = await screen.findByRole("button", { name: /a cat, sunset/ }, SLOW)
    expect(entry).toHaveTextContent("3 generations")
    expect(promptToolsApi.history.list).toHaveBeenCalledWith(expect.objectContaining({ field: "prompt" }), expect.anything())
    fireEvent.click(entry)
    expect(onChange).toHaveBeenLastCalledWith("a cat, sunset")
  }, 20_000)
})
