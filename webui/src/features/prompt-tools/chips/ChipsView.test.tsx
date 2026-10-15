import { QueryClientProvider } from "@tanstack/react-query"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import type { ReactNode } from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { queryClient } from "@/api/queries"
import { TEST_CONFIG } from "../testing"
import { type PromptToolsConfig, promptToolsApi } from "../api"
import ChipsView from "./ChipsView"

const CONFIG = TEST_CONFIG

const wrapper = ({ children }: { children: ReactNode }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>

function show(value: string, config: PromptToolsConfig = CONFIG) {
  const setValue = vi.fn()
  const utils = render(
    <ChipsView value={value} setValue={setValue} kind="positive" editor={() => null} label="Prompt" disabled={false} fieldId="p" config={config}>
      {null}
    </ChipsView>,
    { wrapper },
  )
  return { setValue, ...utils }
}

describe("tag chips view", () => {
  beforeEach(() => {
    vi.spyOn(promptToolsApi, "lookup").mockResolvedValue({ long_hair: { category: 0, count: 4_300_000, translation: "长发" } })
  })
  afterEach(() => {
    vi.restoreAllMocks()
    queryClient.clear()
  })

  it("shows one chip per tag with weights, networks, dividers and translations", async () => {
    show("1girl, (long_hair:1.2), <lora:style:0.8>\nBREAK\n/*smile, */blush")
    const list = screen.getByRole("list", { name: "Tags of Prompt" })
    expect(list).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /^1girl$/ })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /^long_hair, Weight 1\.2/ })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /^style, Extra network, Weight 0\.8/ })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "BREAK" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /^smile, Disabled/ })).toBeInTheDocument()
    await waitFor(() => expect(screen.getByText("长发")).toBeInTheDocument())
    expect(promptToolsApi.lookup).toHaveBeenCalledWith(["1girl", "blush", "long_hair", "smile"], expect.anything())
  }, 20_000)

  it("is one tab stop; arrows move the focus, Alt+arrows move the tag, Delete removes it", () => {
    const { setValue } = show("a, b, c")
    const a = screen.getByRole("button", { name: "a" })
    const b = screen.getByRole("button", { name: "b" })
    expect(a).toHaveAttribute("tabindex", "0")
    expect(b).toHaveAttribute("tabindex", "-1")
    fireEvent.keyDown(a, { key: "ArrowRight", altKey: true })
    expect(setValue).toHaveBeenLastCalledWith("b, a, c")
    fireEvent.keyDown(b, { key: "Delete" })
    expect(setValue).toHaveBeenLastCalledWith("a, c")
    expect(screen.getByRole("status")).toHaveTextContent("Removed “b”")
  })

  it("edits the weight, disables and deletes from the chip's popover", async () => {
    const { setValue } = show("a, (b:1.2), c")
    act(() => screen.getByRole("button", { name: /^b, Weight 1\.2/ }).click())
    const increase = await screen.findByRole("button", { name: "Increase the weight" })
    fireEvent.click(increase)
    expect(setValue).toHaveBeenLastCalledWith("a, (b:1.3), c")
    fireEvent.click(screen.getByRole("button", { name: "Reset the weight" }))
    expect(setValue).toHaveBeenLastCalledWith("a, b, c")
    fireEvent.click(screen.getByRole("button", { name: "Disable" }))
    expect(setValue).toHaveBeenLastCalledWith("a, /*(b:1.2), */c")
    fireEvent.click(screen.getByRole("button", { name: "Delete" }))
    expect(setValue).toHaveBeenLastCalledWith("a, c")
  }, 20_000)

  it("offers no disabling when prompt comments are off", async () => {
    show("a, b", { ...CONFIG, tools: { ...CONFIG.tools, comments: false } })
    act(() => screen.getByRole("button", { name: "a" }).click())
    await screen.findByRole("button", { name: "Delete" })
    expect(screen.queryByRole("button", { name: "Disable" })).toBeNull()
  }, 20_000)

  it("suggests tags while typing; picking one writes it like a completion and counts its use", async () => {
    const complete = vi.spyOn(promptToolsApi, "complete").mockResolvedValue([{ type: "tag", text: "name_(series)", category: 4, count: 1200 }])
    const recordUse = vi.spyOn(promptToolsApi, "recordUse").mockResolvedValue({ ok: true })
    const setValue = vi.fn()
    render(
      <ChipsView value="a" setValue={setValue} kind="negative" editor={() => null} label="Negative" disabled={false} fieldId="n" config={CONFIG}>
        {null}
      </ChipsView>,
      { wrapper },
    )
    const input = screen.getByRole("combobox", { name: "Add tags" })
    fireEvent.change(input, { target: { value: "b, nam" } })
    const option = await screen.findByRole("option", { name: /name \(series\)/ })
    expect(complete).toHaveBeenCalledWith({ mode: "tag", q: "nam", limit: 8, negative: true }, expect.anything())
    act(() => option.click())
    expect(setValue).toHaveBeenLastCalledWith("a, b, name \\(series\\)")
    expect(recordUse).toHaveBeenCalledWith({ name: "name_(series)", type: "tag", negative: true })
  }, 20_000)

  it("adds typed tags at the end", () => {
    const { setValue } = show("a")
    const input = screen.getByRole("combobox", { name: "Add tags" })
    fireEvent.change(input, { target: { value: "b, c" } })
    fireEvent.keyDown(input, { key: "Enter" })
    expect(setValue).toHaveBeenLastCalledWith("a, b, c")
  })
})
