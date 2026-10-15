import { QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { toast } from "sonner"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { keys, queryClient } from "@/api/queries"
import { sendToPage } from "@/pages/ext/deliver"
import { bootstrapWith } from "@/test/api-fixtures"
import { FormProvider, OwnerProvider } from "@/ui/context"
import { createFormStore, getFormStore } from "@/ui/form-store"
import type { CustomNode } from "@/ui/schema"
import { widgetKeys } from "../_lib/i18n"
import SendButton from "./index"
import zh from "./locales/zh-CN.json"
import source from "./index.tsx?raw"
import { appendText, openUrl, sendToPrompt } from "./send"

vi.mock("@/pages/ext/deliver", () => ({ sendToPage: vi.fn(async () => 1) }))

const OWNER = "page:workshop"
const NODE: CustomNode = { type: "custom", name: "send", label: "", widget: "send-button", props: {}, default: null }

function setup(props: Record<string, unknown>, values: Record<string, unknown> = {}) {
  const store = createFormStore(`page:send-${Math.random()}`, { persistDrafts: false })
  store.getState().load([{ owner: OWNER, layout: null, defaults: { text: "", url: "", image: null, ...values } }])
  render(
    <QueryClientProvider client={queryClient}>
      <FormProvider formId={OWNER} store={store}>
        <OwnerProvider owner={OWNER}>
          {/* oxlint-disable-next-line typescript/no-explicit-any -- each test passes one props shape */}
          <SendButton value={null} onChange={() => {}} props={props as any} node={NODE} disabled={false} id="send" label="" />
        </OwnerProvider>
      </FormProvider>
    </QueryClientProvider>,
  )
  return store
}

describe("appendText", () => {
  it("joins like the old Gradio buttons", () => {
    expect(appendText("", "cat")).toBe("cat")
    expect(appendText("a dog  \n", "cat")).toBe("a dog, cat")
    expect(appendText("a dog,", "cat")).toBe("a dog, cat")
  })
})

describe("sendToPrompt", () => {
  beforeEach(() => {
    getFormStore("txt2img").getState().setValues("core", { prompt: "" })
    history.replaceState(null, "", "#/ext/workshop")
  })

  it("appends to or replaces the txt2img prompt, as a draft when the form is not loaded", () => {
    const store = getFormStore("txt2img")
    store.getState().setValues("core", { prompt: "masterpiece," })
    expect(sendToPrompt("txt2img", "  1girl  ", "append", false)).toBe(true)
    expect(store.getState().values.core?.prompt).toBe("masterpiece, 1girl")
    expect(location.hash).toBe("#/ext/workshop")
    expect(sendToPrompt("txt2img", "solo", "replace", true)).toBe(true)
    expect(store.getState().values.core?.prompt).toBe("solo")
    expect(location.hash).toBe("#/txt2img")
    expect(sendToPrompt("txt2img", "   ", "replace", true)).toBe(false)
  })
})

describe("openUrl", () => {
  it("opens http(s) links only", () => {
    const open = vi.spyOn(window, "open").mockImplementation(() => null)
    expect(openUrl(" https://danbooru.donmai.us/posts/1 ")).toBe(true)
    expect(open).toHaveBeenCalledWith("https://danbooru.donmai.us/posts/1", "_blank", "noopener,noreferrer")
    expect(openUrl("javascript:alert(1)")).toBe(false)
    expect(openUrl("not a url")).toBe(false)
    expect(open).toHaveBeenCalledTimes(1)
  })
})

describe("send-button widget", () => {
  beforeEach(() => {
    queryClient.setQueryData(keys.app, { ...bootstrapWith({}), pages: [{ id: "joycaption", title: "JoyCaption" }] })
    vi.mocked(sendToPage).mockClear()
    history.replaceState(null, "", "#/ext/workshop")
  })
  afterEach(() => {
    vi.restoreAllMocks()
    queryClient.clear()
  })

  it("sends a text param to the txt2img prompt", () => {
    getFormStore("txt2img").getState().setValues("core", { prompt: "a" })
    setup({ to: "prompt", source: "text", mode: "append", navigate: false, label: "→ 1. Quality" }, { text: "best quality" })
    const success = vi.spyOn(toast, "success")
    fireEvent.click(screen.getByRole("button", { name: "→ 1. Quality" }))
    expect(getFormStore("txt2img").getState().values.core?.prompt).toBe("a, best quality")
    expect(success).toHaveBeenCalledWith("Added to the txt2img prompt")
  })

  it("says when there is nothing to send", () => {
    const info = vi.spyOn(toast, "info")
    setup({ to: "prompt", source: "text", label: "Send" })
    fireEvent.click(screen.getByRole("button", { name: "Send" }))
    expect(info).toHaveBeenCalledWith("Nothing to send")
  })

  it("runs the action first, then sends the image to the page", async () => {
    const calls: string[] = []
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      calls.push(decodeURIComponent(new URL((input as Request).url).pathname))
      return new Response(JSON.stringify({ image: { value: "up-1" } }), { headers: { "Content-Type": "application/json" } })
    })
    setup({ to: "page", page: "joycaption", param: "image", source: "image", action: "download", label: "Send to JoyCaption" })
    fireEvent.click(screen.getByRole("button", { name: "Send to JoyCaption" }))
    await waitFor(() => expect(sendToPage).toHaveBeenCalledWith("joycaption", { image: "up-1" }))
    expect(calls).toEqual([`/api/v2/actions/${OWNER}/download`])
  })

  it("sends nothing when the action fails (the source may hold an older image)", async () => {
    const calls: string[] = []
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      calls.push(new URL((input as Request).url).pathname)
      const answer = calls.length === 1 ? { __toast__: { level: "error", message: "❌ Download failed" } } : { __toast__: { level: "warning", message: "❗ No image URL" } }
      return new Response(JSON.stringify(answer), { headers: { "Content-Type": "application/json" } })
    })
    setup({ to: "page", page: "joycaption", param: "image", source: "image", action: "download", label: "Send to JoyCaption" }, { image: "up-old" })
    const button = screen.getByRole("button", { name: "Send to JoyCaption" })
    fireEvent.click(button)
    await waitFor(() => expect(calls).toHaveLength(1))
    await waitFor(() => expect(button).toBeEnabled())
    fireEvent.click(button)
    await waitFor(() => expect(calls).toHaveLength(2))
    await waitFor(() => expect(button).toBeEnabled())
    expect(sendToPage).not.toHaveBeenCalled()
  })

  it("is not offered when the target page is missing", () => {
    queryClient.setQueryData(keys.app, bootstrapWith({}))
    setup({ to: "page", page: "joycaption", param: "image", source: "image", label: "Send to JoyCaption" })
    expect(screen.queryByRole("button")).not.toBeInTheDocument()
  })

  it("opens the post URL", () => {
    const open = vi.spyOn(window, "open").mockImplementation(() => null)
    setup({ to: "url", source: "url", label: "Open" }, { url: "https://x.example/p/1" })
    fireEvent.click(screen.getByRole("button", { name: "Open" }))
    expect(open).toHaveBeenCalledOnce()
  })

  it("translates every chrome string", () => {
    const catalog = zh as Record<string, string>
    expect(widgetKeys(source).filter((key) => !catalog[key])).toEqual([])
  })
})
