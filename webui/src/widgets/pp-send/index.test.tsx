import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { toast } from "sonner"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { useSession } from "@/features/gallery/session-store"
import { deliver } from "@/features/send-to/send-to"
import { FormProvider, OwnerProvider } from "@/ui/context"
import { createFormStore } from "@/ui/form-store"
import type { CustomNode } from "@/ui/schema"
import { widgetKeys } from "../_lib/i18n"
import PpSend from "./index"
import { sourceItem } from "./source"
import zh from "./locales/zh-CN.json"

vi.mock("@/features/send-to/send-to", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/features/send-to/send-to")>()),
  deliver: vi.fn(async () => 2),
}))

const OWNER = "post_processing_suite"
const PROPS = { action: "apply_send", target: "img2img" as const }
const NODE: CustomNode = { type: "custom", name: "send", label: "", widget: "pp-send", props: PROPS, default: null }

function setup(props: typeof PROPS | { action: string; target: "inpaint" } = PROPS) {
  const store = createFormStore(`pp-send-${Math.random()}`, { persistDrafts: false })
  store.getState().load([{ owner: OWNER, layout: null, defaults: { enable: true, tone_enabled: true, send: null, preset: "" } }])
  render(
    <FormProvider formId="txt2img" store={store}>
      <OwnerProvider owner={OWNER}>
        <PpSend value={null} onChange={() => {}} props={props} node={NODE} disabled={false} id="send" label="" />
      </OwnerProvider>
    </FormProvider>,
  )
  return store
}

describe("pp-send widget", () => {
  beforeEach(() => {
    useSession.getState().clear()
    vi.mocked(deliver).mockClear()
  })
  afterEach(() => vi.restoreAllMocks())

  function answer(sent: string) {
    vi.spyOn(globalThis, "fetch").mockImplementation(
      async () => new Response(JSON.stringify({ __outputs__: { sent } }), { headers: { "Content-Type": "application/json" } }),
    )
  }

  it("waits for a result to send", () => {
    setup()
    expect(screen.getByRole("button", { name: "Apply PP → Send to img2img" })).toBeDisabled()
    expect(screen.getByText("Generate an image first")).toBeInTheDocument()
  })

  it("post-processes the selected result on the server and sends the copy to img2img", async () => {
    const calls: Record<string, unknown>[] = []
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const request = input as Request
      expect(new URL(request.url).pathname).toBe("/api/v2/actions/post_processing_suite/apply_send")
      calls.push((await request.json()) as Record<string, unknown>)
      return new Response(JSON.stringify({ __outputs__: { sent: "pp-copy" } }), { headers: { "Content-Type": "application/json" } })
    })
    act(() =>
      useSession.getState().addJob({
        jobId: "j1",
        tab: "txt2img",
        outputs: [
          { id: "out-a", kind: "image" },
          { id: "out-b", kind: "image" },
        ],
      }),
    )
    act(() => useSession.getState().select("j1:1"))
    const store = setup()
    expect(screen.getByRole("img", { name: "Selected result" })).toHaveAttribute("src", expect.stringContaining("/thumbs/out-b"))

    fireEvent.click(screen.getByRole("button", { name: "Apply PP → Send to img2img" }))
    await waitFor(() => expect(deliver).toHaveBeenCalled())
    expect(calls[0]).toMatchObject({ file_id: "out-b", values: { enable: true, tone_enabled: true } })
    expect(deliver).toHaveBeenCalledWith("img2img", [{ owner: "core", values: { init_img: "pp-copy", mode: "img2img" } }])
    expect(store.getState().outputs[OWNER]?.sent).toBe("pp-copy")
  })

  it("uses this tab's results only", () => {
    act(() => {
      useSession.getState().addJob({ jobId: "t", tab: "txt2img", outputs: [{ id: "txt", kind: "image" }] })
      useSession.getState().addJob({ jobId: "i", tab: "img2img", outputs: [{ id: "img", kind: "image" }] })
    })
    // The newest (selected) result is img2img's: the txt2img one is sent
    expect(sourceItem(useSession.getState(), "txt2img")?.fileId).toBe("txt")
    act(() => useSession.getState().select("t:0"))
    expect(sourceItem(useSession.getState(), "txt2img")?.fileId).toBe("txt")
    expect(sourceItem({ items: useSession.getState().items.filter((item) => item.tab === "img2img"), selected: null }, "txt2img")).toBeNull()
  })

  it("names the inpaint target and reports success only when something was delivered", async () => {
    answer("pp-copy")
    act(() => useSession.getState().addJob({ jobId: "j2", tab: "txt2img", outputs: [{ id: "out", kind: "image" }] }))
    setup({ action: "apply_send", target: "inpaint" })
    vi.mocked(deliver).mockResolvedValueOnce(0)
    const success = vi.spyOn(toast, "success")
    const failure = vi.spyOn(toast, "error")
    fireEvent.click(screen.getByRole("button", { name: "Apply PP → Send to inpaint" }))
    await waitFor(() => expect(failure).toHaveBeenCalledWith("Could not send the post-processed image", expect.anything()))
    expect(deliver).toHaveBeenCalledWith("img2img", [{ owner: "core", values: { init_img: "pp-copy", mode: "Inpaint" } }])
    expect(success).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole("button", { name: "Apply PP → Send to inpaint" }))
    await waitFor(() => expect(success).toHaveBeenCalledWith("Post-processed image sent to inpaint", expect.anything()))
  })

  it("has a zh-CN translation for every chrome string", () => {
    const sources = import.meta.glob<string>(["./*.tsx", "!./*.test.*"], { query: "?raw", import: "default", eager: true })
    const keys = Object.values(sources).flatMap(widgetKeys)
    expect(keys.length).toBeGreaterThan(0)
    expect(keys.filter((key) => !(zh as Record<string, string>)[key])).toEqual([])
  })
})
