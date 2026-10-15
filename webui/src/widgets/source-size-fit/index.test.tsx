import { act, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { api } from "@/api/endpoints"
import { FormProvider, OwnerProvider } from "@/ui/context"
import { createFormStore, type FormStore } from "@/ui/form-store"
import type { CustomNode } from "@/ui/schema"
import { widgetKeys } from "../_lib/i18n"
import zh from "./locales/zh-CN.json"
import SourceSizeFit from "./index"

const OWNER = "anima_random_resolution"
const NODE: CustomNode = { type: "custom", name: "fit_source", label: "", widget: "source-size-fit", props: { source: "core.init_img", action: "fit_to_source", manual_action: "fit_to_source_now" }, default: true }

type Setup = { initImg?: string | null; value?: boolean; disabled?: boolean; formId?: string }

function mount({ initImg = null, value = true, disabled = false, formId = `img2img-${Math.random()}` }: Setup = {}) {
  const store = createFormStore(formId, { persistDrafts: false })
  store.getState().load([
    { owner: "core", layout: null, defaults: { init_img: initImg, width: 1024, height: 1024 } },
    { owner: OWNER, layout: { type: "group", children: [NODE] }, defaults: { fit_source: true } },
  ])
  const onChange = vi.fn()
  const view = (current: boolean) => (
    <FormProvider formId={formId} store={store}>
      <OwnerProvider owner={OWNER}>
        <SourceSizeFit value={current} onChange={onChange} props={NODE.props as never} node={NODE} disabled={disabled} id="fit" label="" />
      </OwnerProvider>
    </FormProvider>
  )
  const result = render(view(value))
  return { store, onChange, formId, rerender: (current: boolean) => result.rerender(view(current)), unmount: result.unmount }
}

const setSource = (store: FormStore, id: string | null) => act(() => store.getState().setValue("core", "init_img", id))

afterEach(() => vi.restoreAllMocks())

// The first render (jsdom, Radix) can take seconds on a loaded machine
describe("source-size-fit", { timeout: 20_000 }, () => {
  it("fits once per new source image and applies the size patch", async () => {
    const action = vi.spyOn(api, "action").mockResolvedValue({ "core.width": { value: 1216 }, "core.height": { value: 1664 } })
    const { store } = mount()
    expect(action).not.toHaveBeenCalled()
    setSource(store, "upload-1")
    expect(action).toHaveBeenCalledTimes(1)
    const [owner, name, body] = action.mock.calls[0]!
    expect([owner, name]).toEqual([OWNER, "fit_to_source"])
    expect(body.core).toMatchObject({ init_img: "upload-1" })
    await vi.waitFor(() => expect(store.getState().values.core).toMatchObject({ width: 1216, height: 1664 }))
    // The same source again (a re-render, the size changing) does not refit
    act(() => store.getState().setValue("core", "width", 512))
    expect(action).toHaveBeenCalledTimes(1)
    setSource(store, "upload-2")
    expect(action).toHaveBeenCalledTimes(2)
  })

  it("fits a source delivered before it mounted, once", () => {
    const action = vi.spyOn(api, "action").mockResolvedValue({})
    const first = mount({ initImg: "sent-1", formId: "img2img-delivered" })
    expect(action).toHaveBeenCalledTimes(1)
    first.unmount()
    // Back on the page with the same source: nothing to do
    mount({ initImg: "sent-1", formId: "img2img-delivered" })
    expect(action).toHaveBeenCalledTimes(1)
  })

  it("stays off when switched off, and fits on demand", () => {
    const action = vi.spyOn(api, "action").mockResolvedValue({})
    const { store, onChange } = mount({ value: false })
    setSource(store, "upload-3")
    expect(action).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole("button", { name: "Fit now" }))
    expect(action).toHaveBeenCalledTimes(1)
    expect(action.mock.calls[0]![1]).toBe("fit_to_source_now")
    fireEvent.click(screen.getByRole("switch"))
    expect(onChange).toHaveBeenCalledWith(true)
  })

  it("cannot fit without a source", () => {
    mount()
    expect(screen.getByRole("button", { name: "Fit now" })).toBeDisabled()
  })

  it("has a zh-CN translation for every chrome string", () => {
    const sources = import.meta.glob<string>(["./*.tsx", "!./*.test.*"], { query: "?raw", import: "default", eager: true })
    const keys = Object.values(sources).flatMap(widgetKeys)
    expect(keys.length).toBeGreaterThan(0)
    expect(keys.filter((key) => !(zh as Record<string, string>)[key])).toEqual([])
  })
})
