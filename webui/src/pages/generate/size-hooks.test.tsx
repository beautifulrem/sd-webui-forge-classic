import { fireEvent, render, renderHook } from "@testing-library/react"
import type { ReactNode } from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { tabSchema } from "@/api/mock/fixtures"
import { FormProvider } from "@/ui/context"
import { createFormStore, type FormStore } from "@/ui/form-store"
import { loadSchema } from "./use-form-loader"

const effects = vi.hoisted(() => ({ runChangeEffects: vi.fn() }))
vi.mock("@/ui/actions", async (importOriginal) => ({ ...(await importOriginal<typeof import("@/ui/actions")>()), ...effects }))

const { usePasteSize, useSetSize } = await import("./size-hooks")

let store: FormStore

const wrapper = ({ children }: { children: ReactNode }) => (
  <FormProvider formId="txt2img" store={store}>
    {children}
  </FormProvider>
)

const size = () => ({ width: store.getState().values.core?.width, height: store.getState().values.core?.height })

describe("useSetSize", () => {
  beforeEach(() => {
    effects.runChangeEffects.mockClear()
    store = createFormStore("size-hooks-test", { persistDrafts: false })
    loadSchema(store, tabSchema("txt2img"))
  })

  it("writes width and height as a user edit and runs their change effects", () => {
    const { result } = renderHook(() => useSetSize(), { wrapper })
    result.current({ width: 832.4, height: 1215.6 })
    expect(size()).toEqual({ width: 832, height: 1216 })
    expect(effects.runChangeEffects).toHaveBeenCalledWith({ store, tab: "txt2img" }, "core", ["width", "height"])
  })

  it("clamps to the sliders' bounds, an action's override first", () => {
    const { result } = renderHook(() => useSetSize(), { wrapper })
    result.current({ width: 10, height: 99_999 })
    expect(size()).toEqual({ width: 64, height: 4096 }) // the mock txt2img sliders: 64 to 4096

    store.setState({ overrides: { core: { width: { maximum: 1024 }, height: { minimum: 512 } } } })
    result.current({ width: 1600, height: 100 })
    expect(size()).toEqual({ width: 1024, height: 512 })
  })

  it("does nothing in a form without width and height", () => {
    store = createFormStore("size-hooks-empty", { persistDrafts: false })
    const { result } = renderHook(() => useSetSize(), { wrapper })
    result.current({ width: 512, height: 512 })
    expect(store.getState().values.core).toBeUndefined()
    expect(effects.runChangeEffects).not.toHaveBeenCalled()
  })
})

function PasteTarget() {
  const onPaste = usePasteSize()
  return (
    <div onPaste={onPaste}>
      <span data-testid="plain">size here</span>
      <textarea aria-label="notes" />
      <div data-slot="prompt-editor">
        <span data-testid="prompt">prompt</span>
      </div>
    </div>
  )
}

const paste = (target: Element, text: string) => {
  const event = new Event("paste", { bubbles: true, cancelable: true })
  Object.assign(event, { clipboardData: { getData: (type: string) => (type === "text" ? text : "") } })
  fireEvent(target, event)
  return event
}

describe("usePasteSize", () => {
  beforeEach(() => {
    effects.runChangeEffects.mockClear()
    store = createFormStore("size-paste-test", { persistDrafts: false })
    loadSchema(store, tabSchema("txt2img"))
  })

  it("takes a pasted W x H as the image size", () => {
    const view = render(<PasteTarget />, { wrapper })
    const event = paste(view.getByTestId("plain"), "768 x 1344")
    expect(event.defaultPrevented).toBe(true)
    expect(size()).toEqual({ width: 768, height: 1344 })
  })

  it("leaves other text, and pastes into the prompt or a text area, alone", () => {
    const view = render(<PasteTarget />, { wrapper })
    const before = size()
    expect(paste(view.getByTestId("plain"), "a cat on a sofa").defaultPrevented).toBe(false)
    expect(paste(view.getByRole("textbox", { name: "notes" }), "512x512").defaultPrevented).toBe(false)
    expect(paste(view.getByTestId("prompt"), "512x512").defaultPrevented).toBe(false)
    expect(size()).toEqual(before)
    expect(effects.runChangeEffects).not.toHaveBeenCalled()
  })
})
