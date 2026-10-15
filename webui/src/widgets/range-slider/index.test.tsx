import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { i18n } from "@/i18n"
import { createFormStore, type FormStore } from "@/ui/form-store"
import { PageForm } from "@/ui/SchemaForm"
import type { Json, PageSchema } from "@/ui/schema"
import { widgetKeys } from "../_lib/i18n"
import zh from "./locales/zh-CN.json"
import source from "./index.tsx?raw"

const LABEL = "Timestep Range"

const UNIT = { minimum: 0, maximum: 1, step: 0.05 }

function schemaWith(value: unknown, props: Record<string, unknown> = UNIT, extra: Record<string, unknown> = {}): PageSchema {
  return {
    version: 1,
    id: "range",
    title: "Range",
    owner: "page:range",
    layout: { type: "custom", name: "percent", label: LABEL, widget: "range-slider", props, default: [0.15, 0.85], ...extra } as PageSchema["layout"],
    defaults: { percent: value as Json },
  }
}

function renderWidget(value: unknown = [0.15, 0.85], props?: Record<string, unknown>, extra?: Record<string, unknown>): FormStore {
  const store = createFormStore(`range-slider-${Math.random()}`, { persistDrafts: false })
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, enabled: false } } })
  render(
    <QueryClientProvider client={client}>
      <PageForm schema={schemaWith(value, props, extra)} store={store} />
    </QueryClientProvider>,
  )
  return store
}

const percent = (store: FormStore) => store.getState().values["page:range"]?.percent

describe("range-slider widget", () => {
  it("renders the pair as two thumbs and two number boxes", async () => {
    renderWidget()
    const group = await screen.findByRole("group", { name: LABEL }, { timeout: 5000 })
    expect(group.querySelectorAll("input[type=range]")).toHaveLength(2)
    expect(screen.getByRole("textbox", { name: `${LABEL}: from` })).toHaveValue("0.15")
    expect(screen.getByRole("textbox", { name: `${LABEL}: to` })).toHaveValue("0.85")
    // the field label points at the first box
    expect(screen.getByLabelText(LABEL, { selector: "input" })).toBe(screen.getByRole("textbox", { name: `${LABEL}: from` }))
  })

  it("commits typed ends, clamped so they never cross", async () => {
    const store = renderWidget()
    const from = await screen.findByRole("textbox", { name: `${LABEL}: from` }, { timeout: 5000 })
    fireEvent.change(from, { target: { value: "0.3" } })
    fireEvent.blur(from)
    expect(percent(store)).toEqual([0.3, 0.85])
    const to = screen.getByRole("textbox", { name: `${LABEL}: to` })
    fireEvent.change(to, { target: { value: "0.1" } })
    fireEvent.blur(to)
    expect(percent(store)).toEqual([0.3, 0.3])
  })

  it("moves a thumb with the keyboard by one step", async () => {
    const store = renderWidget([0.2, 0.6])
    const group = await screen.findByRole("group", { name: LABEL }, { timeout: 5000 })
    const [low] = group.querySelectorAll("input[type=range]")
    fireEvent.keyDown(low!, { key: "ArrowRight" })
    expect(percent(store)).toEqual([0.25, 0.6])
  })

  it("shows a malformed value as the full range and repairs it on the first edit", async () => {
    const store = renderWidget("garbage")
    const to = await screen.findByRole("textbox", { name: `${LABEL}: to` }, { timeout: 5000 })
    expect(to).toHaveValue("1")
    fireEvent.change(to, { target: { value: "0.5" } })
    fireEvent.blur(to)
    expect(percent(store)).toEqual([0, 0.5])
  })

  it("is disabled with its field", async () => {
    renderWidget([0.2, 0.6], UNIT, { enabled_if: false })
    const from = await screen.findByRole("textbox", { name: `${LABEL}: from` }, { timeout: 5000 })
    expect(from).toBeDisabled()
    expect(screen.getByRole("textbox", { name: `${LABEL}: to` })).toBeDisabled()
  })

  it("translates its chrome from its own catalog", async () => {
    expect(widgetKeys(source).sort()).toEqual(Object.keys(zh).sort())
    renderWidget()
    await screen.findByRole("group", { name: LABEL }, { timeout: 5000 })
    await act(async () => {
      await i18n.changeLanguage("zh-CN")
    })
    expect(screen.getByRole("textbox", { name: `${LABEL}：起点` })).toBeInTheDocument()
    await act(async () => {
      await i18n.changeLanguage("en")
    })
  })
})
