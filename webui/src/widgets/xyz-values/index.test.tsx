import { act, render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"
import { FormProvider, OwnerProvider } from "@/ui/context"
import { createFormStore } from "@/ui/form-store"
import type { CustomNode } from "@/ui/schema"
import { widgetKeys } from "../_lib/i18n"
import XyzValues from "./index"
import zh from "./locales/zh-CN.json"

const OWNER = "x_y_z_plot"
const PROPS = { axis: "x_type", action: "axis_choices" }
const NODE: CustomNode = { type: "custom", name: "x_values_dropdown", label: "X values", widget: "xyz-values", props: PROPS, default: [] }
const CHOICES = { Sampler: ["Euler", "Euler a", "DPM++ 2M"], "Schedule type": ["Automatic", "Karras"] }

type Call = { path: string; body: Record<string, unknown> }

function mockActions(answer: unknown = { __outputs__: { axis_choices: CHOICES } }) {
  const calls: Call[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const request = input as Request
    calls.push({ path: new URL(request.url).pathname, body: (await request.json()) as Record<string, unknown> })
    return new Response(JSON.stringify(answer), { headers: { "Content-Type": "application/json" } })
  })
  return calls
}

function setup(value: string[] = []) {
  const store = createFormStore(`xyz-values-${Math.random()}`, { persistDrafts: false })
  store.getState().load([{ owner: OWNER, layout: null, defaults: { x_type: "Sampler", x_values_dropdown: value } }])
  const onChange = vi.fn((next: string[] | null) => store.getState().setValue(OWNER, "x_values_dropdown", next))
  const view = () => (
    <FormProvider formId="txt2img" store={store}>
      <OwnerProvider owner={OWNER}>
        <XyzValues
          value={(store.getState().values[OWNER]?.x_values_dropdown as string[]) ?? []}
          onChange={onChange}
          props={PROPS}
          node={NODE}
          disabled={false}
          id="xv"
          label="X values"
        />
      </OwnerProvider>
    </FormProvider>
  )
  const result = render(view())
  return { store, onChange, rerender: () => result.rerender(view()) }
}

/** The combobox popup is driven with user-event, which is slow on a loaded machine. */
const SLOW = { timeout: 20_000 }

describe("xyz-values widget", () => {
  afterEach(() => vi.restoreAllMocks())

  it("loads every axis' values once and offers the selected axis' ones", SLOW, async () => {
    const calls = mockActions()
    const { store } = setup()
    await waitFor(() => expect(store.getState().outputs[OWNER]?.axis_choices).toEqual(CHOICES))
    expect(calls).toHaveLength(1)
    expect(calls[0]!.path).toBe("/api/v2/actions/x_y_z_plot/axis_choices")
    expect(calls[0]!.body.values).toMatchObject({ x_type: "Sampler" })

    const user = userEvent.setup()
    await user.click(screen.getByRole("combobox", { name: "X values" }))
    expect(await screen.findByRole("option", { name: "Euler a" })).toBeInTheDocument()
    expect(screen.queryByRole("option", { name: "Karras" })).toBeNull()

    act(() => store.getState().setValue(OWNER, "x_type", "Schedule type"))
    expect(await screen.findByRole("option", { name: "Karras" })).toBeInTheDocument()
    expect(calls).toHaveLength(1) // the cached answer serves every axis
  })

  it("selects values and keeps ones outside the list as chips", SLOW, async () => {
    mockActions()
    const { onChange, store, rerender } = setup(["Not listed"])
    await waitFor(() => expect(store.getState().outputs[OWNER]?.axis_choices).toBeDefined())
    expect(screen.getByText("Not listed", { selector: "[data-slot=combobox-chip]" })).toBeInTheDocument()
    const user = userEvent.setup()
    await user.click(screen.getByRole("combobox", { name: "X values" }))
    await user.keyboard("dpm")
    await user.click(await screen.findByRole("option", { name: "DPM++ 2M" }))
    expect(onChange).toHaveBeenLastCalledWith(["Not listed", "DPM++ 2M"])
    rerender()
    expect(screen.getByText("DPM++ 2M", { selector: "[data-slot=combobox-chip]" })).toBeInTheDocument()
  })

  it("says so when the values cannot be loaded", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(JSON.stringify({ detail: "boom" }), { status: 500, headers: { "Content-Type": "application/json" } }))
    setup()
    await waitFor(() => expect(screen.getByRole("combobox", { name: "X values" })).toHaveAttribute("placeholder", "Could not load the values"))
  })

  it("shares one load between the X, Y and Z dropdowns", async () => {
    const calls = mockActions()
    const store = createFormStore(`xyz-values-${Math.random()}`, { persistDrafts: false })
    store.getState().load([{ owner: OWNER, layout: null, defaults: { x_type: "Sampler", y_type: "Schedule type" } }])
    render(
      <FormProvider formId="txt2img" store={store}>
        <OwnerProvider owner={OWNER}>
          {(["x", "y"] as const).map((axis) => (
            <XyzValues key={axis} value={[]} onChange={vi.fn()} props={{ axis: `${axis}_type`, action: "axis_choices" }} node={NODE} disabled={false} id={axis} label={axis} />
          ))}
        </OwnerProvider>
      </FormProvider>,
    )
    await waitFor(() => expect(store.getState().outputs[OWNER]?.axis_choices).toEqual(CHOICES))
    expect(calls).toHaveLength(1)
  })

  it("prefers the fresh list an axis action returned", SLOW, async () => {
    mockActions()
    const { store } = setup()
    await waitFor(() => expect(store.getState().outputs[OWNER]?.axis_choices).toEqual(CHOICES))
    act(() => store.getState().setOutputs(OWNER, { "axis_choices/Sampler": ["Euler", "New sampler"] }))
    const user = userEvent.setup()
    await user.click(screen.getByRole("combobox", { name: "X values" }))
    expect(await screen.findByRole("option", { name: "New sampler" })).toBeInTheDocument()
    expect(screen.queryByRole("option", { name: "Euler a" })).toBeNull()
  })

  it("reports an axis whose list failed without hiding the others", async () => {
    mockActions({ __outputs__: { axis_choices: { ...CHOICES, Sampler: null } } })
    const { store } = setup()
    await waitFor(() => expect(screen.getByRole("combobox", { name: "X values" })).toHaveAttribute("placeholder", "Could not load the values"))
    act(() => store.getState().setValue(OWNER, "x_type", "Schedule type"))
    await waitFor(() => expect(screen.getByRole("combobox", { name: "X values" })).toHaveAttribute("placeholder", "Select values…"))
  })

  it("retries a failed load when the dropdown opens", SLOW, async () => {
    const fetch = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(new Response(JSON.stringify({ detail: "boom" }), { status: 500, headers: { "Content-Type": "application/json" } }))
      .mockResolvedValue(new Response(JSON.stringify({ __outputs__: { axis_choices: CHOICES } }), { headers: { "Content-Type": "application/json" } }))
    setup()
    const input = screen.getByRole("combobox", { name: "X values" })
    await waitFor(() => expect(input).toHaveAttribute("placeholder", "Could not load the values"))
    const user = userEvent.setup()
    await user.click(input)
    expect(await screen.findByRole("option", { name: "Euler a" })).toBeInTheDocument()
    expect(fetch).toHaveBeenCalledTimes(2)
  })

  it("has a zh-CN translation for every chrome string", () => {
    const sources = import.meta.glob<string>(["./*.tsx", "!./*.test.*"], { query: "?raw", import: "default", eager: true })
    const keys = Object.values(sources).flatMap(widgetKeys)
    expect(keys.length).toBeGreaterThan(0)
    expect(keys.filter((key) => !(zh as Record<string, string>)[key])).toEqual([])
  })
})
