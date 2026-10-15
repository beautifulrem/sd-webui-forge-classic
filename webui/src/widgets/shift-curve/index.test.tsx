import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { createFormStore, type FormStore } from "@/ui/form-store"
import { TabForm } from "@/ui/SchemaForm"
import type { TabSchema } from "@/ui/schema"
import { widgetKeys } from "../_lib/i18n"
import zh from "./locales/zh-CN.json"
import source from "./index.tsx?raw"
import { computeSigmas, sigmaPath } from "./schedule"

// Widgets load lazily; give the chunk time on a busy machine
const LOAD = { timeout: 15_000 }

// compute_sigmas() of lib_dynshift/schedule.py for the same arguments
const PYTHON = [
  { n: 10, start: 5.0, end: 1.5, curve: "Cosine", sigmas: [1.0, 0.975124248745666, 0.9414783750191298, 0.8920364741641338, 0.8165258684912106, 0.7025757712544307, 0.5436011649820114, 0.3540079371581851, 0.1683993602337732, 0.0014992503748125952, 0.0] },
  { n: 8, start: 2.0, end: 6.0, curve: "Linear", sigmas: [1.0, 0.9391971202754521, 0.8872369588002904, 0.8322444965095916, 0.7631338278630222, 0.6609783293741213, 0.47674157583618104, 0.005970149253731349, 0.0] },
  { n: 6, start: 3.0, end: 1.0, curve: "Exponential", sigmas: [1.0, 0.9060587583928099, 0.7438925322452158, 0.5091187847558428, 0.23838042171594404, 0.0010000000000000009, 0.0] },
  { n: 1, start: 3.0, end: 3.0, curve: "Linear", sigmas: [1.0, 0.0] },
  { n: 4, start: 0.001, end: 2.0, curve: "Linear", sigmas: [0.9999999999999991, 0.5742305036780688, 0.40131961382418346, 0.001998001998002, 0.0] },
]

describe("computeSigmas", () => {
  it.each(PYTHON)("matches the Python schedule ($n steps, $start → $end, $curve)", ({ n, start, end, curve, sigmas }) => {
    const got = computeSigmas(n, start, end, curve)
    expect(got).toHaveLength(sigmas.length)
    got.forEach((value, i) => expect(value).toBeCloseTo(sigmas[i]!, 12))
  })
})

const schema: TabSchema = {
  version: 1,
  tab: "txt2img",
  core: { owner: "core", layout: { type: "column", children: [{ type: "slot", section: "sampler" }] }, defaults: {} },
  scripts: [
    {
      id: "sampler",
      title: "Sampler",
      kind: "alwayson",
      section: "sampler",
      order: 0,
      legacy: false,
      layout: { type: "slider", name: "steps", label: "Sampling Steps", minimum: 1, maximum: 150, step: 1, integer: true },
      defaults: { steps: 20 },
      args: ["steps"],
    },
    {
      id: "dynamic_shift",
      title: "Dynamic Shift",
      kind: "alwayson",
      section: "sampler",
      order: 5,
      legacy: false,
      layout: {
        type: "group",
        children: [
          { type: "slider", name: "shift_start", label: "Shift @ first step", minimum: 0.1, maximum: 12, step: 0.05, integer: false },
          { type: "slider", name: "shift_end", label: "Shift @ last step", minimum: 0.1, maximum: 12, step: 0.05, integer: false },
          {
            type: "custom",
            name: "curve",
            label: "Curve",
            widget: "shift-curve",
            props: { choices: ["Linear", "Cosine", "Exponential"], start: "shift_start", end: "shift_end", steps: "sampler.steps" },
          },
        ],
      },
      defaults: { shift_start: 5, shift_end: 1.5, curve: "Cosine" },
      args: ["shift_start", "shift_end", "curve"],
    },
  ],
}

function renderForm(): FormStore {
  const store = createFormStore(`shift-curve-${Math.random()}`, { persistDrafts: false })
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, enabled: false } } })
  render(
    <QueryClientProvider client={client}>
      <TabForm schema={schema} formId="txt2img" store={store} />
    </QueryClientProvider>,
  )
  return store
}

describe("shift-curve widget", { timeout: 30_000 }, () => {
  it("edits the curve and plots the schedule of the sibling shifts and the steps", async () => {
    const store = renderForm()
    const plot = await screen.findByRole("img", { name: "Sigma schedule over 20 steps: shift 5 to 1.5, Cosine" }, LOAD)
    expect(screen.getByRole("radiogroup", { name: "Curve" })).toBeInTheDocument()
    expect(plot.querySelector("[data-testid=dynamic-schedule]")?.getAttribute("d")).toBe(sigmaPath(computeSigmas(20, 5, 1.5, "Cosine").slice(0, -1), 360, 140, 8))

    fireEvent.click(screen.getByRole("radio", { name: "Exponential" }))
    expect(store.getState().values.dynamic_shift?.curve).toBe("Exponential")

    act(() => {
      store.getState().setValues("dynamic_shift", { shift_start: 3 })
      store.getState().setValues("sampler", { steps: 8 })
    })
    expect(screen.getByRole("img", { name: "Sigma schedule over 8 steps: shift 3 to 1.5, Exponential" })).toBeInTheDocument()
    expect(screen.getByText("Constant 3")).toBeInTheDocument()
  })

  it("translates every chrome string", () => {
    const missing = widgetKeys(source).filter((key) => !(key in zh))
    expect(missing).toEqual([])
  })
})
