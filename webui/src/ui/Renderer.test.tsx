/** Renderer smoke tests: every node kind renders accessibly and binds to the form store. */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { tabSchema } from "@/api/mock/fixtures"
import { createFormStore, type FormStore } from "./form-store"
import { TabForm } from "./SchemaForm"
import type { Node, OwnerDefaults, ScriptSchema, TabSchema } from "./schema"

function schemaOf(layout: Node, defaults: OwnerDefaults = {}, scripts: ScriptSchema[] = []): TabSchema {
  return { version: 1, tab: "txt2img", core: { owner: "core", layout, defaults }, scripts }
}

function renderSchema(schema: TabSchema): FormStore {
  const store = createFormStore(`test-${Math.random()}`, { persistDrafts: false })
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, enabled: false } } })
  render(
    <QueryClientProvider client={client}>
      <TabForm schema={schema} formId="txt2img" store={store} />
    </QueryClientProvider>,
  )
  return store
}

const one = (node: Node, defaults: OwnerDefaults = {}) => renderSchema(schemaOf({ type: "column", children: [node] }, defaults))
const core = (store: FormStore) => store.getState().values.core ?? {}

describe("input kinds", () => {
  it("toggle", () => {
    const store = one({ type: "toggle", name: "on", label: "Enable thing" }, { on: false })
    const toggle = screen.getByRole("switch", { name: "Enable thing" })
    fireEvent.click(toggle)
    expect(core(store).on).toBe(true)
    // Modified → a reset button appears and restores the default.
    fireEvent.click(screen.getByRole("button", { name: "Reset Enable thing to default" }))
    expect(core(store).on).toBe(false)
  })

  it("slider with numeric box", () => {
    const store = one({ type: "slider", name: "steps", label: "Steps", minimum: 1, maximum: 150, step: 1, integer: true }, { steps: 20 })
    // Thumbs stay visibility:hidden until measured (never in jsdom), so query the range input directly.
    expect(screen.getByLabelText("Steps", { selector: "input[type=range]" })).toHaveValue("20")
    const box = screen.getByLabelText("Steps", { selector: "input:not([type=range])" })
    fireEvent.change(box, { target: { value: "42" } })
    fireEvent.blur(box)
    expect(core(store).steps).toBe(42)
  })

  it("number", () => {
    const store = one({ type: "number", name: "shift", label: "Shift", minimum: 0, maximum: 10, step: 0.1, precision: 2 }, { shift: 3 })
    const input = screen.getByLabelText("Shift")
    expect(input).toHaveValue("3")
    fireEvent.change(input, { target: { value: "4.25" } })
    fireEvent.blur(input)
    expect(core(store).shift).toBe(4.25)
  })

  it("text (single and multi line)", () => {
    const store = renderSchema(
      schemaOf(
        {
          type: "column",
          children: [
            { type: "text", name: "name", label: "Name", placeholder: "Your name" },
            { type: "text", name: "notes", label: "Notes", lines: 3, info: "Free text." },
          ],
        },
        { name: "", notes: "" },
      ),
    )
    fireEvent.change(screen.getByLabelText("Name"), { target: { value: "Remi" } })
    fireEvent.change(screen.getByLabelText("Notes"), { target: { value: "a\nb" } })
    expect(core(store)).toMatchObject({ name: "Remi", notes: "a\nb" })
    expect(screen.getByLabelText("Notes")).toHaveAccessibleDescription("Free text.")
  })

  it("text with prompt=true mounts the prompt editor", async () => {
    one({ type: "text", name: "prompt", label: "Prompt", prompt: true, lines: 2 }, { prompt: "a (cat:1.2)" })
    const editor = await screen.findByRole("textbox", { name: "Prompt" }, { timeout: 5000 })
    expect(editor.textContent).toBe("a (cat:1.2)")
    expect(editor.querySelector(".cm-prompt-weight")?.textContent).toBe(":1.2")
  })

  it("choice: dropdown, radio, searchable", () => {
    const short = [
      { value: "a", label: "Alpha" },
      { value: "b", label: "Beta" },
    ]
    const long = Array.from({ length: 20 }, (_, i) => ({ value: i, label: `Option ${i}` }))
    const store = renderSchema(
      schemaOf(
        {
          type: "column",
          children: [
            { type: "choice", name: "drop", label: "Dropdown", choices: short },
            { type: "choice", name: "radio", label: "Radio", choices: short, style: "radio" },
            { type: "choice", name: "search", label: "Searchable", choices: long },
          ],
        },
        { drop: "a", radio: "a", search: 3 },
      ),
    )
    expect(screen.getByRole("combobox", { name: "Dropdown" })).toHaveTextContent("Alpha")
    const radios = within(screen.getByRole("radiogroup", { name: "Radio" })).getAllByRole("radio")
    expect(radios).toHaveLength(2)
    fireEvent.click(radios[1]!)
    expect(core(store).radio).toBe("b")
    expect(screen.getByLabelText("Searchable")).toHaveTextContent("Option 3")
  })

  it("multichoice: checkboxes and chips", () => {
    const choices = [
      { value: "x", label: "Ex" },
      { value: "y", label: "Why" },
    ]
    const store = renderSchema(
      schemaOf(
        {
          type: "column",
          children: [
            { type: "multichoice", name: "checks", label: "Checks", choices, style: "checkboxes" },
            { type: "multichoice", name: "chips", label: "Chips", choices },
          ],
        },
        { checks: [], chips: ["y"] },
      ),
    )
    fireEvent.click(screen.getByRole("checkbox", { name: "Why" }))
    fireEvent.click(screen.getByRole("checkbox", { name: "Ex" }))
    expect(core(store).checks).toEqual(["x", "y"]) // declared order
    expect(screen.getByText("Why", { selector: "[data-slot=combobox-chip]" })).toBeInTheDocument()
  })

  it("seed: dice resets to -1", () => {
    const store = one({ type: "seed", name: "seed", label: "Seed" }, { seed: 1234 })
    expect(screen.getByLabelText("Seed")).toHaveValue("1234")
    fireEvent.click(screen.getByRole("button", { name: "Random seed" }))
    expect(core(store).seed).toBe(-1)
    expect(screen.getByRole("button", { name: "Reuse seed from the last image" })).toBeDisabled()
  })

  it("color", () => {
    const store = one({ type: "color", name: "tint", label: "Tint" }, { tint: "#ff0000" })
    const hex = screen.getByLabelText("Tint")
    expect(hex).toHaveValue("#ff0000")
    fireEvent.change(hex, { target: { value: "#00FF00" } })
    expect(core(store).tint).toBe("#00ff00")
    fireEvent.change(hex, { target: { value: "#00F" } })
    expect(core(store).tint).toBe("#00f")
    fireEvent.change(hex, { target: { value: "nope" } })
    expect(core(store).tint).toBe("#00f")
    expect(hex).toHaveAttribute("aria-invalid", "true")
  })

  it("image, image list and file render upload targets", () => {
    renderSchema(
      schemaOf(
        {
          type: "column",
          children: [
            { type: "image", name: "img", label: "Image", kind: "mask" },
            { type: "image_list", name: "imgs", label: "Images" },
            { type: "file", name: "f", label: "Model", accept: ".safetensors" },
          ],
        },
        { img: null, imgs: [], f: null },
      ),
    )
    // Drop zones are named by the field label and described by what they do.
    expect(screen.getByRole("button", { name: "Image" })).toHaveAccessibleDescription("Drop an image, paste, or click to browse")
    expect(screen.getByRole("button", { name: "Images" })).toHaveAccessibleDescription("Drop images, paste, or click to browse")
    expect(screen.getByRole("button", { name: "Model Choose a file (.safetensors)" })).toBeInTheDocument()
  })

  it("image upload posts to /uploads and stores the id", async () => {
    const fetchSpy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(new Response(JSON.stringify({ id: "up-1", width: 8, height: 8 }), { headers: { "Content-Type": "application/json" } }))
    const store = one({ type: "image", name: "img", label: "Image" }, { img: null })
    const zone = screen.getByRole("button", { name: "Image" })
    const file = new File([new Uint8Array([1])], "a.png", { type: "image/png" })
    fireEvent.drop(zone, { dataTransfer: { files: [file] } })
    await waitFor(() => expect(core(store).img).toBe("up-1"))
    const request = fetchSpy.mock.calls[0]![0] as Request
    expect(request.method).toBe("POST")
    expect(new URL(request.url).pathname).toBe("/api/v2/uploads")
    expect(await screen.findByRole("button", { name: "Remove Image" })).toBeInTheDocument()
    expect(zone).toHaveAccessibleDescription("Replace image")
  })

  it("json", () => {
    const store = one({ type: "json", name: "cfg", label: "Config" }, { cfg: { a: 1 } })
    const area = screen.getByLabelText("Config")
    fireEvent.change(area, { target: { value: '{"a": 2}' } })
    expect(core(store).cfg).toEqual({ a: 2 })
    fireEvent.change(area, { target: { value: "{bad" } })
    expect(screen.getByRole("alert")).toHaveTextContent("Invalid JSON")
    expect(core(store).cfg).toEqual({ a: 2 })
  })

  it("custom widgets load from the registry; unknown widgets fall back to JSON", async () => {
    renderSchema(
      schemaOf(
        {
          type: "column",
          children: [
            { type: "custom", name: "w", label: "Weights", widget: "block-weights", props: { blocks: ["IN", "OUT"] } },
            { type: "custom", name: "u", label: "Unknown", widget: "does-not-exist" },
          ],
        },
        { w: [1, 0.5], u: [1, 2] },
      ),
    )
    const group = await screen.findByRole("group", { name: "Weights" }, { timeout: 5000 })
    expect(group.querySelectorAll("input[type=range]")).toHaveLength(2)
    expect(screen.getByLabelText("Unknown")).toHaveValue("[\n  1,\n  2\n]")
  })

  it("state and hidden inputs are not rendered but keep their values", () => {
    const store = renderSchema(
      schemaOf(
        {
          type: "column",
          children: [
            { type: "state", name: "s", label: "State" },
            { type: "text", name: "h", label: "Hidden", hidden: true },
          ],
        },
        { s: { v: 1 }, h: "x" },
      ),
    )
    expect(screen.queryByText("State")).toBeNull()
    expect(screen.queryByText("Hidden")).toBeNull()
    expect(core(store)).toMatchObject({ s: { v: 1 }, h: "x" })
  })
})

describe("layout and display nodes", () => {
  it("section with enable toggle, open state and section reset", () => {
    const store = one(
      {
        type: "section",
        title: "Hires fix",
        enable: { type: "toggle", name: "enable_hr", label: "Hires fix" },
        children: [{ type: "slider", name: "scale", label: "Scale", minimum: 1, maximum: 4, step: 0.5 }],
      },
      { enable_hr: false, scale: 2 },
    )
    const trigger = screen.getByRole("button", { name: "Hires fix" })
    expect(trigger).toHaveAttribute("aria-expanded", "false")
    fireEvent.click(screen.getByRole("switch", { name: "Enable Hires fix" }))
    expect(core(store).enable_hr).toBe(true)
    // Enabling opens the section.
    expect(trigger).toHaveAttribute("aria-expanded", "true")
    fireEvent.click(screen.getByRole("button", { name: "Reset Hires fix to default" }))
    expect(core(store).enable_hr).toBe(false)
    fireEvent.click(trigger)
    expect(trigger).toHaveAttribute("aria-expanded", "false")
  })

  it("group, row, column, separator, markdown, html", () => {
    renderSchema(
      schemaOf({
        type: "column",
        children: [
          { type: "group", title: "Advanced", children: [{ type: "markdown", text: "Some **bold** help" }] },
          { type: "group", children: [{ type: "html", text: "<em>trusted</em><script>bad()</script>" }] },
          { type: "row", children: [{ type: "column", width: 6, children: [{ type: "separator" }] }] },
        ],
      }),
    )
    expect(screen.getByRole("button", { name: "Advanced" })).toHaveAttribute("aria-expanded", "true")
    expect(screen.getByText("bold").tagName).toBe("STRONG")
    expect(screen.getByText("trusted").tagName).toBe("EM")
    expect(document.querySelector("script")).toBeNull()
    expect(screen.getByRole("separator")).toBeInTheDocument()
  })

  it("visible_if and enabled_if react to values", () => {
    const store = renderSchema(
      schemaOf(
        {
          type: "column",
          children: [
            { type: "toggle", name: "advanced", label: "Advanced" },
            { type: "text", name: "extra", label: "Extra", visible_if: { param: "advanced" } },
            { type: "group", enabled_if: { param: "advanced" }, children: [{ type: "text", name: "locked", label: "Locked" }] },
          ],
        },
        { advanced: false, extra: "", locked: "" },
      ),
    )
    expect(screen.queryByLabelText("Extra")).toBeNull()
    expect(screen.getByLabelText("Locked")).toBeDisabled()
    act(() => store.getState().setValue("core", "advanced", true))
    expect(screen.getByLabelText("Extra")).toBeInTheDocument()
    expect(screen.getByLabelText("Locked")).toBeEnabled()
  })

  it("tabs: unbound and bound to a choice", () => {
    const store = renderSchema(
      schemaOf(
        {
          type: "column",
          children: [
            { type: "choice", name: "mode", label: "Mode", choices: [{ value: "a", label: "A" }, { value: "b", label: "B" }], hidden: true },
            {
              type: "tabs",
              bind: "mode",
              children: [
                { type: "tab", title: "Tab A", value: "a", children: [{ type: "markdown", text: "content A" }] },
                { type: "tab", title: "Tab B", value: "b", children: [{ type: "markdown", text: "content B" }] },
              ],
            },
          ],
        },
        { mode: "a" },
      ),
    )
    expect(screen.getByText("content A")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("tab", { name: "Tab B" }))
    expect(core(store).mode).toBe("b")
    expect(screen.getByText("content B")).toBeInTheDocument()
    act(() => store.getState().setValue("core", "mode", "a"))
    expect(screen.getByRole("tab", { name: "Tab A" })).toHaveAttribute("aria-selected", "true")
  })

  it("tabs honour their own visible_if and enabled_if", () => {
    const store = renderSchema(
      schemaOf(
        {
          type: "column",
          children: [
            { type: "toggle", name: "advanced", label: "Advanced" },
            {
              type: "tabs",
              children: [
                { type: "tab", title: "Expert", visible_if: { param: "advanced" }, children: [{ type: "markdown", text: "expert content" }] },
                { type: "tab", title: "Basic", children: [{ type: "markdown", text: "basic content" }] },
                { type: "tab", title: "Locked", enabled_if: { param: "advanced" }, children: [{ type: "text", name: "note", label: "Note" }] },
              ],
            },
          ],
        },
        { advanced: false, note: "" },
      ),
    )
    // The hidden first tab has neither trigger nor content; the strip falls back to the first visible tab.
    expect(screen.queryByRole("tab", { name: "Expert" })).toBeNull()
    expect(screen.queryByText("expert content")).toBeNull()
    expect(screen.getByText("basic content")).toBeInTheDocument()
    expect(screen.getByRole("tab", { name: "Locked" })).toHaveAttribute("aria-disabled", "true")
    act(() => store.getState().setTab("core::tabs::Expert|Basic|Locked", "2"))
    expect(screen.getByLabelText("Note")).toBeDisabled()
    act(() => store.getState().setValue("core", "advanced", true))
    expect(screen.getByLabelText("Note")).toBeEnabled()
    expect(screen.getByRole("tab", { name: "Expert" })).toBeInTheDocument()
  })

  it("validation errors mark the field and describe the control", () => {
    const store = one({ type: "text", name: "name", label: "Name", info: "Your name." }, { name: "" })
    act(() => store.getState().setErrors([{ owner: "core", param: "name", message: "required" }]))
    const input = screen.getByLabelText("Name")
    expect(input).toHaveAttribute("aria-invalid", "true")
    expect(input).toHaveAccessibleDescription("Your name. required")
    fireEvent.change(input, { target: { value: "Remi" } })
    expect(input).not.toHaveAttribute("aria-invalid")
    expect(screen.queryByText("required")).toBeNull()
  })

  it("patches can clear info and label with null", () => {
    const store = one({ type: "text", name: "w", label: "Weight", info: "Static help." }, { w: "" })
    act(() => store.getState().applyPatches([["core", "w", { info: "Dynamic help." }]]))
    expect(screen.getByText("Dynamic help.")).toBeInTheDocument()
    act(() => store.getState().applyPatches([["core", "w", { info: null }]]))
    expect(screen.queryByText("Static help.")).toBeNull()
    expect(screen.queryByText("Dynamic help.")).toBeNull()
  })

  it("color picker has an accessible name", () => {
    one({ type: "color", name: "tint", label: "Tint" }, { tint: "#ff0000" })
    expect(screen.getByLabelText("Pick a color for Tint")).toHaveAttribute("type", "color")
  })

  it("button calls the owner's action and applies the patch", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({
          width: { value: 768 },
          height: { visible: false },
          __outputs__: { result: "swapped", pics: ["mock-1-120"] },
        }),
        { headers: { "Content-Type": "application/json" } },
      ),
    )
    const store = renderSchema(
      schemaOf(
        {
          type: "column",
          children: [
            { type: "number", name: "width", label: "Width" },
            { type: "number", name: "height", label: "Height" },
            { type: "button", label: "Swap", action: "swap_size", variant: "tool" },
            { type: "output", name: "result", kind: "text", label: "Result" },
            { type: "gallery", name: "pics", label: "Pictures" },
          ],
        },
        { width: 512, height: 768 },
      ),
    )
    expect(screen.getByText("No images yet")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Swap" }))
    await waitFor(() => expect(core(store).width).toBe(768))
    const request = fetchSpy.mock.calls[0]![0] as Request
    expect(new URL(request.url).pathname).toBe("/api/v2/actions/core/swap_size")
    expect(await request.json()).toEqual({ tab: "txt2img", values: { width: 512, height: 768 } })
    expect(screen.queryByLabelText("Height")).toBeNull()
    expect(screen.getByText("swapped")).toBeInTheDocument()
    expect(screen.getByRole("list", { name: "Pictures" }).querySelector("img")?.getAttribute("src")).toBe("/api/v2/files/mock-1-120")
  })

  it("slots place scripts; selectable scripts follow core.script", () => {
    const store = renderSchema(tabSchema("txt2img"))
    // Script sections render in their slots; untitled layouts get a titled frame.
    expect(screen.getByRole("button", { name: "Anima" })).toBeInTheDocument()
    expect(within(screen.getByRole("group", { name: "Sampler" })).getByRole("combobox", { name: "Sampling method" })).toBeInTheDocument()
    expect(within(screen.getByRole("group", { name: "Seed" })).getByLabelText("Seed")).toBeInTheDocument()
    expect(document.querySelector("[data-slot-section=accordions] [data-script=refiner]")).not.toBeNull()
    // Unknown slot falls back to the default "scripts" slot.
    const misc = screen.getByRole("group", { name: "Misc extension" })
    expect(misc.closest("[data-slot-section]")).toHaveAttribute("data-slot-section", "scripts")
    expect(within(misc).getByText("Flag")).toBeInTheDocument()
    expect(screen.queryByText("X values")).toBeNull()
    act(() => store.getState().setValue("core", "script", "X/Y/Z plot"))
    expect(screen.getByLabelText("X values")).toBeInTheDocument()
    // Cross-owner condition: the ControlNet note needs anima.enable and its own enable.
    expect(screen.queryByText(/control weights are rescaled/)).toBeNull()
  })
})
