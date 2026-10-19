/** The completion source on a real CodeMirror view (API mocked). */
import { completionStatus, currentCompletions, startCompletion } from "@codemirror/autocomplete"
import { EditorState } from "@codemirror/state"
import { EditorView, runScopeHandlers } from "@codemirror/view"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { queryClient } from "@/api/queries"
import { registerCompletionSource } from "@/features/prompt-editor/completion-registry"
import { completionExtension, completionSlot } from "@/features/prompt-editor/extensions"
import { promptKind, type PromptScope, promptScope } from "@/features/prompt-editor/facets"
import { type CompletionItem, type PromptToolsConfig, promptToolsApi } from "../api"
import { toolKeys } from "../queries"
import { TEST_CONFIG } from "../testing"
import { clearCompletionCache, matchRanges, promptToolsCompletions } from "./source"

const CONFIG: PromptToolsConfig = { ...TEST_CONFIG, tags: { ...TEST_CONFIG.tags, negative: false } }

const TAGS: CompletionItem[] = [
  { type: "tag", text: "long_hair", category: 0, count: 4_300_000, translation: "长发" },
  { type: "tag", text: "very_long_hair", category: 0, count: 800_000 },
]

const views: EditorView[] = []
let off: () => void = () => {}

function editor(doc: string, kind: "positive" | "negative" = "positive", scope: PromptScope = { form: "txt2img", extension: false }) {
  const view = new EditorView({
    parent: document.body,
    state: EditorState.create({ doc, selection: { anchor: doc.length }, extensions: [promptKind.of(kind), promptScope.of(scope), completionSlot.of(completionExtension())] }),
  })
  views.push(view)
  return view
}

const pause = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))
/** Waits for the completion to open (past CodeMirror's 75 ms accept delay). */
async function opened(view: EditorView) {
  await vi.waitFor(() => expect(completionStatus(view.state)).toBe("active"), { timeout: 3000 })
  await pause(120)
}
const settle = () => pause(300)
const press = (view: EditorView, key: string) => runScopeHandlers(view, new KeyboardEvent("keydown", { key }), "editor")

describe("tag completion source", () => {
  beforeEach(() => {
    queryClient.setQueryData(toolKeys.config, structuredClone(CONFIG))
    clearCompletionCache()
    off = registerCompletionSource("prompt-tools-test", promptToolsCompletions)
  })
  afterEach(() => {
    off()
    views.splice(0).forEach((view) => view.destroy())
    queryClient.clear()
    vi.restoreAllMocks()
  })

  it("asks the server for the word being typed and keeps its order", async () => {
    const complete = vi.spyOn(promptToolsApi, "complete").mockResolvedValue(TAGS)
    const view = editor("1girl, long ha")
    startCompletion(view)
    await opened(view)
    expect(complete).toHaveBeenCalledWith(expect.objectContaining({ mode: "tag", q: "long ha", negative: false, limit: 8 }), expect.anything())
    expect(completionStatus(view.state)).toBe("active")
    expect(currentCompletions(view.state).map((c) => c.label)).toEqual(["long hair", "very long hair"])
    // translation and post count are rendered next to the label
    expect(view.dom.querySelector(".cm-tooltip-autocomplete")?.textContent).toContain("长发")
  })

  it("Tab accepts with the insertion rules and records the use", async () => {
    vi.spyOn(promptToolsApi, "complete").mockResolvedValue(TAGS)
    const record = vi.spyOn(promptToolsApi, "recordUse").mockResolvedValue({ ok: true, message: null })
    const view = editor("1girl, long ha")
    startCompletion(view)
    await opened(view)
    expect(press(view, "Tab")).toBe(true)
    expect(view.state.doc.toString()).toBe("1girl, long hair, ")
    expect(view.state.selection.main.head).toBe(view.state.doc.length)
    expect(record).toHaveBeenCalledWith({ name: "long_hair", type: "tag", negative: false })
  })

  it("serves repeated queries from the cache", async () => {
    const complete = vi.spyOn(promptToolsApi, "complete").mockResolvedValue(TAGS)
    const first = editor("long ha")
    startCompletion(first)
    await opened(first)
    const second = editor("long ha")
    startCompletion(second)
    await opened(second)
    expect(complete).toHaveBeenCalledTimes(1)
  })

  it("stays quiet in negative prompts when told to, and when tags are off", async () => {
    const complete = vi.spyOn(promptToolsApi, "complete").mockResolvedValue(TAGS)
    const negative = editor("long ha", "negative")
    startCompletion(negative)
    await settle()
    expect(completionStatus(negative.state)).toBeNull()
    queryClient.setQueryData(toolKeys.config, { ...CONFIG, tags: { ...CONFIG.tags, enabled: false } })
    const off = editor("long ha")
    startCompletion(off)
    await settle()
    expect(complete).not.toHaveBeenCalled()
  })

  it("follows the per-form and extension-field switches (tac_activeIn_*)", async () => {
    const complete = vi.spyOn(promptToolsApi, "complete").mockResolvedValue(TAGS)
    queryClient.setQueryData(toolKeys.config, { ...CONFIG, tags: { ...CONFIG.tags, txt2img: false, third_party: false } })
    for (const scope of [
      { form: "txt2img", extension: false },
      { form: "txt2img", extension: true },
      { form: "img2img", extension: true },
      { form: "page:prompt-workshop", extension: true },
    ]) {
      const view = editor("long ha", "positive", scope)
      startCompletion(view)
      await settle()
      expect(completionStatus(view.state), JSON.stringify(scope)).toBeNull()
    }
    expect(complete).not.toHaveBeenCalled()
    // img2img's main prompts and the style editor (no form) still complete
    for (const scope of [
      { form: "img2img", extension: false },
      { form: null, extension: false },
    ]) {
      const view = editor("long ha", "positive", scope)
      startCompletion(view)
      await opened(view)
    }
    expect(complete).toHaveBeenCalledTimes(1) // the second answer came from the cache
  })

  it("labels a tag found by an alias with the alias alone when asked (tac_alias_onlyShowAlias)", async () => {
    vi.spyOn(promptToolsApi, "complete").mockResolvedValue([{ type: "tag", text: "breasts", category: 0, count: 3_400_000, alias: "oppai" }])
    vi.spyOn(promptToolsApi, "recordUse").mockResolvedValue({ ok: true, message: null })
    const view = editor("opp")
    startCompletion(view)
    await opened(view)
    expect(currentCompletions(view.state).map((c) => [c.label, c.detail])).toEqual([["breasts", "← oppai"]])
    view.destroy()
    clearCompletionCache()
    queryClient.setQueryData(toolKeys.config, { ...CONFIG, tags: { ...CONFIG.tags, only_alias: true } })
    const aliasOnly = editor("opp")
    startCompletion(aliasOnly)
    await opened(aliasOnly)
    expect(currentCompletions(aliasOnly.state).map((c) => [c.label, c.detail])).toEqual([["oppai", undefined]])
    press(aliasOnly, "Enter")
    expect(aliasOnly.state.doc.toString()).toBe("breasts, ") // the tag is inserted
  })

  it("waits tac_delayTime before asking the server", async () => {
    const complete = vi.spyOn(promptToolsApi, "complete").mockResolvedValue(TAGS)
    queryClient.setQueryData(toolKeys.config, { ...CONFIG, tags: { ...CONFIG.tags, delay: 400 } })
    const view = editor("long ha")
    startCompletion(view)
    await pause(200)
    expect(complete).not.toHaveBeenCalled()
    await opened(view)
    expect(complete).toHaveBeenCalledTimes(1)
  })

  it("highlights the typed text in labels", () => {
    expect(matchRanges("long hair", "long_ha")).toEqual([0, 7])
    expect(matchRanges("very long hair", "*hair")).toEqual([10, 14])
    expect(matchRanges("breasts", "oppai")).toEqual([])
  })

  it("routes `<lora:` and inserts the LoRA tag", async () => {
    const complete = vi.spyOn(promptToolsApi, "complete").mockResolvedValue([{ type: "lora", text: "styleAlias", label: "styles/style", weight: 0.8 }])
    vi.spyOn(promptToolsApi, "recordUse").mockResolvedValue({ ok: true, message: null })
    const view = editor("1girl, <lora:sty")
    startCompletion(view)
    await opened(view)
    expect(complete).toHaveBeenCalledWith(expect.objectContaining({ mode: "lora", q: "sty" }), expect.anything())
    expect(currentCompletions(view.state)[0]?.label).toBe("styles/style")
    press(view, "Enter")
    expect(view.state.doc.toString()).toBe("1girl, <lora:styleAlias:0.8>")
  })
})
