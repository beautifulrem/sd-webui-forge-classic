/** Prompt-editor commands on a real CodeMirror view (no React). */
import { completionStatus, currentCompletions, startCompletion } from "@codemirror/autocomplete"
import { forceLinting } from "@codemirror/lint"
import { EditorState } from "@codemirror/state"
import { EditorView, runScopeHandlers } from "@codemirror/view"
import { afterEach, describe, expect, it } from "vitest"
import { DEFAULT_ATTENTION_OPTIONS } from "@/ui/prompt/attention"
import type { PromptKeyOptions } from "@/ui/prompt/editing"
import { registerCompletionSource } from "./completion-registry"
import { bracketDiagnostics, bracketLint, completionExtension, completionSlot, promptCommands } from "./extensions"

const views: EditorView[] = []
afterEach(() => views.splice(0).forEach((view) => view.destroy()))

function editor(doc: string, cursor: number | [number, number], options: Partial<PromptKeyOptions> = {}) {
  const keyOptions: PromptKeyOptions = { attention: DEFAULT_ATTENTION_OPTIONS, move: true, ...options }
  const view = new EditorView({
    parent: document.body,
    state: EditorState.create({
      doc,
      selection: Array.isArray(cursor) ? { anchor: cursor[0], head: cursor[1] } : { anchor: cursor },
      extensions: [promptCommands(() => keyOptions), completionSlot.of(completionExtension())],
    }),
  })
  views.push(view)
  const press = (key: string, mods: { ctrl?: boolean; alt?: boolean; shift?: boolean } = {}) =>
    runScopeHandlers(view, new KeyboardEvent("keydown", { key, ctrlKey: mods.ctrl, altKey: mods.alt, shiftKey: mods.shift }), "editor")
  return { view, press, doc: () => view.state.doc.toString(), selection: () => view.state.selection.main }
}

describe("attention weight (Ctrl/Cmd+↑/↓)", () => {
  it("wraps the word under the cursor and steps by the attention precision", () => {
    const e = editor("a cat, blue sky", 13)
    e.press("ArrowUp", { ctrl: true })
    expect(e.doc()).toBe("a cat, blue (sky:1.1)")
    e.press("ArrowUp", { ctrl: true })
    expect(e.doc()).toBe("a cat, blue (sky:1.2)")
    // The selection stays on the weighted text.
    expect(e.doc().slice(e.selection().from, e.selection().to)).toBe("sky")
  })

  it("uses the configured precision and unwraps at 1.0", () => {
    const e = editor("(sky:1.25)", 2, { attention: { ...DEFAULT_ATTENTION_OPTIONS, precisionAttention: 0.25 } })
    e.press("ArrowDown", { ctrl: true })
    expect(e.doc()).toBe("sky")
  })

  it("steps extra networks by the extra precision", () => {
    const e = editor("<lora:style:0.8>", 3, { attention: { ...DEFAULT_ATTENTION_OPTIONS, precisionExtra: 0.05 } })
    e.press("ArrowUp", { ctrl: true })
    expect(e.doc()).toBe("<lora:style:0.85>")
  })

  it("converts legacy emphasis to an explicit weight", () => {
    const e = editor("((cat))", 3)
    e.press("ArrowUp", { ctrl: true })
    expect(e.doc()).toBe("(cat:1.3)")
  })

  it("is undoable in one step", () => {
    const e = editor("sky", 1)
    e.press("ArrowUp", { ctrl: true })
    e.press("z", { ctrl: true })
    expect(e.doc()).toBe("sky")
  })
})

describe("tag order (Alt+←/→)", () => {
  it("moves the tag under the cursor", () => {
    const e = editor("a, b, c", 4)
    e.press("ArrowRight", { alt: true })
    expect(e.doc()).toBe("a, c, b")
    e.press("ArrowLeft", { alt: true })
    expect(e.doc()).toBe("a, b, c")
  })

  it("moves a selection of tags together", () => {
    const e = editor("a, b, c, d", [3, 7])
    e.press("ArrowLeft", { alt: true })
    expect(e.doc()).toBe(" b, c,a, d")
  })

  it("is off when keyedit_move is off", () => {
    const e = editor("a, b, c", 4, { move: false })
    e.press("ArrowRight", { alt: true })
    expect(e.doc()).toBe("a, b, c")
  })
})

describe("bracket lint", () => {
  it("turns unbalanced brackets into warnings", () => {
    expect(bracketDiagnostics("(a, [b)").map((d) => [d.from, d.severity, d.message])).toEqual([
      [0, "warning", "“(” is never closed"],
      [4, "warning", "“[” is never closed"],
      [6, "warning", "Unmatched “)”"],
    ])
  })

  it("runs as a CodeMirror linter", async () => {
    const view = new EditorView({ parent: document.body, state: EditorState.create({ doc: "cat)", extensions: [bracketLint] }) })
    views.push(view)
    forceLinting(view)
    await new Promise((resolve) => setTimeout(resolve, 0))
    expect(view.dom.querySelector(".cm-lintRange-warning")?.textContent).toBe(")")
  })
})

describe("registerCompletionSource", () => {
  it("reaches open editors and stops when unregistered", async () => {
    const e = editor("ca", 2)
    const off = registerCompletionSource("test-tags", (ctx) => {
      const word = ctx.matchBefore(/\w+/)
      return word ? { from: word.from, options: [{ label: "cat" }, { label: "car" }] } : null
    })
    e.view.dispatch({ effects: completionSlot.reconfigure(completionExtension()) })
    startCompletion(e.view)
    await new Promise((resolve) => setTimeout(resolve, 80))
    expect(completionStatus(e.view.state)).toBe("active")
    expect(currentCompletions(e.view.state).map((c) => c.label)).toEqual(["car", "cat"])
    off()
    // Nothing registered: no autocompletion extension at all.
    expect(completionExtension()).toEqual([])
  })
})
