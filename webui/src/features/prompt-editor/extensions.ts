/**
 * CodeMirror extensions of the prompt editor, usable without React (tests
 * build an `EditorView` from them): key commands, highlighting, bracket lint
 * and registered completion sources.
 */
import { autocompletion } from "@codemirror/autocomplete"
import { defaultKeymap, history, historyKeymap } from "@codemirror/commands"
import { linter, type Diagnostic } from "@codemirror/lint"
import { Compartment, type Extension, Prec, RangeSet } from "@codemirror/state"
import { Decoration, type DecorationSet, EditorView, keymap, ViewPlugin, type ViewUpdate } from "@codemirror/view"
import { t } from "@/i18n"
import { type Edit, editAttention } from "@/ui/prompt/attention"
import { minimalChange, type PromptKeyOptions } from "@/ui/prompt/editing"
import { moveTags } from "@/ui/prompt/reorder"
import { completionSources } from "./completion-registry"
import { bracketProblems, type PromptTokenKind, tokenizePrompt } from "./tokenize"

function applyEdit(view: EditorView, edit: Edit | null, userEvent: string): boolean {
  if (!edit) return false
  view.dispatch({
    changes: minimalChange(view.state.doc.toString(), edit.text),
    selection: { anchor: edit.from, head: edit.to },
    userEvent,
    scrollIntoView: true,
  })
  return true
}

/** Ctrl/Cmd+↑/↓ attention weight, Alt+←/→ tag order. Options are read per key press. */
export function promptCommands(options: () => PromptKeyOptions): Extension {
  const attention = (view: EditorView, direction: 1 | -1) => {
    const { from, to } = view.state.selection.main
    // Always consumed, as in A1111: no cursor jump when there is nothing to weight.
    applyEdit(view, editAttention(view.state.doc.toString(), from, to, direction, options().attention), "input.attention")
    return true
  }
  const move = (view: EditorView, direction: 1 | -1) => {
    if (!options().move) return false
    const { from, to } = view.state.selection.main
    return applyEdit(view, moveTags(view.state.doc.toString(), from, to, direction), "move.tags")
  }
  return [
    history({ newGroupDelay: 600 }),
    Prec.high(
      keymap.of([
        { key: "Mod-ArrowUp", run: (v) => attention(v, 1) },
        { key: "Mod-ArrowDown", run: (v) => attention(v, -1) },
        { key: "Alt-ArrowLeft", run: (v) => move(v, -1) },
        { key: "Alt-ArrowRight", run: (v) => move(v, 1) },
      ]),
    ),
    // Mod-Enter belongs to the page (Generate), not insertBlankLine.
    keymap.of([...historyKeymap, ...defaultKeymap.filter((binding) => binding.key !== "Mod-Enter")]),
  ]
}

const CLASS: Record<PromptTokenKind, string> = {
  bracket: "cm-prompt-bracket",
  "bracket-error": "cm-prompt-bracket-error",
  weight: "cm-prompt-weight",
  schedule: "cm-prompt-schedule",
  extra: "cm-prompt-extra",
  "extra-weight": "cm-prompt-weight",
  keyword: "cm-prompt-keyword",
  wildcard: "cm-prompt-wildcard",
  comment: "cm-prompt-comment",
  escape: "cm-prompt-escape",
}

const marks = new Map<string, Decoration>()
function mark(className: string): Decoration {
  let decoration = marks.get(className)
  if (!decoration) marks.set(className, (decoration = Decoration.mark({ class: className })))
  return decoration
}

function decorate(text: string): DecorationSet {
  const ranges = tokenizePrompt(text)
    .filter((token) => token.to > token.from)
    .map((token) => {
      const base = CLASS[token.kind]
      const className = token.kind === "bracket" ? `${base} ${base}-${(token.depth ?? 0) % 3}` : base
      return mark(className).range(token.from, token.to)
    })
  return RangeSet.of(ranges, true)
}

/** Syntax colouring from `tokenizePrompt` (the whole prompt: brackets need context). */
export const promptHighlight = ViewPlugin.fromClass(
  class {
    decorations: DecorationSet
    constructor(view: EditorView) {
      this.decorations = decorate(view.state.doc.toString())
    }
    update(update: ViewUpdate) {
      if (update.docChanged) this.decorations = decorate(update.state.doc.toString())
    }
  },
  { decorations: (plugin) => plugin.decorations },
)

/** Bracket-balance diagnostics as warnings. */
export function bracketDiagnostics(text: string): Diagnostic[] {
  return bracketProblems(text).map((problem) => ({
    from: problem.from,
    to: problem.to,
    severity: "warning",
    source: "brackets",
    message:
      problem.kind === "unclosed"
        ? t("“{{char}}” is never closed", { char: problem.char })
        : t("Unmatched “{{char}}”", { char: problem.char }),
  }))
}

export const bracketLint: Extension = linter((view) => bracketDiagnostics(view.state.doc.toString()), { delay: 250 })

/** Autocompletion over the registered sources; nothing at all while none is registered. */
export function completionExtension(): Extension {
  const sources = completionSources()
  if (sources.length === 0) return []
  return autocompletion({ override: sources.map((entry) => entry.source), icons: false })
}

/** A compartment holding `completionExtension()`; reconfigure it when the registry changes. */
export const completionSlot = new Compartment()
