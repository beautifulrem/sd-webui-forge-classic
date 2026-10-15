/**
 * Prompt editor (CodeMirror 6; its own lazy chunk). Wave 1 scope: weight and
 * extra-network highlighting, Ctrl/Cmd+↑/↓ attention editing, Alt+←/→ tag
 * reordering, undo/redo history. Tag completion, LoRA insertion and token
 * counting build on this component later.
 */
import { history, historyKeymap, defaultKeymap } from "@codemirror/commands"
import { Annotation, Compartment, EditorState, Prec, type Extension } from "@codemirror/state"
import {
  Decoration,
  type DecorationSet,
  EditorView,
  keymap,
  MatchDecorator,
  placeholder as placeholderExt,
  ViewPlugin,
  type ViewUpdate,
} from "@codemirror/view"
import { useQuery } from "@tanstack/react-query"
import { useEffect, useRef } from "react"
import { bootstrapQuery } from "@/api/queries"
import { cn } from "@/lib/utils"
import { type Edit, editAttention } from "./attention"
import { keyOptionsFrom, minimalChange, type PromptKeyOptions } from "./editing"
import { moveTags } from "./reorder"

export type PromptEditorProps = {
  id?: string
  value: string
  onChange: (value: string) => void
  placeholder?: string
  disabled?: boolean
  minLines?: number
  maxLines?: number
  ariaLabel?: string
  ariaDescribedBy?: string
  ariaInvalid?: boolean
  className?: string
}

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

/** Highlights `<lora:…>` blocks, `:1.2` weights, brackets and BREAK/AND. */
const promptMatcher = new MatchDecorator({
  regexp: /<[^<>\n]+>|:\s*-?\d*\.?\d+(?=\s*[)\]>])|\b(?:BREAK|AND)\b|[()[\]{}]/g,
  decoration: (match) => {
    const text = match[0]
    if (text.startsWith("<")) return Decoration.mark({ class: "cm-prompt-extra" })
    if (text.startsWith(":")) return Decoration.mark({ class: "cm-prompt-weight" })
    if (text === "BREAK" || text === "AND") return Decoration.mark({ class: "cm-prompt-keyword" })
    return Decoration.mark({ class: "cm-prompt-bracket" })
  },
})

const promptHighlight = ViewPlugin.fromClass(
  class {
    decorations: DecorationSet
    constructor(view: EditorView) {
      this.decorations = promptMatcher.createDeco(view)
    }
    update(update: ViewUpdate) {
      this.decorations = promptMatcher.updateDeco(update, this.decorations)
    }
  },
  { decorations: (plugin) => plugin.decorations },
)

const theme = EditorView.theme({
  "&": {
    fontSize: "0.875rem",
    borderRadius: "var(--radius-lg)",
    border: "1px solid var(--input)",
    backgroundColor: "color-mix(in oklch, var(--input) 30%, transparent)",
    transition: "border-color 120ms, box-shadow 120ms",
  },
  "&.cm-focused": {
    outline: "none",
    borderColor: "var(--ring)",
    boxShadow: "0 0 0 3px color-mix(in oklch, var(--ring) 50%, transparent)",
  },
  ".cm-scroller": { fontFamily: "var(--font-sans)", lineHeight: "1.25rem", overflow: "auto" },
  ".cm-content": { padding: "0.4rem 0", caretColor: "var(--foreground)" },
  ".cm-line": { padding: "0 0.625rem" },
  ".cm-placeholder": { color: "var(--muted-foreground)" },
  ".cm-cursor": { borderLeftColor: "var(--foreground)" },
  "&.cm-focused .cm-selectionBackground, .cm-selectionBackground, ::selection": {
    backgroundColor: "color-mix(in oklch, var(--primary) 28%, transparent) !important",
  },
  ".cm-prompt-weight": { color: "var(--primary)", fontVariantNumeric: "tabular-nums" },
  ".cm-prompt-bracket": { color: "var(--muted-foreground)" },
  ".cm-prompt-extra": { color: "var(--chart-2)", fontWeight: "500" },
  ".cm-prompt-keyword": { color: "var(--chart-3)", fontWeight: "600" },
})

export default function PromptEditor(props: PromptEditorProps) {
  const host = useRef<HTMLDivElement>(null)
  const view = useRef<EditorView | null>(null)
  const onChange = useRef(props.onChange)
  const bootstrap = useQuery(bootstrapQuery)
  const keyOptions = useRef<PromptKeyOptions>(keyOptionsFrom(undefined))
  const editable = useRef(new Compartment())
  const placeholderSlot = useRef(new Compartment())
  const aria = useRef(new Compartment())

  useEffect(() => {
    onChange.current = props.onChange
    keyOptions.current = keyOptionsFrom(bootstrap.data?.options)
  })

  // Create the view once; later prop changes are dispatched into it.
  useEffect(() => {
    if (!host.current) return
    const extensions: Extension[] = [
      history({ newGroupDelay: 600 }),
      Prec.high(
        keymap.of([
          // Attention keys are always consumed, as in A1111 (no cursor jump on a no-op).
          { key: "Mod-ArrowUp", run: (v) => applyEdit(v, attentionEdit(v, 1), "input.attention") || true },
          { key: "Mod-ArrowDown", run: (v) => applyEdit(v, attentionEdit(v, -1), "input.attention") || true },
          { key: "Alt-ArrowLeft", run: (v) => keyOptions.current.move && applyEdit(v, tagMove(v, -1), "move.tags") },
          { key: "Alt-ArrowRight", run: (v) => keyOptions.current.move && applyEdit(v, tagMove(v, 1), "move.tags") },
        ]),
      ),
      // Mod-Enter belongs to the page (Generate), not insertBlankLine.
      keymap.of([...historyKeymap, ...defaultKeymap.filter((binding) => binding.key !== "Mod-Enter")]),
      EditorView.lineWrapping,
      promptHighlight,
      theme,
      editable.current.of([EditorView.editable.of(!props.disabled), EditorState.readOnly.of(Boolean(props.disabled))]),
      placeholderSlot.current.of(props.placeholder ? placeholderExt(props.placeholder) : []),
      aria.current.of(contentAttributes(props.id, props.ariaLabel, props.ariaDescribedBy, props.ariaInvalid)),
      EditorView.updateListener.of((update) => {
        if (update.docChanged && !update.transactions.some((tr) => tr.annotation(externalSync))) {
          onChange.current(update.state.doc.toString())
        }
      }),
    ]
    const editor = new EditorView({
      parent: host.current,
      state: EditorState.create({ doc: props.value, extensions }),
    })
    view.current = editor

    function attentionEdit(v: EditorView, direction: 1 | -1) {
      const { from, to } = v.state.selection.main
      return editAttention(v.state.doc.toString(), from, to, direction, keyOptions.current.attention)
    }
    function tagMove(v: EditorView, direction: 1 | -1) {
      const { from, to } = v.state.selection.main
      return moveTags(v.state.doc.toString(), from, to, direction)
    }

    return () => {
      editor.destroy()
      view.current = null
    }
    // The view is created once; the effects below keep it in sync.
    // oxlint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // External value changes (reset, paste, action patch) replace the document.
  useEffect(() => {
    const editor = view.current
    if (!editor) return
    const current = editor.state.doc.toString()
    if (current === props.value) return
    editor.dispatch({ changes: minimalChange(current, props.value), annotations: externalSync.of(true) })
  }, [props.value])

  useEffect(() => {
    view.current?.dispatch({
      effects: [
        editable.current.reconfigure([EditorView.editable.of(!props.disabled), EditorState.readOnly.of(Boolean(props.disabled))]),
        placeholderSlot.current.reconfigure(props.placeholder ? placeholderExt(props.placeholder) : []),
        aria.current.reconfigure(contentAttributes(props.id, props.ariaLabel, props.ariaDescribedBy, props.ariaInvalid)),
      ],
    })
  }, [props.disabled, props.placeholder, props.ariaLabel, props.ariaDescribedBy, props.ariaInvalid, props.id])

  const minLines = Math.max(props.minLines ?? 2, 1)
  return (
    <div
      ref={host}
      data-slot="prompt-editor"
      data-invalid={props.ariaInvalid || undefined}
      className={cn(
        "min-w-0 [&_.cm-editor]:min-h-(--prompt-min) [&_.cm-scroller]:max-h-(--prompt-max)",
        "data-invalid:[&_.cm-editor]:border-destructive data-invalid:[&_.cm-editor]:ring-3 data-invalid:[&_.cm-editor]:ring-destructive/20",
        props.disabled && "opacity-60",
        props.className,
      )}
      style={
        {
          "--prompt-min": `${minLines * 1.25 + 0.9}rem`,
          "--prompt-max": props.maxLines ? `${props.maxLines * 1.25 + 0.9}rem` : "24rem",
        } as React.CSSProperties
      }
    />
  )
}

/** Marks transactions that mirror the React value, so they are not echoed back. */
const externalSync = Annotation.define<boolean>()

function contentAttributes(id?: string, label?: string, describedBy?: string, invalid?: boolean): Extension {
  const attrs: Record<string, string> = { spellcheck: "false", autocorrect: "off", autocapitalize: "off", "aria-multiline": "true" }
  if (id) attrs.id = id
  if (label) attrs["aria-label"] = label
  if (describedBy) attrs["aria-describedby"] = describedBy
  if (invalid) attrs["aria-invalid"] = "true"
  return EditorView.contentAttributes.of(attrs)
}
