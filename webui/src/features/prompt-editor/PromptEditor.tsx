/**
 * Prompt editor (CodeMirror 6, its own lazy chunk): prompt-syntax colouring,
 * bracket-balance warnings, Ctrl/Cmd+↑/↓ attention, Alt+←/→ tag order,
 * undo/redo, registered completion sources, and image drops handed to the
 * page (generation parameters). Ctrl/Cmd+Enter is left to the page.
 */
import { Annotation, Compartment, EditorState, type Extension } from "@codemirror/state"
import { EditorView, placeholder as placeholderExt } from "@codemirror/view"
import { useQuery } from "@tanstack/react-query"
import { type Ref, useEffect, useImperativeHandle, useRef } from "react"
import { bootstrapQuery } from "@/api/queries"
import { cn } from "@/lib/utils"
import { keyOptionsFrom, minimalChange, type PromptKeyOptions } from "@/ui/prompt/editing"
import { subscribeCompletionSources } from "./completion-registry"
import { bracketLint, completionExtension, completionSlot, promptCommands, promptHighlight } from "./extensions"
import { type PromptKind, promptKind, type PromptScope, promptScope } from "./facets"

export type PromptEditorHandle = {
  focus: () => void
  /** Inserts text at the cursor (replacing the selection), as a user edit. */
  insertText: (text: string) => void
  view: () => EditorView | null
}

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
  kind?: PromptKind
  /** Where the editor is (read by completion sources; fixed at creation). */
  scope?: PromptScope
  /** Bracket-balance warnings (default on). */
  lint?: boolean
  /** Image files dropped on the editor (other drops are CodeMirror's). */
  onDropFiles?: (files: File[]) => void
  onFocusChange?: (focused: boolean) => void
  ref?: Ref<PromptEditorHandle>
}

/** Marks transactions that mirror the React value, so they are not echoed back. */
const externalSync = Annotation.define<boolean>()

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
  ".cm-prompt-weight": { color: "var(--primary)", fontVariantNumeric: "tabular-nums", fontWeight: "500" },
  ".cm-prompt-bracket": { fontWeight: "600" },
  ".cm-prompt-bracket-0": { color: "color-mix(in oklch, var(--primary) 70%, var(--muted-foreground))" },
  ".cm-prompt-bracket-1": { color: "color-mix(in oklch, var(--chart-2) 72%, var(--foreground))" },
  ".cm-prompt-bracket-2": { color: "color-mix(in oklch, var(--chart-4) 72%, var(--foreground))" },
  ".cm-prompt-bracket-error": {
    color: "var(--destructive)",
    fontWeight: "700",
    backgroundColor: "color-mix(in oklch, var(--destructive) 14%, transparent)",
    borderRadius: "2px",
  },
  ".cm-prompt-schedule": { color: "color-mix(in oklch, var(--chart-3) 72%, var(--foreground))", fontWeight: "600" },
  ".cm-prompt-extra": {
    color: "color-mix(in oklch, var(--chart-2) 72%, var(--foreground))",
    backgroundColor: "color-mix(in oklch, var(--chart-2) 10%, transparent)",
    borderRadius: "3px",
  },
  ".cm-prompt-keyword": { color: "color-mix(in oklch, var(--chart-3) 72%, var(--foreground))", fontWeight: "700", letterSpacing: "0.02em" },
  ".cm-prompt-wildcard": { color: "color-mix(in oklch, var(--chart-5) 72%, var(--foreground))", fontStyle: "italic" },
  ".cm-prompt-comment": { color: "var(--muted-foreground)", fontStyle: "italic" },
  ".cm-prompt-escape": { color: "var(--muted-foreground)" },
  ".cm-lintRange-warning": {
    backgroundImage: "none",
    textDecoration: "underline wavy var(--warning)",
    textUnderlineOffset: "3px",
  },
  ".cm-tooltip": {
    border: "none",
    borderRadius: "var(--radius-md)",
    backgroundColor: "var(--popover)",
    color: "var(--popover-foreground)",
    boxShadow: "0 4px 16px rgb(0 0 0 / 0.18)",
    outline: "1px solid color-mix(in oklch, var(--foreground) 10%, transparent)",
  },
  ".cm-tooltip-autocomplete > ul > li[aria-selected]": { backgroundColor: "var(--accent)", color: "var(--accent-foreground)" },
  ".cm-diagnostic": { padding: "0.25rem 0.5rem", fontSize: "0.75rem" },
  ".cm-diagnostic-warning": { borderLeftColor: "var(--warning)" },
})

function contentAttributes(id?: string, label?: string, describedBy?: string, invalid?: boolean): Extension {
  const attrs: Record<string, string> = { spellcheck: "false", autocorrect: "off", autocapitalize: "off", "aria-multiline": "true" }
  if (id) attrs.id = id
  if (label) attrs["aria-label"] = label
  if (describedBy) attrs["aria-describedby"] = describedBy
  if (invalid) attrs["aria-invalid"] = "true"
  return EditorView.contentAttributes.of(attrs)
}

const editableExt = (disabled?: boolean) => [EditorView.editable.of(!disabled), EditorState.readOnly.of(Boolean(disabled))]

export default function PromptEditor(props: PromptEditorProps) {
  const host = useRef<HTMLDivElement>(null)
  const view = useRef<EditorView | null>(null)
  const callbacks = useRef({ onChange: props.onChange, onDropFiles: props.onDropFiles, onFocusChange: props.onFocusChange })
  const bootstrap = useQuery(bootstrapQuery)
  const keyOptions = useRef<PromptKeyOptions>(keyOptionsFrom(undefined))
  const compartments = useRef({ editable: new Compartment(), placeholder: new Compartment(), aria: new Compartment(), lint: new Compartment() })

  useEffect(() => {
    callbacks.current = { onChange: props.onChange, onDropFiles: props.onDropFiles, onFocusChange: props.onFocusChange }
    keyOptions.current = keyOptionsFrom(bootstrap.data?.options)
  })

  useImperativeHandle(
    props.ref,
    () => ({
      focus: () => view.current?.focus(),
      insertText: (text) => {
        const editor = view.current
        if (!editor) return
        editor.dispatch(editor.state.replaceSelection(text), { userEvent: "input.paste", scrollIntoView: true })
        editor.focus()
      },
      view: () => view.current,
    }),
    [],
  )

  // Create the view once; later prop changes are dispatched into it.
  useEffect(() => {
    if (!host.current) return
    const c = compartments.current
    const editor = new EditorView({
      parent: host.current,
      state: EditorState.create({
        doc: props.value,
        extensions: [
          promptCommands(() => keyOptions.current),
          EditorView.lineWrapping,
          promptHighlight,
          theme,
          promptKind.of(props.kind ?? "positive"),
          promptScope.of(props.scope ?? { form: null, extension: false }),
          c.lint.of(props.lint === false ? [] : bracketLint),
          completionSlot.of(completionExtension()),
          c.editable.of(editableExt(props.disabled)),
          c.placeholder.of(props.placeholder ? placeholderExt(props.placeholder) : []),
          c.aria.of(contentAttributes(props.id, props.ariaLabel, props.ariaDescribedBy, props.ariaInvalid)),
          EditorView.domEventHandlers({
            drop: (event) => {
              const files = [...(event.dataTransfer?.files ?? [])].filter((file) => file.type.startsWith("image/"))
              if (files.length === 0 || !callbacks.current.onDropFiles) return false
              event.preventDefault()
              callbacks.current.onDropFiles(files)
              return true
            },
          }),
          EditorView.focusChangeEffect.of((_state, focused) => {
            callbacks.current.onFocusChange?.(focused)
            return null
          }),
          EditorView.updateListener.of((update) => {
            if (update.docChanged && !update.transactions.some((tr) => tr.annotation(externalSync))) {
              callbacks.current.onChange(update.state.doc.toString())
            }
          }),
        ],
      }),
    })
    view.current = editor
    const offCompletions = subscribeCompletionSources(() => {
      editor.dispatch({ effects: completionSlot.reconfigure(completionExtension()) })
    })
    return () => {
      offCompletions()
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
    const c = compartments.current
    view.current?.dispatch({
      effects: [
        c.editable.reconfigure(editableExt(props.disabled)),
        c.placeholder.reconfigure(props.placeholder ? placeholderExt(props.placeholder) : []),
        c.aria.reconfigure(contentAttributes(props.id, props.ariaLabel, props.ariaDescribedBy, props.ariaInvalid)),
        c.lint.reconfigure(props.lint === false ? [] : bracketLint),
      ],
    })
  }, [props.disabled, props.placeholder, props.ariaLabel, props.ariaDescribedBy, props.ariaInvalid, props.id, props.lint])

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
