/**
 * Generate-page keyboard shortcuts: Ctrl/Cmd+Enter generates (or interrupts
 * and restarts, see `run-machine.ts`), Ctrl/Cmd+Shift+Enter adds the form to
 * the queue (the Agent Scheduler's "Enqueue" shortcut), Alt+Enter skips,
 * Escape interrupts unless it belongs to something else (a text field,
 * popup, dialog, menu). In a prompt editor Escape interrupts as in A1111
 * once the editor has no use for it: CodeMirror consumes (preventDefault)
 * the Escape that closes a completion list or drops a selection.
 */
import { useEffect, useRef } from "react"
import { enqueueForm } from "@/features/queue/edit-session"
import { dispatchRun, type GenerateTab, useRuns } from "@/features/progress/run-controller"
import { isBusy } from "@/features/progress/run-machine"

const TEXT_TARGETS = 'input, textarea, select, [contenteditable=""], [contenteditable="true"], .cm-editor'
const OWN_ESCAPE = '[role="dialog"], [role="alertdialog"], [role="menu"], [data-slot$="-content"]'
/** The text of a prompt editor (not a decorator's own fields around it). */
const PROMPT_TEXT = '[data-slot="prompt-editor"] .cm-editor'
const IN_OVERLAY = '[role="dialog"], [role="alertdialog"], [role="menu"]'
const OPEN_POPUP = '[data-open]:is([role="dialog"], [role="alertdialog"], [role="menu"], [role="listbox"], [data-side])'
const OPEN_MODAL = '[data-open]:is([role="dialog"], [role="alertdialog"])'

/** A modal dialog (style editor, viewer, confirmations) keeps the generate keys to itself. */
const modalOpen = () => document.querySelector(OPEN_MODAL) !== null

/** Whether Escape is free to interrupt generation (nothing else uses it). */
export function escapeIsFree(event: KeyboardEvent): boolean {
  if (event.defaultPrevented || event.isComposing) return false
  const target = event.target instanceof Element ? event.target : null
  if (target?.closest(PROMPT_TEXT)) {
    // A prompt editor in a dialog (style editor) leaves Escape to the dialog.
    if (target.closest(IN_OVERLAY)) return false
  } else if (target?.closest(TEXT_TARGETS) || target?.closest(OWN_ESCAPE)) return false
  return document.querySelector(OPEN_POPUP) === null
}

export function useGenerateShortcuts(tab: GenerateTab) {
  const current = useRef(tab)
  useEffect(() => {
    current.current = tab
  })
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const tab = current.current
      if (event.key === "Enter" && (event.metaKey || event.ctrlKey) && !event.altKey) {
        if (event.repeat || event.isComposing || event.defaultPrevented || modalOpen()) return
        event.preventDefault()
        if (event.shiftKey) void enqueueForm(tab)
        else dispatchRun(tab, { type: "shortcut" })
      } else if (event.key === "Enter" && event.altKey && !event.metaKey && !event.ctrlKey) {
        if (event.repeat || event.isComposing || event.defaultPrevented || modalOpen()) return
        event.preventDefault()
        dispatchRun(tab, { type: "skip" })
      } else if (event.key === "Escape" && !event.metaKey && !event.ctrlKey && !event.altKey && !event.shiftKey) {
        if (!isBusy(useRuns.getState().runs[tab]) || !escapeIsFree(event)) return
        event.preventDefault()
        dispatchRun(tab, { type: "interrupt" })
      }
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [])
}
