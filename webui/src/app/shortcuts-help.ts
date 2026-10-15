/** Whether the keyboard shortcuts dialog is open, and the "?" key that opens it. */
import { create } from "zustand"

export const useShortcutsHelp = create<{ open: boolean; loaded: boolean; setOpen: (open: boolean) => void }>()((set) => ({
  open: false,
  loaded: false,
  setOpen: (open) => set((s) => ({ open, loaded: s.loaded || open })),
}))

function isTyping(target: EventTarget | null): boolean {
  return target instanceof HTMLElement && (target.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName) || target.closest(".cm-editor") !== null)
}

/**
 * "?" outside text fields and dialogs opens the dialog. Decided after the
 * event has run its course: a handler that used the key (the image editor's
 * own "?" list) prevents its default.
 */
export function onShortcutsKey(event: KeyboardEvent) {
  if (event.key !== "?" || event.ctrlKey || event.metaKey || event.altKey || event.repeat || isTyping(event.target)) return
  if (document.querySelector('[role="dialog"], [role="alertdialog"]')) return
  setTimeout(() => {
    if (!event.defaultPrevented) useShortcutsHelp.getState().setOpen(true)
  })
}
