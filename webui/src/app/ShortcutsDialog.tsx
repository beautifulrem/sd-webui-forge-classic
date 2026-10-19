/**
 * The app's keyboard shortcuts in one dialog: "?" outside text fields, the
 * command palette ("Keyboard shortcuts") or the appearance menu. The image
 * editor lists its own with "?" while it is in use.
 */
import { useTranslation } from "react-i18next"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Kbd, KbdGroup } from "@/components/ui/kbd"
import { formatShortcut, isMac } from "./commands"
import { useShortcutsHelp } from "./shortcuts-help"

type Entry = { label: string; keys: string[] }

/** English labels and keys (`Mod` is ⌘ on macOS, Ctrl elsewhere); one key per Kbd, alternatives as separate entries. */
const SHORTCUT_GROUPS: { group: string; entries: Entry[] }[] = [
  {
    group: "Everywhere",
    entries: [
      { label: "Command palette", keys: ["Mod+K"] },
      { label: "Keyboard shortcuts", keys: ["?"] },
      { label: "Skip to content", keys: ["Tab"] },
    ],
  },
  {
    group: "Generate",
    entries: [
      { label: "Generate (while running: queue another)", keys: ["Mod+Enter"] },
      { label: "Add to queue", keys: ["Mod+Shift+Enter"] },
      { label: "Skip the current image", keys: ["Alt+Enter"] },
      { label: "Interrupt", keys: ["Esc"] },
      { label: "Networks panel", keys: ["Alt+N"] },
      { label: "Results: select, open, remove", keys: ["←", "→", "Enter", "Delete"] },
    ],
  },
  {
    group: "Prompt editor",
    entries: [
      { label: "More / less attention", keys: ["Mod+↑", "Mod+↓"] },
      { label: "Move the tag left / right", keys: ["Alt+←", "Alt+→"] },
      { label: "Undo / redo", keys: ["Mod+Z", isMac ? "Mod+Shift+Z" : "Mod+Y"] },
      { label: "Accept a suggestion", keys: ["Tab", "Enter"] },
    ],
  },
  {
    group: "Image viewer",
    entries: [
      { label: "Previous / next image", keys: ["←", "→"] },
      { label: "First / last image", keys: ["Home", "End"] },
      { label: "Zoom in / out", keys: ["+", "-"] },
      { label: "Fit / actual size", keys: ["0", "1"] },
      { label: "Tile preview", keys: ["T"] },
      { label: "Download", keys: ["S"] },
      { label: "Close", keys: ["Esc"] },
    ],
  },
  {
    group: "Gallery",
    entries: [
      { label: "Move between images", keys: ["←", "→", "↑", "↓"] },
      { label: "Select (in selection mode)", keys: ["Space"] },
      { label: "Star", keys: ["F"] },
    ],
  },
  {
    group: "Networks",
    entries: [
      { label: "Search", keys: ["/"] },
      { label: "Insert the card", keys: ["Enter"] },
      { label: "Metadata / edit", keys: ["I", "E"] },
      { label: "Card menu", keys: ["Shift+F10"] },
    ],
  },
  {
    group: "Settings",
    entries: [
      { label: "Search", keys: ["/"] },
      { label: "Apply settings", keys: ["Mod+S"] },
    ],
  },
]

export default function ShortcutsDialog() {
  const { t } = useTranslation()
  const open = useShortcutsHelp((s) => s.open)
  const setOpen = useShortcutsHelp((s) => s.setOpen)
  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogContent className="max-h-[min(40rem,calc(100dvh-2rem))] overflow-y-auto sm:max-w-3xl">
        <DialogHeader>
          <DialogTitle>{t("Keyboard shortcuts")}</DialogTitle>
          <DialogDescription>{t("The image editor lists its own: press ? while it is in use.")}</DialogDescription>
        </DialogHeader>
        <div className="grid gap-x-8 gap-y-5 sm:grid-cols-2">
          {SHORTCUT_GROUPS.map(({ group, entries }) => (
            <section key={group} aria-label={t(group)} className="flex flex-col gap-1.5">
              <h3 className="text-xs font-medium text-muted-foreground">{t(group)}</h3>
              <dl className="flex flex-col gap-1">
                {entries.map((entry) => (
                  <div key={entry.label} className="flex items-center justify-between gap-3">
                    <dt>{t(entry.label)}</dt>
                    <dd className="shrink-0">
                      <KbdGroup>
                        {entry.keys.map((key) => (
                          <Kbd key={key}>{formatShortcut(key)}</Kbd>
                        ))}
                      </KbdGroup>
                    </dd>
                  </div>
                ))}
              </dl>
            </section>
          ))}
        </div>
      </DialogContent>
    </Dialog>
  )
}
