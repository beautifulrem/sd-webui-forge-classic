/** Ctrl/Cmd+K command palette (lazy chunk; cmdk is only loaded on first open). */
import { useTranslation } from "react-i18next"
import {
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandShortcut,
  Command as CommandRoot,
} from "@/components/ui/command"
import { type Command, formatShortcut, useCommands } from "./commands"

function groupBy(commands: Command[]): [string, Command[]][] {
  const groups = new Map<string, Command[]>()
  for (const command of commands) {
    const list = groups.get(command.group)
    if (list) list.push(command)
    else groups.set(command.group, [command])
  }
  return [...groups]
}

export default function CommandPalette() {
  const { t } = useTranslation()
  const open = useCommands((s) => s.palette)
  const setOpen = useCommands((s) => s.setPalette)
  const sources = useCommands((s) => s.sources)
  const groups = groupBy(Object.values(sources).flat())

  const run = (command: Command) => {
    setOpen(false)
    // Let the dialog close (and restore focus) before the command moves focus.
    requestAnimationFrame(() => command.run())
  }

  return (
    <CommandDialog
      open={open}
      onOpenChange={setOpen}
      title={t("Command palette")}
      description={t("Search for a page or an action")}
      className="sm:max-w-lg"
    >
      <CommandRoot loop label={t("Search for a page or an action")}>
        <CommandInput placeholder={t("Type a command or search…")} />
        <CommandList className="max-h-[min(60vh,26rem)]">
          <CommandEmpty>{t("No results.")}</CommandEmpty>
          {groups.map(([group, commands]) => (
            <CommandGroup key={group} heading={t(group)}>
              {commands.map((command) => {
                const label = t(command.label)
                const Icon = command.icon
                return (
                  <CommandItem
                    key={command.id}
                    value={`${command.id} ${label}`}
                    keywords={[command.label, ...(command.keywords ?? [])]}
                    onSelect={() => run(command)}
                  >
                    {Icon ? <Icon aria-hidden /> : null}
                    <span className="truncate">{label}</span>
                    {command.shortcut ? <CommandShortcut className="touch-only:hidden">{formatShortcut(command.shortcut)}</CommandShortcut> : null}
                  </CommandItem>
                )
              })}
            </CommandGroup>
          ))}
        </CommandList>
      </CommandRoot>
    </CommandDialog>
  )
}
