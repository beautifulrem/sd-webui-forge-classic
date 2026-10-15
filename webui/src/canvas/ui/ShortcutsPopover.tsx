/** Keyboard shortcuts of the editor (also opened with "?"). */
import { Keyboard } from "lucide-react"
import { useTranslation } from "react-i18next"
import { Button } from "@/components/ui/button"
import { Kbd, KbdGroup } from "@/components/ui/kbd"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { isMac, shortcutList } from "../core/shortcuts"
import { Hint } from "./controls"

const GROUPS = ["Tools", "Brush", "Edit", "View"] as const

/** Key names that are words get translated; symbols and letters stay. */
function useKeyName() {
  const { t } = useTranslation()
  return (key: string) => {
    switch (key) {
      case "Shift":
        return t("Shift")
      case "Space":
        return t("Space")
      case "Wheel":
        return t("Wheel")
      case "Drag":
        return t("Drag")
      case "Right drag":
        return t("Right drag")
      default:
        return key
    }
  }
}

export function ShortcutsPopover({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) {
  const { t } = useTranslation()
  const keyName = useKeyName()
  const list = shortcutList(isMac())
  return (
    <Popover open={open} onOpenChange={onOpenChange}>
      <Hint label={t("Keyboard shortcuts")} keys={["?"]} disabled={open}>
        <PopoverTrigger render={<Button variant="ghost" size="icon-sm" aria-label={t("Keyboard shortcuts")} />}>
          <Keyboard />
        </PopoverTrigger>
      </Hint>
      <PopoverContent side="top" align="end" className="max-h-[min(32rem,70vh)] w-[min(34rem,calc(100vw-2rem))] overflow-y-auto">
        <h3 className="text-sm font-medium">{t("Keyboard shortcuts")}</h3>
        <div className="grid gap-x-6 gap-y-3 sm:grid-cols-2">
          {GROUPS.map((group) => (
            <section key={group} aria-label={t(group)} className="flex flex-col gap-1">
              <h4 className="text-xs font-medium text-muted-foreground">{t(group)}</h4>
              <dl className="flex flex-col gap-1">
                {list
                  .filter((item) => item.group === group)
                  .map((item) => (
                    <div key={`${item.label}-${item.keys.join("+")}`} className="flex items-center justify-between gap-3 text-sm">
                      <dt>{t(item.label)}</dt>
                      <dd>
                        <KbdGroup>
                          {item.keys.map((key) => (
                            <Kbd key={key}>{keyName(key)}</Kbd>
                          ))}
                        </KbdGroup>
                      </dd>
                    </div>
                  ))}
              </dl>
            </section>
          ))}
        </div>
      </PopoverContent>
    </Popover>
  )
}
