/** Header menu: theme, density, accent colour and language. */
import { Check, Keyboard, Languages, Monitor, Moon, Palette, Rows3, Rows4, Sun } from "lucide-react"
import { Menu as MenuPrimitive } from "@base-ui/react/menu"
import { useTranslation } from "react-i18next"
import type { LocaleInfo } from "@/api/types"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuShortcut,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { usePopupTooltip } from "@/components/popup-tooltip"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { useLocale } from "@/i18n"
import { ACCENTS, type Accent, type Density, type Theme, useAppearance } from "@/lib/theme"
import { cn } from "@/lib/utils"
import { useShortcutsHelp } from "./shortcuts-help"

const THEME_OPTIONS = [
  { value: "light", label: "Light", icon: Sun },
  { value: "dark", label: "Dark", icon: Moon },
  { value: "system", label: "System", icon: Monitor },
] as const satisfies readonly { value: Theme; label: string; icon: unknown }[]

const DENSITY_OPTIONS = [
  { value: "comfortable", label: "Comfortable", icon: Rows3 },
  { value: "compact", label: "Compact", icon: Rows4 },
] as const satisfies readonly { value: Density; label: string; icon: unknown }[]

const ACCENT_LABELS: Record<Accent, string> = {
  iris: "Iris",
  blue: "Blue",
  teal: "Teal",
  green: "Green",
  amber: "Amber",
  coral: "Coral",
  rose: "Rose",
}

export default function AppearanceMenu({ locales }: { locales: LocaleInfo[] }) {
  const { t } = useTranslation()
  const { theme, density, accent, setTheme, setDensity, setAccent } = useAppearance()
  const locale = useLocale((s) => s.locale)
  const setLocale = useLocale((s) => s.setLocale)
  // Closing the menu refocuses the button: keep its tooltip from popping up and staying.
  const tip = usePopupTooltip()

  return (
    <DropdownMenu open={tip.popupOpen} onOpenChange={tip.onPopupOpenChange}>
      <Tooltip {...tip.tooltip}>
        <TooltipTrigger
          render={
            <DropdownMenuTrigger render={<Button variant="ghost" size="icon" aria-label={t("Appearance")} {...tip.trigger} />} />
          }
        >
          <Palette />
        </TooltipTrigger>
        <TooltipContent>{t("Appearance")}</TooltipContent>
      </Tooltip>
      <DropdownMenuContent align="end" className="w-60">
        <DropdownMenuGroup>
          <DropdownMenuLabel>{t("Theme")}</DropdownMenuLabel>
          <DropdownMenuRadioGroup value={theme} onValueChange={(v) => setTheme(v as Theme)}>
            {THEME_OPTIONS.map(({ value, label, icon: Icon }) => (
              <DropdownMenuRadioItem key={value} value={value}>
                <Icon aria-hidden />
                {/* "System" here means "follow the system", not the settings category */}
                {value === "system" ? t("System", { context: "theme" }) : t(label)}
              </DropdownMenuRadioItem>
            ))}
          </DropdownMenuRadioGroup>
        </DropdownMenuGroup>
        <DropdownMenuSeparator />
        <DropdownMenuGroup>
          <DropdownMenuLabel>{t("Density")}</DropdownMenuLabel>
          <DropdownMenuRadioGroup value={density} onValueChange={(v) => setDensity(v as Density)}>
            {DENSITY_OPTIONS.map(({ value, label, icon: Icon }) => (
              <DropdownMenuRadioItem key={value} value={value}>
                <Icon aria-hidden />
                {t(label)}
              </DropdownMenuRadioItem>
            ))}
          </DropdownMenuRadioGroup>
        </DropdownMenuGroup>
        <DropdownMenuSeparator />
        <DropdownMenuGroup>
          <DropdownMenuLabel>{t("Accent color")}</DropdownMenuLabel>
          <DropdownMenuRadioGroup
            value={accent}
            onValueChange={(v) => setAccent(v as Accent)}
            aria-label={t("Accent color")}
            className="flex flex-wrap gap-1.5 px-1.5 pt-0.5 pb-1.5"
          >
            {ACCENTS.map((value) => (
              <AccentSwatch key={value} accent={value} label={t(ACCENT_LABELS[value])} selected={accent === value} />
            ))}
          </DropdownMenuRadioGroup>
        </DropdownMenuGroup>
        {locales.length > 1 ? (
          <>
            <DropdownMenuSeparator />
            <DropdownMenuGroup>
              <DropdownMenuLabel className="flex items-center gap-1.5">
                <Languages className="size-3.5" aria-hidden />
                {t("Language")}
              </DropdownMenuLabel>
              <DropdownMenuRadioGroup value={locale} onValueChange={(v) => setLocale(String(v))}>
                {locales.map((item) => (
                  <DropdownMenuRadioItem key={item.id} value={item.id} lang={item.id}>
                    {item.name}
                  </DropdownMenuRadioItem>
                ))}
              </DropdownMenuRadioGroup>
            </DropdownMenuGroup>
          </>
        ) : null}
        <DropdownMenuSeparator />
        <DropdownMenuItem onClick={() => useShortcutsHelp.getState().setOpen(true)}>
          <Keyboard aria-hidden />
          {t("Keyboard shortcuts")}
          <DropdownMenuShortcut>?</DropdownMenuShortcut>
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

function AccentSwatch(props: { accent: Accent; label: string; selected: boolean }) {
  // A menu radio item (a plain radiogroup is not allowed inside role="menu"); Base UI handles keys and checking.
  return (
    <Tooltip>
      <TooltipTrigger
        render={
          <MenuPrimitive.RadioItem
            value={props.accent}
            aria-label={props.label}
            data-accent={props.accent}
            className={cn(
              "grid size-6 place-items-center rounded-full bg-primary text-primary-foreground outline-none transition-transform",
              "hover:scale-110 focus-visible:ring-3 focus-visible:ring-ring/60 data-highlighted:scale-110 data-highlighted:ring-3 data-highlighted:ring-ring/60",
              props.selected && "ring-2 ring-foreground/70 ring-offset-2 ring-offset-popover",
            )}
          />
        }
      >
        {props.selected ? <Check className="size-3.5" aria-hidden /> : null}
      </TooltipTrigger>
      <TooltipContent>{props.label}</TooltipContent>
    </Tooltip>
  )
}
