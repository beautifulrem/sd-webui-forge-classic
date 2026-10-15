/** Header menu: theme, density, accent colour and language. */
import { Check, Languages, Monitor, Moon, Palette, Rows3, Rows4, Sun } from "lucide-react"
import { useTranslation } from "react-i18next"
import type { LocaleInfo } from "@/api/types"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { useLocale } from "@/i18n"
import { ACCENTS, type Accent, type Density, type Theme, useAppearance } from "@/lib/theme"
import { cn } from "@/lib/utils"

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

  return (
    <DropdownMenu>
      <Tooltip>
        <TooltipTrigger
          render={
            <DropdownMenuTrigger render={<Button variant="ghost" size="icon" aria-label={t("Appearance")} />} />
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
                {t(label)}
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
          <div role="radiogroup" aria-label={t("Accent color")} className="flex flex-wrap gap-1.5 px-1.5 pt-0.5 pb-1.5">
            {ACCENTS.map((value) => (
              <AccentSwatch
                key={value}
                accent={value}
                label={t(ACCENT_LABELS[value])}
                selected={accent === value}
                onSelect={() => setAccent(value)}
              />
            ))}
          </div>
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
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

function AccentSwatch(props: { accent: Accent; label: string; selected: boolean; onSelect: () => void }) {
  return (
    <Tooltip>
      <TooltipTrigger
        render={
          <button
            type="button"
            role="radio"
            aria-checked={props.selected}
            aria-label={props.label}
            data-accent={props.accent}
            onClick={props.onSelect}
            className={cn(
              "grid size-6 place-items-center rounded-full bg-primary text-primary-foreground outline-none transition-transform",
              "hover:scale-110 focus-visible:ring-3 focus-visible:ring-ring/60",
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
