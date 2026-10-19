/**
 * The prompt tools bar under a prompt editor: text ↔ tag view, quick tag
 * groups, prompt history, favorites, translation and, with an
 * OpenAI-compatible provider, writing a prompt from a description (each
 * panel is its own lazy chunk, loaded when first opened).
 */
import { BrushCleaning, History, KeyRound, Languages, LayoutGrid, Sparkles, Star, Tags, TextCursorInput } from "lucide-react"
import { lazy, type ReactNode, Suspense, useState } from "react"
import { useTranslation } from "react-i18next"
import { toast } from "sonner"
import { Button } from "@/components/ui/button"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuSeparator, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { usePopupTooltip } from "@/components/popup-tooltip"
import { Spinner } from "@/components/ui/spinner"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import type { PromptDecoratorProps } from "@/features/prompt-editor/decorators"
import { cn } from "@/lib/utils"
import type { PromptToolsConfig } from "./api"
import { parsePrompt, tidyPrompt } from "./chips/parse"
import type { PanelProps } from "./panels/helpers"
import { translatePromptToEnglish, translateTags } from "./translate"
import { type PromptView, usePromptViews } from "./view-store"

const TranslationKeyDialog = lazy(() => import("./TranslationKeyDialog").then((module) => ({ default: module.TranslationKeyDialog })))
const GroupsPanel = lazy(() => import("./panels/GroupsPanel"))
const HistoryPanel = lazy(() => import("./panels/HistoryPanel"))
const FavoritesPanel = lazy(() => import("./panels/FavoritesPanel"))
const GeneratePanel = lazy(() => import("./panels/GeneratePanel"))

export type PromptToolbarProps = PromptDecoratorProps & { config: PromptToolsConfig; viewKey: string; view: PromptView }

/** Providers reached over the network (they may need an API key). */
const NETWORK_PROVIDERS = new Set(["libretranslate", "openai", "deepl"])
const PROVIDER_NAMES: Record<string, string> = { libretranslate: "LibreTranslate", openai: "OpenAI-compatible API", deepl: "DeepL" }

const LANGUAGE_NAMES: Record<string, string> = {
  zh_CN: "Simplified Chinese",
  zh_TW: "Traditional Chinese",
  ja: "Japanese",
  ko: "Korean",
  ru: "Russian",
  de: "German",
  fr: "French",
  es: "Spanish",
}

function ToolButton({ label, pressed, disabled, onClick, children }: { label: string; pressed?: boolean; disabled?: boolean; onClick: () => void; children: ReactNode }) {
  return (
    <Tooltip>
      <TooltipTrigger
        render={
          <Button
            variant="ghost"
            size="icon-xs"
            aria-label={label}
            aria-pressed={pressed}
            disabled={disabled}
            onClick={onClick}
            className={cn("text-muted-foreground", pressed && "bg-primary/10 text-primary hover:bg-primary/15 hover:text-primary")}
          />
        }
      >
        {children}
      </TooltipTrigger>
      <TooltipContent>{label}</TooltipContent>
    </Tooltip>
  )
}

/** A bar button opening a panel in a popover. */
function PanelTool({ label, icon, disabled, wide, children }: { label: string; icon: ReactNode; disabled?: boolean; wide?: boolean; children: (close: () => void) => ReactNode }) {
  const tip = usePopupTooltip()
  return (
    <Popover open={tip.popupOpen} onOpenChange={tip.onPopupOpenChange}>
      <Tooltip {...tip.tooltip}>
        <TooltipTrigger
          render={
            <PopoverTrigger
              render={<Button variant="ghost" size="icon-xs" aria-label={label} disabled={disabled} className="text-muted-foreground" {...tip.trigger} />}
            />
          }
        >
          {icon}
        </TooltipTrigger>
        <TooltipContent>{label}</TooltipContent>
      </Tooltip>
      <PopoverContent
        align="start"
        aria-label={label}
        className={cn("max-h-[min(36rem,70dvh)] w-[min(28rem,calc(100vw-1rem))] gap-0 overflow-hidden p-0", wide && "w-[min(36rem,calc(100vw-1rem))]")}
      >
        {tip.popupOpen ? (
          <Suspense
            fallback={
              <div className="flex h-40 items-center justify-center">
                <Spinner />
              </div>
            }
          >
            {children(() => tip.onPopupOpenChange(false))}
          </Suspense>
        ) : null}
      </PopoverContent>
    </Popover>
  )
}

function TranslateTool({ value, setValue, disabled, config, viewKey }: PromptToolbarProps) {
  const { t } = useTranslation()
  const [busy, setBusy] = useState(false)
  const setView = usePromptViews((state) => state.setView)
  const tip = usePopupTooltip()
  const [keyDialog, setKeyDialog] = useState(false)
  const language = config.tools.language
  const provider = config.tools.translation_provider
  const usable = Boolean(config.tools.translation)
  const network = NETWORK_PROVIDERS.has(provider)

  const run = async (action: () => Promise<void>) => {
    setBusy(true)
    try {
      await action()
    } catch (error) {
      toast.error(t("Translation failed"), { description: error instanceof Error ? error.message : String(error) })
    } finally {
      setBusy(false)
    }
  }

  const toEnglish = () =>
    run(async () => {
      const result = await translatePromptToEnglish(value)
      if (result.translated === 0) {
        toast.info(t("Nothing to translate: every tag is already in English, or no translation was found."))
        return
      }
      setValue(result.text)
      toast.success(t("Translated {{count}} tags into English", { count: result.translated }), { description: t("Undo with Ctrl+Z in the editor.") })
    })

  const showTranslations = () =>
    run(async () => {
      const tags = parsePrompt(value).chips.filter((chip) => chip.kind === "tag" && !chip.disabled).map((chip) => chip.text)
      const found = await translateTags(tags, language)
      setView(viewKey, "chips")
      if (found === 0) toast.info(t("No new translations: the tag view already shows what is known."))
    })

  return (
    <>
      <DropdownMenu open={tip.popupOpen} onOpenChange={tip.onPopupOpenChange}>
        <Tooltip {...tip.tooltip}>
          <TooltipTrigger
            render={
              <DropdownMenuTrigger
                render={<Button variant="ghost" size="icon-xs" aria-label={t("Translate")} disabled={disabled || busy} className="text-muted-foreground" {...tip.trigger} />}
              />
            }
          >
            {busy ? <Spinner /> : <Languages />}
          </TooltipTrigger>
          <TooltipContent>{t("Translate")}</TooltipContent>
        </Tooltip>
        <DropdownMenuContent align="start" className="w-64">
          {usable ? null : (
            <p className="px-2 py-1.5 text-xs text-muted-foreground">{t("{{provider}} needs more settings: an API key, or the service URL and model in Settings.", { provider: t(PROVIDER_NAMES[provider] ?? provider) })}</p>
          )}
          <DropdownMenuItem disabled={!usable} onClick={() => void toEnglish()}>
            {t("Write the prompt in English")}
          </DropdownMenuItem>
          {network ? (
            <DropdownMenuItem disabled={!usable} onClick={() => void showTranslations()}>
              {t("Show the tags in {{language}}", { language: t(LANGUAGE_NAMES[language] ?? language) })}
            </DropdownMenuItem>
          ) : null}
          {network ? (
            <>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={() => setKeyDialog(true)}>
                <KeyRound />
                {config.tools.translation_key ? t("Change the API key…") : t("Set the API key…")}
              </DropdownMenuItem>
            </>
          ) : null}
        </DropdownMenuContent>
      </DropdownMenu>
      {keyDialog ? (
        <Suspense fallback={null}>
          <TranslationKeyDialog open={keyDialog} onOpenChange={setKeyDialog} configured={config.tools.translation_key} />
        </Suspense>
      ) : null}
    </>
  )
}

export default function PromptToolbar(props: PromptToolbarProps) {
  const { t } = useTranslation()
  const setView = usePromptViews((state) => state.setView)
  const { config, view, viewKey } = props
  const chips = view === "chips"
  const panel = (close: () => void): PanelProps => ({ value: props.value, setValue: props.setValue, kind: props.kind, label: props.label, disabled: props.disabled, config, close })

  return (
    <div role="group" aria-label={t("Prompt tools")} className="-mt-0.5 flex min-h-6 flex-wrap items-center gap-0.5">
      <ToolButton label={chips ? t("Edit as text") : t("Show as tags")} pressed={chips} onClick={() => setView(viewKey, chips ? "text" : "chips")}>
        {chips ? <TextCursorInput /> : <Tags />}
      </ToolButton>
      <ToolButton
        label={t("Tidy up the separators")}
        disabled={props.disabled || !props.value.trim()}
        onClick={() => {
          const tidy = tidyPrompt(props.value)
          if (tidy === props.value) {
            toast.info(t("The prompt is already tidy."))
            return
          }
          props.setValue(tidy)
          toast.success(t("Separators tidied up"), { description: t("Undo with Ctrl+Z in the editor.") })
        }}
      >
        <BrushCleaning />
      </ToolButton>
      {config.tools.groups ? (
        <PanelTool label={t("Quick tags")} icon={<LayoutGrid />} disabled={props.disabled} wide>
          {(close) => <GroupsPanel {...panel(close)} />}
        </PanelTool>
      ) : null}
      {config.tools.history ? (
        <PanelTool label={t("Prompt history")} icon={<History />}>
          {(close) => <HistoryPanel {...panel(close)} />}
        </PanelTool>
      ) : null}
      <PanelTool label={t("Favorites")} icon={<Star />}>
        {(close) => <FavoritesPanel {...panel(close)} />}
      </PanelTool>
      {config.tools.translation || NETWORK_PROVIDERS.has(config.tools.translation_provider) ? <TranslateTool {...props} /> : null}
      {config.tools.generate ? (
        <PanelTool label={props.kind === "negative" ? t("Write a negative prompt") : t("Write a prompt")} icon={<Sparkles />} disabled={props.disabled}>
          {(close) => <GeneratePanel {...panel(close)} />}
        </PanelTool>
      ) : null}
    </div>
  )
}
