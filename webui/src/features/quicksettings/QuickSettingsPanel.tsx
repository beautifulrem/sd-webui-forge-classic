/** The quick settings popover's content: the pinned settings, rendered and applied as they change (lazy chunk: it carries the schema renderer). */
import { useQuery } from "@tanstack/react-query"
import { Loader2, PinOff } from "lucide-react"
import { useEffect, useState } from "react"
import { useTranslation } from "react-i18next"
import { optionsQuery } from "@/api/queries"
import { Button } from "@/components/ui/button"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { FormProvider, OwnerProvider } from "@/ui/context"
import { RenderNode } from "@/ui/Renderer"
import { createQuickForm, QUICK_OWNER, type QuickForm } from "./quick-form"
import { setPinned } from "./pinned"

export default function QuickSettingsPanel({ keys: pinned }: { keys: string[] }) {
  const { t } = useTranslation()
  const options = useQuery(optionsQuery)
  const [form] = useState<QuickForm>(() => createQuickForm())
  const [loadedFrom, setLoadedFrom] = useState<object | null>(null)
  const signature = pinned.join("\n")

  useEffect(() => form.start(), [form])
  useEffect(() => {
    if (!options.data) return
    form.load(options.data, pinned)
    setLoadedFrom(options.data)
    // `signature` stands for `pinned`.
    // oxlint-disable-next-line react-hooks/exhaustive-deps
  }, [form, options.data, signature])

  if (options.isError) return <p className="px-1 py-4 text-sm text-destructive">{options.error.message}</p>
  if (!options.data || loadedFrom === null) {
    return (
      <div className="grid place-items-center py-8" aria-busy="true">
        <Loader2 className="size-5 animate-spin text-muted-foreground" aria-label={t("Loading…")} />
      </div>
    )
  }
  const missing = pinned.filter((key) => !form.nodes.some((node) => node.name === key))
  return (
    <FormProvider formId="quicksettings" store={form.store}>
      <OwnerProvider owner={QUICK_OWNER}>
        <ul className="flex max-h-[min(32rem,70dvh)] flex-col gap-1 overflow-y-auto overscroll-contain pr-1" aria-label={t("Quick settings")}>
          {form.nodes.map((node) => (
            <li key={node.name} className="group/quick flex items-start gap-1 rounded-lg px-2 py-2 hover:bg-muted/50">
              <div className="min-w-0 flex-1">
                <RenderNode node={node} />
              </div>
              <Tooltip>
                <TooltipTrigger
                  render={
                    <Button
                      variant="ghost"
                      size="icon-xs"
                      className="opacity-0 group-focus-within/quick:opacity-100 group-hover/quick:opacity-100 focus-visible:opacity-100"
                      aria-label={t("Unpin {{label}}", { label: t(node.label || node.name) })}
                      onClick={() => void setPinned(node.name, false)}
                    />
                  }
                >
                  <PinOff />
                </TooltipTrigger>
                <TooltipContent>{t("Unpin")}</TooltipContent>
              </Tooltip>
            </li>
          ))}
          {missing.map((key) => (
            <li key={key} className="flex items-center justify-between gap-2 px-2 py-1.5 text-xs text-muted-foreground">
              <span>
                {t("Unknown setting")} <code className="font-mono">{key}</code>
              </span>
              <Button variant="ghost" size="xs" onClick={() => void setPinned(key, false)}>
                {t("Unpin")}
              </Button>
            </li>
          ))}
        </ul>
      </OwnerProvider>
    </FormProvider>
  )
}
