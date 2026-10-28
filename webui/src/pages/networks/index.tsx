/**
 * Networks: the full-screen network browser. Prompt cards go into the chosen
 * generation form (txt2img / img2img, positive or negative prompt), which
 * keeps them for when its page opens; checkpoint and module cards switch the
 * model at once.
 */
import { useTranslation } from "react-i18next"
import { HeaderSlot } from "@/app/header-slot"
import { routeHref } from "@/app/router"
import { Button } from "@/components/ui/button"
import { NetworkBrowser } from "@/features/networks/NetworkBrowser"
import { useNetworksUi } from "@/features/networks/store"
import { FormTargetSwitch, PromptTargetSwitch } from "@/features/networks/TargetSwitch"

export default function NetworksPage() {
  const { t } = useTranslation()
  const form = useNetworksUi((s) => s.pageForm)
  return (
    <>
      <HeaderSlot>
        <div className="hidden items-center gap-2 2xl:flex">
          <span className="shrink-0 text-xs text-muted-foreground">{t("Insert into")}</span>
          <FormTargetSwitch className="shrink-0" />
          <PromptTargetSwitch className="shrink-0" />
          <Button variant="ghost" size="sm" render={<a href={routeHref({ name: "generate", tab: form })} />} nativeButton={false} className="text-muted-foreground">
            {t("Open {{form}}", { form: t(form) })}
          </Button>
        </div>
      </HeaderSlot>
      <div className="flex h-full min-h-0 flex-col">
        {/* below 2xl the header (model bar, quick settings) has no room: the targets get their own row */}
        <div className="flex items-center gap-2 overflow-x-auto border-b px-3 py-1.5 2xl:hidden" role="group" aria-label={t("Insert into")}>
          <span className="shrink-0 text-xs text-muted-foreground">{t("Insert into")}</span>
          <FormTargetSwitch className="shrink-0" />
          <PromptTargetSwitch className="shrink-0" />
          <Button variant="ghost" size="sm" render={<a href={routeHref({ name: "generate", tab: form })} />} nativeButton={false} className="hidden shrink-0 text-muted-foreground sm:inline-flex">
            {t("Open {{form}}", { form: t(form) })}
          </Button>
        </div>
        <NetworkBrowser surface="page" form={form} toastOnApply className="min-h-0 flex-1" />
      </div>
    </>
  )
}
