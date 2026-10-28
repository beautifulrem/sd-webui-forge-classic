/** The parameters of one generation form: prompts, img2img source, curated core form, scripts. */
import { lazy, type ReactNode, Suspense } from "react"
import { useTranslation } from "react-i18next"
import type { GenerateTab } from "@/features/progress/run-controller"
import { Skeleton } from "@/components/ui/skeleton"
import { selectedItem, useSession } from "@/features/gallery/session-store"
import { FormProvider, OwnerProvider, SeedSourceProvider } from "@/ui/context"
import { getFormStore } from "@/ui/form-store"
import { ScriptsProvider } from "@/ui/layout/Slot"
import type { TabSchema } from "@/ui/schema"
import { CoreForm, SourceSlot } from "./CoreForm"
import { curateCore } from "./curate"
import { PromptsPanel } from "./PromptsPanel"
import { ScriptsPanel } from "./ScriptsPanel"
import { useFormLoader } from "./use-form-loader"

// Phones only (the header shows the bar elsewhere); shares the header bar's chunk.
const ModelBarStacked = lazy(() => import("@/features/model-bar/ModelBar").then((m) => ({ default: m.ModelBarStacked })))

/** "Reuse seed" takes the seed of the result the stage shows. */
function SelectedSeedSource({ children }: { children: ReactNode }) {
  const infotext = useSession((s) => selectedItem(s)?.infotext ?? null)
  return <SeedSourceProvider infotext={infotext}>{children}</SeedSourceProvider>
}

export function ParamsPane({ schema, tab, showModelBar = false }: { schema: TabSchema; tab: GenerateTab; showModelBar?: boolean }) {
  const { t } = useTranslation()
  const store = getFormStore(tab)
  const ready = useFormLoader(store, schema)
  const curated = curateCore(schema.core.layout)

  if (!ready) {
    return (
      <div className="flex flex-col gap-3" aria-busy>
        <Skeleton className="h-28 rounded-md" />
        <Skeleton className="h-16 rounded-md" />
        <Skeleton className="h-40 rounded-md" />
      </div>
    )
  }

  return (
    <FormProvider formId={tab} store={store}>
      <SelectedSeedSource>
        <ScriptsProvider scripts={schema.scripts}>
          <div className="flex flex-col gap-3" aria-label={t("Parameters")} role="form">
            {showModelBar ? (
              <Suspense fallback={<Skeleton className="h-16 rounded-md" />}>
                <ModelBarStacked />
              </Suspense>
            ) : null}
            <OwnerProvider owner="core">
              <PromptsPanel prompt={curated.prompt} negative={curated.negative} styles={curated.styles} />
            </OwnerProvider>
            <SourceSlot curated={curated} />
            <CoreForm curated={curated} />
            <ScriptsPanel schema={schema} scriptChoice={curated.script} />
          </div>
        </ScriptsProvider>
      </SelectedSeedSource>
    </FormProvider>
  )
}
