/**
 * The img2img surfaces of the Generate page, loaded with the img2img tab:
 * txt2img (the default view) never renders them, so their code stays out of
 * its first load (docs/research/ui-performance.md, "Bundle").
 */
import { type ComponentProps, lazy, Suspense } from "react"
import { Skeleton } from "@/components/ui/skeleton"
import { cn } from "@/lib/utils"

const SourcePanel = lazy(() => import("./SourcePanel").then((m) => ({ default: m.Img2imgSourcePanel })))
const Stage = lazy(() => import("./Img2imgStage").then((m) => ({ default: m.Img2imgStage })))

type SourcePanelProps = ComponentProps<typeof import("./SourcePanel").Img2imgSourcePanel>
type StageProps = ComponentProps<typeof import("./Img2imgStage").Img2imgStage>

/** The source card's content (`SourcePanel`), fetched the first time it renders. */
export function Img2imgSourcePanel(props: SourcePanelProps) {
  return (
    <Suspense fallback={<Skeleton className="h-40 rounded-lg" />}>
      <SourcePanel {...props} />
    </Suspense>
  )
}

/** The img2img stage (`Img2imgStage`), fetched the first time it renders. */
export function Img2imgStage(props: StageProps) {
  return (
    <Suspense fallback={<Skeleton className={cn("rounded-md", props.className)} />}>
      <Stage {...props} />
    </Suspense>
  )
}
