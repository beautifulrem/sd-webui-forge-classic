/** Compare two images: a before / after divider, or side by side. */
import { useState } from "react"
import { useTranslation } from "react-i18next"
import { api } from "@/api/endpoints"
import { Segmented } from "@/components/Segmented"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { CompareSlider } from "@/features/compare/CompareSlider"

export type CompareImage = { id: string; name: string; width?: number | null; height?: number | null }

type CompareDialogProps = {
  images: readonly [CompareImage, CompareImage] | null
  onOpenChange: (open: boolean) => void
}

export function CompareDialog({ images, onOpenChange }: CompareDialogProps) {
  const { t } = useTranslation()
  const [mode, setMode] = useState<"slider" | "side">("slider")
  const [a, b] = images ?? []
  const sameSize = a && b && a.width === b.width && a.height === b.height
  return (
    <Dialog open={images !== null} onOpenChange={onOpenChange}>
      <DialogContent className="flex max-h-[94dvh] w-[min(96vw,1400px)] max-w-none flex-col gap-3 sm:max-w-none">
        <DialogHeader className="flex-row flex-wrap items-center justify-between gap-2">
          <div className="min-w-0">
            <DialogTitle>{t("Compare images")}</DialogTitle>
            <DialogDescription className="truncate">
              {a && b ? `${a.name} · ${b.name}` : null}
              {a && b && !sameSize ? ` · ${t("Different sizes: both are fitted to the frame")}` : null}
            </DialogDescription>
          </div>
          <Segmented
            size="sm"
            aria-label={t("Comparison view")}
            value={mode}
            onValueChange={setMode}
            options={[
              { value: "slider", label: t("Divider") },
              { value: "side", label: t("Side by side") },
            ]}
          />
        </DialogHeader>
        {a && b ? (
          mode === "slider" ? (
            <div className="flex min-h-0 flex-1 items-center justify-center">
              <CompareSlider
                before={api.fileUrl(a.id)}
                after={api.fileUrl(b.id)}
                beforeLabel={t("A")}
                afterLabel={t("B")}
                aspect={a.width && a.height ? a.width / a.height : undefined}
                fitHeight="76dvh"
              />
            </div>
          ) : (
            <div className="grid min-h-0 flex-1 grid-cols-2 gap-2">
              {[a, b].map((image, index) => (
                <figure key={image.id} className="flex min-h-0 flex-col gap-1">
                  <img src={api.fileUrl(image.id)} alt={image.name} className="min-h-0 flex-1 rounded-lg bg-checker object-contain" />
                  <figcaption className="truncate text-center text-xs text-muted-foreground">
                    {index === 0 ? t("A") : t("B")} · {image.name}
                    {image.width && image.height ? ` · ${image.width}×${image.height}` : ""}
                  </figcaption>
                </figure>
              ))}
            </div>
          )
        ) : null}
      </DialogContent>
    </Dialog>
  )
}
