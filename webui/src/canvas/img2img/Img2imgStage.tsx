/**
 * The img2img stage: a Source / Result switch over the image editor bound
 * to the img2img form, or the results. Batch modes have no canvas: the
 * stage shows the results only.
 */
import { Images, PencilRuler } from "@/components/icons"
import { type ReactNode, useEffect } from "react"
import { useTranslation } from "react-i18next"
import { cn } from "@/lib/utils"
import { ImageEditor } from "../ImageEditor"
import { preloadImageEditor } from "../preload"
import { useImg2imgStageView } from "./stage-view"
import { IMG2IMG, setCoreValues, setSizeFromImage, setSourceImage, useImg2imgBinding, useImg2imgEditorProps, useModeLayers } from "./use-img2img"

function ViewSwitch({ view, onChange }: { view: "source" | "result"; onChange: (view: "source" | "result") => void }) {
  const { t } = useTranslation()
  const options = [
    { value: "source" as const, label: t("Source"), icon: PencilRuler },
    { value: "result" as const, label: t("Result"), icon: Images },
  ]
  return (
    <div role="tablist" aria-label={t("Image view")} className="inline-flex shrink-0 items-center gap-0.5 rounded-lg bg-muted p-0.5">
      {options.map(({ value, label, icon: Icon }) => (
        <button
          key={value}
          type="button"
          role="tab"
          aria-selected={view === value}
          // Only the shown view's panel is mounted.
          aria-controls={view === value ? `img2img-stage-${value}` : undefined}
          id={`img2img-stage-tab-${value}`}
          onClick={() => onChange(value)}
          onKeyDown={(event) => {
            if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
              event.preventDefault()
              const next = value === "source" ? "result" : "source"
              onChange(next)
              document.getElementById(`img2img-stage-tab-${next}`)?.focus()
            }
          }}
          tabIndex={view === value ? 0 : -1}
          className={cn(
            "flex h-6 items-center gap-1.5 rounded-md px-2.5 text-xs font-medium whitespace-nowrap text-muted-foreground transition-colors outline-none select-none",
            "hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring/60",
            "aria-selected:bg-background aria-selected:text-foreground aria-selected:shadow-sm dark:aria-selected:bg-accent",
          )}
        >
          <Icon className="size-3.5" aria-hidden />
          {label}
        </button>
      ))}
    </div>
  )
}

type Img2imgStageProps = {
  /** The result view (image and parameters). */
  result: ReactNode
  /** The result's actions, shown in the header with the result view. */
  actions?: ReactNode
  /** Header items before / after the switch (pane toggles, the run bar). */
  leading?: ReactNode
  trailing?: ReactNode
  className?: string
}

export function Img2imgStage({ result, actions, leading, trailing, className }: Img2imgStageProps) {
  const { t } = useTranslation()
  const binding = useImg2imgBinding()
  const layers = useModeLayers(binding)
  const editor = useImg2imgEditorProps(binding, layers)
  const [view, setView] = useImg2imgStageView(binding ? [binding.background, ...binding.layers.map((layer) => layer.param)] : undefined)
  const canvas = Boolean(binding && layers.canvas)
  const shown = canvas ? view : "result"

  // The editor chunk is fetched with the page, so switching to Source is instant.
  useEffect(() => {
    void preloadImageEditor()
  }, [])

  return (
    <div className={cn("flex min-h-0 flex-col gap-2", className)}>
      <div className="flex min-h-7 flex-wrap items-center gap-1">
        {leading}
        {canvas ? <ViewSwitch view={shown} onChange={setView} /> : null}
        {shown === "result" && actions ? actions : <div className="min-w-0 flex-1" />}
        {trailing}
      </div>
      {shown === "source" && binding ? (
        <div role="tabpanel" id="img2img-stage-source" aria-labelledby="img2img-stage-tab-source" className="flex min-h-0 flex-1 flex-col">
          <ImageEditor
            docKey={IMG2IMG}
            pendingKey={IMG2IMG}
            label={t("Source image")}
            className="min-h-64 flex-1"
            fill
            {...editor}
            onImageChange={(image) => {
              void setSourceImage(binding, image?.id ?? null, image)
            }}
            onLayerChange={(key, id) => setCoreValues({ [key]: id })}
            onUseImageSize={setSizeFromImage}
          />
        </div>
      ) : (
        <div role={canvas ? "tabpanel" : undefined} id="img2img-stage-result" aria-labelledby={canvas ? "img2img-stage-tab-result" : undefined} className="flex min-h-0 flex-1 flex-col">
          {result}
        </div>
      )}
    </div>
  )
}
