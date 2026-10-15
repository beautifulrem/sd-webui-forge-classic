/** The options of the current tool (brush size, opacity, …) and the layer actions. */
import { Blend, CircleDot, Contrast, Feather, SlidersHorizontal, Trash2 } from "lucide-react"
import { useTranslation } from "react-i18next"
import { Segmented } from "@/components/Segmented"
import { cn } from "@/lib/utils"
import type { LayerKind } from "../core/modes"
import { sizeToSlider, sliderToSize, type Tool, useTools } from "../state/tools"
import { ColorControl } from "./ColorControl"
import { Divider, IconButton, PANEL, ValueSlider } from "./controls"

type OptionsBarProps = {
  tool: Tool
  kind: LayerKind
  /** Colour fixed by the embedding widget (no picker). */
  fixedColor?: string
  fixedSize?: number
  fixedOpacity?: number
  fixedSoftness?: number
  canClear: boolean
  /** The shortcut modifier's name (Ctrl or ⌘). */
  mod: string
  onInvert: () => void
  onClear: () => void
  className?: string
}

const percent = (value: number) => `${Math.round(value * 100)}%`

export function OptionsBar({ tool, kind, fixedColor, fixedSize, fixedOpacity, fixedSoftness, canClear, mod, onInvert, onClear, className }: OptionsBarProps) {
  const { t } = useTranslation()
  const tools = useTools()
  const brush = tool === "eraser" ? tools.eraser : tools.brush
  const setBrush = tool === "eraser" ? tools.setEraser : tools.setBrush
  const sketch = kind === "sketch"
  const showColor = sketch && !fixedColor && (tool === "brush" || tool === "fill" || tool === "rect" || tool === "lasso")
  const showSize = (tool === "brush" || tool === "eraser") && fixedSize === undefined

  let hint: string | null = null
  if (tool === "rect" || tool === "lasso") hint = kind === "mask" ? t("Drag to mask an area; hold Shift to unmask") : t("Drag to fill an area; hold Shift to erase")
  else if (tool === "eyedropper") hint = t("Click the image to pick a color")
  else if (tool === "hand") hint = t("Drag to pan; scroll to zoom")

  return (
    <div role="toolbar" aria-label={t("Tool options")} className={cn(PANEL, "flex h-10 max-w-full items-center gap-1.5 overflow-x-auto px-1.5 [scrollbar-width:none] @md:gap-2", className)}>
      {showColor ? (
        <ColorControl color={tools.color} recent={tools.recent} onChange={tools.setColor} onEyedropper={() => tools.setTool("eyedropper")} />
      ) : null}
      {showSize ? (
        <ValueSlider
          label={t("Size")}
          icon={<CircleDot />}
          value={sizeToSlider(brush.size)}
          min={0}
          max={100}
          step={0.5}
          onChange={(value) => setBrush({ size: sliderToSize(value) })}
          format={() => `${brush.size}`}
          valueText={() => t("{{size}} px", { size: brush.size })}
        />
      ) : null}
      {sketch && fixedOpacity === undefined && (tool === "brush" || tool === "eraser" || tool === "fill" || tool === "rect" || tool === "lasso") ? (
        <ValueSlider
          label={t("Opacity")}
          icon={<Blend />}
          value={(tool === "eraser" ? tools.eraser : tools.brush).opacity}
          min={0.01}
          max={1}
          step={0.01}
          onChange={(value) => (tool === "eraser" ? tools.setEraser({ opacity: value }) : tools.setBrush({ opacity: value }))}
          format={percent}
          valueText={percent}
        />
      ) : null}
      {(tool === "brush" || tool === "eraser") && fixedSoftness === undefined ? (
        <ValueSlider label={t("Softness")} icon={<Feather />} value={brush.softness} min={0} max={1} step={0.01} onChange={(value) => setBrush({ softness: value })} format={percent} />
      ) : null}
      {tool === "fill" ? (
        <>
          <ValueSlider label={t("Tolerance")} icon={<SlidersHorizontal />} value={tools.fillTolerance} min={0} max={255} onChange={(value) => tools.set({ fillTolerance: value })} format={(v) => String(Math.round(v))} />
          <Segmented
            size="sm"
            aria-label={t("Fill by the colors of")}
            value={tools.fillSample}
            onValueChange={(value) => tools.set({ fillSample: value })}
            options={[
              { value: "layer", label: t("Layer") },
              { value: "image", label: t("Image") },
            ]}
            className="shrink-0 flex-nowrap"
          />
        </>
      ) : null}
      {hint ? <span className="hidden shrink-0 px-1 text-xs whitespace-nowrap text-muted-foreground @lg:inline">{hint}</span> : null}
      <div className="ml-auto flex shrink-0 items-center">
        <Divider />
        {kind === "mask" ? (
          <IconButton label={t("Invert mask")} keys={[mod, "I"]} onClick={onInvert} side="bottom">
            <Contrast />
          </IconButton>
        ) : null}
        <IconButton label={kind === "mask" ? t("Clear mask") : t("Clear sketch")} keys={[mod, "X"]} onClick={onClear} disabled={!canClear} side="bottom">
          <Trash2 />
        </IconButton>
      </div>
    </div>
  )
}
