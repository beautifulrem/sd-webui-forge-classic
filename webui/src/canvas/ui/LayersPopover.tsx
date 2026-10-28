/** Layer visibility and opacity, mask colour, previews and input preferences. */
import { Radio } from "@base-ui/react/radio"
import { RadioGroup } from "@base-ui/react/radio-group"
import { Eye, EyeOff, ImageIcon, Layers } from "@/components/icons"
import { useId, useState } from "react"
import { useTranslation } from "react-i18next"
import { Segmented } from "@/components/Segmented"
import { Button } from "@/components/ui/button"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { Switch } from "@/components/ui/switch"
import { cn } from "@/lib/utils"
import type { LayerKind } from "../core/modes"
import { MASK_COLOR_NAMES, MASK_COLORS, useTools } from "../state/tools"
import { Hint, ValueSlider } from "./controls"

function VisibilityToggle({ visible, onChange, label }: { visible: boolean; onChange: (visible: boolean) => void; label: string }) {
  const { t } = useTranslation()
  return (
    <Button variant="ghost" size="icon-xs" aria-pressed={visible} aria-label={visible ? t("Hide {{layer}}", { layer: label }) : t("Show {{layer}}", { layer: label })} onClick={() => onChange(!visible)}>
      {visible ? <Eye /> : <EyeOff className="text-muted-foreground" />}
    </Button>
  )
}

function SwitchRow({ label, checked, onChange }: { label: string; checked: boolean; onChange: (checked: boolean) => void }) {
  const id = useId()
  return (
    <div className="flex items-center justify-between gap-3">
      <label htmlFor={id} className="text-sm">
        {label}
      </label>
      <Switch id={id} size="sm" checked={checked} onCheckedChange={onChange} />
    </div>
  )
}

type LayersPopoverProps = {
  /** The edited layer (null: none in this mode). */
  layer: { kind: LayerKind; label: string } | null
  /** Inpaint previews apply (an inpaint mode). */
  inpaint: boolean
  onlyMasked: boolean
}

export function LayersPopover({ layer, inpaint, onlyMasked }: LayersPopoverProps) {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)
  const colorLabel = useId()
  const tools = useTools()
  const mask = layer?.kind === "mask"
  const opacity = mask ? tools.maskOpacity : tools.sketchOpacity

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <Hint label={t("Layers and view")} disabled={open}>
        <PopoverTrigger render={<Button variant="ghost" size="icon-sm" aria-label={t("Layers and view")} />}>
          <Layers />
        </PopoverTrigger>
      </Hint>
      <PopoverContent aria-label={t("Layers and view")} side="top" align="end" className="w-72 gap-3">
        <section aria-label={t("Layers")} className="flex flex-col gap-1">
          <h3 className="text-xs font-medium text-muted-foreground">{t("Layers")}</h3>
          {layer ? (
            <div className="flex flex-col gap-2 rounded-lg border bg-card p-2">
              <div className="flex items-center gap-2">
                <span
                  aria-hidden
                  className={cn("size-3.5 rounded-sm border border-black/10", !mask && "bg-checker")}
                  style={mask ? { backgroundColor: tools.maskColor } : undefined}
                />
                <span className="text-sm font-medium">{layer.label}</span>
                <span className="ml-auto" />
                <VisibilityToggle visible={tools.layerVisible} label={layer.label} onChange={(visible) => tools.set({ layerVisible: visible })} />
              </div>
              <ValueSlider
                label={t("Opacity")}
                value={opacity}
                min={0.05}
                max={1}
                step={0.01}
                onChange={(value) => tools.set(mask ? { maskOpacity: value } : { sketchOpacity: value })}
                format={(v) => `${Math.round(v * 100)}%`}
                className="[&_[data-slot=slider]]:flex-1"
              />
              {mask ? (
                <div className="flex items-center gap-1.5">
                  <span id={colorLabel} className="mr-1 text-xs text-muted-foreground">
                    {t("Color")}
                  </span>
                  <RadioGroup aria-labelledby={colorLabel} value={tools.maskColor} onValueChange={(value) => tools.set({ maskColor: value as string })} className="flex items-center gap-1.5">
                    {MASK_COLORS.map((color) => (
                      <Radio.Root
                        key={color}
                        value={color}
                        nativeButton
                        render={<button type="button" title={t(MASK_COLOR_NAMES[color])} />}
                        aria-label={t(MASK_COLOR_NAMES[color])}
                        className={cn(
                          "size-5 rounded-full border border-black/15 outline-none focus-visible:ring-3 focus-visible:ring-ring/50 dark:border-white/20",
                          "data-checked:ring-2 data-checked:ring-primary data-checked:ring-offset-1 data-checked:ring-offset-popover",
                        )}
                        style={{ backgroundColor: color }}
                      />
                    ))}
                  </RadioGroup>
                </div>
              ) : null}
            </div>
          ) : null}
          <div className="flex items-center gap-2 rounded-lg border bg-card p-2">
            <ImageIcon aria-hidden className="size-3.5 text-muted-foreground" />
            <span className="text-sm font-medium">{t("Image")}</span>
            <span className="ml-auto" />
            <VisibilityToggle visible={tools.imageVisible} label={t("Image")} onChange={(visible) => tools.set({ imageVisible: visible })} />
          </div>
        </section>
        {inpaint ? (
          <section aria-label={t("Previews")} className="flex flex-col gap-2">
            <h3 className="text-xs font-medium text-muted-foreground">{t("Previews")}</h3>
            {mask ? <SwitchRow label={t("Preview mask blur")} checked={tools.previewBlur} onChange={(previewBlur) => tools.set({ previewBlur })} /> : null}
            {onlyMasked ? <SwitchRow label={t("Show the inpaint area")} checked={tools.previewArea} onChange={(previewArea) => tools.set({ previewArea })} /> : null}
          </section>
        ) : null}
        <section aria-label={t("Input")} className="flex flex-col gap-2">
          <h3 className="text-xs font-medium text-muted-foreground">{t("Input")}</h3>
          <div className="flex items-center justify-between gap-3">
            <span className="text-sm">{t("Mouse wheel")}</span>
            <Segmented
              size="sm"
              aria-label={t("Mouse wheel")}
              value={tools.wheel}
              onValueChange={(wheel) => tools.set({ wheel })}
              options={[
                { value: "zoom", label: t("Zoom") },
                { value: "pan", label: t("Pan") },
              ]}
              className="shrink-0 flex-nowrap"
            />
          </div>
          <SwitchRow label={t("Pen pressure sets the brush size")} checked={tools.pressure} onChange={(pressure) => tools.set({ pressure })} />
        </section>
      </PopoverContent>
    </Popover>
  )
}
