/**
 * The img2img source block of the parameters pane: the mode picker (tiles
 * in one row or balanced rows, never a wrapping strip of tabs), a compact line
 * for the source image (thumbnail, size, use size, replace, remove, edit on
 * the stage) and the uploaded mask, and the mode's remaining parameters
 * rendered by the schema renderer (batch inputs, resize mode, inpaint
 * options, the scripts of these categories). The image itself is edited on
 * the stage.
 */
import { Radio } from "@base-ui/react/radio"
import { RadioGroup } from "@base-ui/react/radio-group"
import { Brush, ImageIcon, ImageUp, Images, Palette, PaintbrushVertical, PencilRuler, Proportions, Upload, X, type IconComponent } from "@/components/icons"
import { type CSSProperties, type ReactNode, useEffect, useId, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { evaluate } from "@/ui/conditions"
import { OwnerProvider } from "@/ui/context"
import { useForm, useFormState, useParam } from "@/ui/hooks"
import { RenderChildren, RenderNode } from "@/ui/Renderer"
import type { InputNode, Node, SliderNode, TabsNode } from "@/ui/schema"
import { imageFiles, releaseUpload, useUploader, useUploadInfo } from "@/ui/uploads"
import type { EditorBinding } from "../core/modes"
import { scaledSize } from "../core/overlays"
import { useImageSize } from "./image-size"
import { pruneSource } from "./source-layout"
import { useStageView } from "./stage-view"
import { setCoreValues, setSizeFromImage, setSourceImage, useImg2imgBinding, useModeLayers } from "./use-img2img"

/**
 * The core form's modes (`remi.core.forms.IMG2IMG_MODES`): icon, chrome label
 * (the tiles are app chrome, translated without the server catalog) and a
 * short description shown under the picker. Unknown modes use the tab title.
 */
const MODES: Record<string, { icon: IconComponent; label: string; description: string }> = {
  img2img: { icon: ImageIcon, label: "img2img", description: "Redraws the whole image." },
  Sketch: { icon: Brush, label: "Sketch", description: "Paint colours over the image; the result is redrawn." },
  Inpaint: { icon: PaintbrushVertical, label: "Inpaint", description: "Paint a mask; only the masked area is redrawn." },
  "Inpaint sketch": { icon: Palette, label: "Inpaint sketch", description: "Paint colours; the painted area is redrawn with them." },
  "Inpaint upload": { icon: ImageUp, label: "Inpaint upload", description: "Upload a mask image; its white (or opaque) areas are redrawn." },
  Batch: { icon: Images, label: "Batch", description: "Processes many images with the same settings." },
}

/** Columns of the mode grid: one row when there is room, otherwise balanced rows (3 + 3, not 4 + 2). */
function modeColumns(count: number): CSSProperties {
  return { "--cols": count > 4 ? Math.ceil(count / 2) : count, "--cols-wide": count } as CSSProperties
}

function ModePicker({ tabs, modeNode }: { tabs: TabsNode; modeNode: InputNode }) {
  const { t } = useTranslation()
  const param = useParam<string>(modeNode)
  const labelId = useId()
  const visible = useFormState((s) => tabs.children.map((tab) => (evaluate(tab.visible_if, "core", (o, n) => s.values[o]?.[n]) ? "1" : "0")).join(""))
  const options = tabs.children.filter((_, i) => visible[i] === "1" && tabs.children[i]?.value !== undefined)
  const hint = MODES[String(param.value)]?.description
  return (
    <div className="@container flex flex-col gap-2">
      <span id={labelId} className="sr-only">
        {t("Mode")}
      </span>
      <RadioGroup
        value={param.value}
        onValueChange={(value) => param.setValue(value as string)}
        aria-labelledby={labelId}
        disabled={!param.enabled}
        style={modeColumns(options.length)}
        className="grid grid-cols-[repeat(var(--cols),minmax(0,1fr))] gap-1 rounded-md bg-muted p-1 @[34rem]:grid-cols-[repeat(var(--cols-wide),minmax(0,1fr))]"
      >
        {options.map((tab) => {
          const value = String(tab.value)
          const mode = MODES[value]
          const Icon = mode?.icon ?? ImageIcon
          const label = t(mode?.label ?? tab.title)
          return (
            <Radio.Root
              key={value}
              value={value}
              nativeButton
              render={<button type="button" title={label} />}
              className={cn(
                "group/mode flex min-w-0 flex-col items-center justify-start gap-1 rounded-lg px-1 py-1.5 text-muted-foreground transition-colors outline-none select-none",
                "hover:bg-background/60 hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring/60",
                "data-checked:bg-background data-checked:text-foreground data-checked:shadow-sm dark:data-checked:bg-accent",
              )}
            >
              <Icon className="size-4 shrink-0 group-data-checked/mode:text-primary" aria-hidden />
              <span className="line-clamp-2 w-full text-center text-xs leading-tight font-medium break-words">{label}</span>
            </Radio.Root>
          )
        })}
      </RadioGroup>
      {hint ? <p className="px-0.5 text-xs text-muted-foreground">{t(hint)}</p> : null}
    </div>
  )
}

/** A thumbnail with the image's real size (read once it loads). */
function Thumb({ id, onSize }: { id: string; onSize?: (size: { width: number; height: number }) => void }) {
  const info = useUploadInfo(id)
  return info?.url ? (
    <img
      src={info.url}
      alt=""
      draggable={false}
      onLoad={(event) => onSize?.({ width: event.currentTarget.naturalWidth, height: event.currentTarget.naturalHeight })}
      className="size-full object-cover"
    />
  ) : null
}

type ImageRowProps = {
  label: string
  id: string | null
  error?: string
  emptyText: string
  /** Mask previews sit on a checkerboard. */
  mask?: boolean
  onFile: (file: File) => Promise<void> | void
  onRemove: () => void
  busy: boolean
  extra?: (size: { width: number; height: number } | null) => ReactNode
}

/** One image of the source: thumbnail, size, actions; a drop target for a replacement. */
function ImageRow({ label, id, error, emptyText, mask, onFile, onRemove, busy, extra }: ImageRowProps) {
  const { t } = useTranslation()
  const input = useRef<HTMLInputElement>(null)
  const [size, setSize] = useState<{ id: string; width: number; height: number } | null>(null)
  const [dragging, setDragging] = useState(false)
  const errorId = useId()
  const known = size && size.id === id ? size : null
  return (
    <div
      className={cn("flex items-center gap-2.5 rounded-lg border border-transparent p-1 transition-colors", dragging && "border-dashed border-primary bg-primary/5", error && "border-destructive/50")}
      onDragOver={(event) => {
        if (!event.dataTransfer.types.includes("Files")) return
        event.preventDefault()
        setDragging(true)
      }}
      onDragLeave={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget as globalThis.Node | null)) setDragging(false)
      }}
      onDrop={(event) => {
        event.preventDefault()
        setDragging(false)
        const [file] = imageFiles(event.dataTransfer.files)
        if (file) void onFile(file)
      }}
    >
      <div className={cn("grid size-12 shrink-0 place-items-center overflow-hidden rounded-md border", mask ? "bg-checker" : "bg-muted")}>
        {id ? <Thumb id={id} onSize={(s) => setSize({ id, ...s })} /> : <ImageIcon className="size-4 text-muted-foreground" aria-hidden />}
      </div>
      <div className="flex min-w-0 flex-1 flex-col">
        <span className="truncate text-sm font-medium">{label}</span>
        <span id={errorId} className={cn("truncate text-xs tabular-nums", error ? "text-destructive" : "text-muted-foreground")} role={error ? "alert" : undefined}>
          {error ? t(error) : busy ? t("Uploading…") : id ? (known ? `${known.width}×${known.height}` : "…") : emptyText}
        </span>
      </div>
      <div className="flex shrink-0 items-center">
        {extra?.(known)}
        <Button variant="ghost" size="icon-sm" aria-label={id ? t("Replace {{label}}", { label }) : t("Upload {{label}}", { label })} title={id ? t("Replace {{label}}", { label }) : t("Upload {{label}}", { label })} aria-describedby={error ? errorId : undefined} onClick={() => input.current?.click()} disabled={busy}>
          {id ? <ImageUp /> : <Upload />}
        </Button>
        {id ? (
          <Button variant="ghost" size="icon-sm" aria-label={t("Remove {{label}}", { label })} title={t("Remove {{label}}", { label })} onClick={onRemove}>
            <X />
          </Button>
        ) : null}
      </div>
      <input
        ref={input}
        type="file"
        accept="image/*"
        hidden
        onChange={(event) => {
          const [file] = imageFiles(event.target.files)
          event.target.value = ""
          if (file) void onFile(file)
        }}
      />
    </div>
  )
}

/** Shows the stage's editor, scrolled into view where the stage sits below the parameters (phones). */
function showEditor(setView: (view: "source") => void) {
  setView("source")
  requestAnimationFrame(() => document.getElementById("img2img-stage-source")?.scrollIntoView({ behavior: "smooth", block: "nearest" }))
}

function SourceImages({ binding }: { binding: EditorBinding }) {
  const { t } = useTranslation()
  const layers = useModeLayers(binding)
  // One per row: uploading the mask does not mark the source as uploading.
  const source = useUploader()
  const mask = useUploader()
  const image = useFormState((s) => (s.values.core?.[binding.background] as string | null | undefined) ?? null)
  const imageError = useFormState((s) => s.errors.core?.[binding.background])
  const active = layers.active
  const layerValue = useFormState((s) => (active ? ((s.values.core?.[active.param] as string | null | undefined) ?? null) : null))
  const layerError = useFormState((s) => (active ? s.errors.core?.[active.param] : undefined))
  const setView = useStageView((s) => s.setView)

  const uploadTo = async (uploader: ReturnType<typeof useUploader>, file: File, apply: (id: string, size: { width: number; height: number } | null) => void) => {
    const [first] = await uploader.upload([file])
    if (first) apply(first.id, first.width && first.height ? { width: first.width, height: first.height } : null)
  }

  return (
    <div className="flex flex-col gap-1">
      <ImageRow
        label={t("Source image")}
        id={image}
        error={imageError}
        emptyText={t("Drop an image here or on the stage")}
        busy={source.busy}
        onFile={(file) => uploadTo(source, file, (id, size) => void setSourceImage(binding, id, size))}
        onRemove={() => void setSourceImage(binding, null)}
        extra={(size) => (
          <>
            <Button variant="ghost" size="icon-sm" aria-label={t("Edit on the stage")} title={t("Edit on the stage")} onClick={() => showEditor(setView)}>
              <PencilRuler />
            </Button>
            {size ? (
              <Button
                variant="ghost"
                size="icon-sm"
                aria-label={t("Use the image size ({{width}}×{{height}})", size)}
                title={t("Use the image size ({{width}}×{{height}})", size)}
                onClick={() => setSizeFromImage(size)}
              >
                <Proportions />
              </Button>
            ) : null}
          </>
        )}
      />
      {active && !active.painted ? (
        <ImageRow
          label={t("Mask")}
          id={layerValue}
          error={layerError}
          mask
          emptyText={t("White (or opaque) areas are redrawn")}
          busy={mask.busy}
          onFile={(file) =>
            uploadTo(mask, file, (id) => {
              const old = layerValue
              setCoreValues({ [active.param]: id })
              if (old) releaseUpload(old)
            })
          }
          onRemove={() => {
            const old = layerValue
            setCoreValues({ [active.param]: null })
            if (old) releaseUpload(old)
          }}
        />
      ) : active ? (
        <p className={cn("flex items-center gap-1.5 px-1 text-xs", layerError ? "text-destructive" : "text-muted-foreground")} role={layerError ? "alert" : undefined}>
          {layerError ? t(layerError) : active.kind === "mask" ? t("Paint the mask on the stage.") : t("Paint on the stage.")}
          {layerError ? (
            <Button variant="link" size="xs" className="h-auto p-0" onClick={() => showEditor(setView)}>
              {t("Open the editor")}
            </Button>
          ) : null}
        </p>
      ) : null}
    </div>
  )
}

/**
 * Fills the core form's `scale_by_resolution` output (under "Resize by"):
 * the source size and the processing size it gives, snapped like the server.
 */
function useScaleByResolution(binding: EditorBinding | null) {
  const { t } = useTranslation()
  const { store } = useForm()
  const image = useFormState((s) => (binding ? ((s.values.core?.[binding.background] as string | null | undefined) ?? null) : null))
  const scaleBy = useFormState((s) => Number(s.values.core?.scale_by ?? 1))
  const step = useFormState((s) => s.overrides.core?.width?.step ?? (s.index.inputs.get("core")?.get("width") as SliderNode | undefined)?.step ?? 8)
  const revision = useFormState((s) => s.revision)
  const size = useImageSize(image)
  const target = size ? scaledSize(size, scaleBy, step) : null
  const text = size && target ? t("Resize from {{from}} to {{to}}", { from: `${size.width}×${size.height}`, to: `${target.width}×${target.height}` }) : ""
  useEffect(() => {
    if (store.getState().index.inputs.get("core")?.has("scale_by")) store.getState().setOutputs("core", { scale_by_resolution: text })
  }, [store, text, revision])
}

/** The source card's content; renders inside the form's providers (core owner). */
export function Img2imgSourcePanel({ nodes }: { nodes: readonly Node[] }) {
  const binding = useImg2imgBinding()
  useScaleByResolution(binding)
  const layers = useModeLayers(binding)
  const handled = new Set(binding ? [binding.background, ...binding.layers.map((layer) => layer.param)] : [])
  const found: { tabs: TabsNode | null } = { tabs: null }
  const modeParam = "mode"
  const rest = pruneSource(nodes, handled, modeParam, found)
  const modeNode = useFormState((s) => s.index.inputs.get("core")?.get(modeParam) ?? null)
  const mode = useFormState((s) => s.values.core?.[modeParam])
  const selected = found.tabs?.children.find((tab) => tab.value === mode) ?? null

  return (
    <OwnerProvider owner="core">
      <div className="flex flex-col gap-3">
        {found.tabs && modeNode ? <ModePicker tabs={found.tabs} modeNode={modeNode} /> : null}
        {binding && layers.canvas ? <SourceImages binding={binding} /> : null}
        {selected && selected.children.length > 0 ? (
          <div className="flex flex-col gap-3">
            <RenderChildren nodes={selected.children} />
          </div>
        ) : null}
        {rest.map((node, index) => (
          <RenderNode key={index} node={node} />
        ))}
      </div>
    </OwnerProvider>
  )
}
