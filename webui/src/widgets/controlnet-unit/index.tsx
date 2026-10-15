/**
 * `controlnet-unit`: one ControlNet unit (the `unit_<i>` params of the
 * ControlNet script, `lib_controlnet/unit_param.py`). Replaces the Gradio
 * unit group: the control image in the image editor (upload, blank canvas,
 * mask and colour sketch layers, "use image size"), "Use Mask" (a mask image
 * and a painted mask), the preprocessor preview (run on the server, shown in
 * the unit, usable as the input), the OpenPose JSON editor, the "Region"
 * canvas generator, control type → preprocessor / model filtering with a
 * model refresh, the preprocessor's own sliders, weight, timestep range
 * (the shared `range-slider` widget), control / resize / Hires. fix modes,
 * and a line of the enabled units (the old tab badges; `summary.ts` puts
 * their count in the section header, the old accordion badge). A new control
 * image switches the unit on, as the old upload did.
 *
 * value: the unit JSON (see `unit.ts`; the server validates it again).
 * props: `UnitProps` (unit index, tab, labels and choices, the catalog of
 *        preprocessors, models and control types).
 */
import { Download, ImageIcon, LoaderCircle, RefreshCw, Shapes, X, Zap } from "lucide-react"
import { useEffect, useId, useState } from "react"
import { useTranslation } from "react-i18next"
import { api } from "@/api/endpoints"
import type { ActionExtra } from "@/api/types"
import { ImageEditor } from "@/canvas"
import { SearchSelect } from "@/components/SearchSelect"
import { Segmented } from "@/components/Segmented"
import { Button } from "@/components/ui/button"
import { Switch } from "@/components/ui/switch"
import { cn } from "@/lib/utils"
import { runChangeEffects } from "@/ui/actions"
import { Field } from "@/ui/Field"
import { tabsKey } from "@/ui/form-index"
import { useForm, useFormState, useOwner, useRunAction } from "@/ui/hooks"
import type { TabsNode } from "@/ui/schema"
import type { WidgetProps } from "@/ui/widgets"
import RangeSlider from "../range-slider"
import { NS } from "./i18n"
import { catalogOf, watchPastes } from "./paste"
import { PosePanel } from "./PosePanel"
import { SliderRow } from "./SliderRow"
import {
  applyEdit,
  type Catalog,
  closestEight,
  type Control,
  defaultUnit,
  isPasted,
  mergePasted,
  modelsFor,
  modulesFor,
  normalizeUnit,
  ownsImage,
  preprocessorOf,
  selectModule,
  selectType,
  type SliderSpec,
  sliderValue,
  type UnitProps,
  type UnitValue,
} from "./unit"

type Layer = "image_fg" | "sketch"
type Region = { subjects: number; vertical: number; randomize: number; shape: string }

const EMPTY_CATALOG: Catalog = { preprocessors: [], models: ["None"], types: [] }

function numberOr(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback
}

function regionDefaults(controls: Record<string, Control>): Region {
  const of = (key: string, fallback: number) => numberOr(controls[`region_${key}`]?.default, fallback)
  return { subjects: of("subjects", 0), vertical: of("vertical", 0.4), randomize: of("randomize", 0.075), shape: String(controls.region_shape?.default ?? "Rectangle") }
}

/** A labelled switch of the unit's header row. */
function UnitSwitch(props: { label: string; checked: boolean; onChange: (checked: boolean) => void; disabled?: boolean }) {
  const id = useId()
  return (
    <div className="flex min-h-7 items-center gap-2">
      <Switch id={id} checked={props.checked} onCheckedChange={props.onChange} disabled={props.disabled} />
      <label htmlFor={id} className="text-sm leading-5 select-none">
        {props.label}
      </label>
    </div>
  )
}

/** The enabled units of this script, as buttons switching the unit tab (the old tab badges). */
function ActiveUnits({ titles, current, onSelect }: { titles: string[]; current: number; onSelect: (index: number) => void }) {
  const { t } = useTranslation()
  const { t: tw } = useTranslation(NS)
  const owner = useOwner()
  const summary = useFormState((s) =>
    titles
      .map((_, i) => {
        const unit = s.values[owner]?.[`unit_${i}`] as Partial<UnitValue> | undefined
        return unit?.enabled ? `${i}\u0000${typeof unit.type_filter === "string" ? unit.type_filter : "All"}` : ""
      })
      .join("\u0001"),
  )
  const active = summary
    .split("\u0001")
    .filter(Boolean)
    .map((entry) => {
      const [index, type] = entry.split("\u0000")
      return { index: Number(index), type: type ?? "All" }
    })
  if (titles.length < 2 || active.length === 0) return null
  return (
    <div className="flex flex-wrap items-center gap-1.5 text-xs text-muted-foreground">
      <span>{tw("Enabled units:")}</span>
      {active.map(({ index, type }) => (
        <Button
          key={index}
          size="xs"
          variant={index === current ? "secondary" : "ghost"}
          aria-current={index === current ? "true" : undefined}
          onClick={() => onSelect(index)}
        >
          <span aria-hidden className="size-1.5 rounded-full bg-success" />
          {t(titles[index] ?? "")}
          {type !== "All" ? <span className="text-muted-foreground">· {t(type)}</span> : null}
        </Button>
      ))}
    </div>
  )
}

export default function ControlNetUnitWidget({ value, onChange, props, disabled, id, node }: WidgetProps<unknown, UnitProps>) {
  const { t } = useTranslation()
  const { t: tw } = useTranslation(NS)
  const { store, formId } = useForm()
  const owner = useOwner()
  const run = useRunAction()
  const img2img = props.img2img === true
  const controls = props.controls ?? {}
  const catalog = useFormState((s) => catalogOf(s, owner, props)) ?? EMPTY_CATALOG
  const unit = normalizeUnit(value, img2img)
  const width = useFormState((s) => numberOr(s.values.core?.width, 512))
  const height = useFormState((s) => numberOr(s.values.core?.height, 512))
  const hires = useFormState((s) => s.values.core?.enable_hr === true)
  const [busy, setBusy] = useState<string | null>(null)
  const [layerChoice, setLayerChoice] = useState<Layer | null>(null)
  const [region, setRegion] = useState<Region>(() => regionDefaults(controls))
  const titles = props.tabs ?? []
  const stripKey = tabsKey(owner, { children: titles.map((title) => ({ title })) } as TabsNode)

  useEffect(() => watchPastes(store), [store])
  // A paste written before this widget was loaded has no unit left to merge into
  useEffect(() => {
    if (isPasted(value)) onChange(mergePasted(defaultUnit(img2img), value, catalog))
  }, [value, onChange, img2img, catalog])

  const latest = () => normalizeUnit(store.getState().values[owner]?.[node.name], img2img)
  const commit = (patch: Partial<UnitValue>) => onChange(applyEdit(latest(), patch))
  const label = (key: string, fallback: string) => t(controls[key]?.label ?? fallback)

  const call = async (action: string, extra: Record<string, unknown> = {}) => {
    setBusy(action)
    try {
      return await run(action, { unit: props.unit, ...extra } as unknown as ActionExtra)
    } finally {
      setBusy(null)
    }
  }
  const output = <T,>(split: Awaited<ReturnType<typeof call>>, key: string) => {
    const out = split?.outputs?.[key] as (T & { unit?: number }) | undefined
    return out && out.unit === props.unit ? out : null
  }

  const runPreview = async () => {
    const out = output<{ image: string; pose: unknown }>(await call("preview"), "preview")
    if (out) onChange({ ...latest(), generated_image: out.image, pose: out.pose ?? null })
  }
  const applyPose = async (pose: unknown) => {
    const out = output<{ image: string; pose: unknown }>(await call("render_pose", { pose }), "pose")
    if (out) onChange({ ...latest(), generated_image: out.image, pose: out.pose, use_preview_as_input: true })
  }
  const generateCanvas = async () => {
    const out = output<{ image: string }>(await call("region_canvas", region), "canvas")
    if (out) commit({ image: out.image, sketch: null, image_fg: null })
  }
  const useImageSize = (size: { width: number; height: number }) => {
    const state = store.getState()
    state.setValue("core", "width", closestEight(size.width))
    state.setValue("core", "height", closestEight(size.height))
    runChangeEffects({ store, tab: formId }, "core", ["width", "height"])
  }

  const preprocessor = preprocessorOf(catalog, unit.module)
  const owns = ownsImage(unit, img2img)
  const isRegion = unit.type_filter === "Region"
  const layer: Layer = layerChoice ?? (isRegion ? "sketch" : "image_fg")
  const doc = `${formId}:${owner}:${node.name}`
  const frame = { width, height, resizeMode: unit.resize_mode }
  // preprocessor and model names are technical; "None" is a word
  const option = (name: string) => ({ value: name, label: name === "None" ? t("None") : name })
  const choices = (key: string) => (controls[key]?.choices ?? []).map((choice) => ({ value: choice, label: t(choice) }))

  const slider = (key: "processor_res" | "threshold_a" | "threshold_b", spec: SliderSpec | undefined, fallback: string) =>
    spec?.visible && !(key === "processor_res" && unit.pixel_perfect) ? (
      <SliderRow
        label={t(spec.label ?? fallback)}
        value={sliderValue(unit[key], spec)}
        minimum={numberOr(spec.minimum, 0)}
        maximum={numberOr(spec.maximum, 1)}
        step={numberOr(spec.step, 0.01)}
        onCommit={(next) => commit({ [key]: next })}
        disabled={disabled}
      />
    ) : null

  return (
    <div id={id} role="group" aria-label={t(titles[props.unit] ?? "ControlNet")} className="flex min-w-0 flex-col gap-3">
      <ActiveUnits titles={titles} current={props.unit} onSelect={(index) => store.getState().setTab(stripKey, String(index))} />

      <div className="flex flex-wrap items-center gap-x-5 gap-y-1">
        <UnitSwitch label={label("enabled", "Enable")} checked={unit.enabled} onChange={(enabled) => commit({ enabled })} disabled={disabled} />
        <UnitSwitch label={label("pixel_perfect", "Pixel Perfect")} checked={unit.pixel_perfect} onChange={(pixel_perfect) => commit({ pixel_perfect })} disabled={disabled} />
        {img2img ? (
          <UnitSwitch
            label={label("upload_independent", "Upload independent control image")}
            checked={unit.upload_independent === true}
            // Without an image of its own the unit reads the img2img image: its own images go
            onChange={(on) => commit(on ? { upload_independent: true } : { upload_independent: false, image: null, sketch: null, image_fg: null, mask_upload: false })}
            disabled={disabled}
          />
        ) : null}
        {owns ? (
          <UnitSwitch label={label("mask_upload", "Use Mask")} checked={unit.mask_upload} onChange={(mask_upload) => commit(mask_upload ? { mask_upload } : { mask_upload, mask_image: null, mask_image_fg: null })} disabled={disabled} />
        ) : null}
      </div>

      {owns ? null : (
        <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
          <ImageIcon aria-hidden className="size-3.5 shrink-0" />
          {tw("This unit uses the img2img image. Switch on the independent control image to give it its own.")}
        </p>
      )}

      {owns ? (
        <div className="flex flex-col gap-2">
          <ImageEditor
            docKey={`${doc}:input`}
            pendingKey={formId}
            image={unit.image}
            // A new control image switches the unit on, as the old upload did
            onImageChange={(image) => commit({ image: image?.id ?? null, sketch: null, image_fg: null, ...(image ? { enabled: true } : {}) })}
            layers={{
              image_fg: { kind: "mask", value: unit.image_fg, label: tw("Mask") },
              sketch: { kind: "sketch", value: unit.sketch, label: tw("Sketch") },
            }}
            activeLayer={layer}
            onLayerChange={(key, layerId) => commit({ [key]: layerId })}
            blankSize={{ width, height }}
            frame={frame}
            onUseImageSize={useImageSize}
            label={tw("Control image")}
            hint={props.invert_hint ? t(props.invert_hint) : undefined}
          />
          <div className="flex flex-wrap items-center gap-2">
            <Segmented
              size="sm"
              aria-label={tw("Paint on")}
              options={[
                { value: "image_fg", label: tw("Mask") },
                { value: "sketch", label: tw("Sketch") },
              ]}
              value={layer}
              onValueChange={(next) => setLayerChoice(next as Layer)}
              disabled={disabled || !unit.image}
            />
            <Button size="sm" variant="secondary" className="ml-auto" onClick={() => void runPreview()} disabled={disabled || busy !== null || !unit.image}>
              {busy === "preview" ? <LoaderCircle aria-hidden className="animate-spin" /> : <Zap aria-hidden />}
              {tw("Run preprocessor")}
            </Button>
          </div>

          {isRegion ? (
            <div role="group" aria-label={tw("Region canvas")} className="flex flex-col gap-2 rounded-lg border p-2.5">
              <div className="grid gap-x-4 gap-y-2 sm:grid-cols-2">
                {(["subjects", "vertical", "randomize"] as const).map((key) => {
                  const control = controls[`region_${key}`]
                  return control ? (
                    <SliderRow
                      key={key}
                      label={t(control.label)}
                      info={control.info}
                      value={region[key]}
                      minimum={numberOr(control.minimum, 0)}
                      maximum={numberOr(control.maximum, 1)}
                      step={numberOr(control.step, 0.01)}
                      onCommit={(next) => setRegion((current) => ({ ...current, [key]: next }))}
                      disabled={disabled}
                    />
                  ) : null
                })}
                <ModeField label={label("region_shape", "Mask Shape")} options={choices("region_shape")} value={region.shape} onChange={(shape) => setRegion((current) => ({ ...current, shape }))} disabled={disabled} />
              </div>
              <Button size="sm" variant="secondary" className="self-start" onClick={() => void generateCanvas()} disabled={disabled || busy !== null}>
                {busy === "region_canvas" ? <LoaderCircle aria-hidden className="animate-spin" /> : <Shapes aria-hidden />}
                {tw("Generate Canvas")}
              </Button>
            </div>
          ) : null}

          {unit.mask_upload ? (
            <div role="group" aria-label={label("mask_upload", "Use Mask")} className="flex flex-col gap-1.5">
              <span className="text-[0.8rem] font-medium text-foreground/85">{label("mask_upload", "Use Mask")}</span>
              <ImageEditor
                docKey={`${doc}:mask`}
                pendingKey={formId}
                image={unit.mask_image ?? unit.image}
                onImageChange={(image) => commit({ mask_image: image?.id ?? null, mask_image_fg: null })}
                layers={{ mask_image_fg: { kind: "mask", value: unit.mask_image_fg, label: tw("Mask") } }}
                activeLayer="mask_image_fg"
                onLayerChange={(_key, layerId) => commit({ mask_image_fg: layerId })}
                height={256}
                label={tw("Mask image")}
                hint={tw("Upload a mask image, or paint the mask over the control image")}
              />
            </div>
          ) : null}

          {unit.generated_image ? (
            <div role="group" aria-label={tw("Preprocessor preview")} className="flex flex-col gap-1.5">
              <div className="flex flex-wrap items-center gap-2">
                <span className="mr-auto text-[0.8rem] font-medium text-foreground/85">{tw("Preprocessor preview")}</span>
                <UnitSwitch label={label("use_preview_as_input", "Preview as Input")} checked={unit.use_preview_as_input} onChange={(use_preview_as_input) => commit({ use_preview_as_input })} disabled={disabled} />
                <Button size="icon-sm" variant="ghost" aria-label={tw("Download preview")} title={tw("Download preview")} nativeButton={false} render={<a href={api.downloadUrl(unit.generated_image)} download="" rel="noopener" />}>
                  <Download aria-hidden />
                </Button>
                <Button size="icon-sm" variant="ghost" aria-label={tw("Close preview")} title={tw("Close preview")} onClick={() => commit({ generated_image: null, use_preview_as_input: false, pose: null })} disabled={disabled}>
                  <X aria-hidden />
                </Button>
              </div>
              <ImageEditor docKey={`${doc}:preview`} image={unit.generated_image} noUpload noScribble height={256} label={tw("Preprocessor preview")} />
            </div>
          ) : null}

          {unit.pose !== null || unit.type_filter === "OpenPose" ? <PosePanel key={unit.generated_image ?? ""} pose={unit.pose} busy={busy !== null} disabled={disabled} onApply={(pose) => void applyPose(pose)} /> : null}
        </div>
      ) : null}

      <Field id={`${id}-type`} label={label("type_filter", "Control Type")} labelFor={false}>
        <Segmented
          size="sm"
          aria-labelledby={`${id}-type-label`}
          options={catalog.types.map((type) => ({ value: type.name, label: t(type.name) }))}
          value={unit.type_filter}
          onValueChange={(next) => onChange(selectType(latest(), next, catalog))}
          disabled={disabled}
        />
      </Field>

      <div className="grid gap-x-3 gap-y-3 sm:grid-cols-2">
        <Field id={`${id}-module`} label={label("module", "Preprocessor")}>
          <SearchSelect
            id={`${id}-module`}
            size="sm"
            options={modulesFor(unit, catalog).map(option)}
            value={unit.module}
            onValueChange={(next) => onChange(selectModule(latest(), next, catalog))}
            disabled={disabled}
          />
        </Field>
        {preprocessor?.model !== false ? (
          <Field
            id={`${id}-model`}
            label={label("model", "Model")}
            trailing={
              <Button size="icon-xs" variant="ghost" aria-label={tw("Refresh models")} title={tw("Refresh models")} onClick={() => void call("refresh_models")} disabled={busy !== null}>
                <RefreshCw aria-hidden className={cn(busy === "refresh_models" && "animate-spin")} />
              </Button>
            }
          >
            <SearchSelect id={`${id}-model`} size="sm" options={modelsFor(unit, catalog).map(option)} value={unit.model} onValueChange={(model) => commit({ model })} disabled={disabled} />
          </Field>
        ) : null}
      </div>

      <div className="grid gap-x-4 gap-y-3 sm:grid-cols-2">
        <SliderRow
          label={label("weight", "Control Weight")}
          value={unit.weight}
          minimum={numberOr(controls.weight?.minimum, 0)}
          maximum={numberOr(controls.weight?.maximum, 2)}
          step={numberOr(controls.weight?.step, 0.05)}
          onCommit={(weight) => commit({ weight })}
          disabled={disabled}
        />
        <Field id={`${id}-range`} label={label("timestep_range", "Timestep Range")} labelFor={false}>
          <RangeSlider
            id={`${id}-range`}
            label={label("timestep_range", "Timestep Range")}
            value={[unit.guidance_start, unit.guidance_end]}
            onChange={(range) => range && commit({ guidance_start: range[0], guidance_end: range[1] })}
            props={{ minimum: controls.timestep_range?.minimum ?? 0, maximum: controls.timestep_range?.maximum ?? 1, step: controls.timestep_range?.step ?? 0.01 }}
            node={node}
            disabled={disabled}
          />
        </Field>
        {slider("processor_res", preprocessor?.resolution, "Preprocessor resolution")}
        {slider("threshold_a", preprocessor?.slider_1, "Threshold A")}
        {slider("threshold_b", preprocessor?.slider_2, "Threshold B")}
      </div>

      {preprocessor?.control_mode !== false ? (
        <ModeField label={label("control_mode", "Control Mode")} options={choices("control_mode")} value={unit.control_mode} onChange={(control_mode) => commit({ control_mode })} disabled={disabled} />
      ) : null}
      {owns ? <ModeField label={label("resize_mode", "Resize Mode")} options={choices("resize_mode")} value={unit.resize_mode} onChange={(resize_mode) => commit({ resize_mode })} disabled={disabled} /> : null}
      {!img2img && hires ? <ModeField label={label("hr_option", "Hires-Fix Option")} options={choices("hr_option")} value={unit.hr_option} onChange={(hr_option) => commit({ hr_option })} disabled={disabled} /> : null}

    </div>
  )
}

/** A radio row of the unit's modes (control, resize, Hires. fix). */
function ModeField(props: { label: string; options: { value: string; label: string }[]; value: string; onChange: (value: string) => void; disabled?: boolean }) {
  const id = useId()
  return (
    <Field id={id} label={props.label} labelFor={false}>
      <Segmented size="sm" aria-labelledby={`${id}-label`} options={props.options} value={props.value} onValueChange={props.onChange} disabled={props.disabled} />
    </Field>
  )
}
