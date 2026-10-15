/**
 * `source-size-fit`: fits the img2img size to the source image. While on
 * (the widget's value), every new source image (Send to img2img / inpaint,
 * an upload) runs the owner's `props.action` once; "Fit now" runs it on
 * demand. The action reads the source from the generation form and returns
 * the size patch, so the size rule lives on the server; "Fit now" runs
 * `props.manual_action`, which also explains when nothing fits.
 *
 * A source delivered while the form was not on screen (Send to from
 * txt2img) is fitted when the widget mounts: the last source fitted per
 * form is remembered for the session.
 *
 * props: { source: string ("core.init_img"), action: string, manual_action: string }   value: boolean
 */
import { useEffect } from "react"
import { useTranslation } from "react-i18next"
import { Button } from "@/components/ui/button"
import { Switch } from "@/components/ui/switch"
import { splitRef } from "@/ui/conditions"
import { useForm, useFormState, useOwner, useRunAction } from "@/ui/hooks"
import type { WidgetProps } from "@/ui/widgets"
import { registerWidgetCatalogs } from "../_lib/i18n"

const NS = registerWidgetCatalogs("source-size-fit", import.meta.glob("./locales/*.json", { eager: true, import: "default" }))

type Props = { source?: string; action?: string; manual_action?: string }

/** form id → the source image last seen (fitted, or seen while off). */
const seen = new Map<string, string | null>()

export default function SourceSizeFit({ value, onChange, props, disabled, id }: WidgetProps<boolean | null, Props>) {
  const { t: tw } = useTranslation(NS)
  const { formId } = useForm()
  const owner = useOwner()
  const run = useRunAction()
  const [sourceOwner, sourceName] = splitRef(props.source ?? "core.init_img", owner)
  const raw = useFormState((s) => s.values[sourceOwner]?.[sourceName])
  const source = typeof raw === "string" && raw ? raw : null
  const auto = value !== false
  const action = props.action ?? "fit_to_source"
  const manualAction = props.manual_action ?? action

  useEffect(() => {
    const previous = seen.has(formId) ? seen.get(formId) : null
    seen.set(formId, source)
    if (source && source !== previous && auto && !disabled) void run(action)
    // Only a new source triggers the fit; `run` is recreated every render.
    // oxlint-disable-next-line react-hooks/exhaustive-deps
  }, [source, formId])

  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
      <label className="flex items-center gap-2 text-sm">
        <Switch id={id} checked={auto} disabled={disabled} onCheckedChange={(checked) => onChange(checked === true)} />
        {tw("Fit the size to each new source image")}
      </label>
      <Button variant="outline" size="xs" disabled={disabled || !source} onClick={() => void run(manualAction)}>
        {tw("Fit now")}
      </Button>
    </div>
  )
}
