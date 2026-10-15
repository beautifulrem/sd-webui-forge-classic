/**
 * Image layers declared in a schema (schema-format.md §1.3): an `image`
 * input that other `sketch` / `mask` inputs of the same owner name as their
 * `source` is rendered as one `<ImageEditor>` with those layers painted over
 * it; the layer inputs themselves render nothing. A `mask` without `source`
 * stays an upload field. Any script or extension form gets the editor this
 * way, keyed `formId:owner:param` (layers and undo survive remounts).
 *
 * Which layer is edited: the visible ones (their own `visible_if`, `hidden`
 * and patch overrides), the user picks when there are several.
 */
import { useId, useState } from "react"
import { useTranslation } from "react-i18next"
import { Segmented } from "@/components/Segmented"
import { evaluate } from "@/ui/conditions"
import { Field } from "@/ui/Field"
import { useForm, useFormState, useOwner, useParam } from "@/ui/hooks"
import type { ImageNode } from "@/ui/schema"
import { ImageEditor } from "../ImageEditor"
import type { EditorLayer, LayerKind } from "../types"
import { layersOf } from "./image-role"
import { replaceBackground, writeValues } from "./replace-background"

type LayerInfo = { name: string; kind: LayerKind; label: string; value: string | null; visible: boolean; error: string | undefined }

/** The image input of a background with layers: label and info, then the editor. */
export function SchemaImageEditor({ node }: { node: ImageNode }) {
  const { t } = useTranslation()
  const { formId, store } = useForm()
  const owner = useOwner()
  const fieldId = useId()
  const param = useParam<string | null>(node)
  const label = t(param.label)
  const image = typeof param.value === "string" ? param.value : null
  const enabled = param.enabled

  // One primitive per render (re-renders only when a layer's value, visibility or error changes).
  const signature = useFormState((s) =>
    JSON.stringify(
      layersOf(s, owner, node.name).map((layer): LayerInfo => {
        const override = s.overrides[owner]?.[layer.name]
        const visible = !layer.hidden && override?.visible !== false && evaluate(layer.visible_if, owner, (o, n) => s.values[o]?.[n])
        const value = s.values[owner]?.[layer.name]
        const text = override && Object.hasOwn(override, "label") ? (override.label ?? "") : (layer.label ?? "")
        return { name: layer.name, kind: layer.kind, label: text, value: typeof value === "string" ? value : null, visible, error: s.errors[owner]?.[layer.name] }
      }),
    ),
  )
  const layers = JSON.parse(signature) as LayerInfo[]
  const offered = layers.filter((layer) => layer.visible)
  const [picked, setPicked] = useState<string | null>(null)
  const active = offered.find((layer) => layer.name === picked) ?? offered[0] ?? null
  const layerName = (layer: LayerInfo) => (layer.label ? t(layer.label) : layer.kind === "mask" ? t("Mask") : t("Sketch"))

  const editorLayers: Record<string, EditorLayer> = {}
  for (const layer of layers) editorLayers[layer.name] = { kind: layer.kind, value: layer.value, label: layerName(layer) }
  const error = param.error ?? active?.error ?? null

  return (
    <Field
      id={fieldId}
      label={label}
      info={param.info}
      error={param.error}
      modified={param.modified}
      defaultValue={param.defaultValue}
      onReset={() => void replaceBackground({ store, formId, owner, background: node.name, painted: layers.map((layer) => layer.name) }, (param.defaultValue as string | null) ?? null)}
      disabled={!enabled}
      labelFor={false}
    >
      <div className="flex flex-col gap-1.5">
        {offered.length > 1 ? (
          <Segmented
            size="sm"
            aria-label={t("Layer to edit")}
            value={active?.name ?? null}
            onValueChange={setPicked}
            options={offered.map((layer) => ({ value: layer.name, label: layerName(layer) }))}
            className="self-start"
          />
        ) : null}
        <ImageEditor
          docKey={`${formId}:${owner}:${node.name}`}
          pendingKey={formId}
          label={label || t("Image editor")}
          image={image}
          layers={editorLayers}
          activeLayer={enabled ? (active?.name ?? null) : null}
          noUpload={!enabled}
          error={error}
          onImageChange={(next) => void replaceBackground({ store, formId, owner, background: node.name, painted: layers.map((layer) => layer.name) }, next?.id ?? null)}
          onLayerChange={(key, id) => writeValues({ store, formId, owner }, { [key]: id })}
        />
      </div>
    </Field>
  )
}
