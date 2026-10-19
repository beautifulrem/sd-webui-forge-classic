import { ImagePlus, Loader2, X } from "lucide-react"
import { useId } from "react"
import { useTranslation } from "react-i18next"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { useParam } from "../hooks"
import { describedBy, hintRef, labelId } from "../a11y"
import { Field } from "../Field"
import type { ImageListNode, ImageNode } from "../schema"
import { usePasteTarget } from "../paste-targets"
import { imageFiles, releaseUpload, useUploader, useUploadInfo } from "../uploads"
import { DropZone } from "./DropZone"
import { useImageRole } from "@/canvas/form/image-role"
import { SchemaImageEditor } from "@/canvas/form/SchemaImageEditor"

function Thumb({ id, className }: { id: string; className?: string }) {
  const info = useUploadInfo(id)
  return info?.url ? (
    <img src={info.url} alt="" draggable={false} className={className} />
  ) : (
    <div className={className} />
  )
}

function UploadHint({ id, busy, multiple }: { id: string; busy: boolean; multiple?: boolean }) {
  const { t } = useTranslation()
  return (
    <div className="flex flex-col items-center justify-center gap-1.5 px-3 py-6 text-center text-muted-foreground">
      {busy ? <Loader2 className="size-5 animate-spin" aria-hidden /> : <ImagePlus className="size-5" aria-hidden />}
      <p id={id} className="text-sm">
        {busy ? t("Uploading…") : multiple ? t("Drop images, paste, or click to browse") : t("Drop an image, paste, or click to browse")}
      </p>
    </div>
  )
}

/**
 * Single image input. An image with `sketch` / `mask` layers painted over it
 * (their `source`) is the image editor (src/canvas), and those layers render
 * nothing here; any other image, and a `mask` without `source`, is an upload
 * field (`UploadImageInput`).
 */
export function ImageInput({ node }: { node: ImageNode }) {
  const role = useImageRole(node)
  if (role === "editor") return <SchemaImageEditor node={node} />
  if (role === "layer") return null
  return <UploadImageInput node={node} />
}

/** An image upload field: the drop zone is named by the field label and described by its action. */
function UploadImageInput({ node }: { node: ImageNode }) {
  const { t } = useTranslation()
  const id = useId()
  const hintId = `${id}-hint`
  const param = useParam<string | null>(node)
  const { upload, busy } = useUploader()
  const label = t(param.label)
  const value = typeof param.value === "string" ? param.value : null
  const disabled = !param.enabled

  const onFiles = async (files: File[]) => {
    const [first] = await upload(imageFiles(files))
    if (!first) return
    param.setValue(first.id)
    if (value) releaseUpload(value)
  }
  // A paste with no image field hovered or focused goes to the first empty one on screen.
  usePasteTarget(!disabled, { element: () => document.getElementById(id), empty: () => !value, deliver: (files) => void onFiles(files) })

  return (
    <Field
      id={id}
      label={label}
      info={param.info}
      hint={param.hint}
      error={param.error}
      modified={param.modified}
      defaultValue={param.defaultValue}
      onReset={param.reset}
      disabled={disabled}
      labelFor={false}
    >
      <div className="group/image relative">
        <DropZone
          id={id}
          onFiles={onFiles}
          accept="image/*"
          disabled={disabled}
          aria-labelledby={label ? labelId(id) : undefined}
          aria-label={label ? undefined : value ? t("Replace image") : t("Upload image")}
          aria-describedby={describedBy(id, param.info, param.error, hintRef(id, param.hint), hintId)}
          aria-invalid={Boolean(param.error)}
          className="overflow-hidden"
        >
          {value ? (
            <div className="relative bg-checker">
              <span id={hintId} className="sr-only">
                {t("Replace image")}
              </span>
              <Thumb id={value} className="mx-auto max-h-80 min-h-24 w-full object-contain" />
              {node.kind && node.kind !== "image" ? (
                <Badge variant="secondary" className="absolute top-2 left-2">
                  {node.kind === "mask" ? t("Mask") : t("Sketch")}
                </Badge>
              ) : null}
              {busy ? (
                <div className="absolute inset-0 grid place-items-center bg-background/60">
                  <Loader2 className="size-5 animate-spin" aria-hidden />
                </div>
              ) : null}
            </div>
          ) : (
            <UploadHint id={hintId} busy={busy} />
          )}
        </DropZone>
        {value ? (
          <Button
            variant="secondary"
            size="icon-sm"
            aria-label={label ? t("Remove {{label}}", { label }) : t("Remove image")}
            onClick={() => {
              param.setValue(null)
              releaseUpload(value)
            }}
            disabled={disabled}
            className="absolute top-2 right-2 opacity-0 transition-opacity group-focus-within/image:opacity-100 group-hover/image:opacity-100 [@media(hover:none)]:opacity-100"
          >
            <X />
          </Button>
        ) : null}
      </div>
    </Field>
  )
}

export function ImageListInput({ node }: { node: ImageListNode }) {
  const { t } = useTranslation()
  const id = useId()
  const hintId = `${id}-hint`
  const param = useParam<string[]>(node)
  const { upload, busy } = useUploader()
  const ids = Array.isArray(param.value) ? param.value : []
  const label = t(param.label)
  const disabled = !param.enabled

  const onFiles = async (files: File[]) => {
    const uploaded = await upload(imageFiles(files))
    if (uploaded.length > 0) param.setValue([...ids, ...uploaded.map((u) => u.id)])
  }
  usePasteTarget(!disabled, { element: () => document.getElementById(id), empty: () => ids.length === 0, deliver: (files) => void onFiles(files) })

  return (
    <Field
      id={id}
      label={label}
      info={param.info}
      hint={param.hint}
      error={param.error}
      modified={param.modified}
      defaultValue={param.defaultValue}
      onReset={param.reset}
      disabled={disabled}
      labelFor={false}
    >
      {ids.length > 0 ? (
        <ul className="grid grid-cols-[repeat(auto-fill,minmax(5.5rem,1fr))] gap-1.5" aria-labelledby={label ? labelId(id) : undefined}>
          {ids.map((imageId, index) => (
            <li key={`${imageId}-${index}`} className="group/thumb relative aspect-square overflow-hidden rounded-md bg-checker">
              <Thumb id={imageId} className="size-full object-cover" />
              <Button
                variant="secondary"
                size="icon-xs"
                aria-label={t("Remove image {{n}}", { n: index + 1 })}
                onClick={() => {
                  param.setValue(ids.filter((_, i) => i !== index))
                  releaseUpload(imageId)
                }}
                disabled={disabled}
                className="absolute top-1 right-1 opacity-0 group-focus-within/thumb:opacity-100 group-hover/thumb:opacity-100 [@media(hover:none)]:opacity-100"
              >
                <X />
              </Button>
            </li>
          ))}
        </ul>
      ) : null}
      <DropZone
        id={id}
        multiple
        accept="image/*"
        onFiles={onFiles}
        disabled={disabled}
        aria-labelledby={label ? labelId(id) : undefined}
        aria-label={label ? undefined : t("Add images")}
        aria-describedby={describedBy(id, param.info, param.error, hintRef(id, param.hint), hintId)}
        aria-invalid={Boolean(param.error)}
      >
        <UploadHint id={hintId} busy={busy} multiple />
      </DropZone>
    </Field>
  )
}
