import { FileUp, Loader2, X } from "lucide-react"
import { useId } from "react"
import { useTranslation } from "react-i18next"
import { Button } from "@/components/ui/button"
import { useParam } from "../hooks"
import { describedBy, hintRef, labelId } from "../a11y"
import { Field } from "../Field"
import type { FileNode } from "../schema"
import { releaseUpload, useUploader, useUploadInfo } from "../uploads"
import { DropZone } from "./DropZone"

/** One uploaded file (`accept` filters the browser's picker); the drop zone is named by the field label. */
export function FileInput({ node }: { node: FileNode }) {
  const { t } = useTranslation()
  const id = useId()
  const hintId = `${id}-hint`
  const param = useParam<string | null>(node)
  const { upload, busy } = useUploader()
  const value = typeof param.value === "string" ? param.value : null
  const label = t(param.label)
  const uploaded = useUploadInfo(value)
  const disabled = !param.enabled

  const onFiles = async (files: File[]) => {
    const [first] = await upload(files.slice(0, 1))
    if (!first) return
    param.setValue(first.id)
    if (value) releaseUpload(value)
  }

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
      <div className="flex items-center gap-1.5">
        <DropZone
          id={id}
          onFiles={onFiles}
          accept={node.accept ?? undefined}
          disabled={disabled}
          aria-labelledby={label ? `${labelId(id)} ${hintId}` : undefined}
          aria-label={label ? undefined : value ? t("Replace file") : t("Choose a file")}
          aria-describedby={describedBy(id, param.info, param.error, hintRef(id, param.hint))}
          aria-invalid={Boolean(param.error)}
          className="flex h-8 min-w-0 flex-1 items-center gap-2 px-2.5 text-sm"
        >
          {busy ? (
            <Loader2 className="size-4 shrink-0 animate-spin" aria-hidden />
          ) : (
            <FileUp className="size-4 shrink-0 text-muted-foreground" aria-hidden />
          )}
          <span id={hintId} className={value ? "truncate" : "truncate text-muted-foreground"}>
            {value ? (uploaded?.name ?? value) : node.accept ? t("Choose a file ({{types}})", { types: node.accept }) : t("Choose a file")}
          </span>
        </DropZone>
        {value ? (
          <Button
            variant="ghost"
            size="icon"
            aria-label={label ? t("Remove {{label}}", { label }) : t("Remove file")}
            onClick={() => {
              param.setValue(null)
              releaseUpload(value)
            }}
            disabled={disabled}
          >
            <X />
          </Button>
        ) : null}
      </div>
    </Field>
  )
}
