/**
 * `image-list`: an ordered list of uploaded images (e.g. ImageStitch's
 * reference images). Add (drop, paste, browse), select, replace, remove,
 * reorder (drag a thumbnail, Alt+arrow keys or the toolbar) and clear. The
 * order is the order the script receives.
 *
 * Keyboard (list focused): arrows select, Alt+arrows move the selected
 * image, Home / End, Delete or Backspace remove it. The hover remove button
 * on a thumbnail is a mouse shortcut for the same (hidden from assistive tech).
 *
 * Declared as `ui.ImageList(..., widget="image-list")`, so the value keeps the
 * upload semantics (never restored from drafts).
 *
 * props: { columns?: number (default 3) }   value: string[] (upload ids)
 */
import { ArrowLeft, ArrowRight, ImageOff, ImagePlus, Loader2, Replace, Trash2, X } from "@/components/icons"
import { type KeyboardEvent, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { labelId } from "@/ui/a11y"
import { useForm, useOwner } from "@/ui/hooks"
import { DropZone } from "@/ui/inputs/DropZone"
import { imageFiles, releaseUpload, useUploader, useUploadInfo } from "@/ui/uploads"
import type { WidgetProps } from "@/ui/widgets"
import { registerWidgetCatalogs } from "../_lib/i18n"
import { moveItem, removeAt, replaceAt, selectionAfterRemove } from "./list"

const NS = registerWidgetCatalogs("image-list", import.meta.glob("./locales/*.json", { eager: true, import: "default" }))

type Props = { columns?: number }

const EMPTY: string[] = []

const uploadIds = (value: unknown): string[] =>
  Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : EMPTY

function Thumb({ id }: { id: string }) {
  const { t: tw } = useTranslation(NS)
  const info = useUploadInfo(id)
  const [failed, setFailed] = useState<string | null>(null)
  if (failed === id || !info?.url) {
    return (
      <div className="grid size-full place-items-center text-muted-foreground" title={tw("This image is no longer available")}>
        <ImageOff className="size-5" aria-hidden />
      </div>
    )
  }
  return <img src={info.url} alt="" draggable={false} onError={() => setFailed(id)} className="size-full object-contain" />
}

export default function ImageListWidget({ value, onChange, props, disabled, id, label, node }: WidgetProps<string[] | null, Props>) {
  const { t: tw } = useTranslation(NS)
  const { store } = useForm()
  const owner = useOwner()
  const ids = uploadIds(value)
  // After an upload the list may have changed (another upload, a removal): edit the stored list, not this render's
  const latest = () => uploadIds(store.getState().values[owner]?.[node.name])
  const columns = Math.min(Math.max(Math.round(props.columns ?? 3), 1), 8)
  const [selection, setSelection] = useState(-1)
  const [dragFrom, setDragFrom] = useState<number | null>(null)
  const { upload, busy } = useUploader()
  const replaceInput = useRef<HTMLInputElement>(null)
  const list = useRef<HTMLUListElement>(null)
  const selected = Math.min(selection, ids.length - 1)
  const optionId = (index: number) => `${id}-image-${index}`

  const commit = (next: string[], nextSelection: number) => {
    onChange(next)
    setSelection(nextSelection)
  }

  const add = async (files: File[]) => {
    const uploaded = await upload(imageFiles(files))
    if (uploaded.length === 0) return
    const base = latest()
    commit([...base, ...uploaded.map((item) => item.id)], base.length + uploaded.length - 1)
  }

  const move = (from: number, to: number) => {
    if (from < 0 || to < 0 || to >= ids.length || from === to) return
    commit(moveItem(ids, from, to), to)
  }

  const remove = (index: number) => {
    const removed = ids[index]
    if (removed === undefined) return
    commit(removeAt(ids, index), selectionAfterRemove(index, ids.length))
    releaseUpload(removed)
  }

  const replace = async (files: File[]) => {
    const old = ids[selected]
    if (old === undefined) return
    const [uploaded] = await upload(imageFiles(files).slice(0, 1))
    if (!uploaded) return
    const base = latest()
    // Find the replaced image again: it may have moved, or been removed meanwhile (then the upload is dropped)
    const index = base[selected] === old ? selected : base.indexOf(old)
    if (index < 0) {
      releaseUpload(uploaded.id)
      return
    }
    commit(replaceAt(base, index, uploaded.id), index)
    releaseUpload(old)
  }

  const clear = () => {
    commit([], -1)
    for (const item of ids) releaseUpload(item)
  }

  const onKeyDown = (event: KeyboardEvent<HTMLUListElement>) => {
    if (disabled || ids.length === 0) return
    const from = selected < 0 ? 0 : selected
    const step: Record<string, number> = { ArrowLeft: -1, ArrowRight: 1, ArrowUp: -columns, ArrowDown: columns }
    let target: number | null = null
    if (event.key in step) target = Math.min(Math.max(from + step[event.key]!, 0), ids.length - 1)
    else if (event.key === "Home") target = 0
    else if (event.key === "End") target = ids.length - 1
    else if ((event.key === "Delete" || event.key === "Backspace") && selected >= 0) {
      event.preventDefault()
      remove(selected)
      return
    }
    if (target === null) return
    event.preventDefault()
    if (event.altKey && selected >= 0) move(selected, target)
    else setSelection(target)
  }

  return (
    <div className="flex flex-col gap-2">
      {ids.length > 0 ? (
        <>
          <ul
            ref={list}
            role="listbox"
            tabIndex={disabled ? -1 : 0}
            aria-labelledby={label ? labelId(id) : undefined}
            aria-label={label ? undefined : tw("Images")}
            aria-activedescendant={selected >= 0 ? optionId(selected) : undefined}
            aria-disabled={disabled || undefined}
            onKeyDown={onKeyDown}
            // Files dropped on the thumbnails are appended, as on the drop zone
            onDragOver={(event) => {
              if (dragFrom === null && !disabled && event.dataTransfer.types.includes("Files")) event.preventDefault()
            }}
            onDrop={(event) => {
              if (dragFrom !== null || disabled || event.dataTransfer.files.length === 0) return
              event.preventDefault()
              void add(Array.from(event.dataTransfer.files))
            }}
            className="grid gap-1.5 rounded-lg outline-none focus-visible:ring-3 focus-visible:ring-ring/50"
            style={{ gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` }}
          >
            {ids.map((imageId, index) => (
              <li
                key={`${imageId}-${index}`}
                id={optionId(index)}
                role="option"
                aria-selected={index === selected}
                aria-label={tw("Image {{n}}", { n: index + 1 })}
                draggable={!disabled}
                onClick={() => {
                  if (disabled) return
                  setSelection(index)
                  list.current?.focus()
                }}
                onDragStart={(event) => {
                  event.dataTransfer.effectAllowed = "move"
                  event.dataTransfer.setData("text/plain", String(index))
                  setDragFrom(index)
                }}
                onDragEnd={() => setDragFrom(null)}
                onDragOver={(event) => {
                  if (dragFrom === null) return
                  event.preventDefault()
                  event.dataTransfer.dropEffect = "move"
                }}
                onDrop={(event) => {
                  if (dragFrom === null) return
                  event.preventDefault()
                  move(dragFrom, index)
                  setDragFrom(null)
                }}
                className={cn(
                  "group/thumb relative aspect-square cursor-pointer overflow-hidden rounded-md border-2 border-transparent bg-checker transition-colors",
                  index === selected && "border-primary",
                  dragFrom === index && "opacity-50",
                  disabled && "cursor-default",
                )}
              >
                <Thumb id={imageId} />
                <span aria-hidden className="absolute top-1 left-1 rounded bg-background/80 px-1 text-[0.65rem] font-medium tabular-nums">
                  {index + 1}
                </span>
                <Button
                  variant="secondary"
                  size="icon-xs"
                  tabIndex={-1}
                  aria-hidden
                  title={tw("Remove image {{n}}", { n: index + 1 })}
                  disabled={disabled}
                  onClick={(event) => {
                    event.stopPropagation()
                    remove(index)
                  }}
                  className="absolute top-1 right-1 opacity-0 group-hover/thumb:opacity-100 [@media(hover:none)]:opacity-100"
                >
                  <X />
                </Button>
              </li>
            ))}
          </ul>
          <div role="toolbar" aria-label={tw("Image actions")} className="flex flex-wrap items-center gap-1">
            <Button variant="outline" size="xs" disabled={disabled || selected <= 0} onClick={() => move(selected, selected - 1)}>
              <ArrowLeft aria-hidden />
              {tw("Move earlier")}
            </Button>
            <Button variant="outline" size="xs" disabled={disabled || selected < 0 || selected >= ids.length - 1} onClick={() => move(selected, selected + 1)}>
              <ArrowRight aria-hidden />
              {tw("Move later")}
            </Button>
            <Button variant="outline" size="xs" disabled={disabled || selected < 0 || busy} onClick={() => replaceInput.current?.click()}>
              <Replace aria-hidden />
              {tw("Replace")}
            </Button>
            <Button variant="outline" size="xs" disabled={disabled || selected < 0} onClick={() => remove(selected)}>
              <X aria-hidden />
              {tw("Remove")}
            </Button>
            <Button variant="ghost" size="xs" disabled={disabled} onClick={clear} className="ml-auto text-destructive">
              <Trash2 aria-hidden />
              {tw("Clear all")}
            </Button>
            <input
              ref={replaceInput}
              type="file"
              accept="image/*"
              hidden
              data-testid="replace-input"
              onChange={(event) => {
                const files = Array.from(event.target.files ?? [])
                event.target.value = ""
                void replace(files)
              }}
            />
          </div>
        </>
      ) : null}
      <DropZone
        id={id}
        multiple
        accept="image/*"
        onFiles={(files) => void add(files)}
        disabled={disabled}
        aria-label={tw("Add images")}
        className={ids.length > 0 ? "py-1" : undefined}
      >
        <div className="flex items-center justify-center gap-2 px-3 py-4 text-sm text-muted-foreground">
          {busy ? <Loader2 className="size-4 animate-spin" aria-hidden /> : <ImagePlus className="size-4" aria-hidden />}
          {busy ? tw("Uploading…") : tw("Drop images, paste, or click to add")}
        </div>
      </DropZone>
    </div>
  )
}
