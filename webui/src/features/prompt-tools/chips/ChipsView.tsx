/**
 * The tag view of a prompt: one chip per tag (colour by category,
 * translation underneath, weight badge), `BREAK` / `AND` as dividers.
 * Click a chip to edit its text and weight, disable / enable it (a prompt
 * comment) or delete it; drag chips (or Alt+←/→) to reorder; ←/→ move the
 * focus, Delete removes, Ctrl/Cmd+Z undoes through the text editor. Every
 * edit rewrites only the touched chips of the prompt text.
 */
import { keepPreviousData, useQuery } from "@tanstack/react-query"
import { EyeOff, Layers, MessageSquare, Shuffle } from "@/components/icons"
import { type KeyboardEvent, type PointerEvent, useId, useMemo, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import type { PromptDecoratorProps } from "@/features/prompt-editor/decorators"
import { cn } from "@/lib/utils"
import { type PromptToolsConfig, promptToolsApi, type TagInfo } from "../api"
import { categoryColor } from "../completion/colors"
import { toolKeys } from "../queries"
import { translatedKey, translatedTags } from "../translate"
import { AddTags } from "./AddTags"
import { ChipEditor } from "./ChipEditor"
import { type Chip, displayText, moveChip, type PromptDoc, parsePrompt, removeChip } from "./parse"

export type ChipsViewProps = PromptDecoratorProps & { config: PromptToolsConfig }

/** Pointer travel before a press becomes a drag. */
const DRAG_THRESHOLD = 5

function weightLabel(weight: number): string {
  return String(Math.round(weight * 100) / 100)
}

async function undoInEditor(editor: ChipsViewProps["editor"], redo: boolean) {
  const view = editor()?.view()
  if (!view) return
  const commands = await import("@codemirror/commands")
  ;(redo ? commands.redo : commands.undo)(view)
}

type ChipItemProps = {
  chip: Chip
  index: number
  info: TagInfo | undefined
  translation: string | undefined
  config: PromptToolsConfig
  focused: boolean
  dragging: boolean
  dropBefore: boolean
  disabled: boolean
  open: boolean
  onOpenChange: (open: boolean) => void
  onKeyDown: (event: KeyboardEvent<HTMLButtonElement>) => void
  onPointerDown: (event: PointerEvent<HTMLElement>) => void
  onFocus: () => void
  register: (node: HTMLElement | null) => void
  doc: PromptDoc
  setValue: (value: string) => void
}

function ChipItem({
  chip,
  index,
  info,
  translation,
  config,
  focused,
  dragging,
  dropBefore,
  disabled,
  open,
  onOpenChange,
  onKeyDown,
  onPointerDown,
  onFocus,
  register,
  doc,
  setValue,
}: ChipItemProps) {
  const { t } = useTranslation()
  const text = displayText(chip.text)
  const color = chip.kind === "tag" && !chip.disabled ? categoryColor(config.tags.source, info?.category) : null
  const weighted = chip.kind === "network" || Math.abs(chip.weight - 1) > 0.0001
  const describe = [
    text,
    chip.kind === "network" ? t("Extra network") : null,
    weighted ? t("Weight {{weight}}", { weight: weightLabel(chip.weight) }) : null,
    chip.disabled ? t("Disabled") : null,
    translation ?? null,
  ]
    .filter(Boolean)
    .join(", ")

  if (chip.kind === "keyword") {
    return (
      <li
        ref={register}
        className={cn(
          "relative flex basis-full items-center gap-2 py-0.5",
          dropBefore && "before:absolute before:-top-0.5 before:left-0 before:h-0.5 before:w-full before:rounded-full before:bg-primary",
        )}
      >
        <span aria-hidden className="h-px flex-1 bg-border" />
        <button
          type="button"
          tabIndex={focused ? 0 : -1}
          aria-label={describe}
          onKeyDown={onKeyDown}
          onFocus={onFocus}
          onPointerDown={onPointerDown}
          disabled={disabled}
          className="rounded px-1.5 text-[0.65rem] font-semibold tracking-widest text-muted-foreground outline-none select-none focus-visible:ring-3 focus-visible:ring-ring/50"
        >
          {chip.text}
        </button>
        <span aria-hidden className="h-px flex-1 bg-border" />
      </li>
    )
  }

  return (
    <li
      ref={register}
      className={cn(
        "relative max-w-full",
        dragging && "opacity-40",
        dropBefore && "before:absolute before:top-0.5 before:-left-[3px] before:h-[calc(100%-0.25rem)] before:w-0.5 before:rounded-full before:bg-primary",
      )}
    >
      <Popover open={open} onOpenChange={onOpenChange}>
        <PopoverTrigger
          render={
            <button
              type="button"
              tabIndex={focused ? 0 : -1}
              aria-label={describe}
              onKeyDown={onKeyDown}
              onFocus={onFocus}
              onPointerDown={onPointerDown}
              disabled={disabled}
              className={cn(
                "group/chip flex max-w-full touch-pan-y items-center gap-1 rounded-md border bg-background px-1.5 py-0.5 text-left text-xs leading-tight shadow-xs transition-colors outline-none select-none",
                "hover:border-foreground/25 focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 data-popup-open:border-primary/60",
                chip.kind === "network" && "border-chart-2/40 bg-chart-2/8",
                chip.kind === "wildcard" && "border-chart-5/40 bg-chart-5/8 italic",
                chip.kind === "comment" && "border-dashed bg-transparent text-muted-foreground italic",
                chip.disabled && "border-dashed bg-muted/40 opacity-70",
              )}
            />
          }
        >
          {chip.kind === "network" ? <Layers aria-hidden className="size-3 shrink-0 text-chart-2" /> : null}
          {chip.kind === "wildcard" ? <Shuffle aria-hidden className="size-3 shrink-0 text-chart-5" /> : null}
          {chip.kind === "comment" ? <MessageSquare aria-hidden className="size-3 shrink-0" /> : null}
          {chip.disabled && chip.kind !== "comment" ? <EyeOff aria-hidden className="size-3 shrink-0 text-muted-foreground" /> : null}
          <span className="flex min-w-0 flex-col py-px">
            <span className={cn("max-w-64 truncate font-medium", chip.disabled && "line-through decoration-muted-foreground/60")} style={color ? { color } : undefined}>
              {text}
            </span>
            {translation ? <span className="max-w-64 truncate text-[0.65rem] font-normal text-muted-foreground">{translation}</span> : null}
          </span>
          {weighted ? (
            <span className="tabular shrink-0 rounded bg-primary/10 px-1 text-[0.65rem] font-semibold text-primary">{weightLabel(chip.weight)}</span>
          ) : null}
        </PopoverTrigger>
        <PopoverContent align="start" className="w-72" aria-label={t("Edit “{{tag}}”", { tag: text })}>
          {open ? (
            <ChipEditor
              doc={doc}
              index={index}
              info={info}
              translation={translation}
              config={config}
              setValue={setValue}
              close={() => onOpenChange(false)}
            />
          ) : null}
        </PopoverContent>
      </Popover>
    </li>
  )
}

export default function ChipsView({ value, setValue, label, disabled, editor, config, kind }: ChipsViewProps) {
  const { t } = useTranslation()
  const liveId = useId()
  const doc = useMemo(() => parsePrompt(value), [value])
  const [focus, setFocus] = useState(0)
  const [editing, setEditing] = useState<number | null>(null)
  const [drag, setDrag] = useState<{ from: number; drop: number } | null>(null)
  const [announcement, setAnnouncement] = useState("")
  const press = useRef<{ index: number; x: number; y: number; pointer: number } | null>(null)
  const justDragged = useRef(false)
  const nodes = useRef<(HTMLElement | null)[]>([])
  const list = useRef<HTMLUListElement>(null)

  const tags = useMemo(() => [...new Set(doc.chips.filter((chip) => chip.kind === "tag").map((chip) => displayText(chip.text)))].sort(), [doc])
  const lookup = useQuery({
    queryKey: toolKeys.lookup(tags),
    queryFn: ({ signal }) => promptToolsApi.lookup(tags, signal),
    enabled: tags.length > 0,
    staleTime: Infinity,
    placeholderData: keepPreviousData,
  })
  const language = config.tools.language
  const translated = useQuery({ queryKey: translatedKey(language), queryFn: () => translatedTags(language), staleTime: Infinity })

  const count = doc.chips.length
  const focusIndex = Math.min(focus, Math.max(count - 1, 0))

  const focusChip = (index: number) => {
    const next = Math.max(0, Math.min(index, count - 1))
    setFocus(next)
    queueMicrotask(() => nodes.current[next]?.querySelector("button")?.focus())
  }

  const move = (from: number, to: number) => {
    if (to < 0 || to >= count || from === to) return
    setValue(moveChip(doc, from, to))
    setAnnouncement(t("Moved “{{tag}}” to position {{position}} of {{count}}", { tag: displayText(doc.chips[from]!.text), position: to + 1, count }))
    focusChip(to)
  }

  const onKeyDown = (index: number) => (event: KeyboardEvent<HTMLButtonElement>) => {
    const mod = event.ctrlKey || event.metaKey
    if (mod && (event.key === "z" || event.key === "Z" || event.key === "y")) {
      event.preventDefault()
      void undoInEditor(editor, event.key === "y" || event.shiftKey)
      return
    }
    if (event.altKey && (event.key === "ArrowLeft" || event.key === "ArrowRight")) {
      event.preventDefault()
      move(index, index + (event.key === "ArrowLeft" ? -1 : 1))
      return
    }
    switch (event.key) {
      case "ArrowLeft":
      case "ArrowUp":
        event.preventDefault()
        focusChip(index - 1)
        break
      case "ArrowRight":
      case "ArrowDown":
        event.preventDefault()
        focusChip(index + 1)
        break
      case "Home":
        event.preventDefault()
        focusChip(0)
        break
      case "End":
        event.preventDefault()
        focusChip(count - 1)
        break
      case "Delete":
      case "Backspace":
        event.preventDefault()
        setAnnouncement(t("Removed “{{tag}}”", { tag: displayText(doc.chips[index]!.text) }))
        setValue(removeChip(doc, index))
        focusChip(index >= count - 1 ? index - 1 : index)
        break
    }
  }

  // region drag to reorder (pointer events: mouse, pen and touch alike)

  const dropSlot = (x: number, y: number): number => {
    let best = count
    let distance = Infinity
    nodes.current.forEach((node, index) => {
      if (!node) return
      const rect = node.getBoundingClientRect()
      const inRow = y >= rect.top - 2 && y <= rect.bottom + 2
      const dx = x < rect.left ? rect.left - x : x > rect.right ? x - rect.right : 0
      const score = (inRow ? 0 : 10_000 + Math.abs(y - (rect.top + rect.bottom) / 2) * 10) + dx
      if (score < distance) {
        distance = score
        best = x > rect.left + rect.width / 2 ? index + 1 : index
      }
    })
    return best
  }

  const onPointerDown = (index: number) => (event: PointerEvent<HTMLElement>) => {
    if (disabled || event.button !== 0) return
    press.current = { index, x: event.clientX, y: event.clientY, pointer: event.pointerId }
  }

  const onPointerMove = (event: PointerEvent<HTMLUListElement>) => {
    const start = press.current
    if (!start || start.pointer !== event.pointerId) return
    if (!drag) {
      if (Math.hypot(event.clientX - start.x, event.clientY - start.y) < DRAG_THRESHOLD) return
      list.current?.setPointerCapture(event.pointerId)
    }
    setDrag({ from: start.index, drop: dropSlot(event.clientX, event.clientY) })
  }

  const endDrag = (event: PointerEvent<HTMLUListElement>) => {
    const current = drag
    press.current = null
    if (!current) return
    if (list.current?.hasPointerCapture(event.pointerId)) list.current.releasePointerCapture(event.pointerId)
    setDrag(null)
    justDragged.current = true
    setTimeout(() => (justDragged.current = false), 0)
    const to = current.drop > current.from ? current.drop - 1 : current.drop
    move(current.from, to)
  }

  // endregion

  return (
    <div className="flex min-w-0 flex-col gap-1">
      <ul
        ref={list}
        aria-label={t("Tags of {{label}}", { label })}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={() => {
          press.current = null
          setDrag(null)
        }}
        className={cn(
          "flex min-h-16 flex-wrap content-start items-start gap-1 rounded-lg border border-input bg-input/30 p-1.5 transition-colors",
          drag && "cursor-grabbing",
          disabled && "pointer-events-none opacity-60",
        )}
      >
        {doc.chips.map((chip, index) => {
          const key = displayText(chip.text)
          return (
            <ChipItem
              // positional: the same tag may appear twice, and an open editor survives its edits
              key={index}
              chip={chip}
              index={index}
              info={chip.kind === "tag" ? lookup.data?.[key] : undefined}
              translation={chip.kind === "tag" ? (lookup.data?.[key]?.translation ?? translated.data?.[key] ?? undefined) : undefined}
              config={config}
              focused={index === focusIndex}
              dragging={drag?.from === index}
              dropBefore={drag !== null && drag.drop === index && drag.from !== index && drag.from + 1 !== index}
              disabled={disabled}
              open={editing === index}
              onOpenChange={(open) => {
                if (open && justDragged.current) return
                setEditing(open ? index : null)
              }}
              onKeyDown={onKeyDown(index)}
              onPointerDown={onPointerDown(index)}
              onFocus={() => setFocus(index)}
              register={(node) => {
                nodes.current[index] = node
              }}
              doc={doc}
              setValue={setValue}
            />
          )
        })}
        {drag !== null && drag.drop === count && drag.from !== count - 1 ? <li aria-hidden className="h-6 w-0.5 self-center rounded-full bg-primary" /> : null}
        <li className="min-w-32 flex-1">
          <AddTags doc={doc} setValue={setValue} disabled={disabled} empty={count === 0} config={config} kind={kind} />
        </li>
      </ul>
      <p id={liveId} role="status" aria-live="polite" className="sr-only">
        {announcement}
      </p>
    </div>
  )
}
