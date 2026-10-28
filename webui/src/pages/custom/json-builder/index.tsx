/**
 * JSON Builder (upstream `sd_forge_json_builder`, a JavaScript-only tab):
 * a visual editor for MOSS-Image's structured ("Layout") prompts. Set the
 * canvas size, a summary and a list of items (object, character or text,
 * each with a caption and a box drawn on the canvas), then send the compiled
 * prompt with its width and height to txt2img or img2img. "Load prompt"
 * turns a compiled prompt back into a draft; drafts save to and load from
 * JSON files. The draft is kept in the browser (`./store`).
 */
import { Braces, ClipboardCopy, FileDown, FileUp, Images, Plus, Ratio, Trash2, Wand, X } from "@/components/icons"
import { useId, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { toast } from "sonner"
import type { FormTab } from "@/api/types"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { NumberField } from "@/components/ui/number-field"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Textarea } from "@/components/ui/textarea"
import { Segmented } from "@/components/Segmented"
import { copyText } from "@/features/gallery/files"
import { deliver } from "@/features/send-to/send-to"
import { cn } from "@/lib/utils"
import type { CustomPageProps } from "@/pages/ext"
import { CANVAS_TYPES, LABELS } from "./compile"
import { type Box, type CanvasType, decompilePrompt, type Label, splitPrompt } from "./decompile"
import { ASPECT_RATIOS, type AspectRatio, buildDraft, finalPrompt, type Item, LABEL_NAMES, type Missing, missingFields, resolutionFor, round16 } from "./draft"
import { LayoutCanvas } from "./LayoutCanvas"
import { sizeLimit } from "./size-limit"
import { useBuilder } from "./store"

const CANVAS_NAMES: Record<CanvasType, string> = { single: "Single", multi_panel: "Multi-panel", collage: "Collage" }

const message = (error: unknown) => (error instanceof Error ? error.message : String(error))

function Section({ title, children, className, actions }: { title: string; children: React.ReactNode; className?: string; actions?: React.ReactNode }) {
  return (
    <section className={cn("flex flex-col gap-2.5 rounded-md bg-card p-3", className)}>
      <div className="flex min-h-6 items-center justify-between gap-2">
        <h2 className="text-[0.8rem] font-medium text-muted-foreground">{title}</h2>
        {actions}
      </div>
      {children}
    </section>
  )
}

function FieldLabel({ htmlFor, children }: { htmlFor: string; children: React.ReactNode }) {
  return (
    <label htmlFor={htmlFor} className="text-[0.8rem] font-medium text-foreground/85">
      {children}
    </label>
  )
}

function SizeField({ id, label, value, onChange }: { id: string; label: string; value: number; onChange: (value: number, commit: boolean) => void }) {
  return (
    <div className="flex min-w-0 flex-1 flex-col gap-1">
      <FieldLabel htmlFor={id}>{label}</FieldLabel>
      <NumberField
        id={id}
        value={value}
        min={16}
        step={16}
        format={{ maximumFractionDigits: 0, useGrouping: false }}
        onValueChange={(next) => next !== null && next > 0 && onChange(next, false)}
        onValueCommitted={(next) => next !== null && next > 0 && onChange(next, true)}
      />
    </div>
  )
}

function ResolutionSection() {
  const { t } = useTranslation()
  const id = useId()
  const width = useBuilder((s) => s.width)
  const height = useBuilder((s) => s.height)
  const mp = useBuilder((s) => s.mp)
  const ar = useBuilder((s) => s.ar)
  const set = useBuilder((s) => s.set)
  const setSize = useBuilder((s) => s.setSize)
  const applyMp = () => {
    const size = resolutionFor(mp, ar)
    if (size) setSize(size.width, size.height, true)
  }
  return (
    <Section title={t("Resolution")}>
      <div className="flex gap-2">
        <SizeField id={`${id}-w`} label={t("Width")} value={width} onChange={(v, commit) => (commit ? setSize(round16(v), height, true) : setSize(v, height))} />
        <SizeField id={`${id}-h`} label={t("Height")} value={height} onChange={(v, commit) => (commit ? setSize(width, round16(v), true) : setSize(width, v))} />
      </div>
      <div className="flex items-end gap-2">
        <div className="flex w-24 flex-col gap-1">
          <FieldLabel htmlFor={`${id}-mp`}>{t("Megapixels")}</FieldLabel>
          <NumberField id={`${id}-mp`} value={mp} min={0.1} step={0.1} format={{ maximumFractionDigits: 2 }} onValueChange={(next) => next !== null && next > 0 && set({ mp: next })} />
        </div>
        <div className="flex w-24 flex-col gap-1">
          <span id={`${id}-ar`} className="text-[0.8rem] font-medium text-foreground/85">
            {t("Aspect ratio")}
          </span>
          <Select<AspectRatio> items={ASPECT_RATIOS.map((value) => ({ value, label: value }))} value={ar} onValueChange={(next) => next !== null && set({ ar: next })}>
            <SelectTrigger aria-labelledby={`${id}-ar`} className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {ASPECT_RATIOS.map((value) => (
                <SelectItem key={value} value={value}>
                  {value}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <Button variant="secondary" className="flex-1" onClick={applyMp}>
          <Ratio aria-hidden />
          {t("Set resolution")}
        </Button>
      </div>
    </Section>
  )
}

function LoadPromptDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) {
  const { t } = useTranslation()
  const id = useId()
  const load = useBuilder((s) => s.load)
  const set = useBuilder((s) => s.set)
  const width = useBuilder((s) => s.width)
  const height = useBuilder((s) => s.height)
  const [text, setText] = useState("")
  const [size, setSize] = useState({ width: 1024, height: 1024 })
  const [error, setError] = useState<string | null>(null)
  const [wasOpen, setWasOpen] = useState(open)
  if (open !== wasOpen) {
    setWasOpen(open)
    if (open) {
      // Each opening starts empty, sized like the current canvas (upstream).
      setText("")
      setSize({ width, height })
      setError(null)
    }
  }
  const submit = () => {
    if (text.trim() === "") {
      onOpenChange(false)
      return
    }
    try {
      const { json, extra } = splitPrompt(text)
      load(decompilePrompt(json, size.width, size.height))
      if (extra) set({ extra })
      onOpenChange(false)
      toast.success(t("Prompt loaded"))
    } catch (err) {
      setError(message(err))
    }
  }
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>{t("Load prompt")}</DialogTitle>
          <DialogDescription>{t("Paste a compiled MOSS-Image prompt (with the size it was made for) or a saved layout draft.")}</DialogDescription>
        </DialogHeader>
        <form
          className="flex flex-col gap-3"
          onSubmit={(event) => {
            event.preventDefault()
            submit()
          }}
        >
          <div className="flex flex-col gap-3 sm:flex-row">
            <Textarea
              aria-label={t("Compiled prompt or layout draft")}
              placeholder={t("Compiled prompt or layout draft")}
              spellCheck={false}
              autoFocus
              value={text}
              aria-invalid={error !== null || undefined}
              aria-describedby={error ? `${id}-error` : undefined}
              onChange={(event) => {
                setText(event.target.value)
                setError(null)
              }}
              className="field-sizing-fixed h-64 min-w-0 flex-1 resize-none font-mono text-xs md:text-xs"
            />
            <div className="flex gap-2 sm:w-32 sm:flex-col">
              <SizeField id={`${id}-w`} label={t("Width")} value={size.width} onChange={(width) => setSize((s) => ({ ...s, width }))} />
              <SizeField id={`${id}-h`} label={t("Height")} value={size.height} onChange={(height) => setSize((s) => ({ ...s, height }))} />
            </div>
          </div>
          {error ? (
            <p id={`${id}-error`} role="alert" className="rounded-lg bg-destructive/10 px-3 py-2 text-sm break-words text-destructive">
              {error}
            </p>
          ) : null}
          <DialogFooter>
            <Button type="submit">{t("Load")}</Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}

function FileSection() {
  const { t } = useTranslation()
  const load = useBuilder((s) => s.load)
  const fileInput = useRef<HTMLInputElement>(null)
  const [dialog, setDialog] = useState(false)

  const save = () => {
    const blob = new Blob([`${JSON.stringify(buildDraft(useBuilder.getState()), null, 2)}\n`], { type: "application/json" })
    const url = URL.createObjectURL(blob)
    const link = document.createElement("a")
    link.href = url
    link.download = "layout.json"
    document.body.append(link)
    link.click()
    link.remove()
    setTimeout(() => URL.revokeObjectURL(url), 1000)
  }

  const open = async (file: File) => {
    try {
      load(decompilePrompt(await file.text(), Number.NaN, Number.NaN))
      toast.success(t("Layout loaded"))
    } catch (err) {
      toast.error(t("Could not load {{name}}", { name: file.name }), { description: message(err) })
    }
  }

  return (
    <Section title={t("Save / load")}>
      <Button variant="secondary" onClick={() => setDialog(true)}>
        <Braces aria-hidden />
        {t("Load prompt…")}
      </Button>
      <div className="flex gap-2">
        <Button variant="outline" className="flex-1" onClick={save}>
          <FileDown aria-hidden />
          {t("Save JSON")}
        </Button>
        <Button variant="outline" className="flex-1" onClick={() => fileInput.current?.click()}>
          <FileUp aria-hidden />
          {t("Load JSON")}
        </Button>
      </div>
      <input
        ref={fileInput}
        type="file"
        accept=".json,application/json"
        hidden
        data-testid="json-builder-file"
        onChange={(event) => {
          const file = event.target.files?.[0]
          event.target.value = ""
          if (file) void open(file)
        }}
      />
      <LoadPromptDialog open={dialog} onOpenChange={setDialog} />
    </Section>
  )
}

function SummarySection({ missing }: { missing: ReadonlySet<Missing> }) {
  const { t } = useTranslation()
  const id = useId()
  const caption = useBuilder((s) => s.caption)
  const meta = useBuilder((s) => s.meta)
  const extra = useBuilder((s) => s.extra)
  const canvas = useBuilder((s) => s.canvas)
  const set = useBuilder((s) => s.set)
  return (
    <Section title={t("Summary")}>
      <Segmented<CanvasType>
        aria-label={t("Canvas type")}
        className="w-full"
        options={CANVAS_TYPES.map((value) => ({ value, label: t(CANVAS_NAMES[value]) }))}
        value={canvas}
        onValueChange={(value) => set({ canvas: value })}
      />
      <div className="flex flex-col gap-1">
        <FieldLabel htmlFor={`${id}-caption`}>{t("Caption")}</FieldLabel>
        <Textarea
          id={`${id}-caption`}
          rows={2}
          className="min-h-14"
          placeholder={t("The whole picture in a sentence or two")}
          aria-invalid={missing.has("caption") || undefined}
          value={caption}
          onChange={(event) => set({ caption: event.target.value })}
        />
      </div>
      <div className="flex gap-2">
        <div className="flex min-w-0 flex-1 flex-col gap-1">
          <FieldLabel htmlFor={`${id}-meta`}>{t("Meta")}</FieldLabel>
          <Input id={`${id}-meta`} placeholder={t("Tags, comma separated")} value={meta} onChange={(event) => set({ meta: event.target.value })} />
        </div>
        <div className="flex min-w-0 flex-1 flex-col gap-1">
          <FieldLabel htmlFor={`${id}-extra`}>{t("Extra")}</FieldLabel>
          <Input id={`${id}-extra`} placeholder={t("e.g. a LoRA")} title={t("Added after the compiled prompt")} value={extra} onChange={(event) => set({ extra: event.target.value })} />
        </div>
      </div>
    </Section>
  )
}

const CORNERS = ["x1", "y1", "x2", "y2"] as const

function ItemRow({ item, index, active, missing, onActivate }: { item: Item; index: number; active: boolean; missing: ReadonlySet<Missing>; onActivate: (id: number) => void }) {
  const { t } = useTranslation()
  const id = useId()
  const updateItem = useBuilder((s) => s.updateItem)
  const removeItem = useBuilder((s) => s.removeItem)
  return (
    <li
      data-item={item.id}
      onFocusCapture={() => onActivate(item.id)}
      onPointerDown={() => onActivate(item.id)}
      className={cn("flex flex-col gap-1.5 rounded-lg border border-transparent bg-muted/50 p-2 transition-colors hover:border-border", active && "border-primary/50 bg-primary/5")}
    >
      <div className="flex items-center gap-1.5">
        <span className="w-5 shrink-0 text-center text-xs text-muted-foreground tabular-nums">{index + 1}</span>
        <Segmented<Label>
          size="sm"
          aria-label={t("Item type")}
          className="flex-1"
          options={LABELS.map((value) => ({ value, label: t(LABEL_NAMES[value]) }))}
          value={item.label}
          onValueChange={(label) => updateItem(item.id, { label })}
        />
        <Button size="icon-xs" variant="ghost" aria-label={t("Remove item {{n}}", { n: index + 1 })} title={t("Remove")} onClick={() => removeItem(item.id)}>
          <X aria-hidden />
        </Button>
      </div>
      <Textarea
        aria-label={t("Caption of item {{n}}", { n: index + 1 })}
        placeholder={t("Caption")}
        rows={1}
        className="min-h-8 py-1.5"
        aria-invalid={missing.has(`item:${item.id}:caption`) || undefined}
        value={item.caption}
        onChange={(event) => updateItem(item.id, { caption: event.target.value })}
      />
      {item.label === "text" ? (
        <Input
          aria-label={t("Text to render of item {{n}}", { n: index + 1 })}
          placeholder={t("Text to render")}
          aria-invalid={missing.has(`item:${item.id}:text`) || undefined}
          value={item.text}
          onChange={(event) => updateItem(item.id, { text: event.target.value })}
        />
      ) : null}
      <div className="grid grid-cols-4 gap-1.5" role="group" aria-label={t("Box of item {{n}}", { n: index + 1 })}>
        {CORNERS.map((corner, i) => (
          <div key={corner} className="flex min-w-0 items-center gap-1">
            <label htmlFor={`${id}-${corner}`} className="text-[0.7rem] text-muted-foreground">
              {corner}
            </label>
            <NumberField
              id={`${id}-${corner}`}
              size="sm"
              steppers={false}
              className="flex-1"
              value={item.bbox[i]!}
              format={{ maximumFractionDigits: 2, useGrouping: false }}
              onValueChange={(next) => {
                if (next === null) return
                const bbox = item.bbox.slice() as Box
                bbox[i] = next
                updateItem(item.id, { bbox })
              }}
            />
          </div>
        ))}
      </div>
    </li>
  )
}

function ItemsSection({ activeId, onActivate, missing }: { activeId: number | null; onActivate: (id: number) => void; missing: ReadonlySet<Missing> }) {
  const { t } = useTranslation()
  const items = useBuilder((s) => s.items)
  const addItem = useBuilder((s) => s.addItem)
  const clearItems = useBuilder((s) => s.clearItems)
  return (
    <Section
      title={t("Items")}
      className="lg:min-h-48 lg:flex-1"
      actions={
        <div className="flex gap-1">
          <Button size="sm" variant="secondary" onClick={addItem}>
            <Plus aria-hidden />
            {t("Add item")}
          </Button>
          <Button size="sm" variant="ghost" onClick={clearItems}>
            <Trash2 aria-hidden />
            {t("Clear all")}
          </Button>
        </div>
      }
    >
      <ul className="-mr-1 flex min-h-0 flex-col gap-2 overflow-y-auto pr-1 lg:flex-1" aria-label={t("Items")}>
        {items.map((item, index) => (
          <ItemRow key={item.id} item={item} index={index} active={item.id === activeId} missing={missing} onActivate={onActivate} />
        ))}
      </ul>
    </Section>
  )
}

const TARGET_NAMES: Record<Extract<FormTab, "txt2img" | "img2img">, string> = { txt2img: "txt2img", img2img: "img2img" }

export default function JsonBuilderPage(_props: CustomPageProps) {
  const { t } = useTranslation()
  const [activeId, setActiveId] = useState<number | null>(null)
  const [showMissing, setShowMissing] = useState(false)
  const state = useBuilder()
  const missing = showMissing ? missingFields(state) : new Set<Missing>()

  /** The prompt, or null after telling what is wrong. */
  const compile = (): string | null => {
    const empty = missingFields(state)
    if (empty.size > 0) {
      setShowMissing(true)
      toast.error(t("Fill in the highlighted fields"))
      return null
    }
    setShowMissing(false)
    try {
      return finalPrompt(state)
    } catch (err) {
      toast.error(t("Could not build the prompt"), { description: message(err) })
      return null
    }
  }

  const send = (tab: "txt2img" | "img2img") => {
    const prompt = compile()
    if (prompt === null) return
    const target = t(TARGET_NAMES[tab])
    const { width, height } = state
    deliver(tab, [{ owner: "core", values: { prompt, width, height } }])
      .then((count) => {
        if (count === 0) return void toast.error(t("Could not send to {{target}}", { target }))
        const limit = sizeLimit(tab, width, height)
        if (limit === null) toast.success(t("Sent to {{target}}", { target }))
        else toast.warning(t("Sent to {{target}}, but it accepts at most {{max}} px per side: change the size before generating", { target, max: limit }))
      })
      .catch((err: unknown) => toast.error(t("Could not send to {{target}}", { target }), { description: message(err) }))
  }

  const copy = () => {
    const prompt = compile()
    if (prompt !== null) void copyText(prompt).then((ok) => (ok ? toast.success(t("Copied")) : toast.error(t("Could not copy"))))
  }

  return (
    <div className="h-full overflow-y-auto lg:overflow-hidden" data-scroll-root>
      <div className="flex min-h-full flex-col gap-3 p-3 sm:p-4 lg:h-full lg:flex-row">
        <LayoutCanvas className="h-[60vh] min-h-80 shrink-0 lg:h-auto lg:min-h-0 lg:flex-1 lg:shrink" activeId={activeId} onActivate={setActiveId} missing={missing} />
        <div className="flex min-h-0 flex-col gap-3 lg:w-[26rem] lg:shrink-0 lg:overflow-y-auto xl:w-[30rem]">
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-1 xl:grid-cols-2">
            <ResolutionSection />
            <FileSection />
          </div>
          <SummarySection missing={missing} />
          <ItemsSection activeId={activeId} onActivate={setActiveId} missing={missing} />
          <div className="sticky bottom-0 flex gap-2 bg-background/95 py-1 lg:static lg:bg-transparent lg:py-0" data-toast-obstacle>
            <Button className="flex-1" onClick={() => send("txt2img")}>
              <Wand aria-hidden />
              {t("Send to txt2img")}
            </Button>
            <Button variant="secondary" className="flex-1" onClick={() => send("img2img")}>
              <Images aria-hidden />
              {t("Send to img2img")}
            </Button>
            <Button size="icon" variant="outline" aria-label={t("Copy prompt")} title={t("Copy prompt")} onClick={copy}>
              <ClipboardCopy aria-hidden />
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}
