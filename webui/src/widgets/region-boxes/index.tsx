/**
 * `region-boxes`: the Regional Conditioning regions drawn over a preview of
 * the image (the generation form's width × height), moved and resized by
 * pointer or keyboard.
 *
 * The region params stay the source of truth: for every prefix in
 * `props.regions` the widget reads and writes the owner's `<prefix>_active`,
 * `_prompt`, `_x`, `_y`, `_width`, `_height` and `_strength` (the sliders
 * show the same values), and runs their change effects when an edit ends.
 * Its own value is unused (null).
 *
 * props: { regions: string[] }   value: null
 */
import { type KeyboardEvent, type PointerEvent, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import { cn } from "@/lib/utils"
import { runChangeEffects } from "@/ui/actions"
import { useForm, useFormState, useOwner } from "@/ui/hooks"
import type { WidgetProps } from "@/ui/widgets"
import { registerWidgetCatalogs } from "../_lib/i18n"
import { type Box, dragBox, type Handle, isArrowKey, LAYOUTS, nudgeBox, snapTargets } from "./geometry"

const NS = registerWidgetCatalogs("region-boxes", import.meta.glob("./locales/*.json", { eager: true, import: "default" }))

type Props = { regions?: string[] }

/** One colour per region (full class names, so Tailwind keeps them). */
const COLORS = [
  { box: "border-sky-500 bg-sky-500/20", badge: "bg-sky-500", swatch: "bg-sky-500" },
  { box: "border-amber-500 bg-amber-500/20", badge: "bg-amber-500", swatch: "bg-amber-500" },
  { box: "border-emerald-500 bg-emerald-500/20", badge: "bg-emerald-500", swatch: "bg-emerald-500" },
  { box: "border-fuchsia-500 bg-fuchsia-500/20", badge: "bg-fuchsia-500", swatch: "bg-fuchsia-500" },
]
const BOX_FIELDS = ["x", "y", "width", "height"] as const
const CORNERS: Handle[] = ["nw", "ne", "sw", "se"]
const EDGES: Handle[] = ["n", "s", "e", "w"]
/** Preview height cap (rem); the width follows the image's aspect ratio. */
const PREVIEW_REM = 18
const SNAP_THRESHOLD = 0.015

type Region = { prefix: string; number: number; active: boolean; prompt: string; strength: number; box: Box }

type Drag = { index: number; handle: Handle; pointer: number; startX: number; startY: number; start: Box; rect: DOMRect; changed: boolean }

function num(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback
}

const percent = (value: number) => `${(Math.max(0, Math.min(value, 1)) * 100).toFixed(3)}%`
const fixed = (value: number) => value.toFixed(2)

export default function RegionBoxes({ props, disabled, id, label }: WidgetProps<null, Props>) {
  const { t: tw } = useTranslation(NS)
  const { store, formId } = useForm()
  const owner = useOwner()
  const values = useFormState((s) => s.values[owner])
  const imageWidth = num(useFormState((s) => s.values.core?.width), 1024)
  const imageHeight = num(useFormState((s) => s.values.core?.height), 1024)
  const step = useFormState((s) => {
    const node = s.index.inputs.get(owner)?.get(`${props.regions?.[0]}_x`)
    return node && "step" in node && typeof node.step === "number" ? node.step : 0.01
  })
  const [selected, setSelected] = useState(0)
  const drag = useRef<Drag | null>(null)
  const canvas = useRef<HTMLDivElement>(null)

  const regions: Region[] = (props.regions ?? []).map((prefix, index) => ({
    prefix,
    number: index + 1,
    active: Boolean(values?.[`${prefix}_active`]),
    prompt: typeof values?.[`${prefix}_prompt`] === "string" ? (values[`${prefix}_prompt`] as string) : "",
    strength: num(values?.[`${prefix}_strength`], 1),
    box: {
      x: num(values?.[`${prefix}_x`], 0),
      y: num(values?.[`${prefix}_y`], 0),
      width: num(values?.[`${prefix}_width`], 0),
      height: num(values?.[`${prefix}_height`], 0),
    },
  }))
  const current = regions[selected] ? selected : 0

  const names = (prefix: string, fields: readonly string[]) => fields.map((field) => `${prefix}_${field}`)
  const effects = (changed: string[]) => runChangeEffects({ store, tab: formId }, owner, changed)

  const writeBox = (prefix: string, box: Box) => {
    store.getState().setValues(owner, Object.fromEntries(BOX_FIELDS.map((field) => [`${prefix}_${field}`, box[field]])))
  }

  const setActive = (region: Region, active: boolean) => {
    store.getState().setValue(owner, `${region.prefix}_active`, active)
    effects([`${region.prefix}_active`])
  }

  const applyLayout = (boxes: readonly Box[]) => {
    const changed: Record<string, unknown> = {}
    regions.forEach((region, index) => {
      const box = boxes[index]
      changed[`${region.prefix}_active`] = box !== undefined
      if (box) for (const field of BOX_FIELDS) changed[`${region.prefix}_${field}`] = box[field]
    })
    store.getState().setValues(owner, changed)
    effects(Object.keys(changed))
  }

  const begin = (event: PointerEvent<HTMLElement>, index: number, handle: Handle) => {
    if (disabled || event.button !== 0 || !canvas.current) return
    event.preventDefault()
    event.stopPropagation()
    setSelected(index)
    const region = regions[index]
    if (!region) return
    event.currentTarget.setPointerCapture?.(event.pointerId)
    const rect = canvas.current.getBoundingClientRect()
    drag.current = { index, handle, pointer: event.pointerId, startX: event.clientX, startY: event.clientY, start: region.box, rect, changed: false }
    ;(event.currentTarget.closest("[data-region]") as HTMLElement | null)?.focus({ preventScroll: true })
  }

  const move = (event: PointerEvent<HTMLElement>) => {
    const state = drag.current
    if (!state || state.pointer !== event.pointerId) return
    // The release was missed (outside the window, capture unsupported): stop, do not keep dragging on hover
    if ((event.buttons & 1) === 0) return end(event)
    if (state.rect.width <= 0 || state.rect.height <= 0) return
    const region = regions[state.index]
    if (!region) return
    const dx = (event.clientX - state.startX) / state.rect.width
    const dy = (event.clientY - state.startY) / state.rect.height
    const others = regions.filter((other, index) => index !== state.index && other.active).map((other) => other.box)
    // Shift turns snapping off for fine placement
    const box = dragBox(state.start, state.handle, dx, dy, { step, snaps: snapTargets(others), threshold: event.shiftKey ? 0 : SNAP_THRESHOLD })
    state.changed = true
    writeBox(region.prefix, box)
  }

  /** Ends the drag; the change effects run once, when it ends. */
  const end = (event: PointerEvent<HTMLElement>) => {
    const state = drag.current
    if (!state || state.pointer !== event.pointerId) return
    drag.current = null
    const region = regions[state.index]
    if (state.changed && region) effects(names(region.prefix, BOX_FIELDS))
  }

  const onKeyDown = (event: KeyboardEvent<HTMLElement>, index: number) => {
    const region = regions[index]
    if (disabled || !region || !isArrowKey(event.key)) return
    event.preventDefault()
    const amount = (event.altKey ? 10 : 1) * step
    writeBox(region.prefix, nudgeBox(region.box, event.key, amount, event.shiftKey, step))
    effects(names(region.prefix, BOX_FIELDS))
  }

  const describe = (region: Region) =>
    tw("Region {{number}}: x {{x}}, y {{y}}, width {{width}}, height {{height}}", {
      number: region.number,
      x: fixed(region.box.x),
      y: fixed(region.box.y),
      width: fixed(region.box.width),
      height: fixed(region.box.height),
    })

  const unused = (region: Region) => !region.prompt.trim() || region.strength <= 0 || region.box.width <= 0 || region.box.height <= 0
  const keysId = `${id}-keys`
  const ratio = imageWidth / imageHeight

  return (
    <div className="flex flex-col gap-2">
      <div role="toolbar" aria-label={tw("Region layouts")} className="flex flex-wrap gap-1">
        {LAYOUTS.filter((layout) => layout.boxes.length <= regions.length).map((layout) => (
          <Button key={layout.label} variant="outline" size="xs" disabled={disabled} onClick={() => applyLayout(layout.boxes)}>
            {tw(layout.label)}
          </Button>
        ))}
      </div>
      <div className="flex justify-center">
        <div
          ref={canvas}
          id={id}
          role="group"
          aria-label={label || tw("Regions")}
          aria-disabled={disabled || undefined}
          data-testid="region-canvas"
          className={cn("relative touch-none overflow-hidden rounded-md border bg-canvas select-none", disabled && "opacity-60")}
          style={{
            aspectRatio: `${imageWidth} / ${imageHeight}`,
            width: `min(100%, ${(PREVIEW_REM * ratio).toFixed(3)}rem)`,
            backgroundImage:
              "linear-gradient(to right, var(--border) 1px, transparent 1px), linear-gradient(to bottom, var(--border) 1px, transparent 1px)",
            backgroundSize: "25% 25%",
          }}
        >
          <span id={keysId} className="sr-only">
            {tw("Arrow keys move the region, Shift+arrow keys resize it, Alt moves in steps of ten.")}
          </span>
          {/* Fixed DOM order (moving a node drops its focus and pointer capture): z-index stacks
              the inactive regions underneath and the selected one on top */}
          {regions.map((region, index) => {
            const color = COLORS[index % COLORS.length]!
            const isSelected = index === current
            return (
              <div
                key={region.prefix}
                data-region={region.number}
                tabIndex={region.active && !disabled ? 0 : -1}
                role="group"
                aria-roledescription={tw("region")}
                aria-label={describe(region)}
                aria-describedby={keysId}
                aria-keyshortcuts={KEYSHORTCUTS}
                aria-hidden={region.active ? undefined : true}
                className={cn(
                  "absolute flex min-h-0 min-w-0 flex-col items-start gap-0.5 overflow-hidden rounded-sm border-2 p-1 text-[0.65rem] leading-tight outline-none",
                  color.box,
                  region.active ? "z-1 cursor-move focus-visible:ring-3 focus-visible:ring-ring/50" : "pointer-events-none z-0 border-dashed opacity-35",
                  region.active && unused(region) && "border-dashed",
                  isSelected && region.active && "z-10 shadow-md",
                )}
                style={{ left: percent(region.box.x), top: percent(region.box.y), width: percent(Math.min(region.box.width, 1 - region.box.x)), height: percent(Math.min(region.box.height, 1 - region.box.y)) }}
                onPointerDown={(event) => begin(event, index, "move")}
                onPointerMove={move}
                onPointerUp={end}
                onPointerCancel={end}
                onLostPointerCapture={end}
                onFocus={() => setSelected(index)}
                onKeyDown={(event) => onKeyDown(event, index)}
              >
                <span className={cn("rounded px-1 font-semibold text-white tabular", color.badge)}>{region.number}</span>
                <span className={cn("line-clamp-3 break-words text-foreground", !region.prompt.trim() && "text-muted-foreground italic")}>
                  {region.prompt.trim() || tw("Empty prompt: not used")}
                </span>
                {region.active && isSelected && !disabled
                  ? [...EDGES, ...CORNERS].map((handle) => (
                      <span
                        key={handle}
                        aria-hidden
                        data-handle={handle}
                        className={HANDLE_CLASS[handle]}
                        // Move/up/cancel bubble to the region's handlers (the handle holds the capture)
                        onPointerDown={(event) => begin(event, index, handle)}
                      />
                    ))
                  : null}
              </div>
            )
          })}
        </div>
      </div>
      <ul className="flex flex-wrap gap-x-3 gap-y-1" aria-label={tw("Regions")}>
        {regions.map((region, index) => {
          const color = COLORS[index % COLORS.length]!
          return (
            <li key={region.prefix} className="flex items-center gap-1.5 text-xs">
              <Checkbox
                checked={region.active}
                disabled={disabled}
                onCheckedChange={(checked) => setActive(region, checked === true)}
                aria-label={tw("Region {{number}} active", { number: region.number })}
              />
              <span aria-hidden className={cn("size-2.5 rounded-sm", color.swatch)} />
              <button
                type="button"
                className={cn("rounded px-1 tabular hover:bg-accent", index === current && "font-semibold")}
                disabled={disabled || !region.active}
                onClick={() => {
                  setSelected(index)
                  canvas.current?.querySelector<HTMLElement>(`[data-region="${region.number}"]`)?.focus()
                }}
              >
                {tw("Region {{number}}", { number: region.number })}
              </button>
            </li>
          )
        })}
      </ul>
    </div>
  )
}

const KEYSHORTCUTS = ["", "Shift+", "Alt+", "Alt+Shift+"]
  .flatMap((modifier) => ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].map((key) => modifier + key))
  .join(" ")

const HANDLE_BASE = "absolute z-20 touch-none"
const CORNER = `${HANDLE_BASE} size-2.5 rounded-[2px] border border-background bg-foreground`
const HANDLE_CLASS: Record<Handle, string> = {
  move: "",
  nw: `${CORNER} -top-0.5 -left-0.5 cursor-nwse-resize`,
  ne: `${CORNER} -top-0.5 -right-0.5 cursor-nesw-resize`,
  sw: `${CORNER} -bottom-0.5 -left-0.5 cursor-nesw-resize`,
  se: `${CORNER} -right-0.5 -bottom-0.5 cursor-nwse-resize`,
  n: `${HANDLE_BASE} inset-x-2 top-0 h-1.5 cursor-ns-resize`,
  s: `${HANDLE_BASE} inset-x-2 bottom-0 h-1.5 cursor-ns-resize`,
  e: `${HANDLE_BASE} inset-y-2 right-0 w-1.5 cursor-ew-resize`,
  w: `${HANDLE_BASE} inset-y-2 left-0 w-1.5 cursor-ew-resize`,
}
