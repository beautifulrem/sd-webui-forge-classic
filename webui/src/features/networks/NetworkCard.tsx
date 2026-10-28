/**
 * One network as a grid card (preview, name, badges, hover actions) or a list
 * row. The card itself is an `option` of the browser's listbox (selected = on:
 * in the prompt, the loaded model). The hover buttons are pointer shortcuts
 * drawn over it as a sibling, outside the accessibility tree (an option may
 * not contain buttons); the keyboard reaches the same actions through the
 * item menu (see NetworkGrid).
 */
import { AlertTriangle, Box, Check, ImagePlus, Info, Layers, MoreHorizontal, Pencil, Sparkles, Type } from "@/components/icons"
import { type CSSProperties, memo, type MouseEvent, type ReactNode, useState } from "react"
import { useTranslation } from "react-i18next"
import { api } from "@/api/endpoints"
import type { NetworkItem, NetworkKind } from "@/api/types"
import { formatBytes } from "@/features/model-bar/model-options"
import { useMediaQuery } from "@/lib/use-media-query"
import { cn } from "@/lib/utils"
import { isEditable, type MenuAnchor, thumbWidth } from "./items"
import { useArchLabel } from "./labels"

export type ItemAction = "metadata" | "edit" | "preview" | "menu"

export type CardProps = {
  item: NetworkItem
  kind: NetworkKind | undefined
  id: string
  index: number
  active: boolean
  cursor: boolean
  /** The item was made for another architecture than the current preset's. */
  mismatch: boolean
  width: number
  onApply: (item: NetworkItem, index: number) => void
  onAction: (action: ItemAction, item: NetworkItem, index: number, anchor: MenuAnchor) => void
}

/** A hue from the name, for previews that are missing. */
function hueOf(text: string): number {
  let hash = 7
  for (let i = 0; i < text.length; i++) hash = (hash * 31 + text.charCodeAt(i)) % 360
  return hash
}

const gradient = (hue: number): CSSProperties => ({
  backgroundImage: `linear-gradient(135deg, oklch(0.62 0.09 ${hue}), oklch(0.34 0.06 ${(hue + 70) % 360}))`,
})

const KIND_ICON: Record<string, typeof Layers> = { lora: Layers, embeddings: Type, checkpoints: Box, vae: Sparkles, text_encoders: Type }

/**
 * The preview with its blurred placeholder underneath (a muted loop for video
 * previews, still when the user prefers reduced motion); a tinted monogram
 * without one.
 */
export function Preview({ item, width, className, rounded = "" }: { item: NetworkItem; width: number; className?: string; rounded?: string }) {
  const [loaded, setLoaded] = useState(false)
  const [failed, setFailed] = useState(false)
  const still = useMediaQuery("(prefers-reduced-motion: reduce)")
  const dpr = typeof window === "undefined" ? 1 : Math.min(window.devicePixelRatio || 1, 2)
  if (!item.preview || failed) {
    const Icon = KIND_ICON[item.kind] ?? Layers
    return (
      <div aria-hidden className={cn("flex items-center justify-center text-white/80", rounded, className)} style={gradient(hueOf(item.name))}>
        <Icon className="size-[28%] max-h-12 max-w-12 opacity-70" strokeWidth={1.5} />
      </div>
    )
  }
  const media = "absolute inset-0 size-full object-cover opacity-0 transition-opacity duration-200 data-loaded:opacity-100"
  return (
    <div aria-hidden className={cn("relative overflow-hidden bg-muted", rounded, className)}>
      {item.placeholder && !loaded ? <img src={item.placeholder} alt="" className="absolute inset-0 size-full scale-110 object-cover blur-md" draggable={false} /> : null}
      {item.preview_type === "video" ? (
        <video
          src={api.fileUrl(item.preview)}
          muted
          loop
          playsInline
          autoPlay={!still}
          preload={still ? "metadata" : "auto"}
          disablePictureInPicture
          data-loaded={loaded || undefined}
          onLoadedData={() => setLoaded(true)}
          onError={() => setFailed(true)}
          className={media}
        />
      ) : (
        <img
          src={api.thumbUrl(item.preview, thumbWidth(width * dpr))}
          alt=""
          loading="lazy"
          decoding="async"
          draggable={false}
          data-loaded={loaded || undefined}
          onLoad={() => setLoaded(true)}
          onError={() => setFailed(true)}
          className={media}
        />
      )}
    </div>
  )
}

function useAccessibleName(item: NetworkItem, active: boolean, kind: NetworkKind | undefined, mismatch: boolean) {
  const { t } = useTranslation()
  const archLabel = useArchLabel()
  const parts = [item.title]
  if (item.arch) parts.push(archLabel(item.arch))
  if (item.folder) parts.push(item.folder)
  if (mismatch) parts.push(t("made for another architecture"))
  if (active) parts.push(kind?.action === "checkpoint" ? t("loaded") : kind?.action === "module" ? t("selected") : t("in the prompt"))
  return parts.join(", ")
}

/** The pointer shortcuts of a card or row (aria-hidden, out of the tab order: the item menu has them all). */
function HoverActions({ item, kind, index, onAction, variant, className }: Pick<CardProps, "item" | "kind" | "index" | "onAction"> & { variant: "card" | "row"; className?: string }) {
  const { t } = useTranslation()
  const editable = isEditable(item)
  const buttons: { action: ItemAction; label: string; icon: ReactNode }[] = [
    { action: "metadata", label: t("Metadata"), icon: <Info /> },
    ...(editable && (kind?.fields.length ?? 0) > 0 ? [{ action: "edit" as const, label: t("Edit metadata"), icon: <Pencil /> }] : []),
    ...(editable && variant === "card" ? [{ action: "preview" as const, label: t("Set preview"), icon: <ImagePlus /> }] : []),
    { action: "menu", label: t("More actions"), icon: <MoreHorizontal /> },
  ]
  return (
    <div aria-hidden className={className}>
      {buttons.map((button) => (
        <button
          key={button.action}
          type="button"
          tabIndex={-1}
          title={button.label}
          onClick={(event: MouseEvent<HTMLButtonElement>) => {
            event.stopPropagation()
            onAction(button.action, item, index, event.currentTarget)
          }}
          className={
            variant === "card"
              ? "grid size-7 place-items-center rounded-md bg-black/45 text-white transition-colors outline-none hover:bg-black/70 [&_svg]:size-3.5"
              : "grid size-7 place-items-center rounded-md text-muted-foreground outline-none hover:bg-background hover:text-foreground [&_svg]:size-3.5"
          }
        >
          {button.icon}
        </button>
      ))}
    </div>
  )
}

const contextMenu = (onAction: CardProps["onAction"], item: NetworkItem, index: number) => (event: MouseEvent<HTMLElement>) => {
  event.preventDefault()
  onAction("menu", item, index, { x: event.clientX, y: event.clientY, element: event.currentTarget })
}

function CardView({ item, kind, id, index, active, cursor, mismatch, width, onApply, onAction }: CardProps) {
  const archLabel = useArchLabel()
  const label = useAccessibleName(item, active, kind, mismatch)
  return (
    <div role="none" data-cursor={cursor || undefined} className="group/card relative size-full">
      <div
        id={id}
        role="option"
        aria-selected={active}
        aria-label={label}
        data-index={index}
        data-cursor={cursor || undefined}
        data-active={active || undefined}
        onClick={() => onApply(item, index)}
        onContextMenu={contextMenu(onAction, item, index)}
        className={cn(
          "relative size-full cursor-pointer overflow-hidden rounded-md bg-card shadow-xs ring-1 ring-border transition-shadow select-none",
          "group-hover/card:shadow-md group-hover/card:ring-foreground/20",
          "data-active:ring-2 data-active:ring-primary",
          "data-cursor:outline-2 data-cursor:outline-offset-2 data-cursor:outline-ring/70",
        )}
      >
        <Preview item={item} width={width} className={cn("absolute inset-0 transition-[filter]", mismatch && "opacity-70 saturate-50")} />
        <div className="pointer-events-none absolute inset-x-0 bottom-0 bg-linear-to-t from-black/80 via-black/40 to-transparent px-2.5 pt-8 pb-2 text-white">
          <p className="line-clamp-2 text-[0.8rem] leading-snug font-medium [overflow-wrap:anywhere] drop-shadow-sm">{item.title}</p>
          {item.folder ? <p className="truncate text-[0.68rem] leading-4 text-white/70">{item.folder}</p> : null}
        </div>
        <div className="pointer-events-none absolute top-1.5 left-1.5 flex max-w-[70%] flex-wrap gap-1">
          {item.arch ? (
            <span className={cn("rounded-md px-1.5 py-px text-[0.65rem] leading-4 font-medium", mismatch ? "bg-warning/85 text-black" : "bg-black/50 text-white")}>
              {mismatch ? <AlertTriangle className="-mt-px mr-0.5 inline size-2.5" aria-hidden /> : null}
              {archLabel(item.arch)}
            </span>
          ) : null}
          {item.subtype && item.subtype !== "LoRA" ? <span className="rounded-md bg-black/50 px-1.5 py-px text-[0.65rem] leading-4 font-medium text-white">{item.subtype}</span> : null}
        </div>
        {active ? (
          <span aria-hidden className="absolute right-1.5 bottom-1.5 grid size-5 place-items-center rounded-full bg-primary text-primary-foreground shadow-sm ring-2 ring-black/20">
            <Check className="size-3" strokeWidth={3} />
          </span>
        ) : null}
      </div>
      <HoverActions
        item={item}
        kind={kind}
        index={index}
        onAction={onAction}
        variant="card"
        className={cn(
          "absolute top-1.5 right-1.5 flex flex-col gap-1 opacity-0 transition-opacity duration-150",
          // invisible buttons must not take clicks meant for the card
          "pointer-events-none group-hover/card:pointer-events-auto group-hover/card:opacity-100 group-data-cursor/card:pointer-events-auto group-data-cursor/card:opacity-100",
          "[@media(hover:none)]:pointer-events-auto [@media(hover:none)]:opacity-100",
        )}
      />
    </div>
  )
}

function formatDate(seconds: number | null | undefined, locale: string): string | null {
  if (!seconds) return null
  return new Date(seconds * 1000).toLocaleDateString(locale, { year: "numeric", month: "short", day: "numeric" })
}

function RowView({ item, kind, id, index, active, cursor, mismatch, onApply, onAction }: CardProps) {
  const { i18n } = useTranslation()
  const archLabel = useArchLabel()
  const label = useAccessibleName(item, active, kind, mismatch)
  return (
    <div role="none" data-cursor={cursor || undefined} className="group/row relative size-full">
      <div
        id={id}
        role="option"
        aria-selected={active}
        aria-label={label}
        data-index={index}
        data-cursor={cursor || undefined}
        data-active={active || undefined}
        onClick={() => onApply(item, index)}
        onContextMenu={contextMenu(onAction, item, index)}
        className={cn(
          "flex size-full cursor-pointer items-center gap-3 rounded-lg px-2 transition-colors select-none group-hover/row:bg-muted",
          "data-cursor:bg-accent data-cursor:ring-1 data-cursor:ring-ring/50",
        )}
      >
        <div className="relative size-10 shrink-0">
          <Preview item={item} width={48} rounded="rounded-md" className={cn("absolute inset-0", mismatch && "opacity-70 saturate-50")} />
          {active ? (
            <span aria-hidden className="absolute -right-1 -bottom-1 grid size-4 place-items-center rounded-full bg-primary text-primary-foreground ring-2 ring-background">
              <Check className="size-2.5" strokeWidth={3} />
            </span>
          ) : null}
        </div>
        <div className="min-w-0 flex-1">
          <p className={cn("truncate text-sm", active && "font-medium text-primary")}>{item.title}</p>
          <p className="truncate text-xs text-muted-foreground">{item.path}</p>
        </div>
        <div className="hidden shrink-0 items-center gap-1 @md:flex">
          {item.subtype && item.subtype !== "LoRA" ? <span className="rounded-md bg-muted px-1.5 text-[0.68rem] leading-5 text-muted-foreground">{item.subtype}</span> : null}
          {item.arch ? (
            <span className={cn("rounded-md px-1.5 text-[0.68rem] leading-5", mismatch ? "bg-warning/20 text-[color-mix(in_oklch,var(--warning)_70%,var(--foreground))]" : "bg-muted text-muted-foreground")}>
              {archLabel(item.arch)}
            </span>
          ) : null}
        </div>
        <span className="hidden w-16 shrink-0 text-right text-xs text-muted-foreground tabular-nums @lg:block">{formatBytes(item.size) ?? ""}</span>
        <span className="hidden w-24 shrink-0 text-right text-xs text-muted-foreground tabular-nums @xl:block">{formatDate(item.mtime, i18n.language) ?? ""}</span>
      </div>
      <HoverActions
        item={item}
        kind={kind}
        index={index}
        onAction={onAction}
        variant="row"
        // over the size and date columns, like mail rows: hover shows the actions in their place
        className={cn(
          "pointer-events-none absolute inset-y-1 right-1 flex items-center gap-0.5 rounded-md bg-muted pl-1.5 opacity-0 transition-opacity",
          "group-hover/row:pointer-events-auto group-hover/row:opacity-100 group-data-cursor/row:pointer-events-auto group-data-cursor/row:bg-accent group-data-cursor/row:opacity-100",
          "[@media(hover:none)]:pointer-events-auto [@media(hover:none)]:bg-transparent [@media(hover:none)]:opacity-100",
        )}
      />
    </div>
  )
}

/*
 * Memoised: the virtualized grid renders again on every scroll frame, and only the cards of
 * rows coming into view are new; the others keep their props (NetworkGrid passes stable
 * callbacks) and are skipped.
 */
export const NetworkCard = memo(CardView)
export const NetworkRow = memo(RowView)
