/**
 * Gallery: every output the server indexed (txt2img, img2img, Extras,
 * grids, saved copies), filtered and sorted, in a virtualized justified or
 * masonry grid. Select (click the check, Ctrl / Cmd-click, Shift-click for a
 * range, Ctrl / Cmd+A for all loaded) for bulk actions: download as zip
 * (at most 2000 images), star, compare two, remove or delete (sent in
 * batches the server takes). The viewer shows the parameters next
 * to the image with "use parameters", send-to, PNG Info and show in folder.
 */
import { useInfiniteQuery } from "@tanstack/react-query"
import { Columns2, Download, GalleryVerticalEnd, RefreshCw, Star, StarOff, Trash2, X } from "lucide-react"
import { lazy, Suspense, useEffect, useState } from "react"
import { useTranslation } from "react-i18next"
import { toast } from "sonner"
import { api } from "@/api/endpoints"
import { queryClient } from "@/api/queries"
import type { GalleryItem } from "@/api/types"
import { useRegisterCommands } from "@/app/commands"
import { HeaderSlot } from "@/app/header-slot"
import { PagePlaceholder } from "@/components/PagePlaceholder"
import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { CompareDialog, type CompareImage } from "@/features/library/CompareDialog"
import { DeleteDialog } from "@/features/library/DeleteDialog"
import { activeFilters, galleryQuery, useGalleryView } from "@/features/library/filters"
import { GalleryDetails } from "@/features/library/GalleryDetails"
import { GalleryGrid } from "@/features/library/GalleryGrid"
import { errorText } from "@/features/errors/error-text"
import { GalleryToolbar } from "@/features/library/GalleryToolbar"
import { asResult, downloadZip, galleryKeys, galleryListQuery, removeOutputs, starOutputs } from "@/features/library/queries"
import { EMPTY_SELECTION, modeOf, ordered, prune, select, selectAll, type Selection } from "@/features/library/selection"

const Lightbox = lazy(() => import("@/features/lightbox/Lightbox").then((m) => ({ default: m.Lightbox })))

function BulkBar(props: { count: number; onClear: () => void; onZip: () => void; onStar: (starred: boolean) => void; onCompare: (() => void) | null; onDelete: () => void }) {
  const { t } = useTranslation()
  const action = (label: string, onClick: () => void, icon: React.ReactNode, destructive = false) => (
    <Tooltip>
      <TooltipTrigger render={<Button variant={destructive ? "destructive" : "ghost"} size="icon-sm" aria-label={label} onClick={onClick} />}>{icon}</TooltipTrigger>
      <TooltipContent>{label}</TooltipContent>
    </Tooltip>
  )
  return (
    <div
      role="toolbar"
      aria-label={t("Selected images")}
      className="pointer-events-auto flex items-center gap-1 rounded-xl border bg-popover/95 p-1 pl-3 shadow-lg backdrop-blur animate-in fade-in-0 slide-in-from-bottom-2"
    >
      <span className="mr-1 text-sm font-medium tabular-nums">{t("{{count}} selected", { count: props.count })}</span>
      {action(t("Download as zip"), props.onZip, <Download />)}
      {action(t("Star"), () => props.onStar(true), <Star />)}
      {action(t("Unstar"), () => props.onStar(false), <StarOff />)}
      {props.onCompare ? action(t("Compare"), props.onCompare, <Columns2 />) : null}
      {action(t("Remove…"), props.onDelete, <Trash2 />, true)}
      {action(t("Clear selection"), props.onClear, <X />)}
    </div>
  )
}

export default function GalleryPage() {
  const { t } = useTranslation()
  const view = useGalleryView()
  const query = galleryQuery(view)
  const list = useInfiniteQuery(galleryListQuery(query))
  const items = list.data?.pages.flatMap((page) => page.items) ?? []
  const order = items.map((item) => item.id)
  // Once every page is in, the count is what the pages hold: it can never disagree with the tiles.
  const total = list.data && !list.hasNextPage ? items.length : (list.data?.pages[0]?.total ?? null)
  // Pages that came back empty (the server skipped files that are gone) are not the end: read on.
  const { fetchNextPage, isFetchingNextPage } = list
  const readingOn = items.length === 0 && list.hasNextPage && !list.isFetchNextPageError
  useEffect(() => {
    if (readingOn && !isFetchingNextPage) void fetchNextPage()
  }, [readingOn, isFetchingNextPage, fetchNextPage])
  const [stored, setSelection] = useState<Selection>(EMPTY_SELECTION)
  // Selected ids that left the list (deleted, filtered out) do not count.
  const selection = prune(stored, order)
  const [viewing, setViewing] = useState<string | null>(null)
  const [deleting, setDeleting] = useState<readonly string[] | null>(null)
  const [comparing, setComparing] = useState<readonly [CompareImage, CompareImage] | null>(null)
  const [refreshing, setRefreshing] = useState(false)

  const selectedIds = ordered(selection, order)
  const byId = new Map(items.map((item) => [item.id, item]))
  const viewed = viewing ? byId.get(viewing) : undefined

  const rescan = async () => {
    setRefreshing(true)
    try {
      const { added } = await api.scanOutputs()
      toast.success(added > 0 ? t("Found {{count}} new images", { count: added }) : t("The gallery is up to date"))
      // Start the lists over from their first page (refetching every loaded page is slow).
      await queryClient.resetQueries({ queryKey: galleryKeys.all })
    } catch (error) {
      toast.error(t("Could not scan the output folders"), { description: errorText(error) })
    } finally {
      setRefreshing(false)
    }
  }

  useRegisterCommands("gallery", [
    { id: "gallery.rescan", label: "Scan output folders for new images", group: "Gallery", run: () => void rescan() },
    { id: "gallery.starred", label: "Show starred images only", group: "Gallery", run: () => view.set({ starred: true }) },
  ])

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target instanceof Element ? event.target : null
      if (target?.closest("input, textarea, [contenteditable], [role=dialog]")) return
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "a" && order.length > 0) {
        event.preventDefault()
        setSelection(selectAll(order))
      } else if (event.key === "Escape" && selection.ids.size > 0) {
        setSelection(EMPTY_SELECTION)
      } else if (event.key === "Delete" && selection.ids.size > 0) {
        setDeleting(ordered(selection, order))
      }
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  })

  const header = (
    <HeaderSlot>
      <Button variant="ghost" size="sm" onClick={() => void rescan()} disabled={refreshing} title={t("Scan output folders for new images")}>
        <RefreshCw className={refreshing ? "animate-spin" : undefined} aria-hidden />
        <span className="hidden sm:inline">{t("Rescan")}</span>
      </Button>
    </HeaderSlot>
  )

  let body: React.ReactNode
  if (list.isPending || readingOn) {
    body = (
      <div className="grid grid-cols-[repeat(auto-fill,minmax(10rem,1fr))] gap-1.5 p-3" aria-busy="true">
        {Array.from({ length: 18 }, (_, index) => (
          <Skeleton key={index} className="aspect-square" />
        ))}
      </div>
    )
  } else if (list.isError) {
    body = <PagePlaceholder icon={GalleryVerticalEnd} title="Gallery unavailable" description={errorText(list.error)} />
  } else if (items.length === 0) {
    body =
      activeFilters(view) > 0 ? (
        <PagePlaceholder icon={GalleryVerticalEnd} title="No images match" description="Try other filters, or clear them.">
          <Button variant="outline" size="sm" onClick={view.reset}>
            {t("Clear filters")}
          </Button>
        </PagePlaceholder>
      ) : (
        <PagePlaceholder icon={GalleryVerticalEnd} title="No images yet" description="Images you generate appear here. Older images in the output folders are found by a scan.">
          <Button variant="outline" size="sm" onClick={() => void rescan()} disabled={refreshing}>
            {t("Scan output folders")}
          </Button>
        </PagePlaceholder>
      )
  } else {
    body = (
      <GalleryGrid
        // New filters start a new grid: scrolled to the top, the first image is the tab stop.
        key={JSON.stringify(query)}
        items={items}
        layout={view.layout}
        size={view.size}
        total={total ?? undefined}
        selected={selection.ids}
        hasMore={list.hasNextPage}
        loadingMore={list.isFetchingNextPage}
        onLoadMore={() => void list.fetchNextPage()}
        onActivate={(item, event) => {
          if (selection.ids.size > 0 || event.ctrlKey || event.metaKey || event.shiftKey) setSelection(select(selection, order, item.id, modeOf(event)))
          else setViewing(item.id)
        }}
        onToggleSelect={(item, event) => setSelection(select(selection, order, item.id, modeOf({ ...event, ctrlKey: false, metaKey: false })))}
        onStar={(item) => void starOutputs([item.id], !item.starred)}
      />
    )
  }

  const pair = selectedIds.length === 2 ? selectedIds.map((id) => byId.get(id)).filter((item): item is GalleryItem => Boolean(item)) : []

  return (
    <div className="relative flex h-full flex-col">
      {header}
      <GalleryToolbar count={items.length} total={total} />
      <div className="min-h-0 flex-1">{body}</div>
      {selection.ids.size > 0 ? (
        <div className="pointer-events-none absolute inset-x-0 bottom-4 flex justify-center px-3">
          <BulkBar
            count={selection.ids.size}
            onClear={() => setSelection(EMPTY_SELECTION)}
            onZip={() => void downloadZip(selectedIds)}
            onStar={(starred) => void starOutputs(selectedIds, starred)}
            onCompare={pair.length === 2 ? () => setComparing([pair[0]!, pair[1]!]) : null}
            onDelete={() => setDeleting(selectedIds)}
          />
        </div>
      ) : null}

      {viewing !== null ? (
        <Suspense fallback={null}>
          <Lightbox
            open
            onOpenChange={(open) => !open && setViewing(null)}
            items={items.map(asResult)}
            current={viewing}
            onNavigate={(key) => {
              setViewing(key)
              // Near the end of what is loaded: fetch more, so ← / → keep going.
              if (order.indexOf(key) >= order.length - 5 && list.hasNextPage && !list.isFetchingNextPage) void list.fetchNextPage()
            }}
            tab={viewed?.tab === "img2img" ? "img2img" : "txt2img"}
            aside={(result) => {
              const item = byId.get(result.fileId)
              return item ? <GalleryDetails item={item} onDelete={(target) => setDeleting([target.id])} onClose={() => setViewing(null)} /> : null
            }}
          />
        </Suspense>
      ) : null}

      <DeleteDialog
        ids={deleting}
        onCancel={() => setDeleting(null)}
        onConfirm={(ids, deleteFiles) => {
          setDeleting(null)
          if (viewing && ids.includes(viewing)) {
            // Keep the viewer open on the next image.
            const next = order.find((id, index) => index > order.indexOf(viewing) && !ids.includes(id)) ?? order.find((id) => !ids.includes(id)) ?? null
            setViewing(next)
          }
          void removeOutputs(ids, deleteFiles)
        }}
      />
      <CompareDialog images={comparing} onOpenChange={(open) => !open && setComparing(null)} />
    </div>
  )
}
