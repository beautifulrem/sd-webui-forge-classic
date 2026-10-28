/**
 * Set an item's preview from this session's results, the output gallery
 * (paged) or an uploaded image (`PUT /networks/{kind}/{name}/preview` with a
 * file id; the server stores it as `<model>.preview.png`). Each opening starts
 * fresh: nothing picked, on the session's results when there are any.
 */
import { useInfiniteQuery, useMutation } from "@tanstack/react-query"
import { Check, ImageUp, Loader2 } from "@/components/icons"
import { type KeyboardEvent, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { toast } from "sonner"
import { api } from "@/api/endpoints"
import { queryClient } from "@/api/queries"
import type { NetworkItem } from "@/api/types"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Skeleton } from "@/components/ui/skeleton"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { useSession } from "@/features/gallery/session-store"
import { cn } from "@/lib/utils"

const ARROWS: Record<string, number> = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }

/** A radio group of thumbnails: one tab stop, arrows / Home / End move and pick (WAI-ARIA radio group). */
function Thumbs({ ids, selected, onSelect, label }: { ids: readonly string[]; selected: string | null; onSelect: (id: string) => void; label: string }) {
  const group = useRef<HTMLDivElement>(null)
  const selectedIndex = selected ? ids.indexOf(selected) : -1
  const tabStop = Math.max(0, selectedIndex)

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const from = selectedIndex < 0 ? 0 : selectedIndex
    let next: number | null = null
    if (event.key in ARROWS) next = selectedIndex < 0 ? 0 : (from + ARROWS[event.key]! + ids.length) % ids.length
    else if (event.key === "Home") next = 0
    else if (event.key === "End") next = ids.length - 1
    if (next === null || !ids.length) return
    event.preventDefault()
    onSelect(ids[next]!)
    group.current?.querySelectorAll<HTMLElement>("[role=radio]")[next]?.focus()
  }

  return (
    <div ref={group} role="radiogroup" aria-label={label} onKeyDown={onKeyDown} className="grid grid-cols-[repeat(auto-fill,minmax(6.5rem,1fr))] gap-2 p-0.5">
      {ids.map((id, index) => {
        const on = id === selected
        return (
          <button
            key={id}
            type="button"
            role="radio"
            aria-checked={on}
            aria-label={`${label} ${index + 1}`}
            tabIndex={index === tabStop ? 0 : -1}
            onClick={() => onSelect(id)}
            className={cn(
              "relative aspect-square overflow-hidden rounded-lg bg-checker ring-offset-2 ring-offset-background outline-none focus-visible:ring-2 focus-visible:ring-ring",
              on ? "ring-2 ring-primary" : "hover:ring-2 hover:ring-ring/40",
            )}
          >
            <img src={api.thumbUrl(id, 192)} alt="" loading="lazy" decoding="async" className="size-full object-cover" />
            {on ? (
              <span className="absolute right-1 bottom-1 grid size-5 place-items-center rounded-full bg-primary text-primary-foreground">
                <Check className="size-3" strokeWidth={3} aria-hidden />
              </span>
            ) : null}
          </button>
        )
      })}
    </div>
  )
}

function Empty({ text }: { text: string }) {
  return <p className="rounded-lg border border-dashed px-4 py-10 text-center text-sm text-muted-foreground">{text}</p>
}

const galleryQuery = {
  queryKey: ["outputs", "", "network-preview"] as const,
  queryFn: ({ pageParam, signal }: { pageParam: string | null; signal: AbortSignal }) => api.outputs(pageParam ? { cursor: pageParam } : {}, signal),
  initialPageParam: null as string | null,
  getNextPageParam: (page: { next_cursor?: string | null }) => page.next_cursor ?? undefined,
  staleTime: 30_000,
}

export function PreviewDialog({ item, open, onOpenChange }: { item: NetworkItem | null; open: boolean; onOpenChange: (open: boolean) => void }) {
  const { t } = useTranslation()
  const session = useSession((s) => s.items)
  const [tab, setTab] = useState<string>("session")
  const [selected, setSelected] = useState<string | null>(null)
  const file = useRef<HTMLInputElement>(null)

  // Every opening (for this or another item) starts fresh.
  const opening = open && item ? `${item.kind}:${item.name}` : null
  const [shownFor, setShownFor] = useState<string | null>(null)
  if (opening !== shownFor) {
    setShownFor(opening)
    if (opening) {
      setSelected(null)
      setTab(session.length ? "session" : "gallery")
    }
  }

  const gallery = useInfiniteQuery({ ...galleryQuery, enabled: open && tab === "gallery" })

  const apply = useMutation({
    mutationFn: async (source: { fileId: string } | { file: File }) => {
      if (!item) throw new Error("no item")
      const fileId = "fileId" in source ? source.fileId : (await api.upload(source.file)).id
      return api.networks.setPreview(item.kind, item.name, fileId)
    },
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["networks"] })
      toast.success(t("Preview updated"))
      onOpenChange(false)
    },
    onError: (error) => toast.error(t("Could not set the preview"), { description: error.message }),
  })

  const galleryIds = (gallery.data?.pages ?? []).flatMap((page) => page.items).filter((output) => output.kind === "image").map((output) => output.id)

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="flex max-h-[min(90dvh,48rem)] flex-col gap-4 sm:max-w-2xl">
        <DialogHeader className="pr-8">
          <DialogTitle className="truncate">{t("Preview of {{name}}", { name: item?.title ?? "" })}</DialogTitle>
          <DialogDescription>{t("Pick a result or upload an image. It is saved next to the model file.")}</DialogDescription>
        </DialogHeader>
        <Tabs value={tab} onValueChange={(value) => setTab(String(value))} className="flex min-h-0 flex-col">
          <TabsList>
            <TabsTrigger value="session">{t("This session")}</TabsTrigger>
            <TabsTrigger value="gallery">{t("Gallery")}</TabsTrigger>
            <TabsTrigger value="upload">{t("Upload")}</TabsTrigger>
          </TabsList>
          <TabsContent value="session" className="max-h-[50dvh] overflow-y-auto">
            {session.length ? (
              <Thumbs ids={session.map((result) => result.fileId)} selected={selected} onSelect={setSelected} label={t("Result")} />
            ) : (
              <Empty text={t("Results of this session appear here")} />
            )}
          </TabsContent>
          <TabsContent value="gallery" className="max-h-[50dvh] overflow-y-auto">
            {gallery.isPending ? (
              <div className="grid grid-cols-[repeat(auto-fill,minmax(6.5rem,1fr))] gap-2" aria-busy="true">
                {Array.from({ length: 8 }, (_, i) => (
                  <Skeleton key={i} className="aspect-square rounded-lg" />
                ))}
              </div>
            ) : gallery.isError ? (
              <div className="flex flex-col items-center gap-2 py-8 text-sm text-muted-foreground" role="alert">
                {t("Could not load the gallery")}
                <Button variant="outline" size="sm" onClick={() => void gallery.refetch()}>
                  {t("Retry")}
                </Button>
              </div>
            ) : galleryIds.length ? (
              <div className="flex flex-col gap-3">
                <Thumbs ids={galleryIds} selected={selected} onSelect={setSelected} label={t("Image")} />
                {gallery.hasNextPage ? (
                  <Button variant="outline" size="sm" className="self-center" disabled={gallery.isFetchingNextPage} onClick={() => void gallery.fetchNextPage()}>
                    {gallery.isFetchingNextPage ? <Loader2 className="animate-spin" aria-hidden /> : null}
                    {t("Load more")}
                  </Button>
                ) : null}
              </div>
            ) : (
              <Empty text={t("No saved images yet")} />
            )}
          </TabsContent>
          <TabsContent value="upload">
            <button
              type="button"
              onClick={() => file.current?.click()}
              onDragOver={(event) => event.preventDefault()}
              onDrop={(event) => {
                event.preventDefault()
                const image = [...event.dataTransfer.files].find((candidate) => candidate.type.startsWith("image/"))
                if (image) apply.mutate({ file: image })
              }}
              className="flex w-full flex-col items-center justify-center gap-2 rounded-md border-2 border-dashed px-4 py-12 text-sm text-muted-foreground transition-colors outline-none hover:border-ring/50 hover:text-foreground focus-visible:ring-3 focus-visible:ring-ring/50"
            >
              {apply.isPending ? <Loader2 className="size-6 animate-spin" aria-hidden /> : <ImageUp className="size-6" aria-hidden />}
              {t("Drop an image here or click to choose one")}
            </button>
            <input
              ref={file}
              type="file"
              accept="image/*"
              hidden
              data-testid="preview-upload"
              onChange={(event) => {
                const image = event.target.files?.[0]
                event.target.value = ""
                if (image) apply.mutate({ file: image })
              }}
            />
          </TabsContent>
        </Tabs>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            {t("Cancel")}
          </Button>
          <Button disabled={!selected || apply.isPending} onClick={() => selected && apply.mutate({ fileId: selected })}>
            {apply.isPending ? <Loader2 className="animate-spin" aria-hidden /> : null}
            {t("Use as preview")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
