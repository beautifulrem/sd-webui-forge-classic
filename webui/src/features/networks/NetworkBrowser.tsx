/**
 * The network browser shared by the Networks page and the Generate drawer:
 * kind tabs, search (every word must match; "/" focuses it), sort, grid /
 * list view and card size, filter chips (architecture, "compatible with the
 * current preset", folder), a resizable folder tree with counts, the
 * virtualized grid, and the item menu and dialogs. Clicking a card toggles it
 * in the target form's prompt, loads a checkpoint or toggles a VAE / text
 * encoder.
 */
import { useMutation, useQuery } from "@tanstack/react-query"
import {
  ArrowDownAZ,
  ArrowDownWideNarrow,
  ArrowUpAZ,
  ArrowUpNarrowWide,
  Check,
  CircleAlert,
  FolderTree as FolderTreeIcon,
  LayoutGrid,
  List,
  Loader2,
  RefreshCw,
  Search,
  ShieldCheck,
  X,
} from "lucide-react"
import { type ReactNode, type Ref, type RefObject, useEffect, useImperativeHandle, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { toast } from "sonner"
import { api } from "@/api/endpoints"
import { queryClient } from "@/api/queries"
import type { NetworkItem, NetworkKind } from "@/api/types"
import { FadeStrip } from "@/components/FadeStrip"
import { FitTabsList } from "@/components/FitTabsList"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Empty, EmptyContent, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from "@/components/ui/empty"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from "@/components/ui/resizable"
import { Skeleton } from "@/components/ui/skeleton"
import { Tabs, TabsTrigger } from "@/components/ui/tabs"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { presetLabel } from "@/features/model-bar/model-options"
import { useDebouncedValue } from "@/features/prompt-editor/use-debounced-value"
import { useFittingPlaceholder } from "@/lib/use-fitting-placeholder"
import { useMediaQuery } from "@/lib/use-media-query"
import { cn } from "@/lib/utils"
import { applyItem, copyPath, openFolder, useActiveTest, useNetworkSettings } from "./actions"
import { EditMetadataDialog } from "./EditMetadataDialog"
import { FolderTree } from "./FolderTree"
import { ItemMenu, type MenuCommand } from "./ItemMenu"
import { MetadataDialog } from "./MetadataDialog"
import type { MenuAnchor } from "./items"
import type { ItemAction } from "./NetworkCard"
import { type GridHandle, NetworkGrid } from "./NetworkGrid"
import { PreviewDialog } from "./PreviewDialog"
import { usePromptTarget } from "./prompt-target"
import { facetsQuery, type IndexRange, kindsQuery, type ListParams, type SortKey, type SortOrder, useNetworkWindow } from "./queries"
import { type CardSize, type Surface, TREE_WIDTH, useNetworksUi, useSurface } from "./store"

export type BrowserHandle = { focusSearch: () => void }

type Dialog = { type: "metadata" | "edit" | "preview"; item: NetworkItem } | null

function useKindTitle() {
  const { t } = useTranslation()
  return (kind: Pick<NetworkKind, "id" | "title">) => {
    switch (kind.id) {
      case "lora":
        return t("LoRA")
      case "embeddings":
        return t("Textual Inversion")
      case "checkpoints":
        return t("Checkpoints")
      case "vae":
        return t("VAE")
      case "text_encoders":
        return t("Text Encoders")
      default:
        // extension pages: their titles are in the server's catalog
        return t(kind.title, { ns: "server" })
    }
  }
}

function Chip({ on, onClick, children, label }: { on: boolean; onClick: () => void; children: ReactNode; label?: string }) {
  return (
    <button
      type="button"
      aria-pressed={on}
      aria-label={label}
      onClick={onClick}
      className={cn(
        "inline-flex h-6 shrink-0 items-center gap-1 rounded-full border px-2.5 text-xs whitespace-nowrap transition-colors outline-none select-none",
        "focus-visible:ring-2 focus-visible:ring-ring/60",
        on ? "border-primary/40 bg-primary/12 text-foreground" : "border-border text-muted-foreground hover:bg-muted hover:text-foreground",
      )}
    >
      {on ? <Check className="size-3 text-primary" aria-hidden /> : null}
      {children}
    </button>
  )
}

const NUMERIC_SORTS: ReadonlySet<SortKey> = new Set(["date", "created", "size"])

function SortMenu({ sort, order, onChange }: { sort: SortKey; order: SortOrder; onChange: (sort: SortKey, order: SortOrder) => void }) {
  const { t } = useTranslation()
  const labels: Record<SortKey, string> = { name: t("Name"), path: t("Path"), created: t("Date created"), date: t("Date modified"), size: t("Size") }
  const numeric = NUMERIC_SORTS.has(sort)
  const Icon = numeric ? (order === "asc" ? ArrowUpNarrowWide : ArrowDownWideNarrow) : order === "asc" ? ArrowDownAZ : ArrowUpAZ
  return (
    <DropdownMenu>
      <DropdownMenuTrigger render={<Button variant="ghost" size="sm" className="gap-1.5 text-muted-foreground" aria-label={t("Sort: {{key}}", { key: labels[sort] })} />}>
        <Icon aria-hidden />
        <span className="hidden @lg/browser:inline">{labels[sort]}</span>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-44">
        <DropdownMenuGroup>
          <DropdownMenuLabel>{t("Sort by")}</DropdownMenuLabel>
          <DropdownMenuRadioGroup value={sort} onValueChange={(value) => onChange(value as SortKey, NUMERIC_SORTS.has(value as SortKey) ? "desc" : "asc")}>
            {(Object.keys(labels) as SortKey[]).map((key) => (
              <DropdownMenuRadioItem key={key} value={key}>
                {labels[key]}
              </DropdownMenuRadioItem>
            ))}
          </DropdownMenuRadioGroup>
        </DropdownMenuGroup>
        <DropdownMenuSeparator />
        <DropdownMenuGroup>
          <DropdownMenuRadioGroup value={order} onValueChange={(value) => onChange(sort, value as SortOrder)}>
            <DropdownMenuRadioItem value="asc">{numeric ? t("Smallest / oldest first") : t("A to Z")}</DropdownMenuRadioItem>
            <DropdownMenuRadioItem value="desc">{numeric ? t("Largest / newest first") : t("Z to A")}</DropdownMenuRadioItem>
          </DropdownMenuRadioGroup>
        </DropdownMenuGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

function SearchBox({ value, onChange, inputRef, count }: { value: string; onChange: (value: string) => void; inputRef: RefObject<HTMLInputElement | null>; count: number | null }) {
  const { t } = useTranslation()
  // the long hint only where it fits; a narrow box (phone, slim drawer) says just "Search…"
  const placeholder = useFittingPlaceholder(inputRef, [t("Search names, folders, tags…"), t("Search…")])
  return (
    // on a phone the search takes its own row and the view controls wrap below it
    <div className="relative min-w-48 flex-1 basis-48">
      <Search className="pointer-events-none absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2 text-muted-foreground" aria-hidden />
      <input
        ref={inputRef}
        type="search"
        value={value}
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Escape" && value) {
            event.preventDefault()
            event.stopPropagation()
            onChange("")
          }
        }}
        placeholder={placeholder}
        aria-label={t("Search networks")}
        aria-keyshortcuts="/"
        className="h-8 w-full min-w-0 rounded-lg border border-input bg-transparent pr-16 pl-8 text-sm outline-none placeholder:text-muted-foreground focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 dark:bg-input/30 [&::-webkit-search-cancel-button]:hidden"
      />
      <div className="absolute top-1/2 right-1 flex -translate-y-1/2 items-center gap-1">
        {count !== null ? <span className="text-[0.7rem] text-muted-foreground tabular-nums">{count.toLocaleString()}</span> : null}
        {value ? (
          <Button variant="ghost" size="icon-xs" aria-label={t("Clear search")} onClick={() => onChange("")}>
            <X />
          </Button>
        ) : (
          <kbd className="hidden h-5 items-center rounded border px-1.5 font-mono text-[0.65rem] text-muted-foreground sm:inline-flex" aria-hidden>
            /
          </kbd>
        )}
      </div>
    </div>
  )
}

function EmptyState({ filtered, kindTitle, onClear }: { filtered: boolean; kindTitle: string; onClear: () => void }) {
  const { t } = useTranslation()
  return (
    <div className="flex h-full items-center justify-center p-6">
      <Empty className="max-w-sm border-0">
        <EmptyHeader>
          <EmptyMedia variant="icon">
            <Search aria-hidden />
          </EmptyMedia>
          <EmptyTitle>{filtered ? t("Nothing matches") : t("No {{kind}} found", { kind: kindTitle })}</EmptyTitle>
          <EmptyDescription>{filtered ? t("Try other words, or clear the filters.") : t("Put files into the model folder, then refresh.")}</EmptyDescription>
        </EmptyHeader>
        {filtered ? (
          <EmptyContent>
            <Button variant="outline" size="sm" onClick={onClear}>
              {t("Clear filters")}
            </Button>
          </EmptyContent>
        ) : null}
      </Empty>
    </div>
  )
}

function ErrorState({ title, error, onRetry }: { title: string; error: Error | null; onRetry: () => void }) {
  const { t } = useTranslation()
  return (
    <div className="flex h-full items-center justify-center p-6" role="alert">
      <Empty className="max-w-sm border-0">
        <EmptyHeader>
          <EmptyMedia variant="icon" className="text-destructive">
            <CircleAlert aria-hidden />
          </EmptyMedia>
          <EmptyTitle>{title}</EmptyTitle>
          {error?.message ? <EmptyDescription className="line-clamp-3 [overflow-wrap:anywhere]">{error.message}</EmptyDescription> : null}
        </EmptyHeader>
        <EmptyContent>
          <Button variant="outline" size="sm" onClick={onRetry}>
            <RefreshCw aria-hidden />
            {t("Retry")}
          </Button>
        </EmptyContent>
      </Empty>
    </div>
  )
}

type BrowserProps = {
  surface: Surface
  /** The generation form cards insert into. */
  form: string
  /** Extra controls at the end of the toolbar (target switches). */
  toolbarEnd?: ReactNode
  /** Announce applied cards with a toast (the page, where the prompt is not visible). */
  toastOnApply?: boolean
  className?: string
  ref?: Ref<BrowserHandle>
}

export function NetworkBrowser({ surface, form, toolbarEnd, toastOnApply, className, ref }: BrowserProps) {
  const { t } = useTranslation()
  const kindTitle = useKindTitle()
  const state = useSurface(surface)
  const update = useNetworksUi((s) => s.update)
  const setKindFilter = useNetworksUi((s) => s.setKindFilter)
  const treeWidth = useNetworksUi((s) => s.treeWidth)
  const setTreeWidth = useNetworksUi((s) => s.setTreeWidth)
  const settings = useNetworkSettings()
  const target = usePromptTarget()
  const wide = useMediaQuery("(min-width: 64rem)")
  const searchRef = useRef<HTMLInputElement>(null)
  const gridRef = useRef<GridHandle>(null)
  const [dialog, setDialog] = useState<Dialog>(null)
  const [menu, setMenu] = useState<{ item: NetworkItem; anchor: MenuAnchor } | null>(null)
  const [announcement, setAnnouncement] = useState("")
  const [search, setSearch] = useState(state.query)

  useImperativeHandle(ref, () => ({ focusSearch: () => searchRef.current?.focus() }), [])

  const kinds = useQuery(kindsQuery)
  const kindList = kinds.data ?? []
  const kind = kindList.find((candidate) => candidate.id === state.kind) ?? kindList[0]
  const kindId = kind?.id ?? state.kind

  // The store keeps the search; typing reaches it debounced. Only a new
  // debounced value is written, so "Clear filters" (which clears the store
  // at once) is not undone by the value still pending.
  const debounced = useDebouncedValue(search, 180)
  const written = useRef(debounced)
  useEffect(() => {
    if (debounced === written.current) return
    written.current = debounced
    if (debounced !== useNetworksUi.getState().surfaces[surface].query) update(surface, { query: debounced })
  }, [debounced, surface, update])
  // one key for "ink" and "ink " (the server ignores the spaces too)
  const query = state.query.trim().replace(/\s+/g, " ")

  const folder = state.folder[kindId] ?? ""
  const archs = state.archs[kindId] ?? []
  const compatible = state.compatible[kindId] ?? (kindId === "lora" && settings.presetFilter)
  const sort = state.sort ?? settings.defaultSort.sort
  const order = state.order ?? (state.sort ? "asc" : settings.defaultSort.order)
  const params: ListParams = { q: query, folder, archs, compatible, sort, order }
  const facetParams = { q: query, folder, archs, compatible }

  // A new kind, search, filter or sort starts a new grid at the top (no
  // scroll position or keyboard cursor carried over from another list).
  const gridKey = `${kindId}\u0000${JSON.stringify(params)}`
  const [seen, setSeen] = useState<IndexRange & { key: string }>({ key: gridKey, start: 0, end: 0 })
  // windows are requested once scrolling settles (a scrollbar drag across 5,000 items asks for one window)
  const settled = useDebouncedValue(seen, 120)
  const range = settled.key === gridKey ? settled : { start: 0, end: 0 }
  const list = useNetworkWindow(kindId, params, range, Boolean(kind))
  const facets = useQuery({ ...facetsQuery(kindId, facetParams), enabled: Boolean(kind) })
  const { items, total } = list
  const isActive = useActiveTest(kind, form)
  const currentArch = facets.data?.current_arch

  const refresh = useMutation({
    mutationFn: () => api.networks.refresh(kindId),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: ["networks"] }),
    onError: (error) => toast.error(t("Could not refresh"), { description: error.message }),
  })
  const retry = () => {
    if (kinds.isError) void kinds.refetch()
    else list.refetch()
  }

  const filtered = Boolean(query.trim() || folder || archs.length || compatible)
  const clearFilters = () => {
    setSearch("")
    written.current = ""
    update(surface, { query: "" })
    setKindFilter(surface, "folder", kindId, undefined)
    setKindFilter(surface, "archs", kindId, undefined)
    setKindFilter(surface, "compatible", kindId, kindId === "lora" && settings.presetFilter ? false : undefined)
  }

  const apply = (item: NetworkItem) => {
    if (!kind) return
    const outcome = applyItem(item, kind, form, target)
    if (!outcome) return
    let message: string
    if (kind.action === "checkpoint") message = t("Loading {{name}}", { name: item.title })
    else if (kind.action === "module") message = outcome.added ? t("{{name}} added to the modules", { name: item.title }) : t("{{name}} removed from the modules", { name: item.title })
    else if (outcome.added) message = outcome.target === "negative" ? t("{{name}} added to the negative prompt", { name: item.title }) : t("{{name}} added to the prompt", { name: item.title })
    else message = t("{{name}} removed from the prompt", { name: item.title })
    setAnnouncement(message)
    if (toastOnApply || kind.action !== "prompt") toast(message, { duration: 1800 })
  }

  const onAction = (action: ItemAction, item: NetworkItem, anchor: MenuAnchor) => {
    if (action === "menu") setMenu({ item, anchor })
    else setDialog({ type: action, item })
  }

  const onCommand = (command: MenuCommand, item: NetworkItem) => {
    setMenu(null)
    if (command === "apply") apply(item)
    else if (command === "copy") void copyPath(item)
    else if (command === "folder") void openFolder(item)
    else setDialog({ type: command, item })
  }

  // "/" focuses the search from anywhere in the browser.
  const onKeyDown = (event: React.KeyboardEvent) => {
    if (event.key !== "/" || event.ctrlKey || event.metaKey || event.altKey) return
    const element = event.target as HTMLElement
    if (element.closest("input, textarea, [contenteditable=true]")) return
    event.preventDefault()
    searchRef.current?.focus()
  }

  const showTree = state.tree && wide && surface === "page"
  const folderTree = facets.data ? (
    <FolderTree
      folders={facets.data.folders}
      total={facets.data.folder_total ?? facets.data.total}
      selected={folder}
      onSelect={(path) => setKindFilter(surface, "folder", kindId, path || undefined)}
    />
  ) : (
    <div className="flex flex-col gap-1.5 p-2" aria-hidden>
      {Array.from({ length: 6 }, (_, i) => (
        <Skeleton key={i} className="h-5" />
      ))}
    </div>
  )

  const archFacets = facets.data?.archs ?? []
  const sizes: { value: CardSize; label: string }[] = [
    { value: "s", label: t("Small cards") },
    { value: "m", label: t("Medium cards") },
    { value: "l", label: t("Large cards") },
  ]

  const results = (
    <div className="relative flex h-full min-h-0 min-w-0 flex-1 flex-col">
      {kinds.isError ? (
        <ErrorState title={t("Could not load the network kinds")} error={kinds.error} onRetry={retry} />
      ) : list.isPending || !kind ? (
        <div className="grid grid-cols-[repeat(auto-fill,minmax(9rem,1fr))] gap-2.5 p-2" aria-busy="true" aria-label={t("Loading…")}>
          {Array.from({ length: 12 }, (_, i) => (
            <Skeleton key={i} className="aspect-3/4 rounded-xl" />
          ))}
        </div>
      ) : list.isError ? (
        <ErrorState title={t("Could not load {{kind}}", { kind: kindTitle(kind) })} error={list.error} onRetry={retry} />
      ) : total === 0 ? (
        <EmptyState filtered={filtered} kindTitle={kindTitle(kind)} onClear={clearFilters} />
      ) : (
        <NetworkGrid
          key={gridKey}
          ref={gridRef}
          items={items}
          total={total}
          kind={kind}
          view={state.view}
          size={state.size}
          isActive={isActive}
          currentArch={currentArch}
          onRangeChange={(next) => setSeen((old) => (old.key === gridKey && old.start === next.start && old.end === next.end ? old : { key: gridKey, ...next }))}
          onApply={apply}
          onAction={onAction}
          label={kindTitle(kind)}
          className="flex-1 pt-0"
        />
      )}
      {list.isFetching && !list.isPending ? <Loader2 className="pointer-events-none absolute top-2 right-3 size-4 animate-spin text-muted-foreground" aria-hidden /> : null}
    </div>
  )

  return (
    <div onKeyDown={onKeyDown} className={cn("@container/browser flex min-h-0 flex-col", className)}>
      <div className="flex flex-col gap-2 border-b px-3 pt-2 pb-2">
        {kindList.length > 0 ? (
          <Tabs value={kindId} onValueChange={(value) => update(surface, { kind: String(value) })}>
            {/* a narrow drawer or phone shows the kinds as a select rather than cutting labels */}
            <FitTabsList
              aria-label={t("Network kinds")}
              variant="line"
              className="h-8"
              options={kindList.map((candidate) => ({ value: candidate.id, label: kindTitle(candidate) }))}
              value={kindId}
              onValueChange={(value) => update(surface, { kind: value })}
            >
              {kindList.map((candidate) => (
                <TabsTrigger key={candidate.id} value={candidate.id} className="flex-none px-2 text-[0.8rem]">
                  {kindTitle(candidate)}
                </TabsTrigger>
              ))}
            </FitTabsList>
          </Tabs>
        ) : (
          <Skeleton className="h-8 w-64" />
        )}
        <div className="flex flex-wrap items-center gap-1">
          <SearchBox value={search} onChange={setSearch} inputRef={searchRef} count={list.isPending ? null : total} />
          <SortMenu sort={sort} order={order} onChange={(sort, order) => update(surface, { sort, order })} />
          <ToggleGroup
            value={[state.view]}
            onValueChange={(value) => value[0] && update(surface, { view: value[0] as "grid" | "list" })}
            size="sm"
            spacing={0}
            aria-label={t("View")}
          >
            <ToggleGroupItem value="grid" aria-label={t("Grid")} title={t("Grid")}>
              <LayoutGrid />
            </ToggleGroupItem>
            <ToggleGroupItem value="list" aria-label={t("List")} title={t("List")}>
              <List />
            </ToggleGroupItem>
          </ToggleGroup>
          {state.view === "grid" ? (
            <DropdownMenu>
              <DropdownMenuTrigger render={<Button variant="ghost" size="icon-sm" aria-label={t("Card size")} title={t("Card size")} />}>
                <span className="font-mono text-[0.7rem] font-semibold uppercase">{state.size}</span>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-40">
                <DropdownMenuRadioGroup value={state.size} onValueChange={(value) => update(surface, { size: value as CardSize })}>
                  {sizes.map((size) => (
                    <DropdownMenuRadioItem key={size.value} value={size.value}>
                      {size.label}
                    </DropdownMenuRadioItem>
                  ))}
                </DropdownMenuRadioGroup>
              </DropdownMenuContent>
            </DropdownMenu>
          ) : null}
          {surface === "page" && wide ? (
            <Button variant="ghost" size="icon-sm" aria-label={t("Folder tree")} title={t("Folder tree")} aria-pressed={state.tree} onClick={() => update(surface, { tree: !state.tree })}>
              <FolderTreeIcon />
            </Button>
          ) : (
            <Popover>
              <PopoverTrigger render={<Button variant="ghost" size="icon-sm" aria-label={t("Folders")} title={t("Folders")} className={cn(folder && "text-primary")} />}>
                <FolderTreeIcon />
              </PopoverTrigger>
              <PopoverContent aria-label={t("Folders")} align="end" className="max-h-[60dvh] w-72 overflow-y-auto p-1">
                {folderTree}
              </PopoverContent>
            </Popover>
          )}
          <Button variant="ghost" size="icon-sm" aria-label={t("Refresh")} title={t("Refresh")} disabled={refresh.isPending} onClick={() => refresh.mutate()}>
            <RefreshCw className={cn(refresh.isPending && "animate-spin")} />
          </Button>
          {toolbarEnd}
        </div>
        {kind && (archFacets.length > 1 || folder || compatible || (archFacets.length === 1 && archFacets[0]!.id !== "unknown")) ? (
          <FadeStrip active={kindId} className="-mx-3 flex items-center gap-1.5 px-3" role="group" aria-label={t("Filters")}>
            {kind.action === "prompt" || kind.action === "checkpoint" ? (
              <Chip on={compatible} onClick={() => setKindFilter(surface, "compatible", kindId, !compatible)}>
                <ShieldCheck className="size-3" aria-hidden />
                {currentArch ? t("For {{preset}}", { preset: presetLabel(currentArch) }) : t("Compatible")}
              </Chip>
            ) : null}
            {folder ? (
              <Chip on onClick={() => setKindFilter(surface, "folder", kindId, undefined)} label={t("Folder {{folder}}: remove the filter", { folder })}>
                {folder}
                <X className="size-3" aria-hidden />
              </Chip>
            ) : null}
            {archFacets.map((arch) => {
              const on = archs.includes(arch.id)
              return (
                <Chip key={arch.id} on={on} onClick={() => setKindFilter(surface, "archs", kindId, on ? archs.filter((a) => a !== arch.id) : [...archs, arch.id])}>
                  {arch.id === "unknown" ? t("Unknown") : presetLabel(arch.id)}
                  <span className="text-muted-foreground tabular-nums">{arch.count}</span>
                </Chip>
              )
            })}
            {filtered ? (
              <Button variant="ghost" size="xs" className="ml-auto shrink-0 text-muted-foreground" onClick={clearFilters}>
                {t("Clear filters")}
              </Button>
            ) : null}
          </FadeStrip>
        ) : null}
      </div>

      {showTree ? (
        <ResizablePanelGroup orientation="horizontal" className="min-h-0 flex-1">
          <ResizablePanel
            id="tree"
            defaultSize={treeWidth}
            minSize={TREE_WIDTH.min}
            maxSize={TREE_WIDTH.max}
            groupResizeBehavior="preserve-pixel-size"
            onResize={(size) => {
              if (size.inPixels > 0) setTreeWidth(size.inPixels)
            }}
          >
            <aside className="h-full overflow-y-auto px-1.5" aria-label={t("Folders")}>
              {folderTree}
            </aside>
          </ResizablePanel>
          <ResizableHandle aria-label={t("Resize the folder tree")} />
          <ResizablePanel id="items" minSize={240}>
            {results}
          </ResizablePanel>
        </ResizablePanelGroup>
      ) : (
        <div className="flex min-h-0 flex-1">{results}</div>
      )}
      <p className="sr-only" aria-live="polite" aria-atomic="true">
        {announcement}
      </p>

      <ItemMenu
        item={menu?.item ?? null}
        kind={kind}
        anchor={menu?.anchor ?? null}
        active={menu ? isActive(menu.item) : false}
        canOpenFolder={settings.canOpenFolders}
        onClose={() => setMenu(null)}
        onCommand={onCommand}
      />
      <MetadataDialog item={dialog?.type === "metadata" ? dialog.item : null} open={dialog?.type === "metadata"} onOpenChange={(open) => !open && setDialog(null)} />
      <EditMetadataDialog item={dialog?.type === "edit" ? dialog.item : null} kind={kind} open={dialog?.type === "edit"} onOpenChange={(open) => !open && setDialog(null)} />
      <PreviewDialog item={dialog?.type === "preview" ? dialog.item : null} open={dialog?.type === "preview"} onOpenChange={(open) => !open && setDialog(null)} />
    </div>
  )
}
