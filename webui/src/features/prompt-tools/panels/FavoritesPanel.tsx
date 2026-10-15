/**
 * Favorite prompts and tags, in collections: save the current prompt, use or
 * append a favorite, rename, move between collections, reorder, delete;
 * create / rename / delete collections; export and import as JSON.
 */
import { useInfiniteQuery, useQuery, useQueryClient } from "@tanstack/react-query"
import { ArrowUp, Check, Download, FolderInput, FolderPlus, ListPlus, Pencil, Plus, Trash2, Upload, X } from "lucide-react"
import { type ReactNode, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { toast } from "sonner"
import { Button, buttonVariants } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Spinner } from "@/components/ui/spinner"
import { useDebouncedValue } from "@/features/prompt-editor/use-debounced-value"
import { type Collection, type Favorite, type FavoriteKind, type FavoritesExport, promptToolsApi } from "../api"
import { toolKeys } from "../queries"
import { appendText, type PanelProps } from "./helpers"
import { PanelHeader, SearchBox } from "./panel"

/** Every collection, the unsorted favorites, or a collection id. */
type Filter = string
const ALL = "*all"
const UNSORTED = "*unsorted"
/** The menu value of "no collection" (collection ids never start with `*`). */
const NO_COLLECTION = "*none"
const PAGE_SIZE = 100

/** The `collection` query of a filter (`undefined`: all, `""`: unsorted). */
function collectionQuery(filter: Filter): string | undefined {
  return filter === ALL ? undefined : filter === UNSORTED ? "" : filter
}

function Action({ label, onClick, children, disabled }: { label: string; onClick: () => void; children: ReactNode; disabled?: boolean }) {
  return (
    <Button variant="ghost" size="icon-xs" aria-label={label} title={label} onClick={onClick} disabled={disabled}>
      {children}
    </Button>
  )
}

function NameForm({ initial, label, onSave, onCancel }: { initial: string; label: string; onSave: (name: string) => void; onCancel: () => void }) {
  const { t } = useTranslation()
  const [name, setName] = useState(initial)
  return (
    <form
      className="flex min-w-0 flex-1 items-center gap-1"
      onSubmit={(event) => {
        event.preventDefault()
        onSave(name.trim())
      }}
    >
      <Input
        autoFocus
        value={name}
        maxLength={200}
        aria-label={label}
        placeholder={label}
        className="h-7 text-xs"
        onChange={(event) => setName(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Escape") {
            event.stopPropagation()
            onCancel()
          }
        }}
      />
      <Action label={t("Save")} onClick={() => onSave(name.trim())}>
        <Check />
      </Action>
      <Action label={t("Cancel")} onClick={onCancel}>
        <X />
      </Action>
    </form>
  )
}

export default function FavoritesPanel({ value, setValue, kind, disabled, close }: PanelProps) {
  const { t } = useTranslation()
  const client = useQueryClient()
  const fileInput = useRef<HTMLInputElement>(null)
  const [filter, setFilter] = useState<Filter>(ALL)
  const [query, setQuery] = useState("")
  const search = useDebouncedValue(query.trim(), 250)
  const [editing, setEditing] = useState<string | null>(null)
  const [collectionForm, setCollectionForm] = useState<"new" | "rename" | null>(null)
  const [confirmDelete, setConfirmDelete] = useState(false)
  const ownKind: FavoriteKind = kind === "negative" ? "negative" : "prompt"

  const collections = useQuery({ queryKey: toolKeys.collections, queryFn: ({ signal }) => promptToolsApi.collections.list(signal), staleTime: 10_000 })
  // this prompt's own kind and tag snippets (filtered by the server, so paging is exact)
  const favorites = useInfiniteQuery({
    queryKey: toolKeys.favorites(filter, ownKind, search),
    queryFn: ({ pageParam, signal }) =>
      promptToolsApi.favorites.list({ collection: collectionQuery(filter), kind: [ownKind, "tags"], q: search || undefined, cursor: pageParam, limit: PAGE_SIZE }, signal),
    initialPageParam: null as string | null,
    getNextPageParam: (page) => page.next_cursor ?? undefined,
    staleTime: 10_000,
  })
  const list: Collection[] = collections.data ?? []
  const named = list.filter((collection) => collection.id)
  const current = named.find((collection) => collection.id === filter)
  const items = favorites.data?.pages.flatMap((page) => page.items) ?? []
  const filterOptions = [
    { value: ALL, label: t("All favorites") },
    { value: UNSORTED, label: t("Unsorted") },
    ...named.map((collection) => ({ value: collection.id!, label: `${collection.name} (${collection.count})` })),
  ]

  const refresh = async () => {
    await client.invalidateQueries({ queryKey: ["prompt-tools", "favorites"] })
    await client.invalidateQueries({ queryKey: toolKeys.collections })
  }

  const attempt = async (action: () => Promise<unknown>, success?: string) => {
    try {
      await action()
      await refresh()
      if (success) toast.success(success)
    } catch (error) {
      toast.error(t("Something went wrong"), { description: error instanceof Error ? error.message : String(error) })
    }
  }

  const saveCurrent = () =>
    attempt(() => promptToolsApi.favorites.add({ text: value, kind: ownKind, collection_id: current?.id ?? null }), t("Saved to favorites"))

  const use = (item: Favorite) => {
    if (item.kind === "tags") setValue(appendText(value, item.text))
    else {
      setValue(item.text)
      toast.success(t("Prompt replaced"), { description: t("Undo with Ctrl+Z in the editor.") })
    }
    close()
  }

  const moveUp = (item: Favorite, index: number) => {
    const previous = items[index - 1]
    if (previous) void attempt(() => promptToolsApi.favorites.update(item.id, { before: previous.id }))
  }

  const importFile = async (file: File) => {
    try {
      const data = JSON.parse(await file.text()) as FavoritesExport
      const { added } = await promptToolsApi.favorites.import(data)
      await refresh()
      toast.success(t("Favorites imported: {{count}}", { count: added }))
    } catch (error) {
      toast.error(t("Not a favorites file"), { description: error instanceof Error ? error.message : String(error) })
    }
  }

  return (
    <div className="flex max-h-[inherit] min-h-0 flex-col">
      <PanelHeader title={t("Favorites")}>
        <a
          href={promptToolsApi.favorites.exportUrl()}
          download="prompt-favorites.json"
          aria-label={t("Export favorites")}
          title={t("Export favorites")}
          className={buttonVariants({ variant: "ghost", size: "icon-xs" })}
        >
          <Download />
        </a>
        <Action label={t("Import favorites")} onClick={() => fileInput.current?.click()}>
          <Upload />
        </Action>
        <input
          ref={fileInput}
          type="file"
          accept="application/json,.json"
          hidden
          onChange={(event) => {
            const file = event.target.files?.[0]
            event.target.value = ""
            if (file) void importFile(file)
          }}
        />
      </PanelHeader>

      <div className="flex flex-col gap-2 border-b px-3 py-2">
        <div className="flex items-center gap-1.5">
          {collectionForm ? (
            <NameForm
              initial={collectionForm === "rename" ? (current?.name ?? "") : ""}
              label={collectionForm === "rename" ? t("Collection name") : t("New collection")}
              onCancel={() => setCollectionForm(null)}
              onSave={(name) => {
                if (!name) return
                setCollectionForm(null)
                if (collectionForm === "rename" && current?.id) void attempt(() => promptToolsApi.collections.rename(current.id!, name))
                else
                  void attempt(async () => {
                    const created = await promptToolsApi.collections.create(name)
                    if (created.id) setFilter(created.id)
                  })
              }}
            />
          ) : confirmDelete && current ? (
            <span role="alert" className="flex min-w-0 flex-1 items-center gap-1 text-xs">
              <span className="min-w-0 flex-1 truncate text-muted-foreground">{t("Delete “{{name}}” and its {{count}} favorites?", { name: current.name, count: current.count })}</span>
              <Button variant="ghost" size="xs" onClick={() => setConfirmDelete(false)}>
                {t("Cancel")}
              </Button>
              <Button
                variant="destructive"
                size="xs"
                onClick={() => {
                  setConfirmDelete(false)
                  setFilter(ALL)
                  void attempt(() => promptToolsApi.collections.remove(current.id!))
                }}
              >
                {t("Delete")}
              </Button>
            </span>
          ) : (
            <>
              <Select<Filter> items={filterOptions} value={filter} onValueChange={(next) => next !== null && setFilter(next)}>
                <SelectTrigger size="sm" aria-label={t("Collection")} className="min-w-0 flex-1">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {filterOptions.map((option) => (
                    <SelectItem key={option.value} value={option.value}>
                      {option.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Action label={t("New collection")} onClick={() => setCollectionForm("new")}>
                <FolderPlus />
              </Action>
              {current ? (
                <>
                  <Action label={t("Rename the collection")} onClick={() => setCollectionForm("rename")}>
                    <Pencil />
                  </Action>
                  <Action label={t("Delete the collection")} onClick={() => setConfirmDelete(true)}>
                    <Trash2 />
                  </Action>
                </>
              ) : null}
            </>
          )}
        </div>
        <div className="flex items-center gap-1.5">
          <SearchBox value={query} onChange={setQuery} placeholder={t("Search favorites")} className="flex-1" />
          <Button variant="outline" size="sm" onClick={() => void saveCurrent()} disabled={!value.trim()}>
            <Plus />
            {t("Save prompt")}
          </Button>
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto">
        {favorites.isPending ? (
          <div className="flex h-32 items-center justify-center">
            <Spinner aria-label={t("Loading")} />
          </div>
        ) : items.length === 0 ? (
          <Empty className="py-8">
            <EmptyHeader>
              <EmptyTitle>{search ? t("No favorites match “{{query}}”", { query: search }) : t("No favorites here yet")}</EmptyTitle>
              {search ? null : <EmptyDescription>{t("Save the current prompt, or star a tag or a history entry.")}</EmptyDescription>}
            </EmptyHeader>
          </Empty>
        ) : (
          <ul className="divide-y" aria-label={t("Favorites")}>
            {items.map((item, index) => (
              <li key={item.id} className="group/item px-2 py-1.5 hover:bg-muted/50 focus-within:bg-muted/50">
                {editing === item.id ? (
                  <NameForm
                    initial={item.name}
                    label={t("Name")}
                    onCancel={() => setEditing(null)}
                    onSave={(name) => {
                      setEditing(null)
                      void attempt(() => promptToolsApi.favorites.update(item.id, { name }))
                    }}
                  />
                ) : (
                  <div className="flex items-start gap-1">
                    <button
                      type="button"
                      disabled={disabled}
                      onClick={() => use(item)}
                      title={item.kind === "tags" ? t("Add to the prompt") : t("Use this prompt")}
                      className="min-w-0 flex-1 rounded-md px-1 py-0.5 text-left outline-none focus-visible:ring-3 focus-visible:ring-ring/50 disabled:opacity-50"
                    >
                      <span className="flex items-center gap-1.5">
                        {item.name ? <span className="truncate text-xs font-semibold">{item.name}</span> : null}
                        {item.kind === "tags" ? <span className="rounded bg-muted px-1 text-[0.6rem] font-medium text-muted-foreground uppercase">{t("Tags")}</span> : null}
                      </span>
                      <span className="line-clamp-3 text-xs break-words whitespace-pre-wrap">{item.text}</span>
                    </button>
                    <div className="flex shrink-0 items-center opacity-0 transition-opacity group-hover/item:opacity-100 group-focus-within/item:opacity-100 pointer-coarse:opacity-100">
                      {item.kind !== "tags" ? (
                        <Action label={t("Append to the prompt")} disabled={disabled} onClick={() => setValue(appendText(value, item.text))}>
                          <ListPlus />
                        </Action>
                      ) : null}
                      {filter !== ALL && !search && index > 0 ? (
                        <Action label={t("Move up")} onClick={() => moveUp(item, index)}>
                          <ArrowUp />
                        </Action>
                      ) : null}
                      <Action label={t("Rename")} onClick={() => setEditing(item.id)}>
                        <Pencil />
                      </Action>
                      <DropdownMenu>
                        <DropdownMenuTrigger render={<Button variant="ghost" size="icon-xs" aria-label={t("Move to a collection")} title={t("Move to a collection")} />}>
                          <FolderInput />
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end" className="w-52">
                          <DropdownMenuGroup>
                            <DropdownMenuLabel>{t("Move to a collection")}</DropdownMenuLabel>
                            <DropdownMenuRadioGroup
                              value={item.collection_id ?? NO_COLLECTION}
                              onValueChange={(next: string) => {
                                const target = next === NO_COLLECTION ? null : next
                                if (target !== item.collection_id) void attempt(() => promptToolsApi.favorites.update(item.id, { collection_id: target }))
                              }}
                            >
                              <DropdownMenuRadioItem value={NO_COLLECTION}>{t("Unsorted")}</DropdownMenuRadioItem>
                              {named.map((collection) => (
                                <DropdownMenuRadioItem key={collection.id} value={collection.id!}>
                                  <span className="truncate">{collection.name}</span>
                                </DropdownMenuRadioItem>
                              ))}
                            </DropdownMenuRadioGroup>
                          </DropdownMenuGroup>
                        </DropdownMenuContent>
                      </DropdownMenu>
                      <Action label={t("Delete")} onClick={() => void attempt(() => promptToolsApi.favorites.remove(item.id))}>
                        <Trash2 />
                      </Action>
                    </div>
                  </div>
                )}
              </li>
            ))}
          </ul>
        )}
        {favorites.hasNextPage ? (
          <div className="flex justify-center p-2">
            <Button variant="ghost" size="xs" onClick={() => void favorites.fetchNextPage()} disabled={favorites.isFetchingNextPage}>
              {favorites.isFetchingNextPage ? <Spinner aria-label={t("Loading")} /> : null}
              {t("Load more")}
            </Button>
          </div>
        ) : null}
      </div>
    </div>
  )
}
