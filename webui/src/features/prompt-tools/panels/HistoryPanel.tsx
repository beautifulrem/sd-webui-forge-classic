/**
 * Prompt history (recorded by the server for every queued generation):
 * search, use (replace the prompt), append, save as a favorite, name (named
 * entries are kept when the history is trimmed), delete, clear.
 */
import { useInfiniteQuery, useQueryClient } from "@tanstack/react-query"
import { Check, ListPlus, Pencil, Star, Trash2 } from "lucide-react"
import { type ReactNode, useState } from "react"
import { useTranslation } from "react-i18next"
import { toast } from "sonner"
import { Button } from "@/components/ui/button"
import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty"
import { Input } from "@/components/ui/input"
import { Spinner } from "@/components/ui/spinner"
import { useDebouncedValue } from "@/features/prompt-editor/use-debounced-value"
import { useLocale } from "@/i18n"
import { type HistoryItem, promptToolsApi } from "../api"
import { toolKeys } from "../queries"
import { appendText, type PanelProps, timeAgo } from "./helpers"
import { PanelHeader, SearchBox } from "./panel"

function Action({ label, onClick, children, disabled }: { label: string; onClick: () => void; children: ReactNode; disabled?: boolean }) {
  return (
    <Button variant="ghost" size="icon-xs" aria-label={label} title={label} onClick={onClick} disabled={disabled}>
      {children}
    </Button>
  )
}

function failed(t: (key: string) => string, error: unknown) {
  toast.error(t("Something went wrong"), { description: error instanceof Error ? error.message : String(error) })
}

export default function HistoryPanel({ value, setValue, kind, disabled, close }: PanelProps) {
  const { t } = useTranslation()
  const locale = useLocale((state) => state.locale)
  const client = useQueryClient()
  const field = kind === "negative" ? "negative" : "prompt"
  const [query, setQuery] = useState("")
  const search = useDebouncedValue(query.trim(), 250)
  const [renaming, setRenaming] = useState<{ id: string; name: string } | null>(null)
  const [confirmClear, setConfirmClear] = useState(false)

  const history = useInfiniteQuery({
    queryKey: toolKeys.history(field, search),
    queryFn: ({ pageParam, signal }) => promptToolsApi.history.list({ field, q: search || undefined, cursor: pageParam, limit: 40 }, signal),
    initialPageParam: null as string | null,
    getNextPageParam: (page) => page.next_cursor ?? undefined,
    staleTime: 10_000,
  })
  const items = history.data?.pages.flatMap((page) => page.items) ?? []
  const refresh = () => client.invalidateQueries({ queryKey: ["prompt-tools", "history"] })
  const textOf = (item: HistoryItem) => (field === "negative" ? item.negative_prompt : item.prompt)

  const use = (item: HistoryItem) => {
    setValue(textOf(item))
    close()
    toast.success(t("Prompt replaced"), { description: t("Undo with Ctrl+Z in the editor.") })
  }

  const favorite = async (item: HistoryItem) => {
    try {
      await promptToolsApi.favorites.add({ text: textOf(item), kind: field === "negative" ? "negative" : "prompt", name: item.name })
      await client.invalidateQueries({ queryKey: ["prompt-tools", "favorites"] })
      await client.invalidateQueries({ queryKey: toolKeys.collections })
      toast.success(t("Saved to favorites"))
    } catch (error) {
      failed(t, error)
    }
  }

  const rename = async () => {
    if (!renaming) return
    try {
      // the list is grouped by this prompt: the name goes to every entry of the group
      await promptToolsApi.history.rename(renaming.id, renaming.name, field)
      setRenaming(null)
      await refresh()
    } catch (error) {
      failed(t, error)
    }
  }

  const remove = async (item: HistoryItem) => {
    try {
      await promptToolsApi.history.remove(item.id, field)
      await refresh()
    } catch (error) {
      failed(t, error)
    }
  }

  const clear = async () => {
    setConfirmClear(false)
    try {
      const { removed } = await promptToolsApi.history.clear()
      await refresh()
      toast.success(t("Removed {{count}} history entries", { count: removed }))
    } catch (error) {
      failed(t, error)
    }
  }

  return (
    <div className="flex max-h-[inherit] min-h-0 flex-col">
      <PanelHeader title={t("Prompt history")}>
        {confirmClear ? (
          <span role="alert" className="flex items-center gap-1 text-xs">
            <span className="text-muted-foreground">{t("Clear the history? Named entries are kept.")}</span>
            <Button variant="ghost" size="xs" onClick={() => setConfirmClear(false)}>
              {t("Cancel")}
            </Button>
            <Button variant="destructive" size="xs" onClick={() => void clear()}>
              {t("Clear")}
            </Button>
          </span>
        ) : (
          <Button variant="ghost" size="xs" onClick={() => setConfirmClear(true)} disabled={items.length === 0}>
            {t("Clear")}
          </Button>
        )}
      </PanelHeader>
      <div className="border-b px-3 py-2">
        <SearchBox value={query} onChange={setQuery} placeholder={t("Search the history")} />
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto">
        {history.isPending ? (
          <div className="flex h-32 items-center justify-center">
            <Spinner aria-label={t("Loading")} />
          </div>
        ) : items.length === 0 ? (
          <Empty className="py-8">
            <EmptyHeader>
              <EmptyTitle>{search ? t("No prompts match “{{query}}”", { query: search }) : t("No history yet")}</EmptyTitle>
              {search ? null : <EmptyDescription>{t("The prompts of every generation are kept here.")}</EmptyDescription>}
            </EmptyHeader>
          </Empty>
        ) : (
          <ul className="divide-y" aria-label={t("Prompt history")}>
            {items.map((item) => (
              <li key={item.id} className="group/item px-2 py-1.5 hover:bg-muted/50 focus-within:bg-muted/50">
                {renaming?.id === item.id ? (
                  <form
                    className="flex items-center gap-1 px-1"
                    onSubmit={(event) => {
                      event.preventDefault()
                      void rename()
                    }}
                  >
                    <Input
                      autoFocus
                      value={renaming.name}
                      maxLength={200}
                      aria-label={t("Name")}
                      placeholder={t("Name")}
                      className="h-7 text-xs"
                      onChange={(event) => setRenaming({ id: item.id, name: event.target.value })}
                      onKeyDown={(event) => {
                        if (event.key === "Escape") {
                          event.stopPropagation()
                          setRenaming(null)
                        }
                      }}
                    />
                    <Action label={t("Save")} onClick={() => void rename()}>
                      <Check />
                    </Action>
                  </form>
                ) : (
                  <div className="flex items-start gap-1">
                    <button
                      type="button"
                      disabled={disabled}
                      onClick={() => use(item)}
                      title={t("Use this prompt")}
                      className="min-w-0 flex-1 rounded-md px-1 py-0.5 text-left outline-none focus-visible:ring-3 focus-visible:ring-ring/50 disabled:opacity-50"
                    >
                      {item.name ? <span className="block truncate text-xs font-semibold">{item.name}</span> : null}
                      <span className="line-clamp-3 text-xs break-words whitespace-pre-wrap">{textOf(item)}</span>
                      <span className="mt-0.5 block text-[0.65rem] text-muted-foreground">
                        {timeAgo(item.last_used, locale)}
                        {item.uses > 1 ? ` · ${t("{{count}} generations", { count: item.uses })}` : ""}
                        {item.tab ? ` · ${t(item.tab)}` : ""}
                      </span>
                    </button>
                    <div className="flex shrink-0 items-center opacity-0 transition-opacity group-hover/item:opacity-100 group-focus-within/item:opacity-100 pointer-coarse:opacity-100">
                      <Action label={t("Append to the prompt")} disabled={disabled} onClick={() => setValue(appendText(value, textOf(item)))}>
                        <ListPlus />
                      </Action>
                      <Action label={t("Save to favorites")} onClick={() => void favorite(item)}>
                        <Star />
                      </Action>
                      <Action label={t("Rename")} onClick={() => setRenaming({ id: item.id, name: item.name })}>
                        <Pencil />
                      </Action>
                      <Action label={t("Delete")} onClick={() => void remove(item)}>
                        <Trash2 />
                      </Action>
                    </div>
                  </div>
                )}
              </li>
            ))}
          </ul>
        )}
        {history.hasNextPage ? (
          <div className="flex justify-center p-2">
            <Button variant="ghost" size="xs" onClick={() => void history.fetchNextPage()} disabled={history.isFetchingNextPage}>
              {history.isFetchingNextPage ? <Spinner aria-label={t("Loading")} /> : null}
              {t("Load more")}
            </Button>
          </div>
        ) : null}
      </div>
    </div>
  )
}
