/**
 * Quick tag groups: categories of common tags with labels in the UI
 * language. A tag button adds the tag to the prompt (written like an
 * accepted completion: underscores, escaped brackets), or removes it when it
 * is already there (pressed).
 */
import { useQuery } from "@tanstack/react-query"
import { useDeferredValue, useMemo, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty"
import { Spinner } from "@/components/ui/spinner"
import { toServerLocale, useLocale } from "@/i18n"
import { useScrollFade } from "@/lib/use-scroll-fade"
import { cn } from "@/lib/utils"
import { promptToolsApi, type TagCategory } from "../api"
import { findTag, parsePrompt, tagKey, toggleTag } from "../chips/parse"
import { toolKeys } from "../queries"
import { groupTagText, type PanelProps } from "./helpers"
import { PanelHeader, SearchBox } from "./panel"

type FlatTag = { tag: string; label: string | null; group: string; color: string | null }

function matches(item: FlatTag, query: string): boolean {
  const q = query.toLowerCase()
  return item.tag.toLowerCase().includes(q) || tagKey(item.tag).includes(tagKey(q)) || (item.label?.toLowerCase().includes(q) ?? false)
}

function TagButton({ item, pressed, disabled, onToggle }: { item: FlatTag; pressed: boolean; disabled: boolean; onToggle: () => void }) {
  const { t } = useTranslation()
  const shown = item.label && item.label !== item.tag ? item.label : null
  return (
    <button
      type="button"
      aria-pressed={pressed}
      disabled={disabled}
      title={pressed ? t("Remove “{{tag}}” from the prompt", { tag: item.tag }) : t("Add “{{tag}}” to the prompt", { tag: item.tag })}
      onClick={onToggle}
      className={cn(
        "flex max-w-full min-w-0 flex-col items-start rounded-md border px-2 py-1 text-left text-xs leading-tight transition-colors outline-none",
        "hover:bg-muted focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 disabled:opacity-50",
        pressed ? "border-primary/50 bg-primary/10 text-primary hover:bg-primary/15" : "border-border bg-background",
      )}
    >
      <span className="max-w-full truncate font-medium">{shown ?? item.tag}</span>
      {shown ? <span className="max-w-full truncate text-[0.65rem] text-muted-foreground">{item.tag}</span> : null}
    </button>
  )
}

export default function GroupsPanel({ value, setValue, disabled, config }: PanelProps) {
  const { t } = useTranslation()
  const locale = toServerLocale(useLocale((state) => state.locale))
  const groups = useQuery({ queryKey: toolKeys.groups(locale), queryFn: ({ signal }) => promptToolsApi.groups(locale, signal), staleTime: Infinity })
  const [category, setCategory] = useState(0)
  const [query, setQuery] = useState("")
  const search = useDeferredValue(query.trim())
  const doc = useMemo(() => parsePrompt(value), [value])
  const categories: TagCategory[] = useMemo(() => groups.data ?? [], [groups.data])
  const current = categories[Math.min(category, Math.max(categories.length - 1, 0))]

  const found = useMemo(() => {
    if (!search) return null
    const out: FlatTag[] = []
    for (const cat of categories) for (const group of cat.groups) for (const tag of group.tags) {
      const item = { tag: tag.tag, label: tag.label ?? null, group: `${cat.name} · ${group.name}`, color: group.color ?? null }
      if (matches(item, search)) out.push(item)
      if (out.length >= 200) return out
    }
    return out
  }, [categories, search])

  const toggle = (tag: string) => setValue(toggleTag(value, groupTagText(tag, config.tags)))
  const isIn = (tag: string) => findTag(doc, tag) >= 0

  return (
    <div className="flex max-h-[inherit] min-h-0 flex-col">
      <PanelHeader title={t("Quick tags")}>
        <SearchBox value={query} onChange={setQuery} placeholder={t("Search tags")} className="w-44" />
      </PanelHeader>
      {groups.isPending ? (
        <div className="flex h-40 items-center justify-center">
          <Spinner aria-label={t("Loading")} />
        </div>
      ) : categories.length === 0 ? (
        <Empty className="py-8">
          <EmptyHeader>
            <EmptyTitle>{t("No tag groups")}</EmptyTitle>
            <EmptyDescription>{t("Add YAML files to the group_tags folder of the prompt tools extension.")}</EmptyDescription>
          </EmptyHeader>
        </Empty>
      ) : found ? (
        <div className="min-h-0 flex-1 overflow-y-auto p-3">
          {found.length === 0 ? (
            <p className="py-6 text-center text-sm text-muted-foreground">{t("No tags match “{{query}}”", { query: search })}</p>
          ) : (
            <ul className="flex flex-wrap gap-1.5" aria-label={t("Search results")}>
              {found.map((item) => (
                <li key={`${item.group}:${item.tag}`} className="max-w-full" title={item.group}>
                  <TagButton item={item} pressed={isIn(item.tag)} disabled={disabled} onToggle={() => toggle(item.tag)} />
                </li>
              ))}
            </ul>
          )}
        </div>
      ) : (
        <>
          <CategoryStrip categories={categories} current={current} onSelect={setCategory} />
          <div role="tabpanel" aria-label={current?.name} className="min-h-0 flex-1 space-y-3 overflow-y-auto p-3">
            {current?.groups.map((group) => (
              <section key={group.name} aria-label={group.name} className="space-y-1.5">
                <h3 className="flex items-center gap-1.5 text-xs font-semibold text-muted-foreground">
                  <span aria-hidden className="size-2 rounded-full bg-muted-foreground/40" style={group.color ? { background: group.color } : undefined} />
                  {group.name}
                </h3>
                <ul className="flex flex-wrap gap-1.5">
                  {group.tags.map((tag) => {
                    const item = { tag: tag.tag, label: tag.label ?? null, group: group.name, color: group.color ?? null }
                    return (
                      <li key={tag.tag} className="max-w-full">
                        <TagButton item={item} pressed={isIn(tag.tag)} disabled={disabled} onToggle={() => toggle(tag.tag)} />
                      </li>
                    )
                  })}
                </ul>
              </section>
            ))}
          </div>
        </>
      )}
    </div>
  )
}

/** The tag categories: a strip that scrolls sideways, fading at the edges where more categories are. */
function CategoryStrip({ categories, current, onSelect: setCategory }: { categories: readonly TagCategory[]; current: TagCategory | undefined; onSelect: (index: number) => void }) {
  const { t } = useTranslation()
  const strip = useRef<HTMLDivElement>(null)
  const fade = useScrollFade(strip, current?.name)
  return (
    <div
      ref={strip}
      style={fade}
      role="tablist"
      aria-label={t("Tag categories")}
      className="flex shrink-0 gap-1 overflow-x-auto border-b px-2 py-1.5 [scrollbar-width:none]"
      onKeyDown={(event) => {
        const step = event.key === "ArrowRight" ? 1 : event.key === "ArrowLeft" ? -1 : 0
        if (!step && event.key !== "Home" && event.key !== "End") return
        event.preventDefault()
        const at = categories.indexOf(current!)
        const next = event.key === "Home" ? 0 : event.key === "End" ? categories.length - 1 : (at + step + categories.length) % categories.length
        setCategory(next)
        const tabs = event.currentTarget.querySelectorAll<HTMLButtonElement>("[role=tab]")
        tabs[next]?.focus()
      }}
    >
      {categories.map((cat, index) => (
        <button
          key={cat.name}
          type="button"
          role="tab"
          aria-selected={cat === current}
          data-active={cat === current || undefined}
          tabIndex={cat === current ? 0 : -1}
          onClick={() => setCategory(index)}
          className={cn(
            "shrink-0 rounded-md px-2 py-1 text-xs font-medium whitespace-nowrap transition-colors outline-none focus-visible:ring-3 focus-visible:ring-ring/50",
            cat === current ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:bg-muted hover:text-foreground",
          )}
        >
          {cat.name}
        </button>
      ))}
    </div>
  )
}
