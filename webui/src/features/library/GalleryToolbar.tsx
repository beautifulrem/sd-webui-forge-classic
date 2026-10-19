/** Gallery filters: search, form / folder, model, date range, stars, sort, layout and tile size. */
import { useQuery } from "@tanstack/react-query"
import { CalendarRange, FilterX, LayoutGrid, Rows3, Search, Star, X } from "lucide-react"
import { useEffect, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import type { GalleryCategory, GallerySort } from "@/api/types"
import { SearchSelect } from "@/components/SearchSelect"
import { Segmented } from "@/components/Segmented"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { useFittingPlaceholder } from "@/lib/use-fitting-placeholder"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Toggle } from "@/components/ui/toggle"
import { cn } from "@/lib/utils"
import { activeFilters, useGalleryView } from "./filters"
import { CATEGORY_LABELS, daysAgo } from "./labels"
import { galleryFacetsQuery } from "./queries"

const ALL = "__all__"

const SORT_LABELS: Record<GallerySort, string> = {
  newest: "Newest first",
  oldest: "Oldest first",
  name: "File name",
  largest: "Largest first",
}

function PlainSelect<V extends string>(props: { value: V; onChange: (value: V) => void; options: { value: V; label: string }[]; label: string; className?: string }) {
  return (
    <Select<V> items={props.options} value={props.value} onValueChange={(next) => next !== null && props.onChange(next)}>
      <SelectTrigger size="sm" aria-label={props.label} className={props.className}>
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {props.options.map((option) => (
          <SelectItem key={option.value} value={option.value}>
            {option.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}

/** The search box commits 300 ms after typing stops (or on Enter). */
function SearchBox() {
  const { t } = useTranslation()
  const q = useGalleryView((s) => s.q)
  const set = useGalleryView((s) => s.set)
  const [text, setText] = useState(q)
  const [seen, setSeen] = useState(q)
  if (q !== seen) {
    setSeen(q)
    setText(q)
  }
  useEffect(() => {
    if (text === q) return
    const timer = setTimeout(() => set({ q: text }), 300)
    return () => clearTimeout(timer)
  }, [text, q, set])
  const input = useRef<HTMLInputElement>(null)
  const placeholder = useFittingPlaceholder(input, [t("Search prompts, parameters, file names…"), t("Search…")])
  return (
    <div className="relative min-w-60 flex-1 sm:max-w-80">
      <Search className="pointer-events-none absolute top-1/2 left-2 size-3.5 -translate-y-1/2 text-muted-foreground" aria-hidden />
      <Input
        type="search"
        value={text}
        onChange={(event) => setText(event.target.value)}
        onKeyDown={(event) => event.key === "Enter" && set({ q: text })}
        ref={input}
        placeholder={placeholder}
        aria-label={t("Search the gallery")}
        className="h-7 pl-7 text-[0.8rem]"
      />
    </div>
  )
}

function DateRange() {
  const { t, i18n } = useTranslation()
  const from = useGalleryView((s) => s.from)
  const to = useGalleryView((s) => s.to)
  const set = useGalleryView((s) => s.set)
  const format = (date: string) => new Date(`${date}T00:00:00`).toLocaleDateString(i18n.language, { month: "short", day: "numeric" })
  const summary = from && to ? `${format(from)} – ${format(to)}` : from ? t("Since {{date}}", { date: format(from) }) : to ? t("Until {{date}}", { date: format(to) }) : t("Any date")
  return (
    <Popover>
      <PopoverTrigger render={<Button variant="outline" size="sm" aria-label={t("Date range")} className={cn((from || to) && "border-primary/50 text-primary")} />}>
        <CalendarRange aria-hidden />
        <span className="max-w-36 truncate">{summary}</span>
      </PopoverTrigger>
      <PopoverContent aria-label={t("Date range")} align="start" className="w-72">
        <div className="flex flex-col gap-3">
          <div className="grid grid-cols-2 gap-2">
            <label className="flex flex-col gap-1 text-xs font-medium">
              {t("From")}
              <Input type="date" value={from ?? ""} max={to ?? undefined} onChange={(event) => set({ from: event.target.value || null })} className="h-8" />
            </label>
            <label className="flex flex-col gap-1 text-xs font-medium">
              {t("To")}
              <Input type="date" value={to ?? ""} min={from ?? undefined} onChange={(event) => set({ to: event.target.value || null })} className="h-8" />
            </label>
          </div>
          <div className="flex flex-wrap gap-1">
            <Button variant="secondary" size="xs" onClick={() => set({ from: daysAgo(0), to: daysAgo(0) })}>
              {t("Today")}
            </Button>
            <Button variant="secondary" size="xs" onClick={() => set({ from: daysAgo(6), to: null })}>
              {t("Last 7 days")}
            </Button>
            <Button variant="secondary" size="xs" onClick={() => set({ from: daysAgo(29), to: null })}>
              {t("Last 30 days")}
            </Button>
            <Button variant="ghost" size="xs" onClick={() => set({ from: null, to: null })} disabled={!from && !to}>
              {t("Any date")}
            </Button>
          </div>
        </div>
      </PopoverContent>
    </Popover>
  )
}

export function GalleryToolbar({ count, total }: { count: number; total: number | null }) {
  const { t } = useTranslation()
  const view = useGalleryView()
  const facets = useQuery(galleryFacetsQuery)
  const categories = facets.data?.categories ?? []
  const models = facets.data?.models ?? []
  const filtered = activeFilters(view)

  return (
    <div className="flex flex-col gap-2 border-b bg-background/95 px-3 py-2 backdrop-blur">
      <div className="flex flex-wrap items-center gap-1.5">
        <SearchBox />
        <PlainSelect
          label={t("Form or folder")}
          value={view.category ?? ALL}
          onChange={(value) => view.set({ category: value === ALL ? null : (value as GalleryCategory) })}
          options={[
            { value: ALL, label: t("All images") },
            ...categories.filter((entry) => entry.count > 0 || entry.name === view.category).map((entry) => ({ value: entry.name, label: `${t(CATEGORY_LABELS[entry.name as GalleryCategory] ?? entry.name)} · ${entry.count}` })),
          ]}
        />
        <SearchSelect
          size="sm"
          aria-label={t("Model")}
          className="w-44"
          value={view.model ?? ALL}
          onValueChange={(value) => view.set({ model: value === ALL ? null : value })}
          options={[{ value: ALL, label: t("All models") }, ...models.map((model) => ({ value: model.name, label: model.name, hint: String(model.count) }))]}
        />
        <DateRange />
        <Toggle
          size="sm"
          variant="outline"
          pressed={view.starred}
          onPressedChange={(starred) => view.set({ starred })}
          aria-label={t("Starred only")}
          title={t("Starred only")}
        >
          <Star className={cn(view.starred && "fill-amber-400 text-amber-400")} aria-hidden />
        </Toggle>
        {filtered > 0 ? (
          <Button variant="ghost" size="sm" onClick={view.reset}>
            <FilterX aria-hidden />
            {t("Clear filters")}
          </Button>
        ) : null}
        <div className="ml-auto flex items-center gap-1.5">
          <PlainSelect
            label={t("Sort")}
            value={view.sort}
            onChange={(sort) => view.set({ sort })}
            options={(Object.keys(SORT_LABELS) as GallerySort[]).map((value) => ({ value, label: t(SORT_LABELS[value]) }))}
          />
          <Segmented
            size="sm"
            aria-label={t("Layout")}
            value={view.layout}
            onValueChange={(layout) => view.set({ layout })}
            options={[
              { value: "justified", label: t("Rows") },
              { value: "masonry", label: t("Columns") },
            ]}
          />
          <Segmented
            size="sm"
            aria-label={t("Tile size")}
            value={view.size}
            onValueChange={(size) => view.set({ size })}
            options={[
              { value: "s", label: t("S") },
              { value: "m", label: t("M") },
              { value: "l", label: t("L") },
            ]}
          />
        </div>
      </div>
      <p className="flex items-center gap-1.5 text-xs text-muted-foreground tabular-nums" aria-live="polite">
        {view.layout === "justified" ? <Rows3 className="size-3.5" aria-hidden /> : <LayoutGrid className="size-3.5" aria-hidden />}
        {total === null ? t("Loading…") : filtered > 0 ? t("{{count}} of {{total}} images match", { count: total, total: facets.data?.total ?? total }) : t("Images: {{count}}", { count: total })}
        {count < (total ?? 0) ? <span>· {t("{{count}} loaded", { count })}</span> : null}
        {view.q ? (
          <button type="button" className="inline-flex items-center gap-0.5 rounded px-1 hover:bg-muted" onClick={() => view.set({ q: "" })}>
            “{view.q}” <X className="size-3" aria-label={t("Clear search")} />
          </button>
        ) : null}
      </p>
    </div>
  )
}
