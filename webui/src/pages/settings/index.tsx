/**
 * Settings: every option rendered from the settings schema (GET /options) by
 * the schema renderer, in one searchable, virtualized list with a section
 * nav; filters for changed / unsaved / pinned settings; Apply with
 * optimistic updates; import / export; and the System views (server and
 * models, system info, licenses).
 *
 * Route: `#/settings?view=server|sysinfo|licenses&q=<search>&filter=changed|unsaved|pinned&section=<id>`.
 * The search and filter also survive leaving the page (`settings-filters.ts`).
 */
import { useQuery } from "@tanstack/react-query"
import { Download, Ellipsis, FileUp, ListTree, Lock, RefreshCw, Search, Settings as SettingsIcon, X } from "lucide-react"
import { lazy, Suspense, useCallback, useDeferredValue, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { toast } from "sonner"
import { useStore } from "zustand"
import { bootstrapQuery, optionsQuery } from "@/api/queries"
import { navigate, type Route } from "@/app/router"
import { PagePlaceholder } from "@/components/PagePlaceholder"
import { Segmented } from "@/components/Segmented"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { InputGroup, InputGroupAddon, InputGroupButton, InputGroupInput } from "@/components/ui/input-group"
import { Kbd } from "@/components/ui/kbd"
import { Sheet, SheetContent, SheetHeader, SheetTitle } from "@/components/ui/sheet"
import { Skeleton } from "@/components/ui/skeleton"
import { requestServerCommand } from "@/features/server-control/server-control"
import { useUnsaved } from "@/features/server-control/unsaved"
import { QUICKSETTINGS_KEY } from "@/features/quicksettings/pinned"
import { ApplyBar, EffectNotice } from "./ApplyBar"
import { ImportDialog } from "./ImportExport"
import { downloadSettings } from "./settings-file"
import { setSettingsMode, setSettingsQuery, useSettingsFilters, useSyncFiltersWithRoute } from "./settings-filters"
import { buildSettingsIndex, FILTER_MODES, type FilterMode, filterSettings, flattenRows, searchTerms, unsavedKeys } from "./settings-model"
import { applyChanges, discardChanges, SETTINGS_OWNER, settingsForm, syncSchema, useSettingsEditor } from "./settings-store"
import { SettingsList, type SettingsListHandle } from "./SettingsList"
import { SettingsNav } from "./SettingsNav"
import { type SettingsView, SYSTEM_VIEWS } from "./views"
import { serverStatusQuery } from "./queries"
import "./settings.css"

const ServerView = lazy(() => import("./system/ServerView").then((m) => ({ default: m.ServerView })))
const SysinfoView = lazy(() => import("./system/SysinfoView").then((m) => ({ default: m.SysinfoView })))
const LicensesView = lazy(() => import("./system/LicensesView").then((m) => ({ default: m.LicensesView })))

const FILTER_LABELS: Record<FilterMode, string> = { all: "All", changed: "Changed", unsaved: "Unsaved", pinned: "Pinned" }

function parseView(value: string | null | undefined): SettingsView {
  return SYSTEM_VIEWS.some((view) => view.id === value) ? (value as SettingsView) : "settings"
}

/** The settings route for `view`, keeping `keep`'s search and filter. */
function routeFor(view: SettingsView, keep?: URLSearchParams, extra: Record<string, string> = {}): Route {
  const query = new URLSearchParams(extra)
  for (const key of ["q", "filter"]) {
    const value = keep?.get(key)
    if (value && !query.has(key)) query.set(key, value)
  }
  if (view !== "settings") query.set("view", view)
  return { name: "settings", query: query.size > 0 ? query : undefined }
}

const EMPTY: Record<string, unknown> = {}
const NONE: ReadonlySet<string> = new Set()

/** Settings pages mounted (StrictMode mounts twice): the "not applied yet" toast waits for the last to go. */
let mountedPages = 0

/** Ctrl/Cmd+S applies; "/" focuses the search (outside text fields). */
function useShortcuts(onApply: () => void, search: React.RefObject<HTMLInputElement | null>) {
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && !event.altKey && event.key.toLowerCase() === "s") {
        event.preventDefault()
        onApply()
        return
      }
      const target = event.target as HTMLElement | null
      const typing = target?.closest("input, textarea, select, [contenteditable=true], .cm-editor")
      if (event.key === "/" && !typing && !event.metaKey && !event.ctrlKey && !event.altKey) {
        event.preventDefault()
        search.current?.focus()
        search.current?.select()
      }
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [onApply, search])
}

function SettingsSkeleton() {
  return (
    <div className="mx-auto flex w-full max-w-4xl flex-col gap-3 px-6 py-6" aria-busy="true">
      <Skeleton className="h-4 w-32" />
      {Array.from({ length: 6 }, (_, i) => (
        <Skeleton key={i} className="h-20 w-full" />
      ))}
    </div>
  )
}

export default function SettingsPage({ route }: { route: Route }) {
  const { t, i18n } = useTranslation()
  const view = parseView(route.query?.get("view"))
  const options = useQuery(optionsQuery)
  const status = useQuery(serverStatusQuery)
  const bootstrap = useQuery(bootstrapQuery)
  const listRef = useRef<SettingsListHandle>(null)
  const searchRef = useRef<HTMLInputElement>(null)

  const query = useSettingsFilters((s) => s.query)
  const mode = useSettingsFilters((s) => s.mode)
  const setQuery = setSettingsQuery
  const setMode = setSettingsMode
  useSyncFiltersWithRoute(route.query?.get("q") ?? null, route.query?.get("filter") ?? null)
  const [activeSection, setActiveSection] = useState<string | null>(null)
  const [navOpen, setNavOpen] = useState(false)
  const [importOpen, setImportOpen] = useState(false)
  const deferredQuery = useDeferredValue(query)

  useLayoutEffect(() => {
    if (options.data) syncSchema(options.data)
  }, [options.data])

  const schema = useSettingsEditor((s) => s.schema)
  const saved = useSettingsEditor((s) => s.saved)
  const values = useStore(settingsForm, (s) => s.values[SETTINGS_OWNER]) ?? EMPTY
  const index = useMemo(() => (schema ? buildSettingsIndex(schema) : null), [schema])
  // Re-translate the search text when the language changes.
  const language = i18n.language
  const translate = useCallback((text: string) => t(text, { lng: language }), [t, language])
  const pinnedList = bootstrap.data?.options[QUICKSETTINGS_KEY]
  const pinned = useMemo(() => new Set(Array.isArray(pinnedList) ? (pinnedList as string[]) : []), [pinnedList])

  // Only the modes that compare values depend on them: an edit does not refilter "All".
  const filterValues = mode === "changed" || mode === "unsaved" ? values : EMPTY
  const filterSaved = mode === "unsaved" ? saved : EMPTY
  const filterDefaults = mode === "changed" ? schema?.defaults : undefined
  const filterPinned = mode === "pinned" ? pinned : NONE
  const sections = useMemo(
    () =>
      index
        ? filterSettings(index, { query: deferredQuery, mode, values: filterValues, saved: filterSaved, defaults: filterDefaults, pinned: filterPinned, translate })
        : [],
    [index, deferredQuery, mode, filterValues, filterSaved, filterDefaults, filterPinned, translate],
  )
  const rows = useMemo(() => flattenRows(sections), [sections])
  const terms = useMemo(() => searchTerms(deferredQuery), [deferredQuery])
  const filtering = terms.length > 0 || mode !== "all"

  const inputKeys = useSettingsEditor((s) => s.inputs)
  const unsaved = useMemo(() => unsavedKeys(values, saved, new Set(inputKeys.keys())), [values, saved, inputKeys])
  const unsavedBySection = useMemo(() => {
    const map = new Map<string, number>()
    for (const key of unsaved) {
      const section = index?.byKey.get(key)?.section.id
      if (section) map.set(section, (map.get(section) ?? 0) + 1)
    }
    return map
  }, [unsaved, index])

  // Edits stay when leaving the page: the shell asks before the tab closes, and a toast offers to come back.
  const setUnsavedCount = useUnsaved((s) => s.set)
  useEffect(() => setUnsavedCount("settings", unsaved.length), [unsaved.length, setUnsavedCount])
  const unsavedRef = useRef(0)
  const tRef = useRef(t)
  useEffect(() => {
    unsavedRef.current = unsaved.length
    tRef.current = t
  })
  useEffect(() => {
    mountedPages++
    return () => {
      mountedPages--
      const count = unsavedRef.current
      const translate = tRef.current
      // Only when the page is really left (not a StrictMode remount).
      setTimeout(() => {
        if (mountedPages > 0 || count === 0) return
        toast.info(translate("{{count}} settings changes are not applied yet", { count }), {
          id: "settings-unsaved",
          action: { label: translate("Review"), onClick: () => navigate(routeFor("settings", undefined, { filter: "unsaved" })) },
        })
      }, 0)
    }
  }, [])

  // Deep link to a section once the list exists.
  const initialSection = useRef(route.query?.get("section") ?? null)
  useEffect(() => {
    if (view !== "settings" || !initialSection.current || rows.length === 0) return
    const id = initialSection.current
    initialSection.current = null
    requestAnimationFrame(() => listRef.current?.scrollToSection(id))
  }, [view, rows.length])

  const apply = useCallback(async () => {
    if (unsavedRef.current === 0 || useSettingsEditor.getState().applying) return
    const result = await applyChanges()
    if (result.errors.length === 0 && !result.message) toast.success(t("Settings applied"), { id: "settings-apply" })
    else toast.error(t("Not every setting was applied"), { id: "settings-apply", description: result.message })
  }, [t])
  useShortcuts(() => void apply(), searchRef)

  const openView = (next: SettingsView) => {
    setNavOpen(false)
    navigate(routeFor(next, route.query), { replace: true })
  }
  const openSection = (id: string) => {
    setNavOpen(false)
    if (view !== "settings") {
      initialSection.current = id
      navigate(routeFor("settings", route.query), { replace: true })
      return
    }
    listRef.current?.scrollToSection(id)
  }
  const clearFilters = () => {
    setQuery("")
    setMode("all")
  }
  const onSectionInView = useCallback((id: string | null) => setActiveSection(id), [])

  const nav = (
    <SettingsNav
      sections={sections}
      view={view}
      activeSection={activeSection}
      filtering={filtering}
      unsaved={unsavedBySection}
      onSection={openSection}
      onView={openView}
    />
  )

  if (options.isError && !schema) {
    return <PagePlaceholder icon={SettingsIcon} title="Settings unavailable" description={options.error.message} />
  }

  const filterOptions = FILTER_MODES.map((value) => ({
    value,
    label: value === "unsaved" && unsaved.length > 0 ? `${t(FILTER_LABELS[value])} ${unsaved.length}` : t(FILTER_LABELS[value]),
  }))

  return (
    <div className="flex h-full min-h-0">
      <aside className="hidden w-60 shrink-0 overflow-y-auto border-r bg-sidebar/50 lg:block xl:w-64">{nav}</aside>
      <Sheet open={navOpen} onOpenChange={setNavOpen}>
        <SheetContent side="left" className="w-[min(20rem,85vw)] overflow-y-auto p-0">
          <SheetHeader className="border-b">
            <SheetTitle>{t("Settings sections")}</SheetTitle>
          </SheetHeader>
          {nav}
        </SheetContent>
      </Sheet>

      <div className="relative flex min-w-0 flex-1 flex-col">
        {view === "settings" ? (
          <>
            <div className="flex flex-wrap items-center gap-2 border-b px-3 py-2 sm:px-4">
              <Button variant="ghost" size="icon-sm" className="lg:hidden" aria-label={t("Settings sections")} onClick={() => setNavOpen(true)}>
                <ListTree />
              </Button>
              <InputGroup className="max-w-md min-w-48 flex-1">
                <InputGroupAddon>
                  <Search aria-hidden />
                </InputGroupAddon>
                <InputGroupInput
                  ref={searchRef}
                  type="search"
                  className="[&::-webkit-search-cancel-button]:hidden"
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Escape" && query) {
                      event.preventDefault()
                      event.stopPropagation()
                      setQuery("")
                    }
                  }}
                  placeholder={t("Search settings by name, help or key")}
                  aria-label={t("Search settings")}
                  aria-controls="settings-results"
                />
                <InputGroupAddon align="inline-end">
                  {query ? (
                    <InputGroupButton size="icon-xs" aria-label={t("Clear search")} onClick={() => setQuery("")}>
                      <X />
                    </InputGroupButton>
                  ) : (
                    <Kbd className="hidden sm:inline-flex">/</Kbd>
                  )}
                </InputGroupAddon>
              </InputGroup>
              <Segmented options={filterOptions} value={mode} onValueChange={setMode} aria-label={t("Show")} size="sm" className="w-auto shrink-0" />
              <span className="sr-only" role="status" aria-live="polite">
                {filtering ? t("{{count}} settings found", { count: sections.reduce((n, s) => n + s.count, 0) }) : ""}
              </span>
              <DropdownMenu>
                <DropdownMenuTrigger render={<Button variant="ghost" size="icon-sm" className="ml-auto" aria-label={t("More actions")} />}>
                  <Ellipsis />
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end" className="w-56">
                  <DropdownMenuItem onClick={() => downloadSettings(false)}>
                    <Download aria-hidden />
                    {t("Export changed settings")}
                  </DropdownMenuItem>
                  <DropdownMenuItem onClick={() => downloadSettings(true)}>
                    <Download aria-hidden />
                    {t("Export all settings")}
                  </DropdownMenuItem>
                  <DropdownMenuItem onClick={() => setImportOpen(true)}>
                    <FileUp aria-hidden />
                    {t("Import settings…")}
                  </DropdownMenuItem>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem onClick={() => requestServerCommand("reload")}>
                    <RefreshCw aria-hidden />
                    {t("Reload UI")}
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
            {status.data?.settings_frozen ? (
              <p role="status" className="flex items-center gap-2 border-b bg-muted/60 px-4 py-2 text-sm text-muted-foreground sm:px-6">
                <Lock className="size-4" aria-hidden />
                {t("Settings are read-only on this server (--freeze-settings).")}
              </p>
            ) : null}
            <EffectNotice />
            <div id="settings-results" className="min-h-0 flex-1">
              {schema && index ? (
                <SettingsList ref={listRef} rows={rows} terms={terms} onSectionInView={onSectionInView} onClearFilters={clearFilters} filtering={filtering} />
              ) : (
                <SettingsSkeleton />
              )}
            </div>
            <ApplyBar
              count={unsaved.length}
              onApply={() => void apply()}
              onDiscard={() => discardChanges()}
              reviewing={mode === "unsaved"}
              onReview={() => setMode(mode === "unsaved" ? "all" : "unsaved")}
            />
          </>
        ) : (
          <div className="min-h-0 flex-1 overflow-y-auto">
            <div className="flex items-center gap-2 px-3 pt-2 lg:hidden">
              <Button variant="ghost" size="sm" onClick={() => setNavOpen(true)}>
                <ListTree aria-hidden />
                {t("Settings sections")}
              </Button>
            </div>
            <Suspense fallback={<SettingsSkeleton />}>
              {view === "server" ? <ServerView onImport={() => setImportOpen(true)} onExport={downloadSettings} /> : null}
              {view === "sysinfo" ? <SysinfoView /> : null}
              {view === "licenses" ? <LicensesView /> : null}
            </Suspense>
          </div>
        )}
      </div>
      <ImportDialog
        open={importOpen}
        onOpenChange={setImportOpen}
        onStaged={(count) => {
          toast.success(t("{{count}} imported settings are ready to review", { count }))
          if (view !== "settings") navigate(routeFor("settings", route.query), { replace: true })
          setMode("unsaved")
        }}
      />
    </div>
  )
}
