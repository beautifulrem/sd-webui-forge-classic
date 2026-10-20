/**
 * Installed extensions: enable switches and update marks are staged, then
 * applied together ("Apply and reload UI" saves them, updates the marked
 * extensions and reloads). User extensions and built-in ones are listed
 * apart; built-in ones cannot be uninstalled.
 */
import { useMutation, useQuery } from "@tanstack/react-query"
import { ArrowUpCircle, Check, ChevronRight, CircleAlert, ExternalLink, GitBranch, Loader2, MoreHorizontal, RefreshCw, Search, Trash2, X } from "lucide-react"
import { useEffect, useId, useState } from "react"
import { useTranslation } from "react-i18next"
import { toast } from "sonner"
import { create } from "zustand"
import { api } from "@/api/endpoints"
import { queryClient } from "@/api/queries"
import { Segmented } from "@/components/Segmented"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuSeparator, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { InputGroup, InputGroupAddon, InputGroupInput } from "@/components/ui/input-group"
import { Progress } from "@/components/ui/progress"
import { Skeleton } from "@/components/ui/skeleton"
import { Switch } from "@/components/ui/switch"
import { requestServerCommand, runServerCommandAfterApply } from "@/features/server-control/server-control"
import { useUnsaved } from "@/features/server-control/unsaved"
import { cn } from "@/lib/utils"
import {
  adminErrorMessage,
  disabledList,
  type ExtensionInfo,
  type ExtensionList,
  formatDate,
  isEnabled,
  NO_CHANGES,
  pruneStaged,
  type Staged,
  stageEnabled,
  stagedCount,
  stageUpdate,
} from "./extensions-model"
import { extensionKeys, installedQuery } from "./queries"
import { checkForUpdates, stopUpdateCheck, useUpdateCheck } from "./update-check"

/** Staged changes survive switching tabs (not a page reload: the shell asks first). */
const useStaged = create<{ staged: Staged; set: (staged: Staged) => void }>()((set) => ({ staged: NO_CHANGES, set: (staged) => set({ staged }) }))

const DISABLE_ALL = [
  { value: "none", label: "None" },
  { value: "extra", label: "Extra" },
  { value: "all", label: "All" },
] as const

function StatusBadge({ ext, checking }: { ext: ExtensionInfo; checking: boolean }) {
  const { t } = useTranslation()
  if (checking) {
    return (
      <Badge variant="outline" className="gap-1">
        <Loader2 className="animate-spin" aria-hidden />
        {t("Checking…")}
      </Badge>
    )
  }
  if (ext.can_update) {
    return (
      <Badge className="gap-1">
        <ArrowUpCircle aria-hidden />
        {t(ext.status === "behind HEAD" ? "Behind HEAD" : "Update available")}
      </Badge>
    )
  }
  if (ext.status === "latest") {
    return (
      <Badge variant="secondary" className="gap-1">
        <Check aria-hidden />
        {t("Up to date")}
      </Badge>
    )
  }
  if (ext.status?.startsWith("unknown (")) {
    return (
      <Badge variant="destructive" className="gap-1">
        <CircleAlert aria-hidden />
        {t("Check failed")}
      </Badge>
    )
  }
  return null
}

function ExtensionRow(props: { ext: ExtensionInfo; staged: Staged; access: boolean; checking: boolean; onUninstall: (ext: ExtensionInfo) => void }) {
  const { t, i18n } = useTranslation()
  const { ext, staged, access } = props
  const set = useStaged((s) => s.set)
  const enabled = isEnabled(ext, staged)
  const changed = Object.hasOwn(staged.enabled, ext.name)
  const marked = staged.update.includes(ext.name)
  const check = useMutation({
    mutationFn: () => api.extensions.check(ext.name),
    onSuccess: (info) => queryClient.setQueryData<ExtensionList>(extensionKeys.installed, (list) => (list ? { ...list, items: list.items.map((item) => (item.name === info.name ? info : item)) } : list)),
    onError: (error) => toast.error(t("Could not check {{name}}", { name: ext.name }), { description: adminErrorMessage(error, t) }),
  })
  // Folder names may contain spaces: ids from useId stay valid for aria references.
  const switchId = useId()
  return (
    <li className={cn("flex flex-wrap items-center gap-x-3 gap-y-1.5 rounded-lg px-3 py-2.5 transition-colors hover:bg-muted/40", changed && "bg-primary/[0.06]")}>
      <Switch id={switchId} checked={enabled} disabled={!access} onCheckedChange={(on) => set(stageEnabled(staged, ext, on))} aria-describedby={`${switchId}-meta`} />
      <div className="min-w-0 flex-1 basis-48">
        <label htmlFor={switchId} className={cn("block truncate text-sm font-medium", !enabled && "text-muted-foreground line-through decoration-muted-foreground/50")}>
          {ext.name}
        </label>
        <p id={`${switchId}-meta`} className="flex flex-wrap items-center gap-x-2 text-xs text-muted-foreground">
          {ext.remote_url ? (
            <a href={ext.remote_url} target="_blank" rel="noreferrer noopener" className="inline-flex max-w-72 items-center gap-0.5 truncate hover:text-foreground hover:underline">
              {ext.remote_url.replace(/^https:\/\/(www\.)?/, "")}
              <ExternalLink className="size-3 shrink-0" aria-hidden />
            </a>
          ) : ext.builtin ? (
            <span>{t("Built-in")}</span>
          ) : (
            <span>{t("Local folder")}</span>
          )}
          {ext.branch ? (
            <span className="inline-flex items-center gap-0.5">
              <GitBranch className="size-3" aria-hidden />
              {ext.branch}
            </span>
          ) : null}
          {ext.version ? (
            ext.commit_url ? (
              <a href={ext.commit_url} target="_blank" rel="noreferrer noopener" className="font-mono hover:text-foreground hover:underline">
                {ext.version}
              </a>
            ) : (
              <span className="font-mono">{ext.version}</span>
            )
          ) : null}
          {ext.commit_date ? <span className="tabular-nums">{formatDate(ext.commit_date, i18n.language)}</span> : null}
          {changed ? <span className="font-medium text-primary">{enabled ? t("Will be enabled") : t("Will be disabled")}</span> : null}
        </p>
      </div>
      <StatusBadge ext={ext} checking={props.checking || check.isPending} />
      {ext.can_update ? (
        <label className="flex items-center gap-1.5 text-xs">
          <Checkbox checked={marked} disabled={!access} onCheckedChange={(on) => set(stageUpdate(staged, ext.name, Boolean(on)))} />
          {t("Update")}
        </label>
      ) : null}
      <DropdownMenu>
        <DropdownMenuTrigger render={<Button variant="ghost" size="icon-sm" aria-label={t("Actions for {{name}}", { name: ext.name })} />}>
          <MoreHorizontal />
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-52">
          <DropdownMenuItem disabled={!access || !ext.remote || check.isPending} onClick={() => check.mutate()}>
            <RefreshCw aria-hidden />
            {t("Check for updates")}
          </DropdownMenuItem>
          {ext.remote_url ? (
            <DropdownMenuItem render={<a href={ext.remote_url} target="_blank" rel="noreferrer noopener" />}>
              <ExternalLink aria-hidden />
              {t("Open repository")}
            </DropdownMenuItem>
          ) : null}
          {!ext.builtin ? (
            <>
              <DropdownMenuSeparator />
              <DropdownMenuItem variant="destructive" disabled={!access} onClick={() => props.onUninstall(ext)}>
                <Trash2 aria-hidden />
                {t("Uninstall…")}
              </DropdownMenuItem>
            </>
          ) : null}
        </DropdownMenuContent>
      </DropdownMenu>
    </li>
  )
}

function Group(props: { title: string; count: number; defaultOpen: boolean; filtering: boolean; children: React.ReactNode }) {
  const { t } = useTranslation()
  const [open, setOpen] = useState(props.defaultOpen || props.filtering)
  // Typing a filter opens the group so its matches show.
  const [wasFiltering, setWasFiltering] = useState(props.filtering)
  if (wasFiltering !== props.filtering) {
    setWasFiltering(props.filtering)
    if (props.filtering && props.count > 0) setOpen(true)
  }
  return (
    <Collapsible open={open} onOpenChange={setOpen} className="rounded-xl border bg-card">
      <CollapsibleTrigger className="flex w-full items-center gap-2 rounded-xl px-3 py-2.5 text-left text-sm font-semibold outline-none hover:bg-muted/40 focus-visible:ring-2 focus-visible:ring-ring/60">
        <ChevronRight className={cn("size-4 text-muted-foreground transition-transform", open && "rotate-90")} aria-hidden />
        {t(props.title)}
        <span className="text-xs font-normal text-muted-foreground tabular-nums">{props.count}</span>
      </CollapsibleTrigger>
      <CollapsibleContent>
        <ul className="flex flex-col border-t p-1">{props.children}</ul>
      </CollapsibleContent>
    </Collapsible>
  )
}

function UninstallDialog({ ext, onClose }: { ext: ExtensionInfo | null; onClose: () => void }) {
  const { t } = useTranslation()
  const remove = useMutation({
    mutationFn: (name: string) => api.extensions.uninstall(name),
    onSuccess: (_data, name) => {
      toast.success(t("{{name}} was uninstalled", { name }), {
        description: t("Reload the UI to unload its scripts."),
        action: { label: t("Reload UI"), onClick: () => requestServerCommand("reload") },
      })
      onClose()
    },
    onError: (error) => toast.error(t("Could not uninstall"), { description: adminErrorMessage(error, t) }),
    onSettled: () => void queryClient.invalidateQueries({ queryKey: ["extensions"] }),
  })
  return (
    <AlertDialog open={ext !== null} onOpenChange={(open) => !open && onClose()}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>{t("Uninstall {{name}}?", { name: ext?.name ?? "" })}</AlertDialogTitle>
          <AlertDialogDescription>{t("Its folder is deleted, with any files it keeps there. This cannot be undone.")}</AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>{t("Cancel")}</AlertDialogCancel>
          <AlertDialogAction variant="destructive" disabled={remove.isPending} onClick={() => ext && remove.mutate(ext.name)}>
            {remove.isPending ? <Loader2 className="animate-spin" aria-hidden /> : null}
            {t("Uninstall")}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}

export function InstalledView() {
  const { t } = useTranslation()
  const list = useQuery(installedQuery)
  const staged = useStaged((s) => s.staged)
  const setStaged = useStaged((s) => s.set)
  const progress = useUpdateCheck((s) => s.progress)
  const [query, setQuery] = useState("")
  const [uninstalling, setUninstalling] = useState<ExtensionInfo | null>(null)
  const items = list.data?.items ?? []
  const access = list.data?.access ?? false
  const count = stagedCount(staged)
  const setUnsaved = useUnsaved((s) => s.set)
  useEffect(() => setUnsaved("extensions", count), [count, setUnsaved])

  // A refetched list can make staged changes moot (applied elsewhere, extension removed).
  useEffect(() => {
    if (list.data) {
      const pruned = pruneStaged(useStaged.getState().staged, list.data.items)
      if (stagedCount(pruned) !== stagedCount(useStaged.getState().staged)) setStaged(pruned)
    }
  }, [list.data, setStaged])

  const disableAll = useMutation({
    mutationFn: (value: "none" | "extra" | "all") => api.extensions.configure({ disable_all: value }),
    onSuccess: (data) => {
      queryClient.setQueryData(extensionKeys.installed, data)
      toast.success(t("Saved. Reload the UI to apply it."), { action: { label: t("Reload UI"), onClick: () => requestServerCommand("reload") } })
    },
    onError: (error) => toast.error(t("Could not save"), { description: adminErrorMessage(error, t) }),
  })

  const apply = useMutation({
    mutationFn: async (reload: boolean) => {
      const disabled = disabledList(items, staged)
      if (disabled) queryClient.setQueryData(extensionKeys.installed, await api.extensions.configure({ disabled }))
      if (staged.update.length > 0) queryClient.setQueryData(extensionKeys.installed, await api.extensions.update(staged.update))
      return reload
    },
    onSuccess: (reload) => {
      setStaged(NO_CHANGES)
      setUnsaved("extensions", 0)
      if (reload) runServerCommandAfterApply("reload")
      else toast.success(t("Saved. The changes load with the next Reload UI."))
    },
    onError: (error) => {
      toast.error(t("The changes were not applied"), { description: adminErrorMessage(error, t) })
      void queryClient.invalidateQueries({ queryKey: extensionKeys.installed })
    },
  })

  const check = async () => {
    const result = await checkForUpdates(items)
    if (result.busy) return
    if (result.total === 0) toast.info(t("No user extensions to check"), { description: t("Only enabled extensions installed from a Git repository are checked; built-in ones update with the WebUI.") })
    else if (result.failed.length > 0) toast.warning(t("{{count}} extensions could not be checked", { count: result.failed.length }), { description: result.failed.join(", ") })
    else if (result.checked > 0) toast.success(result.available > 0 ? t("{{count}} updates available", { count: result.available }) : t("Everything is up to date"))
  }

  if (list.isPending) {
    return (
      <div className="flex flex-col gap-2" aria-busy="true">
        {Array.from({ length: 6 }, (_, i) => (
          <Skeleton key={i} className="h-14 w-full" />
        ))}
      </div>
    )
  }
  if (list.isError) return <p role="alert" className="text-sm text-destructive">{adminErrorMessage(list.error, t)}</p>

  const terms = query.toLowerCase().split(/\s+/).filter(Boolean)
  const visible = terms.length ? items.filter((ext) => terms.every((term) => `${ext.name} ${ext.remote ?? ""}`.toLowerCase().includes(term))) : items
  const user = visible.filter((ext) => !ext.builtin)
  const builtin = visible.filter((ext) => ext.builtin)
  const checking = new Set(progress?.current ?? [])
  const updatable = items.filter((ext) => ext.can_update).map((ext) => ext.name)
  const row = (ext: ExtensionInfo) => <ExtensionRow key={ext.name} ext={ext} staged={staged} access={access} checking={checking.has(ext.name)} onUninstall={setUninstalling} />

  return (
    <div className="flex flex-col gap-4 pb-20">
      {list.data?.disabled_by ? (
        <p role="status" className="flex items-start gap-2 rounded-lg bg-warning/10 px-3 py-2 text-sm">
          <CircleAlert className="mt-0.5 size-4 shrink-0 text-warning" aria-hidden />
          {list.data.disabled_by.startsWith("--")
            ? t("Extensions are not loaded because the server was started with {{flag}}.", { flag: list.data.disabled_by })
            : list.data.disabled_by === "all"
              ? t("“Disable all extensions” is set to All: no extension is loaded.")
              : t("“Disable all extensions” is set to Extra: only built-in extensions are loaded.")}
        </p>
      ) : null}

      <div className="flex flex-wrap items-center gap-2">
        <InputGroup className="max-w-xs min-w-44 flex-1">
          <InputGroupAddon>
            <Search aria-hidden />
          </InputGroupAddon>
          <InputGroupInput type="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder={t("Filter extensions")} aria-label={t("Filter extensions")} />
        </InputGroup>
        <div className="flex items-center gap-2">
          <span id="disable-all-label" className="text-xs text-muted-foreground">
            {t("Disable all extensions")}
          </span>
          <Segmented
            size="sm"
            aria-labelledby="disable-all-label"
            options={DISABLE_ALL.map((option) => ({ value: option.value, label: t(option.label) }))}
            value={(disableAll.isPending ? disableAll.variables : list.data?.disable_all) as "none" | "extra" | "all"}
            onValueChange={(value) => disableAll.mutate(value)}
            disabled={!access || disableAll.isPending}
          />
        </div>
        <div className="ml-auto flex items-center gap-2">
          {updatable.length > 0 ? (
            <Button variant="ghost" size="sm" disabled={!access} onClick={() => setStaged({ ...staged, update: staged.update.length === updatable.length ? [] : updatable })}>
              {staged.update.length === updatable.length ? t("Unmark updates") : t("Mark all updates")}
            </Button>
          ) : null}
          {progress ? (
            <Button variant="outline" size="sm" onClick={stopUpdateCheck}>
              <X aria-hidden />
              {t("Stop")}
            </Button>
          ) : (
            <Button variant="outline" size="sm" disabled={!access} onClick={() => void check()}>
              <RefreshCw aria-hidden />
              {t("Check for updates")}
            </Button>
          )}
        </div>
      </div>

      {progress ? (
        <Progress value={progress.total ? (progress.done / progress.total) * 100 : 0} aria-label={t("Checking for updates")} className="gap-1.5">
          <span className="w-full text-xs text-muted-foreground tabular-nums">
            {t("Checked {{done}} of {{total}}", { done: progress.done, total: progress.total })}
            {progress.current.length > 0 ? ` · ${progress.current.join(", ")}` : ""}
          </span>
        </Progress>
      ) : null}

      <Group title="Installed" count={user.length} defaultOpen filtering={terms.length > 0}>
        {user.length > 0 ? user.map(row) : <li className="px-3 py-4 text-sm text-muted-foreground">{terms.length ? t("No extension matches.") : t("No extensions installed yet. Find some under Available.")}</li>}
      </Group>
      <Group title="Built-in" count={builtin.length} defaultOpen={false} filtering={terms.length > 0}>
        {builtin.map(row)}
      </Group>

      {count > 0 ? (
        <div role="region" aria-label={t("Pending extension changes")} className="sticky bottom-3 z-10 flex justify-center">
          <div data-toast-obstacle className="flex w-full max-w-2xl flex-wrap items-center gap-2 rounded-xl border bg-popover/95 px-3 py-2 shadow-lg backdrop-blur animate-in fade-in slide-in-from-bottom-2">
            <p className="min-w-0 flex-1 text-sm" aria-live="polite">
              {t("{{toggles}} switched, {{updates}} to update", { toggles: Object.keys(staged.enabled).length, updates: staged.update.length })}
            </p>
            <Button variant="ghost" size="sm" disabled={apply.isPending} onClick={() => setStaged(NO_CHANGES)}>
              {t("Discard")}
            </Button>
            <Button variant="outline" size="sm" disabled={apply.isPending} onClick={() => apply.mutate(false)}>
              {t("Save")}
            </Button>
            <Button size="sm" disabled={apply.isPending} onClick={() => apply.mutate(true)}>
              {apply.isPending ? <Loader2 className="animate-spin" aria-hidden /> : null}
              {t("Apply and reload UI")}
            </Button>
          </div>
        </div>
      ) : null}
      <UninstallDialog ext={uninstalling} onClose={() => setUninstalling(null)} />
    </div>
  )
}
