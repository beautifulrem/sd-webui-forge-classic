/**
 * Settings → Server and models: reload / restart, model memory (unload, load
 * or load again, the weights in memory), checkpoint hashes, browser
 * notifications, the settings file and the localization template.
 */
import { useMutation, useQuery } from "@tanstack/react-query"
import { Bell, Box, Download, FileUp, Fingerprint, HardDriveDownload, Languages, ListTree, Loader2, Power, RefreshCw, Upload } from "lucide-react"
import { type ReactNode, useId, useState } from "react"
import { useTranslation } from "react-i18next"
import { toast } from "sonner"
import { api } from "@/api/endpoints"
import { useProgress } from "@/api/progress"
import { queryClient } from "@/api/queries"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { requestServerCommand } from "@/features/server-control/server-control"
import { cn } from "@/lib/utils"
import { adminErrorMessage } from "../../extensions/extensions-model"
import { downloadLocalizationTemplate } from "../localization-template"
import { hashProgressQuery, loadedModelsQuery, serverStatusQuery } from "../queries"
import { useSettingsEditor } from "../settings-store"

function Panel(props: { icon: typeof Box; title: string; description: string; children: ReactNode; aside?: ReactNode; footer?: ReactNode }) {
  const { t } = useTranslation()
  const headingId = useId()
  const Icon = props.icon
  return (
    <section className="flex flex-col gap-3 rounded-xl border bg-card p-4" aria-labelledby={headingId}>
      <div className="flex items-start gap-3">
        <span className="grid size-9 shrink-0 place-items-center rounded-lg bg-muted text-muted-foreground">
          <Icon className="size-4.5" aria-hidden />
        </span>
        <div className="min-w-0 flex-1">
          <h2 id={headingId} className="text-sm font-semibold">
            {t(props.title)}
          </h2>
          <p className="text-[0.8rem] text-muted-foreground">{t(props.description)}</p>
        </div>
        {props.aside}
      </div>
      <div className="flex flex-wrap items-center gap-2">{props.children}</div>
      {props.footer}
    </section>
  )
}

function useNotificationPermission() {
  const supported = typeof Notification !== "undefined"
  const [permission, setPermission] = useState<NotificationPermission | "unsupported">(supported ? Notification.permission : "unsupported")
  const request = async () => {
    if (!supported) return
    setPermission(await Notification.requestPermission())
  }
  return { permission, request }
}

/** The weights in memory (the former "List loaded models"), fetched when asked for. */
function LoadedWeights() {
  const { t } = useTranslation()
  const [shown, setShown] = useState(false)
  const loaded = useQuery({ ...loadedModelsQuery, enabled: shown })
  if (!shown) {
    return (
      <Button variant="ghost" size="sm" className="self-start" onClick={() => setShown(true)}>
        <ListTree aria-hidden />
        {t("Show the weights in memory")}
      </Button>
    )
  }
  return (
    <div className="flex flex-col gap-1.5 rounded-lg border bg-muted/30 p-2" aria-live="polite" aria-busy={loaded.isFetching}>
      <div className="flex items-center gap-2 px-1 text-xs font-medium text-muted-foreground">
        <span className="flex-1">{t("Weights in memory")}</span>
        <Button variant="ghost" size="icon-xs" aria-label={t("Refresh")} disabled={loaded.isFetching} onClick={() => void loaded.refetch()}>
          <RefreshCw className={cn(loaded.isFetching && "animate-spin")} />
        </Button>
      </div>
      {loaded.isError ? (
        <p role="alert" className="px-1 text-xs text-destructive">
          {adminErrorMessage(loaded.error, t)}
        </p>
      ) : !loaded.data ? (
        <Loader2 className="mx-auto size-4 animate-spin text-muted-foreground" aria-label={t("Loading…")} />
      ) : loaded.data.items.length === 0 ? (
        <p className="px-1 text-xs text-muted-foreground">{t("Nothing is loaded.")}</p>
      ) : (
        <table className="w-full text-left text-xs">
          <caption className="sr-only">{t("Weights in memory")}</caption>
          <thead className="text-muted-foreground">
            <tr>
              <th scope="col" className="px-1 py-1 font-medium">
                {t("Model")}
              </th>
              <th scope="col" className="px-1 py-1 text-right font-medium">
                {t("Memory")}
              </th>
              <th scope="col" className="px-1 py-1 text-right font-medium">
                {t("Device")}
              </th>
            </tr>
          </thead>
          <tbody className="tabular-nums">
            {loaded.data.items.map((item, index) => (
              <tr key={`${item.name}-${index}`} className="border-t">
                <td className="px-1 py-1 font-mono">{item.name}</td>
                <td className="px-1 py-1 text-right">{item.memory_mb != null ? t("{{size}} MiB", { size: item.memory_mb.toLocaleString() }) : "—"}</td>
                <td className="px-1 py-1 text-right font-mono">{item.device ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}

const THREADS = [1, 2, 4, 8, 16]

/** Hash every checkpoint (the former "Calculate hash for all checkpoints"), with progress. */
function CheckpointHashes() {
  const { t } = useTranslation()
  const [threads, setThreads] = useState(1)
  const progress = useQuery(hashProgressQuery)
  const start = useMutation({
    mutationFn: () => api.system.hashCheckpoints(threads),
    onSuccess: (data) => queryClient.setQueryData(hashProgressQuery.queryKey, data),
    onError: (error) => toast.error(t("Could not start hashing"), { description: adminErrorMessage(error, t) }),
  })
  const state = progress.data
  const running = state?.running ?? false
  return (
    <Panel
      icon={Fingerprint}
      title="Checkpoint hashes"
      description="Calculate the hash of every checkpoint that has none yet (used in infotexts and to match models). Large files take a while."
      footer={
        state && (running || state.finished_at) ? (
          <div className="flex flex-col gap-1" aria-live="polite">
            <Progress value={state.total ? (state.done / state.total) * 100 : 100} aria-label={t("Hashing checkpoints")} />
            <p className="text-xs text-muted-foreground tabular-nums">
              {running
                ? t("Hashed {{done}} of {{total}}", { done: state.done, total: state.total })
                : t("Finished: {{done}} checkpoints hashed", { done: state.done - state.failed.length })}
              {state.failed.length > 0 ? ` · ${t("{{count}} failed", { count: state.failed.length })}: ${state.failed.join(", ")}` : ""}
            </p>
          </div>
        ) : null
      }
    >
      <Select<number> items={THREADS.map((value) => ({ value, label: t("{{count}} at a time", { count: value }) }))} value={threads} onValueChange={(value) => value && setThreads(value)} disabled={running}>
        <SelectTrigger size="sm" aria-label={t("Checkpoints hashed at once")} className="min-w-32">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {THREADS.map((value) => (
            <SelectItem key={value} value={value}>
              {t("{{count}} at a time", { count: value })}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <Button variant="outline" disabled={running || start.isPending} onClick={() => start.mutate()}>
        {running || start.isPending ? <Loader2 className="animate-spin" aria-hidden /> : <Fingerprint aria-hidden />}
        {t("Hash all checkpoints")}
      </Button>
    </Panel>
  )
}

export function ServerView({ onImport, onExport }: { onImport: () => void; onExport: (all: boolean) => void }) {
  const { t } = useTranslation()
  const status = useQuery(serverStatusQuery)
  const modelLoading = useProgress((s) => s.modelLoading)
  const notifications = useNotificationPermission()
  const model = useMutation({
    mutationFn: (action: "unload" | "load" | "reload") => (action === "unload" ? api.system.unloadModel() : api.system.reloadModel(action === "reload")),
    onSuccess: (_data, action) => toast.success(action === "unload" ? t("The model was unloaded") : t("The model is loaded")),
    onError: (error) => toast.error(t("The model action failed"), { description: adminErrorMessage(error, t) }),
    onSettled: () => {
      void status.refetch()
      void queryClient.invalidateQueries({ queryKey: loadedModelsQuery.queryKey })
    },
  })
  const template = useMutation({
    mutationFn: () => downloadLocalizationTemplate(useSettingsEditor.getState().schema),
    onSuccess: (count) => toast.success(t("Strings downloaded: {{count}}", { count })),
    onError: (error) => toast.error(t("Could not build the template"), { description: adminErrorMessage(error, t) }),
  })
  const restartable = status.data?.restartable ?? false
  const loaded = status.data?.model_loaded
  const loadAction = loaded ? "reload" : "load"

  return (
    <div className="mx-auto flex w-full max-w-3xl flex-col gap-4 px-4 py-6 sm:px-6">
      <header>
        <h1 className="text-lg font-semibold tracking-tight">{t("Server and models")}</h1>
        <p className="text-sm text-muted-foreground">{status.data ? t("Version {{version}}", { version: status.data.version }) : " "}</p>
      </header>

      <Panel icon={RefreshCw} title="Reload UI" description="Reloads scripts and extensions and rebuilds the app in this process. Use it after enabling, installing or updating extensions, or after editing a script.">
        <Button onClick={() => requestServerCommand("reload")}>
          <RefreshCw aria-hidden />
          {t("Reload UI")}
        </Button>
      </Panel>

      <Panel
        icon={Power}
        title="Restart server"
        description="Stops the WebUI process and starts it again (needed for settings marked “Requires restart”)."
        aside={!status.isPending && !restartable ? <Badge variant="outline">{t("Not available")}</Badge> : null}
      >
        <Button variant="outline" onClick={() => requestServerCommand("restart")} disabled={!restartable}>
          <Power aria-hidden />
          {t("Restart server")}
        </Button>
        {!status.isPending && !restartable ? (
          <p className="text-xs text-muted-foreground">{t("Start the WebUI with webui.sh / webui.bat to restart it from here.")}</p>
        ) : null}
      </Panel>

      <Panel
        icon={Box}
        title="Model memory"
        description="Unload the model to free RAM and VRAM; load the selected model now instead of at the next generation, or load it again from disk."
        aside={
          <Badge variant={loaded ? "secondary" : "outline"} className="max-w-56 truncate" title={loaded ?? undefined}>
            {modelLoading ? t("Loading…") : loaded ? loaded.replace(/\s*\[[0-9a-f]{6,}\]$/i, "") : t("No model loaded")}
          </Badge>
        }
        footer={<LoadedWeights />}
      >
        <Button variant="outline" disabled={model.isPending || !loaded} onClick={() => model.mutate("unload")}>
          {model.isPending && model.variables === "unload" ? <Loader2 className="animate-spin" aria-hidden /> : <HardDriveDownload aria-hidden />}
          {t("Unload model")}
        </Button>
        <Button variant="outline" disabled={model.isPending || modelLoading !== null} onClick={() => model.mutate(loadAction)}>
          {model.isPending && model.variables !== "unload" ? <Loader2 className="animate-spin" aria-hidden /> : <Upload aria-hidden />}
          {loaded ? t("Reload model from disk") : t("Load model")}
        </Button>
      </Panel>

      <CheckpointHashes />

      <Panel
        icon={Bell}
        title="Browser notifications"
        description="Notify when a job finishes while this tab is in the background."
        aside={
          <Badge variant={notifications.permission === "granted" ? "secondary" : "outline"}>
            {notifications.permission === "granted"
              ? t("Allowed")
              : notifications.permission === "denied"
                ? t("Blocked")
                : notifications.permission === "unsupported"
                  ? t("Not supported")
                  : t("Not asked yet")}
          </Badge>
        }
      >
        <Button variant="outline" disabled={notifications.permission !== "default"} onClick={() => void notifications.request()}>
          <Bell aria-hidden />
          {t("Allow notifications")}
        </Button>
        {notifications.permission === "denied" ? <p className="text-xs text-muted-foreground">{t("Allow them in the browser's site settings.")}</p> : null}
      </Panel>

      <Panel icon={FileUp} title="Settings file" description="Back up your settings as a JSON file, or bring settings over from another installation.">
        <Button variant="outline" onClick={() => onExport(false)}>
          <Download aria-hidden />
          {t("Export changed settings")}
        </Button>
        <Button variant="ghost" onClick={() => onExport(true)}>
          {t("Export all settings")}
        </Button>
        <Button variant="outline" onClick={onImport} className="sm:ml-auto">
          <FileUp aria-hidden />
          {t("Import settings…")}
        </Button>
      </Panel>

      <Panel
        icon={Languages}
        title="Localization template"
        description="Every English string of the UI as JSON, with the current language's translations filled in: a start for a new localizations/<language>.json."
      >
        <Button variant="outline" disabled={template.isPending} onClick={() => template.mutate()}>
          {template.isPending ? <Loader2 className="animate-spin" aria-hidden /> : <Download aria-hidden />}
          {t("Download localization template")}
        </Button>
      </Panel>
    </div>
  )
}
