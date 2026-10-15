/**
 * Backup and restore of the extension commits (config states). A backup is
 * shown next to the current state, differences highlighted; restoring resets
 * every extension to its recorded commit during the next Reload UI. The
 * WebUI commit is shown for reference only: its checkout is never reset
 * from here (a reset would drop local work and old backups predate Remi).
 */
import { useMutation, useQuery } from "@tanstack/react-query"
import { Archive, Info, Loader2, RotateCcw, Save, Trash2 } from "lucide-react"
import { useState } from "react"
import { useTranslation } from "react-i18next"
import { toast } from "sonner"
import { api } from "@/api/endpoints"
import { queryClient } from "@/api/queries"
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
import { Empty, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from "@/components/ui/empty"
import { Input } from "@/components/ui/input"
import { Skeleton } from "@/components/ui/skeleton"
import { runServerCommandAfterApply } from "@/features/server-control/server-control"
import { cn } from "@/lib/utils"
import { adminErrorMessage, type ConfigStateDetail, formatDate } from "./extensions-model"
import { extensionKeys, stateQuery, statesQuery } from "./queries"

type Repo = { commit_hash?: string | null; commit_url?: string | null; branch?: string | null; remote?: string | null; enabled?: boolean } | null | undefined

function Commit({ repo }: { repo: Repo }) {
  if (!repo?.commit_hash) return <span className="text-muted-foreground">—</span>
  const short = repo.commit_hash.slice(0, 8)
  return repo.commit_url ? (
    <a href={repo.commit_url} target="_blank" rel="noreferrer noopener" className="font-mono hover:underline">
      {short}
    </a>
  ) : (
    <span className="font-mono">{short}</span>
  )
}

function differs(saved: Repo, current: Repo): boolean {
  if (!saved || !current) return true
  return saved.commit_hash !== current.commit_hash || saved.branch !== current.branch || saved.enabled !== current.enabled
}

function StateDetail({ detail }: { detail: ConfigStateDetail }) {
  const { t } = useTranslation()
  const changed = detail.items.filter((row) => differs(row.saved, row.current)).length
  const webuiDiffers = detail.webui.saved.commit_hash !== detail.webui.current.commit_hash
  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap gap-x-6 gap-y-1 text-sm">
        <span>
          {t("WebUI")}: <Commit repo={detail.webui.saved} />
          {webuiDiffers ? (
            <>
              {" → "}
              <Commit repo={detail.webui.current} /> <Badge variant="outline">{t("now")}</Badge>
            </>
          ) : null}
        </span>
        <span className="text-muted-foreground">{t("{{count}} extensions differ from now", { count: changed })}</span>
      </div>
      <div className="overflow-x-auto rounded-lg border">
        <table className="w-full min-w-[32rem] text-left text-xs">
          <caption className="sr-only">{t("Extensions in the backup and now")}</caption>
          <thead className="bg-muted/50 text-muted-foreground">
            <tr>
              <th scope="col" className="px-3 py-2 font-medium">
                {t("Extension")}
              </th>
              <th scope="col" className="px-3 py-2 font-medium">
                {t("In the backup")}
              </th>
              <th scope="col" className="px-3 py-2 font-medium">
                {t("Now")}
              </th>
            </tr>
          </thead>
          <tbody>
            {detail.items.map((row) => {
              const diff = differs(row.saved, row.current)
              return (
                <tr key={row.name} className={cn("border-t", diff && "bg-primary/[0.05]")}>
                  <th scope="row" className="px-3 py-1.5 font-medium">
                    {row.name}
                    {row.builtin ? <span className="ml-1.5 font-normal text-muted-foreground">{t("built-in")}</span> : null}
                  </th>
                  {[row.saved, row.current].map((repo, index) => (
                    <td key={index} className="px-3 py-1.5">
                      {repo ? (
                        <span className="flex flex-wrap items-center gap-1.5">
                          <Commit repo={repo} />
                          {repo.branch ? <span className="text-muted-foreground">{repo.branch}</span> : null}
                          {repo.enabled === false ? <Badge variant="outline" className="h-4 px-1 text-[0.65rem]">{t("disabled")}</Badge> : null}
                        </span>
                      ) : (
                        <span className="text-muted-foreground">{index === 0 ? t("not in the backup") : t("not installed")}</span>
                      )}
                    </td>
                  ))}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export function BackupView({ access }: { access: boolean }) {
  const { t, i18n } = useTranslation()
  const states = useQuery(statesQuery)
  const [selected, setSelected] = useState<string | null>(null)
  const [name, setName] = useState("")
  const [confirm, setConfirm] = useState<"restore" | "delete" | null>(null)
  const current = selected ?? states.data?.[0]?.id ?? null
  const detail = useQuery({ ...stateQuery(current ?? ""), enabled: current !== null })

  const save = useMutation({
    mutationFn: () => api.extensions.states.save(name.trim()),
    onSuccess: (info) => {
      setName("")
      setSelected(info.id)
      toast.success(t("Backup saved"))
      void queryClient.invalidateQueries({ queryKey: extensionKeys.states })
    },
    onError: (error) => toast.error(t("Could not save the backup"), { description: adminErrorMessage(error, t) }),
  })
  const remove = useMutation({
    mutationFn: (id: string) => api.extensions.states.remove(id),
    onSuccess: () => {
      setConfirm(null)
      setSelected(null)
      toast.success(t("Backup deleted"))
      void queryClient.invalidateQueries({ queryKey: extensionKeys.states })
    },
    onError: (error) => toast.error(t("Could not delete the backup"), { description: adminErrorMessage(error, t) }),
  })
  const restore = useMutation({
    mutationFn: (id: string) => api.extensions.states.restore(id),
    onSuccess: () => {
      setConfirm(null)
      runServerCommandAfterApply("reload")
    },
    onError: (error) => toast.error(t("Could not restore the backup"), { description: adminErrorMessage(error, t) }),
  })

  return (
    <div className="grid gap-4 lg:grid-cols-[18rem_1fr]">
      <div className="flex flex-col gap-3">
        <form
          className="flex gap-2"
          onSubmit={(event) => {
            event.preventDefault()
            save.mutate()
          }}
        >
          <Input value={name} onChange={(event) => setName(event.target.value)} placeholder={t("Backup name")} aria-label={t("Backup name")} disabled={!access} maxLength={100} />
          <Button type="submit" disabled={!access || save.isPending} className="shrink-0">
            {save.isPending ? <Loader2 className="animate-spin" aria-hidden /> : <Save aria-hidden />}
            {t("Back up")}
          </Button>
        </form>
        {states.isPending ? (
          <Skeleton className="h-40 w-full" />
        ) : states.isError ? (
          <p role="alert" className="text-sm text-destructive">{adminErrorMessage(states.error, t)}</p>
        ) : states.data.length === 0 ? (
          <p className="text-sm text-muted-foreground">{t("No backups yet. One is made before every update.")}</p>
        ) : (
          <ul role="listbox" aria-label={t("Backups")} className="flex max-h-[60dvh] flex-col gap-0.5 overflow-y-auto rounded-xl border bg-card p-1">
            {states.data.map((state) => (
              <li
                key={state.id}
                role="option"
                aria-selected={state.id === current}
                tabIndex={0}
                onClick={() => setSelected(state.id)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault()
                    setSelected(state.id)
                  }
                }}
                className={cn(
                  "cursor-default rounded-lg px-3 py-2 outline-none hover:bg-muted/60 focus-visible:ring-2 focus-visible:ring-ring/60",
                  state.id === current && "bg-accent",
                )}
              >
                <p className="truncate text-sm font-medium">{state.name}</p>
                <p className="text-xs text-muted-foreground tabular-nums">
                  {state.created_at ? new Date(state.created_at).toLocaleString(i18n.language) : ""} · {t("{{count}} extensions", { count: state.extensions })}
                </p>
              </li>
            ))}
          </ul>
        )}
      </div>
      <div className="min-w-0">
        {current === null ? (
          <Empty className="border">
            <EmptyHeader>
              <EmptyMedia variant="icon">
                <Archive aria-hidden />
              </EmptyMedia>
              <EmptyTitle>{t("Backups")}</EmptyTitle>
              <EmptyDescription>{t("A backup records the commit of every extension, so a bad update can be undone.")}</EmptyDescription>
            </EmptyHeader>
          </Empty>
        ) : detail.isPending ? (
          <Skeleton className="h-64 w-full" />
        ) : detail.isError ? (
          <p role="alert" className="text-sm text-destructive">{adminErrorMessage(detail.error, t)}</p>
        ) : (
          <div className="flex flex-col gap-4">
            <div className="flex flex-wrap items-center gap-2">
              <div className="min-w-0 flex-1">
                <h2 className="truncate text-base font-semibold">{detail.data.name}</h2>
                <p className="text-xs text-muted-foreground">{formatDate(detail.data.created_at, i18n.language)}</p>
              </div>
              <Button disabled={!access} onClick={() => setConfirm("restore")}>
                <RotateCcw aria-hidden />
                {t("Restore extensions")}
              </Button>
              <Button variant="ghost" size="icon" aria-label={t("Delete this backup")} disabled={!access || remove.isPending} onClick={() => setConfirm("delete")}>
                <Trash2 />
              </Button>
            </div>
            <StateDetail detail={detail.data} />
            <p className="flex items-start gap-2 text-xs text-muted-foreground">
              <Info className="mt-0.5 size-3.5 shrink-0" aria-hidden />
              {t("The WebUI commit is shown for reference: restoring never resets the WebUI itself. Update it with git from its folder.")}
            </p>
          </div>
        )}
      </div>
      <AlertDialog open={confirm === "restore"} onOpenChange={(open) => !open && setConfirm(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{t("Restore the extensions of this backup?")}</AlertDialogTitle>
            <AlertDialogDescription>
              {t("Every extension is reset to its recorded commit, and those not in the backup are disabled. The UI reloads to apply it.")}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>{t("Cancel")}</AlertDialogCancel>
            <AlertDialogAction disabled={restore.isPending || current === null} onClick={() => current && restore.mutate(current)}>
              {restore.isPending ? <Loader2 className="animate-spin" aria-hidden /> : null}
              {t("Restore extensions")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
      <AlertDialog open={confirm === "delete"} onOpenChange={(open) => !open && setConfirm(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{t("Delete this backup?")}</AlertDialogTitle>
            <AlertDialogDescription>
              {t("“{{name}}” is deleted. This cannot be undone.", { name: detail.data?.name ?? "" })}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>{t("Cancel")}</AlertDialogCancel>
            <AlertDialogAction variant="destructive" disabled={remove.isPending || current === null} onClick={() => current && remove.mutate(current)}>
              {remove.isPending ? <Loader2 className="animate-spin" aria-hidden /> : null}
              {t("Delete")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  )
}
