/**
 * Queue: the running job (live preview, interrupt / skip), the queued jobs
 * in run order (drag to reorder, priorities, edit in their form, duplicate,
 * cancel), pause / resume, and the history (search, status, stars, form;
 * queue again, open in its form, view outputs). The menu imports the Agent
 * Scheduler and Prompt Queue data, stops generate-forever jobs and clears.
 */
import { useInfiniteQuery, useQuery } from "@tanstack/react-query"
import { Ban, Eraser, Import, Infinity as InfinityIcon, ListOrdered, MoreHorizontal, Pause, Play, Search, Star } from "lucide-react"
import { lazy, Suspense, useEffect, useState } from "react"
import { useTranslation } from "react-i18next"
import { api } from "@/api/endpoints"
import { queueItemsQuery } from "@/api/queries"
import type { FormTab, Job, QueueStatus } from "@/api/types"
import { HeaderSlot } from "@/app/header-slot"
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
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuSeparator, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Skeleton } from "@/components/ui/skeleton"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Toggle } from "@/components/ui/toggle"
import { useMediaQuery } from "@/lib/use-media-query"
import { cn } from "@/lib/utils"
import type { ResultItem } from "@/features/gallery/session-store"
import { errorText } from "@/features/errors/error-text"
import { HistoryList } from "@/features/queue/HistoryList"
import { ImportDialog } from "@/features/queue/ImportDialog"
import { PendingList } from "@/features/queue/PendingList"
import { act, HISTORY_STATUSES, type HistoryFilter, historyQuery } from "@/features/queue/queries"
import { RunningCard } from "@/features/queue/RunningCard"
import { useQueueStatus } from "@/features/queue/status"

const Lightbox = lazy(() => import("@/features/lightbox/Lightbox").then((m) => ({ default: m.Lightbox })))

const ALL_TABS = "__all__"

function useDebounced<T>(value: T, ms: number): T {
  const [debounced, setDebounced] = useState(value)
  useEffect(() => {
    const timer = setTimeout(() => setDebounced(value), ms)
    return () => clearTimeout(timer)
  }, [value, ms])
  return debounced
}

function Pending({ status }: { status: QueueStatus | undefined }) {
  const { t } = useTranslation()
  const items = useQuery(queueItemsQuery)
  const jobs = items.data ?? []
  return (
    <div className="flex flex-col gap-3">
      <RunningCard paused={status?.paused ?? false} queued={jobs.length} />
      <section aria-labelledby="queue-pending-title" className="flex flex-col gap-2">
        <h2 id="queue-pending-title" className="flex items-center gap-2 text-sm font-semibold">
          {t("Up next")}
          <span className="rounded-full bg-muted px-1.5 text-xs font-medium text-muted-foreground tabular-nums">{jobs.length}</span>
        </h2>
        {items.isPending ? (
          <Skeleton className="h-24 w-full" />
        ) : jobs.length === 0 ? (
          <p className="rounded-lg border border-dashed px-4 py-6 text-center text-sm text-muted-foreground">
            {t("No jobs are waiting. Use “Add to queue” next to Generate to line jobs up.")}
          </p>
        ) : (
          <PendingList jobs={jobs} />
        )}
      </section>
    </div>
  )
}

function History({ onView }: { onView: (job: Job, index: number) => void }) {
  const { t } = useTranslation()
  const [text, setText] = useState("")
  const q = useDebounced(text.trim(), 300)
  const [filter, setFilter] = useState<HistoryFilter>("all")
  const [starred, setStarred] = useState(false)
  const [tab, setTab] = useState<FormTab | typeof ALL_TABS>(ALL_TABS)
  const query = { status: HISTORY_STATUSES[filter], ...(q ? { q } : {}), ...(starred ? { starred: true } : {}), ...(tab !== ALL_TABS ? { tab } : {}) }
  const history = useInfiniteQuery(historyQuery(query))
  const jobs = history.data?.pages.flatMap((page) => page.items) ?? []
  const total = history.data?.pages[0]?.total ?? null
  const tabs = [
    { value: ALL_TABS, label: t("All forms") },
    { value: "txt2img", label: t("txt2img") },
    { value: "img2img", label: t("img2img") },
    { value: "extras", label: t("Extras") },
  ]
  return (
    <section aria-labelledby="queue-history-title" className="flex min-h-0 flex-1 flex-col gap-2">
      <div className="flex flex-wrap items-center gap-1.5">
        <h2 id="queue-history-title" className="mr-1 text-sm font-semibold">
          {t("History")}
          {total !== null ? " " : null}
          {total !== null ? <span className="ml-1 text-xs font-normal text-muted-foreground tabular-nums">{total}</span> : null}
        </h2>
        <div className="relative min-w-36 flex-1">
          <Search className="pointer-events-none absolute top-1/2 left-2 size-3.5 -translate-y-1/2 text-muted-foreground" aria-hidden />
          <Input type="search" value={text} onChange={(event) => setText(event.target.value)} placeholder={t("Search prompts and parameters…")} aria-label={t("Search the history")} className="h-7 pl-7 text-[0.8rem]" />
        </div>
        <Toggle size="sm" variant="outline" pressed={starred} onPressedChange={setStarred} aria-label={t("Starred only")} title={t("Starred only")}>
          <Star className={cn(starred && "fill-amber-400 text-amber-400")} aria-hidden />
        </Toggle>
      </div>
      <div className="flex flex-wrap items-center gap-1.5">
        <Segmented
          size="sm"
          aria-label={t("Status")}
          value={filter}
          onValueChange={setFilter}
          options={[
            { value: "all", label: t("All") },
            { value: "completed", label: t("Completed") },
            { value: "failed", label: t("Failed") },
            { value: "stopped", label: t("Stopped") },
          ]}
        />
        <Select items={tabs} value={tab} onValueChange={(value) => value !== null && setTab(value as FormTab | typeof ALL_TABS)}>
          <SelectTrigger size="sm" aria-label={t("Form")}>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {tabs.map((option) => (
              <SelectItem key={option.value} value={option.value}>
                {option.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      {history.isPending ? (
        <div className="flex flex-col gap-1.5">
          {Array.from({ length: 4 }, (_, index) => (
            <Skeleton key={index} className="h-20 w-full" />
          ))}
        </div>
      ) : history.isError ? (
        <p role="alert" className="text-sm text-destructive">
          {errorText(history.error)}
        </p>
      ) : jobs.length === 0 ? (
        <p className="rounded-lg border border-dashed px-4 py-8 text-center text-sm text-muted-foreground">
          {q || starred || filter !== "all" || tab !== ALL_TABS ? t("No jobs match.") : t("Finished jobs appear here.")}
        </p>
      ) : (
        <HistoryList jobs={jobs} total={total} hasMore={history.hasNextPage} loadingMore={history.isFetchingNextPage} onLoadMore={() => void history.fetchNextPage()} onView={onView} />
      )}
    </section>
  )
}

type Confirm = "clear-history" | "cancel-all" | null

export default function QueuePage() {
  const { t } = useTranslation()
  const status = useQueueStatus()
  const wide = useMediaQuery("(min-width: 64rem)")
  const [importing, setImporting] = useState(false)
  const [confirm, setConfirm] = useState<Confirm>(null)
  const [keepStarred, setKeepStarred] = useState(true)
  const [viewing, setViewing] = useState<{ items: ResultItem[]; current: string; tab: FormTab } | null>(null)
  const paused = status?.paused ?? false

  const view = (job: Job, index: number) => {
    const items = job.outputs
      .filter((output) => (output.kind ?? "image") === "image")
      .map<ResultItem>((output, position) => ({
        key: `${job.id}:${position}`,
        fileId: output.id,
        jobId: job.id,
        tab: job.tab,
        index: position,
        infotext: output.infotext ?? job.infotext ?? null,
        width: output.width ?? undefined,
        height: output.height ?? undefined,
        createdAt: Date.parse(job.created_at) || 0,
      }))
    if (items[index]) setViewing({ items, current: items[index].key, tab: job.tab })
  }

  const header = (
    <HeaderSlot>
      <Button
        variant={paused ? "default" : "outline"}
        size="sm"
        aria-pressed={paused}
        onClick={() => void act(() => (paused ? api.queue.resume() : api.queue.pause()), paused ? t("Could not resume the queue") : t("Could not pause the queue"))}
      >
        {paused ? <Play aria-hidden /> : <Pause aria-hidden />}
        {paused ? t("Resume") : t("Pause")}
      </Button>
      <DropdownMenu>
        <DropdownMenuTrigger render={<Button variant="ghost" size="icon-sm" aria-label={t("Queue settings")} />}>
          <MoreHorizontal />
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="min-w-60">
          <DropdownMenuItem onClick={() => setImporting(true)}>
            <Import />
            {t("Import from Agent Scheduler or Prompt Queue…")}
          </DropdownMenuItem>
          <DropdownMenuSeparator />
          <DropdownMenuItem onClick={() => void act(() => api.queue.stopForever(), t("Could not stop generate forever"), t("Generate forever stops after the current run"))}>
            <InfinityIcon />
            {t("Stop every generate forever")}
          </DropdownMenuItem>
          <DropdownMenuItem onClick={() => setConfirm("cancel-all")} disabled={(status?.size ?? 0) === 0 && !status?.running}>
            <Ban />
            {t("Stop and cancel every queued job…")}
          </DropdownMenuItem>
          <DropdownMenuItem variant="destructive" onClick={() => setConfirm("clear-history")}>
            <Eraser />
            {t("Clear the history…")}
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </HeaderSlot>
  )

  const pending = <Pending status={status} />
  const history = <History onView={view} />

  return (
    <div className="flex h-full flex-col">
      {header}
      {wide ? (
        <div className="mx-auto grid min-h-0 w-full max-w-7xl flex-1 grid-cols-[minmax(0,5fr)_minmax(0,6fr)] gap-6 p-4">
          <div className="min-h-0 overflow-y-auto pr-1">{pending}</div>
          <div className="flex min-h-0 flex-col">{history}</div>
        </div>
      ) : (
        <Tabs defaultValue="queue" className="flex min-h-0 flex-1 flex-col gap-0 p-3">
          <TabsList variant="line" className="w-full justify-start border-b pb-px">
            <TabsTrigger value="queue" className="flex-none px-2.5">
              <ListOrdered aria-hidden />
              {t("Queue")}
              {status?.size ? <span className="text-xs text-muted-foreground tabular-nums">{status.size}</span> : null}
            </TabsTrigger>
            <TabsTrigger value="history" className="flex-none px-2.5">
              {t("History")}
            </TabsTrigger>
          </TabsList>
          <TabsContent value="queue" className="min-h-0 flex-1 overflow-y-auto pt-3">
            {pending}
          </TabsContent>
          <TabsContent value="history" className="flex min-h-0 flex-1 flex-col pt-3">
            {history}
          </TabsContent>
        </Tabs>
      )}

      <ImportDialog open={importing} onOpenChange={setImporting} />

      <AlertDialog open={confirm !== null} onOpenChange={(open) => !open && setConfirm(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{confirm === "clear-history" ? t("Clear the history?") : t("Stop and cancel every queued job?")}</AlertDialogTitle>
            <AlertDialogDescription>
              {confirm === "clear-history"
                ? t("Finished jobs are removed from the history. Their images stay in the gallery.")
                : t("The running job is interrupted (its images are kept) and every waiting job is cancelled.")}
            </AlertDialogDescription>
          </AlertDialogHeader>
          {confirm === "clear-history" ? (
            <label className="flex items-center gap-2 text-sm">
              <Checkbox checked={keepStarred} onCheckedChange={(checked) => setKeepStarred(checked === true)} />
              {t("Keep starred jobs")}
            </label>
          ) : null}
          <AlertDialogFooter>
            <AlertDialogCancel>{t("Cancel")}</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              onClick={() => {
                const which = confirm
                setConfirm(null)
                if (which === "clear-history") void act(() => api.jobs.clearHistory(keepStarred), t("Could not clear the history"), t("History cleared"))
                else void act(() => api.queue.interruptAll(true), t("Could not clear the queue"), t("Queue cleared"))
              }}
            >
              {confirm === "clear-history" ? t("Clear") : t("Stop and cancel")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {viewing ? (
        <Suspense fallback={null}>
          <Lightbox
            open
            onOpenChange={(open) => !open && setViewing(null)}
            items={viewing.items}
            current={viewing.current}
            onNavigate={(key) => setViewing((v) => (v ? { ...v, current: key } : v))}
            tab={viewing.tab}
          />
        </Suspense>
      ) : null}
    </div>
  )
}

