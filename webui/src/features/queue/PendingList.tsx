/**
 * Queued jobs in run order, virtualized (an imported queue can hold
 * thousands): only the rows near the viewport of the page's scroller exist.
 * Reorder by dragging the grip, or with the keyboard: focus the grip,
 * Space / Enter picks the job up, ↑ / ↓ move it, Space / Enter drops,
 * Escape cancels (announced to screen readers). Each job can be edited
 * (opens its form), duplicated, given a priority, moved to the top or
 * bottom, or cancelled. Moves are sent by id against the latest queue (see
 * `reorder.ts`), so a job that starts mid-drag does not shift the target.
 */
import { defaultRangeExtractor, useVirtualizer } from "@tanstack/react-virtual"
import { ArrowDownToLine, ArrowUpToLine, Copy, Flag, GripVertical, MoreHorizontal, PencilLine, X } from "lucide-react"
import { useEffect, useLayoutEffect, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { api } from "@/api/endpoints"
import { keys, queryClient } from "@/api/queries"
import type { Job } from "@/api/types"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { cn } from "@/lib/utils"
import { openJob } from "./edit-session"
import { act, duplicate, formLabel, jobTitle, removeJob, setPriority } from "./queries"
import { dropTarget, moveIndex, planMove, PRIORITIES, priorityLabel } from "./reorder"

type Grab = { id: string; order: readonly Job[]; via: "pointer" | "keyboard" }
type MoveTexts = { failure: string; started: string }

/** Row height estimate (a row is one card of two text lines plus the gap). */
const ROW = 58
/** Pointer distance from the scroller's edge that scrolls it while dragging. */
const EDGE = 48

/** Moves the job at `from` of the latest queue to `to`, showing the new order right away. */
async function commitMove(latest: readonly Job[], from: number, to: number, texts: MoveTexts) {
  const plan = planMove(latest, from, to)
  if (!plan) return
  const byId = new Map(latest.map((job) => [job.id, job]))
  queryClient.setQueryData<Job[]>(
    keys.queueItems,
    plan.order.map((id) => (id === plan.id && plan.priority !== null ? { ...byId.get(id)!, priority: plan.priority } : byId.get(id)!)),
  )
  await act(() => api.queue.move({ id: plan.id, after: plan.after, before: plan.before, priority: plan.priority }), texts.failure, undefined, { 409: texts.started })
}

/** The nearest scrolling ancestor (the page column), else the document. */
function scrollParent(element: HTMLElement | null): HTMLElement | null {
  for (let node = element?.parentElement; node; node = node.parentElement) {
    const { overflowY } = getComputedStyle(node)
    if (overflowY === "auto" || overflowY === "scroll") return node
  }
  return element ? ((document.scrollingElement as HTMLElement | null) ?? document.documentElement) : null
}

function JobMenu({ job, jobs }: { job: Job; jobs: readonly Job[] }) {
  const { t } = useTranslation()
  const failure: MoveTexts = { failure: t("Could not move the job"), started: t("The job started meanwhile and was not moved.") }
  const index = jobs.findIndex((item) => item.id === job.id)
  return (
    <DropdownMenu>
      <DropdownMenuTrigger render={<Button variant="ghost" size="icon-sm" aria-label={t("More actions for job #{{seq}}", { seq: job.seq })} />}>
        <MoreHorizontal />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="min-w-52">
        <DropdownMenuItem onClick={() => void duplicate(job)} disabled={!job.replayable}>
          <Copy />
          {t("Duplicate")}
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => void commitMove(jobs, index, 0, failure)} disabled={index <= 0}>
          <ArrowUpToLine />
          {t("Move to top")}
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => void commitMove(jobs, index, jobs.length - 1, failure)} disabled={index === -1 || index === jobs.length - 1}>
          <ArrowDownToLine />
          {t("Move to bottom")}
        </DropdownMenuItem>
        <DropdownMenuSub>
          <DropdownMenuSubTrigger>
            <Flag />
            {t("Priority")}
          </DropdownMenuSubTrigger>
          <DropdownMenuSubContent>
            <DropdownMenuRadioGroup value={String(job.priority)} onValueChange={(value) => void setPriority(job, Number(value))}>
              {PRIORITIES.map((preset) => (
                <DropdownMenuRadioItem key={preset.value} value={String(preset.value)}>
                  {t(preset.label)}
                </DropdownMenuRadioItem>
              ))}
            </DropdownMenuRadioGroup>
          </DropdownMenuSubContent>
        </DropdownMenuSub>
        <DropdownMenuSeparator />
        <DropdownMenuItem variant="destructive" onClick={() => void removeJob(job)}>
          <X />
          {t("Cancel job")}
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

export function PendingList({ jobs }: { jobs: readonly Job[] }) {
  // TanStack Virtual's virtualizer is a mutable object the React Compiler must not memoise.
  "use no memo"
  const { t, i18n } = useTranslation()
  const [grab, setGrab] = useState<Grab | null>(null)
  const [announcement, setAnnouncement] = useState("")
  const rows = useRef(new Map<string, HTMLLIElement>())
  // While a job is held, show the order being arranged (minus jobs that left the queue meanwhile).
  const latestIds = new Set(jobs.map((job) => job.id))
  const list = grab ? grab.order.filter((job) => job.id === grab.id || latestIds.has(job.id)) : jobs
  const grabbedIndex = grab ? list.findIndex((job) => job.id === grab.id) : -1

  const [listElement, setListElement] = useState<HTMLOListElement | null>(null)
  const [scroller, setScroller] = useState<HTMLElement | null>(null)
  // Where the list starts in the scroller's content (the running card above it changes height).
  const [margin, setMargin] = useState(0)
  useLayoutEffect(() => setScroller(scrollParent(listElement)), [listElement])
  useLayoutEffect(() => {
    if (!listElement || !scroller) return
    const update = () => setMargin(Math.round(listElement.getBoundingClientRect().top - scroller.getBoundingClientRect().top + scroller.scrollTop))
    update()
    const observer = new ResizeObserver(update)
    observer.observe(scroller.firstElementChild ?? scroller)
    return () => observer.disconnect()
  }, [listElement, scroller])

  // oxlint-disable-next-line react/incompatible-library -- opted out of memoisation above
  const virtualizer = useVirtualizer({
    count: list.length,
    getScrollElement: () => scroller,
    estimateSize: () => ROW,
    overscan: 8,
    scrollMargin: margin,
    getItemKey: (index) => list[index]?.id ?? index,
    // The held job stays rendered (and focused) wherever it is moved.
    rangeExtractor: (range) => {
      const indexes = defaultRangeExtractor(range)
      if (grabbedIndex >= 0 && !indexes.includes(grabbedIndex)) return [...indexes, grabbedIndex].sort((a, b) => a - b)
      return indexes
    },
  })

  // Moving a row can drop the focus: keep it on the held job's grip.
  const grabbedId = grab?.via === "keyboard" ? grab.id : null
  const orderKey = list.map((job) => job.id).join()
  useEffect(() => {
    if (!grabbedId) return
    const handle = rows.current.get(grabbedId)?.querySelector<HTMLButtonElement>("button[aria-roledescription]")
    if (handle && document.activeElement !== handle) handle.focus({ preventScroll: true })
  }, [grabbedId, orderKey])
  const failure: MoveTexts = { failure: t("Could not move the job"), started: t("The job started meanwhile and was not moved.") }

  const moveTo = (to: number) => {
    if (!grab) return
    const current = list.findIndex((job) => job.id === grab.id)
    const target = Math.max(0, Math.min(list.length - 1, to))
    if (current === -1 || current === target) return
    const order = moveIndex(list, current, target)
    setGrab({ ...grab, order })
    if (grab.via === "keyboard") {
      virtualizer.scrollToIndex(target, { align: "auto" })
      setAnnouncement(t("Position {{index}} of {{count}}", { index: target + 1, count: order.length }))
    }
  }

  const drop = () => {
    if (!grab) return
    setGrab(null)
    const target = dropTarget(jobs, list, grab.id)
    if (!target) {
      setAnnouncement(t("The job started meanwhile and was not moved."))
      return
    }
    setAnnouncement(t("Dropped at position {{index}}", { index: target.to + 1 }))
    void commitMove(jobs, target.from, target.to, failure)
  }

  const cancel = () => {
    if (!grab) return
    setGrab(null)
    setAnnouncement(t("Move cancelled"))
  }

  /** Where the held job goes for a pointer at `clientY`: after every other row whose middle is above it (from the layout model, no DOM reads per row). */
  const indexAt = (clientY: number): number => {
    const top = listElement?.getBoundingClientRect().top ?? 0
    const y = clientY - top
    let count = 0
    for (const measurement of virtualizer.measurementsCache) {
      if (measurement.index === grabbedIndex) continue
      if (measurement.start - margin + measurement.size / 2 < y) count++
      else break
    }
    return count
  }

  const autoScroll = (clientY: number) => {
    if (!scroller) return
    const rect = scroller.getBoundingClientRect()
    if (clientY < rect.top + EDGE) scroller.scrollBy({ top: -ROW / 2 })
    else if (clientY > rect.bottom - EDGE) scroller.scrollBy({ top: ROW / 2 })
  }

  return (
    <>
      <ol ref={setListElement} aria-label={t("Queued jobs")} className="relative" style={{ height: virtualizer.getTotalSize() }}>
        {virtualizer.getVirtualItems().map((virtual) => {
          const job = list[virtual.index]
          if (!job) return null
          const index = virtual.index
          const grabbed = grab?.id === job.id
          const priority = priorityLabel(job.priority)
          const created = new Date(job.created_at)
          return (
            <li
              key={job.id}
              data-index={index}
              aria-posinset={index + 1}
              aria-setsize={list.length}
              ref={(element) => {
                if (element) {
                  rows.current.set(job.id, element)
                  virtualizer.measureElement(element)
                } else rows.current.delete(job.id)
              }}
              className={cn("absolute inset-x-0 top-0 pb-1.5", grabbed && "z-10")}
              style={{ transform: `translateY(${virtual.start - margin}px)` }}
            >
              <div
                className={cn(
                  "flex items-center gap-2 rounded-lg border bg-card py-1.5 pr-1.5 pl-1 transition-shadow",
                  grabbed && "border-primary/60 shadow-lg ring-2 ring-primary/30",
                )}
              >
                <button
                  type="button"
                  aria-roledescription={t("sortable")}
                  aria-label={t("Move job #{{seq}}", { seq: job.seq })}
                  aria-pressed={grabbed}
                  aria-describedby="queue-reorder-help"
                  className="grid h-9 w-6 shrink-0 cursor-grab touch-none place-items-center rounded-md text-muted-foreground outline-none hover:bg-muted hover:text-foreground focus-visible:ring-3 focus-visible:ring-ring/50 active:cursor-grabbing"
                  onPointerDown={(event) => {
                    if (event.button !== 0 || grab) return
                    event.currentTarget.setPointerCapture?.(event.pointerId)
                    setGrab({ id: job.id, order: [...jobs], via: "pointer" })
                  }}
                  onPointerMove={(event) => {
                    if (grab?.via !== "pointer") return
                    autoScroll(event.clientY)
                    moveTo(indexAt(event.clientY))
                  }}
                  onPointerUp={() => grab?.via === "pointer" && drop()}
                  onPointerCancel={() => grab?.via === "pointer" && cancel()}
                  onKeyDown={(event) => {
                    if (event.key === " " || event.key === "Enter") {
                      event.preventDefault()
                      if (grab?.id === job.id) drop()
                      else if (!grab) {
                        setGrab({ id: job.id, order: [...jobs], via: "keyboard" })
                        setAnnouncement(t("Picked up job #{{seq}} at position {{index}}. Use the arrow keys to move it, Space to drop, Escape to cancel.", { seq: job.seq, index: index + 1 }))
                      }
                    } else if (grab?.id === job.id && (event.key === "ArrowUp" || event.key === "ArrowDown")) {
                      event.preventDefault()
                      moveTo(index + (event.key === "ArrowUp" ? -1 : 1))
                    } else if (event.key === "Escape" && grab) {
                      event.preventDefault()
                      cancel()
                    } else if (event.key === "Tab" && grab) {
                      cancel()
                    }
                  }}
                >
                  <GripVertical className="size-4" aria-hidden />
                </button>
                <span className="w-7 shrink-0 text-center text-xs font-medium text-muted-foreground tabular-nums" aria-hidden>
                  {index + 1}
                </span>
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm" title={job.summary ?? undefined}>
                    {jobTitle(job)}
                  </p>
                  <p className="flex items-center gap-x-1.5 truncate text-xs text-muted-foreground tabular-nums">
                    <span>#{job.seq}</span>
                    <span>{formLabel(job.tab)}</span>
                    <span>{Number.isNaN(created.getTime()) ? "" : created.toLocaleTimeString(i18n.language, { hour: "2-digit", minute: "2-digit" })}</span>
                    {job.forever ? <span>· {t("Generate forever")}</span> : null}
                    {job.source ? <span>· {t("Imported")}</span> : null}
                  </p>
                </div>
                {priority ? (
                  <Badge variant={job.priority > 0 ? "default" : "outline"} className="shrink-0">
                    {t(priority)}
                  </Badge>
                ) : null}
                <Button variant="ghost" size="icon-sm" aria-label={t("Edit job #{{seq}}", { seq: job.seq })} title={t("Edit")} onClick={() => void openJob(job.id)}>
                  <PencilLine />
                </Button>
                <JobMenu job={job} jobs={jobs} />
              </div>
            </li>
          )
        })}
      </ol>
      <p id="queue-reorder-help" className="sr-only">
        {t("Press Space to pick the job up, the arrow keys to move it, Space again to drop it.")}
      </p>
      <p role="status" aria-live="assertive" className="sr-only">
        {announcement}
      </p>
    </>
  )
}
