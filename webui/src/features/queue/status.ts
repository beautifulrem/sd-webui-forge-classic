/**
 * The queue state (size, paused, running job) kept current from the
 * WebSocket: `queue.changed` carries the whole state, so no refetch is
 * needed. Used by the Queue page and by the navigation's queued-count badge.
 */
import { useQuery } from "@tanstack/react-query"
import { useEffect } from "react"
import { queryClient } from "@/api/queries"
import type { QueueStatus } from "@/api/types"
import { onWsEvent } from "@/api/ws"
import { queueKeys, queueStatusQuery } from "./queries"

export function useQueueStatus(enabled = true): QueueStatus | undefined {
  const status = useQuery({ ...queueStatusQuery, enabled })
  useEffect(() => {
    if (!enabled) return
    return onWsEvent("queue.changed", (e) =>
      queryClient.setQueryData<QueueStatus>(queueKeys.status, (prev) => ({ size: e.size ?? prev?.size ?? 0, paused: e.paused ?? prev?.paused ?? false, running: e.running ?? null })),
    )
  }, [enabled])
  return status.data
}

/** Jobs waiting in the queue (0 while `enabled` is false). */
export function useQueuedCount(enabled = true): number {
  return useQueueStatus(enabled)?.size ?? 0
}
