/**
 * "Check for updates" with progress: one request per extension, a few at a
 * time, so the list fills in as answers arrive and the user can stop it.
 */
import { create } from "zustand"
import { api } from "@/api/endpoints"
import type { ExtensionInfo } from "./extensions-model"
import { putExtension } from "./queries"

export type CheckProgress = { done: number; total: number; current: string[]; failed: string[] }

/**
 * Runs `check` for every name, `concurrency` at a time; `onProgress` after
 * each. Stops scheduling when `signal` aborts. Returns the names that failed.
 */
export async function runChecks(
  names: readonly string[],
  check: (name: string) => Promise<unknown>,
  { concurrency = 4, onProgress, signal }: { concurrency?: number; onProgress?: (progress: CheckProgress) => void; signal?: AbortSignal } = {},
): Promise<string[]> {
  const queue = [...names]
  const progress: CheckProgress = { done: 0, total: names.length, current: [], failed: [] }
  const report = () => onProgress?.({ ...progress, current: [...progress.current], failed: [...progress.failed] })
  const worker = async () => {
    // oxlint-disable-next-line eslint/no-unmodified-loop-condition -- the caller aborts `signal` while the workers await
    while (queue.length > 0 && !signal?.aborted) {
      const name = queue.shift()!
      progress.current.push(name)
      report()
      try {
        await check(name)
      } catch {
        progress.failed.push(name)
      }
      progress.current = progress.current.filter((item) => item !== name)
      progress.done++
      report()
    }
  }
  await Promise.all(Array.from({ length: Math.max(1, Math.min(concurrency, names.length)) }, worker))
  return progress.failed
}

type CheckState = { progress: CheckProgress | null; controller: AbortController | null }

/** The running check (kept when the page is left and reopened). */
export const useUpdateCheck = create<CheckState>()(() => ({ progress: null, controller: null }))

/** Checks the given extensions (default: enabled ones with a remote, not built-in). */
export type CheckResult = {
  /** Extensions there were to check (0: none is enabled, installed from a repository and not built in). */
  total: number
  checked: number
  available: number
  failed: string[]
  /** Another check was running: this one did nothing. */
  busy?: boolean
}

export async function checkForUpdates(items: readonly ExtensionInfo[]): Promise<CheckResult> {
  if (useUpdateCheck.getState().controller) return { total: 0, checked: 0, available: 0, failed: [], busy: true }
  const names = items.filter((ext) => ext.enabled && ext.remote && !ext.builtin).map((ext) => ext.name)
  if (names.length === 0) return { total: 0, checked: 0, available: 0, failed: [] }
  const controller = new AbortController()
  let available = 0
  useUpdateCheck.setState({ controller, progress: { done: 0, total: names.length, current: [], failed: [] } })
  try {
    const failed = await runChecks(
      names,
      async (name) => {
        const info = await api.extensions.check(name)
        putExtension(info)
        if (info.can_update) available++
      },
      { signal: controller.signal, onProgress: (progress) => useUpdateCheck.setState({ progress }) },
    )
    return { total: names.length, checked: names.length - failed.length, available, failed }
  } finally {
    useUpdateCheck.setState({ controller: null, progress: null })
  }
}

export function stopUpdateCheck() {
  useUpdateCheck.getState().controller?.abort()
}
