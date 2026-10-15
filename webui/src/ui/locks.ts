/**
 * Param locks: a toggle with `locks: "owner.param"` protects that param while
 * it is on. The form store drops programmatic writes to it (`setValues`:
 * pasted parameters, "Send to", reused jobs, carried-over defaults;
 * `applyPatches`: presets and action patches); user edits (`setValue`) and
 * resets still apply. Display parts of a patch (choices, enabled, …) apply.
 */
import { splitRef } from "./conditions"
import type { FormIndex } from "./form-index"

export type ParamLock = {
  /** The lock toggle. */
  owner: string
  name: string
  /** The protected param. */
  target: readonly [owner: string, name: string]
}

export function collectLocks(index: FormIndex): ParamLock[] {
  const locks: ParamLock[] = []
  for (const [owner, inputs] of index.inputs) {
    for (const [name, node] of inputs) {
      if (node.type === "toggle" && typeof node.locks === "string" && node.locks) {
        locks.push({ owner, name, target: splitRef(node.locks, owner) })
      }
    }
  }
  return locks
}

/** Names of `owner` that are locked now (their lock toggle is on). */
export function lockedNames(locks: readonly ParamLock[], values: Record<string, Record<string, unknown> | undefined>, owner: string): Set<string> {
  const names = new Set<string>()
  for (const lock of locks) {
    if (lock.target[0] === owner && values[lock.owner]?.[lock.name] === true) names.add(lock.target[1])
  }
  return names
}
