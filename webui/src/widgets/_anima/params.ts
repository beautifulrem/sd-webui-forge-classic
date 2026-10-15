/**
 * Reading the form a widget sits in: sibling or other owners' params by
 * reference ("name", "owner.name", "core.name"), and the owner's action
 * outputs. Not a widget (no `index.tsx`).
 */
import { useFormState, useOwner } from "@/ui/hooks"

/** `[owner, name]` of a param reference relative to `owner`. */
export function locate(ref: string, owner: string): [string, string] {
  const dot = ref.indexOf(".")
  return dot > 0 ? [ref.slice(0, dot), ref.slice(dot + 1)] : [owner, ref]
}

/** The first of `refs` holding a usable number (`accept` filters, e.g. positive step counts). */
export function useFirstNumber(refs: readonly string[] | undefined, fallback: number, accept: (value: number) => boolean = () => true): number {
  const owner = useOwner()
  return useFormState((s) => {
    for (const ref of refs ?? []) {
      const [o, n] = locate(ref, owner)
      const value = s.values[o]?.[n]
      if (typeof value === "number" && Number.isFinite(value) && accept(value)) return value
    }
    return fallback
  })
}

/**
 * The Shift a run reads from the first of `refs` holding a number, clamped
 * to 1-24 as the Python side does (`_runtime_shift_from_processing`,
 * `stage_window`): 0 reads as 1, not as the next ref. 3 when none is set.
 */
export function useShift(refs: readonly string[] | undefined): number {
  return Math.min(24, Math.max(1, useFirstNumber(refs, 3)))
}

/** A param of this widget's owner (or "owner.name"). */
export function useSibling<T>(ref: string | undefined): T | undefined {
  const owner = useOwner()
  return useFormState((s) => {
    if (!ref) return undefined
    const [o, n] = locate(ref, owner)
    return s.values[o]?.[n] as T | undefined
  })
}

/** An action output of this widget's owner (`__outputs__`), if any. */
export function useOwnerOutput<T>(name: string): T | undefined {
  const owner = useOwner()
  return useFormState((s) => s.outputs[owner]?.[name] as T | undefined)
}
