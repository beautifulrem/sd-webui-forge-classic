/**
 * Values a form is still producing (e.g. a canvas layer being exported and
 * uploaded after the last brush stroke). A submit awaits them first, so the
 * job carries what the user sees. Writers register per form id.
 */

type Flush = () => Promise<void>

const writers = new Map<string, Set<Flush>>()

/** Registers `flush` for form `formId`; returns the unregister function. */
export function registerPendingWrite(formId: string, flush: Flush): () => void {
  let set = writers.get(formId)
  if (!set) writers.set(formId, (set = new Set()))
  set.add(flush)
  return () => void set.delete(flush)
}

/** Resolves when every pending value of the form is written (rejects if one fails). */
export async function flushPendingWrites(formId: string): Promise<void> {
  const set = writers.get(formId)
  if (!set || set.size === 0) return
  await Promise.all([...set].map((flush) => flush()))
}
