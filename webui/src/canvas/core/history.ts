/**
 * Undo/redo stack with a memory budget (pure). Entries report their size in
 * bytes; the oldest undo entries are dropped when the stack holds more than
 * `maxEntries` or more than `maxBytes` (the newest entry always stays, so a
 * single huge edit can still be undone once). Pushing after an undo drops the
 * redo branch.
 *
 * Undo and redo can be scoped with a predicate (the editor undoes the edited
 * layer's entries only): the newest matching entry moves, entries of other
 * scopes stay where they are. That is sound when edits of different scopes
 * commute (they touch different layers). `push` then drops only the redo
 * entries of its own scope.
 */

export type HistoryEntry = { readonly bytes: number }

/** Which entries an operation looks at (all when omitted). */
export type Scope<E> = (entry: E) => boolean

export type HistoryLimits = {
  /** Undo states kept at most (default 64). */
  maxEntries?: number
  /** Total size of the kept entries, undo and redo (default 384 MiB). */
  maxBytes?: number
}

export const DEFAULT_MAX_ENTRIES = 64
export const DEFAULT_MAX_BYTES = 384 * 1024 * 1024

export class History<E extends HistoryEntry> {
  private done: E[] = []
  private undone: E[] = []
  private size = 0
  readonly maxEntries: number
  readonly maxBytes: number

  constructor({ maxEntries = DEFAULT_MAX_ENTRIES, maxBytes = DEFAULT_MAX_BYTES }: HistoryLimits = {}) {
    this.maxEntries = Math.max(1, maxEntries)
    this.maxBytes = Math.max(0, maxBytes)
  }

  get canUndo() {
    return this.done.length > 0
  }

  get canRedo() {
    return this.undone.length > 0
  }

  /** Undo states available. */
  get depth() {
    return this.done.length
  }

  get redoDepth() {
    return this.undone.length
  }

  /** Bytes held by every kept entry. */
  get bytes() {
    return this.size
  }

  /** Whether an entry of `scope` can be undone. */
  canUndoIn(scope?: Scope<E>) {
    return scope ? this.done.some(scope) : this.done.length > 0
  }

  /** Whether an entry of `scope` can be redone. */
  canRedoIn(scope?: Scope<E>) {
    return scope ? this.undone.some(scope) : this.undone.length > 0
  }

  /** Records an edit; the redo entries of `scope` (all when omitted) are dropped. */
  push(entry: E, scope?: Scope<E>) {
    this.undone = this.undone.filter((dropped) => {
      if (scope && !scope(dropped)) return true
      this.size -= dropped.bytes
      return false
    })
    this.done.push(entry)
    this.size += entry.bytes
    this.trim()
  }

  /** The newest entry of `scope` to revert, moved to the redo side; null when there is none. */
  undo(scope?: Scope<E>): E | null {
    const entry = takeLast(this.done, scope)
    if (!entry) return null
    this.undone.push(entry)
    return entry
  }

  /** The newest undone entry of `scope` to apply again, moved back to the undo side. */
  redo(scope?: Scope<E>): E | null {
    const entry = takeLast(this.undone, scope)
    if (!entry) return null
    this.done.push(entry)
    return entry
  }

  /** Newest undo entry, if any. */
  peek(): E | null {
    return this.done.at(-1) ?? null
  }

  clear() {
    this.done = []
    this.undone = []
    this.size = 0
  }

  /**
   * Drops every entry `predicate` matches (e.g. the edits of a layer whose
   * content was replaced from outside). Entries of other layers stay: edits
   * of different layers commute.
   */
  removeWhere(predicate: (entry: E) => boolean) {
    const keep = (list: E[]) =>
      list.filter((entry) => {
        if (!predicate(entry)) return true
        this.size -= entry.bytes
        return false
      })
    this.done = keep(this.done)
    this.undone = keep(this.undone)
  }

  private trim() {
    // Redo entries are gone after a push, so only undo entries are trimmed.
    while (this.done.length > 1 && (this.done.length > this.maxEntries || this.size > this.maxBytes)) {
      const dropped = this.done.shift()!
      this.size -= dropped.bytes
    }
  }
}

/** Removes and returns the last entry of `list` that `scope` matches. */
function takeLast<E>(list: E[], scope?: Scope<E>): E | null {
  for (let i = list.length - 1; i >= 0; i--) {
    if (scope && !scope(list[i]!)) continue
    return list.splice(i, 1)[0]!
  }
  return null
}
