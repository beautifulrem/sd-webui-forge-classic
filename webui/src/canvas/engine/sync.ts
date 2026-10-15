/**
 * Keeps a layer's form value (an upload id) in step with its pixels: after
 * an edit, once the user pauses, the layer is encoded (PNG, wire format),
 * hashed, and uploaded unless the same content was uploaded before (undo
 * back to an earlier state, or the same mask twice), then committed. `flush()` exports
 * at once and resolves when the value is current (before a submit).
 */
import { hashBytes } from "../core/pixels"

export type UploadCache = {
  get(hash: string): string | undefined
  set(hash: string, id: string): void
}

/**
 * How long an upload is reused. The server prunes uploads no job refers to
 * after `remi_upload_retention_days` (hourly housekeeping); an id older than
 * this is uploaded again rather than risk "unknown upload" on submit.
 */
export const UPLOAD_REUSE_MS = 60 * 60 * 1000

/** A small LRU of content hash → upload id; entries expire after `maxAge` ms (from their upload). */
export function createUploadCache(limit = 48, maxAge = UPLOAD_REUSE_MS, now: () => number = Date.now): UploadCache {
  const map = new Map<string, { id: string; at: number }>()
  return {
    get(hash) {
      const entry = map.get(hash)
      if (!entry) return undefined
      map.delete(hash)
      if (now() - entry.at > maxAge) return undefined
      map.set(hash, entry)
      return entry.id
    },
    set(hash, id) {
      map.delete(hash)
      map.set(hash, { id, at: now() })
      while (map.size > limit) map.delete(map.keys().next().value!)
    },
  }
}

export type SyncDeps = {
  /** The layer as a PNG in wire format, null when empty (its content is taken when called). */
  png: () => Promise<Blob | null>
  upload: (blob: Blob) => Promise<string>
  cache: UploadCache
  /** Debounce after the last edit, ms. */
  delay?: number
  /** An edit is in progress (a stroke): automatic exports wait for it. */
  busy?: () => boolean
}

type IdleWindow = { requestIdleCallback?: (fn: () => void, options?: { timeout: number }) => number }

/** Runs `fn` when the main thread is idle (an export reads and hashes the whole layer). */
function whenIdle(fn: () => void) {
  const idle = (globalThis as IdleWindow).requestIdleCallback
  if (idle) idle(fn, { timeout: 1000 })
  else fn()
}

export class LayerSync {
  /** Bumped by every edit. */
  private version = 0
  /** Version whose content the committed value holds. */
  private synced = 0
  private id: string | null = null
  private timer: ReturnType<typeof setTimeout> | undefined
  private running: Promise<void> | null = null
  private commitFn: (id: string | null) => void = () => {}
  private errorFn: (error: unknown) => void = () => {}
  private readonly deps: SyncDeps
  private readonly listeners = new Set<() => void>()

  constructor(deps: SyncDeps) {
    this.deps = deps
  }

  /** Where values go (the form param); the latest binding wins. */
  bind(commit: (id: string | null) => void, onError?: (error: unknown) => void) {
    this.commitFn = commit
    if (onError) this.errorFn = onError
  }

  /** The id the layer's content was last committed as. */
  get value(): string | null {
    return this.id
  }

  /** Edits not committed yet. */
  get pending(): boolean {
    return this.synced !== this.version
  }

  subscribe(listener: () => void) {
    this.listeners.add(listener)
    return () => void this.listeners.delete(listener)
  }

  private emit() {
    for (const listener of this.listeners) listener()
  }

  /** The content changed by an edit: schedule an export. */
  touch() {
    this.version++
    this.schedule()
    this.emit()
  }

  private schedule() {
    clearTimeout(this.timer)
    this.timer = setTimeout(
      () =>
        whenIdle(() => {
          // Painting again: wait for the pause after this stroke.
          if (this.deps.busy?.()) this.schedule()
          else if (this.pending) void this.run().catch(this.errorFn)
        }),
      this.deps.delay ?? 600,
    )
  }

  /**
   * The content now is what `id` holds (loaded from it, or cleared with
   * `null`): nothing to upload, pending edits are dropped.
   */
  adopt(id: string | null) {
    clearTimeout(this.timer)
    this.version++
    this.synced = this.version
    this.id = id
    this.emit()
  }

  /** Exports now and resolves once the committed value matches the content. */
  async flush(): Promise<void> {
    clearTimeout(this.timer)
    await this.run()
  }

  dispose() {
    clearTimeout(this.timer)
    this.listeners.clear()
  }

  private run(): Promise<void> {
    if (!this.running) {
      this.running = this.loop().finally(() => {
        this.running = null
        this.emit()
      })
      return this.running
    }
    // A run is in flight: wait for it, then make sure the newest edit went out too.
    return this.running.then(() => (this.pending ? this.run() : undefined))
  }

  private async loop() {
    while (this.synced !== this.version) {
      const version = this.version
      const png = await this.deps.png()
      let id: string | null = null
      if (png) {
        // The same pixels encode to the same PNG: an earlier upload of them is reused.
        const hash = hashBytes(new Uint8Array(await png.arrayBuffer()))
        id = this.deps.cache.get(hash) ?? null
        if (id === null) {
          id = await this.deps.upload(png)
          this.deps.cache.set(hash, id)
        }
      }
      // Adopted (replaced from outside) meanwhile: that value wins.
      if (this.synced > version) continue
      this.synced = version
      if (id !== this.id) {
        this.id = id
        this.commitFn(id)
      }
    }
  }
}
