/**
 * WebSocket client store: one connection to /api/v2/ws, JSON events fanned out
 * to subscribers, binary preview frames turned into object URLs (the previous
 * URL is revoked as soon as a newer frame arrives), automatic reconnect with
 * exponential backoff and a feature handshake. The page's visibility goes to
 * the server (in the hello and on every change): a hidden page gets no
 * previews, and while no page is shown the server encodes none.
 */
import { create } from "zustand"
import { API_BASE } from "./http"
import { normalizeEvent, type WsEvent, type WsEventType } from "./events"
import { FrameError, parseFrame } from "./frames"
import { flushPaintBatch, holdPaint, paintBatch } from "./paint-batch"

/** What this client understands; sent in the hello message. */
export const CLIENT_FEATURES = ["events.v1", "preview.binary", "preview.jpeg", "preview.webp"] as const

export type WsStatus = "idle" | "connecting" | "open" | "reconnecting" | "closed"

export type LivePreview = { jobSeq: number; step: number; url: string }

/** The subset of WebSocket used here; the mock socket implements it too. */
export interface SocketLike {
  binaryType: BinaryType
  readonly readyState: number
  send(data: string): void
  close(code?: number, reason?: string): void
  onopen: ((ev: Event) => void) | null
  onmessage: ((ev: MessageEvent) => void) | null
  onclose: ((ev: CloseEvent) => void) | null
  onerror: ((ev: Event) => void) | null
}

type Handler<K extends WsEventType> = (event: Extract<WsEvent, { type: K }>) => void
type AnyHandler = (event: WsEvent) => void

type WsState = {
  status: WsStatus
  /** Features announced by the server's hello. */
  serverFeatures: readonly string[]
  /** Consecutive failed attempts; resets on a successful open. */
  attempt: number
  /** The newest live preview frame, if any. */
  preview: LivePreview | null
  connect: () => void
  disconnect: () => void
  send: (message: object) => boolean
  clearPreview: () => void
}

const BACKOFF_BASE_MS = 500
const BACKOFF_MAX_MS = 15_000

/** Delay before reconnect attempt `n` (0-based): exponential, capped, ±25 % jitter. */
export function backoffDelay(attempt: number, random: () => number = Math.random): number {
  const ideal = Math.min(BACKOFF_MAX_MS, BACKOFF_BASE_MS * 2 ** attempt)
  return Math.round(ideal * (0.75 + random() * 0.5))
}

function wsUrl(): string {
  const scheme = location.protocol === "https:" ? "wss:" : "ws:"
  return `${scheme}//${location.host}${API_BASE}/ws`
}

async function openSocket(): Promise<SocketLike> {
  // Read the env directly (not http.ts's MOCK) so the bundler drops the mock chunk.
  if (import.meta.env.VITE_MOCK === "1") {
    const { MockSocket } = await import("./mock/socket")
    return new MockSocket()
  }
  return new WebSocket(wsUrl())
}

const handlers = new Map<WsEventType | "*", Set<AnyHandler>>()
const frameListeners = new Set<(frame: { jobSeq: number; step: number }) => void>()

/** Called for each preview frame as it arrives (before it is decoded and shown); returns the unsubscribe function. */
export function onPreviewFrame(listener: (frame: { jobSeq: number; step: number }) => void): () => void {
  frameListeners.add(listener)
  return () => frameListeners.delete(listener)
}

/** Subscribes to one event type (or "*" for all); returns the unsubscribe function. */
export function onWsEvent<K extends WsEventType>(type: K, handler: Handler<K>): () => void
export function onWsEvent(type: "*", handler: AnyHandler): () => void
export function onWsEvent(type: WsEventType | "*", handler: AnyHandler): () => void {
  let set = handlers.get(type)
  if (!set) handlers.set(type, (set = new Set()))
  set.add(handler)
  return () => set.delete(handler)
}

function dispatch(event: WsEvent) {
  for (const key of [event.type, "*"] as const) {
    for (const handler of handlers.get(key) ?? []) {
      try {
        handler(event)
      } catch (error) {
        console.error(`[ws] handler for ${event.type} failed`, error)
      }
    }
  }
}

/** Whether the page is shown (no previews are sent to a hidden one). */
function pageVisible(): boolean {
  return typeof document === "undefined" || document.visibilityState !== "hidden"
}

let socket: SocketLike | null = null
let retryTimer: ReturnType<typeof setTimeout> | undefined
let wanted = false
/**
 * Bumped by every `start()` and `disconnect()`. Opening a socket is async (the
 * mock is a lazy import), so a connect → disconnect → connect sequence (React
 * StrictMode) or a wake-up during a pending open would otherwise leave two
 * sockets dispatching every event twice: only the newest attempt may adopt
 * its socket.
 */
let attemptToken = 0

/**
 * Resolves once `url`'s image is decoded (and in the memory cache), so the
 * `<img>` that switches to it draws it in the same frame; at once where the
 * browser cannot decode ahead.
 */
function decoded(url: string): Promise<void> {
  if (typeof Image === "undefined") return Promise.resolve()
  const image = new Image()
  if (typeof image.decode !== "function") return Promise.resolve()
  image.src = url
  return image.decode().catch(() => {})
}

export const useWs = create<WsState>()((set, get) => {
  /** Bumped by `clearPreview`: a frame still decoding from before must not show. */
  let epoch = 0
  /**
   * Preview frames in the order they arrived, and the arrival number of the one shown. The
   * socket delivers frames in the order the server sent them (its preview slot keeps only the
   * newest), so the latest arrival is the newest image; but frames are decoded concurrently and
   * a later one can finish first, so a frame is shown only if no later arrival is on screen.
   * Not the step: it starts again at 0 for every image of a batch (and every pass of a hires
   * fix), and an earlier step of the job's next image is the newest frame.
   */
  let arrived = 0
  let shown = 0

  function setPreview(next: LivePreview | null) {
    const previous = get().preview
    set({ preview: next })
    if (previous) URL.revokeObjectURL(previous.url)
  }

  function handleBinary(buffer: ArrayBuffer) {
    try {
      const frame = parseFrame(buffer)
      const order = ++arrived
      const url = URL.createObjectURL(new Blob([frame.bytes], { type: frame.mime }))
      const from = epoch
      // Decoded first, then shown with the step's progress in one frame (api/paint-batch).
      const release = holdPaint()
      for (const listener of frameListeners) listener({ jobSeq: frame.jobSeq, step: frame.step })
      void decoded(url).then(() => {
        paintBatch(() => {
          if (from !== epoch || order < shown) URL.revokeObjectURL(url)
          else {
            shown = order
            setPreview({ jobSeq: frame.jobSeq, step: frame.step, url })
          }
        })
        release()
      })
    } catch (error) {
      if (error instanceof FrameError) console.warn("[ws]", error.message)
      else throw error
    }
  }

  function handleText(text: string) {
    let raw: unknown
    try {
      raw = JSON.parse(text)
    } catch {
      console.warn("[ws] non-JSON text frame ignored")
      return
    }
    const event = normalizeEvent(raw)
    if (!event) return
    if (event.type === "hello") set({ serverFeatures: event.features ?? [] })
    dispatch(event)
  }

  function scheduleReconnect() {
    if (!wanted) return
    const attempt = get().attempt
    set({ status: "reconnecting", attempt: attempt + 1 })
    clearTimeout(retryTimer)
    retryTimer = setTimeout(() => void start(), backoffDelay(attempt))
  }

  async function start() {
    if (!wanted || socket) return
    const token = ++attemptToken
    set({ status: get().attempt > 0 ? "reconnecting" : "connecting" })
    let ws: SocketLike
    try {
      ws = await openSocket()
    } catch {
      if (token === attemptToken) scheduleReconnect()
      return
    }
    if (!wanted || socket || token !== attemptToken) {
      ws.close(1000, "superseded")
      return
    }
    socket = ws
    ws.binaryType = "arraybuffer"
    ws.onopen = () => {
      set({ status: "open", attempt: 0 })
      ws.send(JSON.stringify({ type: "hello", features: CLIENT_FEATURES, visible: pageVisible() }))
    }
    ws.onmessage = (ev) => {
      if (typeof ev.data === "string") handleText(ev.data)
      else if (ev.data instanceof ArrayBuffer) handleBinary(ev.data)
    }
    ws.onclose = () => {
      // A socket that was already replaced or dropped must not schedule anything.
      if (socket !== ws) return
      socket = null
      if (wanted) scheduleReconnect()
      else set({ status: "closed" })
    }
    // onclose always follows onerror; reconnect is handled there.
    ws.onerror = () => {}
  }

  return {
    status: "idle",
    serverFeatures: [],
    attempt: 0,
    preview: null,
    connect() {
      wanted = true
      clearTimeout(retryTimer)
      // Reconnect immediately when asked explicitly (e.g. tab became visible).
      if (!socket) {
        set({ attempt: 0 })
        void start()
      }
    },
    disconnect() {
      wanted = false
      attemptToken++
      clearTimeout(retryTimer)
      const ws = socket
      socket = null
      ws?.close(1000, "client disconnect")
      set({ status: "closed" })
    },
    send(message) {
      if (!socket || socket.readyState !== 1) return false
      socket.send(JSON.stringify(message))
      return true
    },
    clearPreview() {
      // Queued updates first: a frame of the ended job must not show after this.
      flushPaintBatch()
      epoch++
      setPreview(null)
    },
  }
})

// A hidden tab tells the server, which stops sending it previews (and encoding them when no page
// is shown); shown again, the next preview arrives within a step (the server's encoder looks at
// the state at least once a second). The hello of a new connection carries the current state.
if (typeof document !== "undefined") {
  document.addEventListener("visibilitychange", () => void useWs.getState().send({ type: "visibility", visible: pageVisible() }))
}

/** Whether the server announced a feature in its hello. */
export function useServerFeature(feature: string): boolean {
  return useWs((s) => s.serverFeatures.includes(feature))
}
