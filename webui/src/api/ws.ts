/**
 * WebSocket client store: one connection to /api/v2/ws, JSON events fanned out
 * to subscribers, binary preview frames turned into object URLs (the previous
 * URL is revoked as soon as a newer frame arrives), automatic reconnect with
 * exponential backoff and a feature handshake.
 */
import { create } from "zustand"
import { API_BASE } from "./http"
import { normalizeEvent, type WsEvent, type WsEventType } from "./events"
import { FrameError, parseFrame } from "./frames"

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

export const useWs = create<WsState>()((set, get) => {
  function setPreview(next: LivePreview | null) {
    const previous = get().preview
    set({ preview: next })
    if (previous) URL.revokeObjectURL(previous.url)
  }

  function handleBinary(buffer: ArrayBuffer) {
    try {
      const frame = parseFrame(buffer)
      const current = get().preview
      // Frames for an older job or step can arrive late; keep the newest.
      if (current && current.jobSeq === frame.jobSeq && frame.step < current.step) return
      const url = URL.createObjectURL(new Blob([frame.bytes], { type: frame.mime }))
      setPreview({ jobSeq: frame.jobSeq, step: frame.step, url })
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
      ws.send(JSON.stringify({ type: "hello", features: CLIENT_FEATURES }))
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
      setPreview(null)
    },
  }
})

/** Whether the server announced a feature in its hello. */
export function useServerFeature(feature: string): boolean {
  return useWs((s) => s.serverFeatures.includes(feature))
}
