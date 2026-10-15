/** In-browser stand-in for /api/v2/ws in mock mode. */
import type { SocketLike } from "../ws"
import { queueStatus } from "./server"

const sockets = new Set<MockSocket>()

export class MockSocket implements SocketLike {
  binaryType: BinaryType = "arraybuffer"
  readyState = 0
  onopen: ((ev: Event) => void) | null = null
  onmessage: ((ev: MessageEvent) => void) | null = null
  onclose: ((ev: CloseEvent) => void) | null = null
  onerror: ((ev: Event) => void) | null = null

  constructor() {
    setTimeout(() => {
      this.readyState = 1
      sockets.add(this)
      this.onopen?.(new Event("open"))
    }, 150)
  }

  send(data: string) {
    const message = JSON.parse(data) as { type?: string }
    if (message.type !== "hello") return
    this.deliver(JSON.stringify({ type: "hello", features: ["events.v1", "preview.binary"], version: "remi-mock" }))
    // Like the server: a snapshot of the queue follows the hello.
    this.deliver(JSON.stringify({ type: "queue.changed", ...queueStatus() }))
  }

  close() {
    this.readyState = 3
    sockets.delete(this)
    this.onclose?.(new CloseEvent("close", { code: 1000 }))
  }

  deliver(data: string | ArrayBuffer) {
    if (this.readyState === 1) this.onmessage?.(new MessageEvent("message", { data }))
  }
}

export function broadcast(event: object) {
  const text = JSON.stringify(event)
  for (const socket of sockets) socket.deliver(text)
}

export function broadcastBinary(buffer: ArrayBuffer) {
  for (const socket of sockets) socket.deliver(buffer)
}
