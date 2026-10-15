/** A controllable WebSocket double for tests (stub it with `vi.stubGlobal("WebSocket", FakeSocket)`). */
export class FakeSocket {
  static instances: FakeSocket[] = []
  binaryType = "blob"
  readyState = 0
  sent: string[] = []
  onopen: ((ev: Event) => void) | null = null
  onmessage: ((ev: MessageEvent) => void) | null = null
  onclose: ((ev: CloseEvent) => void) | null = null
  onerror: ((ev: Event) => void) | null = null
  readonly url: string
  constructor(url: string) {
    this.url = url
    FakeSocket.instances.push(this)
  }
  send(data: string) {
    this.sent.push(data)
  }
  close() {
    this.readyState = 3
    this.onclose?.(new CloseEvent("close"))
  }
  open() {
    this.readyState = 1
    this.onopen?.(new Event("open"))
  }
  receive(data: string | ArrayBuffer) {
    this.onmessage?.(new MessageEvent("message", { data }))
  }
  drop() {
    this.readyState = 3
    this.onclose?.(new CloseEvent("close", { code: 1006 }))
  }
}
