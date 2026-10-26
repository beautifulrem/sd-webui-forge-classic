import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { FakeSocket } from "@/test/fake-socket"

describe("useWs", () => {
  beforeEach(() => {
    vi.resetModules()
    vi.useFakeTimers()
    FakeSocket.instances = []
    vi.stubGlobal("WebSocket", FakeSocket)
  })
  afterEach(() => {
    vi.useRealTimers()
    vi.unstubAllGlobals()
  })

  async function setup() {
    const ws = await import("./ws")
    const { encodeFrame, FrameFormat } = await import("./frames")
    ws.useWs.getState().connect()
    await vi.waitFor(() => expect(FakeSocket.instances).toHaveLength(1))
    return { ...ws, encodeFrame, FrameFormat, socket: FakeSocket.instances[0]! }
  }

  it("connects to /api/v2/ws and sends the hello handshake", async () => {
    const { useWs, socket, CLIENT_FEATURES } = await setup()
    expect(socket.url).toMatch(/^ws:\/\/.+\/api\/v2\/ws$/)
    socket.open()
    expect(useWs.getState().status).toBe("open")
    expect(JSON.parse(socket.sent[0]!)).toEqual({ type: "hello", features: [...CLIENT_FEATURES], visible: true })
    socket.receive(JSON.stringify({ type: "hello", features: ["jobs", "preview.binary"] }))
    expect(useWs.getState().serverFeatures).toEqual(["jobs", "preview.binary"])
    useWs.getState().disconnect()
  })

  it("tells the server when the page is hidden or shown (no previews for a hidden page)", async () => {
    const { useWs, socket } = await setup()
    socket.open()
    const visibility = vi.spyOn(document, "visibilityState", "get").mockReturnValue("hidden")
    document.dispatchEvent(new Event("visibilitychange"))
    visibility.mockReturnValue("visible")
    document.dispatchEvent(new Event("visibilitychange"))
    const sent = socket.sent.map((text) => JSON.parse(text) as { type: string; visible?: boolean }).filter((message) => message.type === "visibility")
    expect(sent).toEqual([
      { type: "visibility", visible: false },
      { type: "visibility", visible: true },
    ])
    useWs.getState().disconnect()
  })

  it("dispatches JSON events to subscribers", async () => {
    const { onWsEvent, useWs, socket } = await setup()
    socket.open()
    const progress = vi.fn()
    const all = vi.fn()
    const off = onWsEvent("job.progress", progress)
    onWsEvent("*", all)
    socket.receive(JSON.stringify({ type: "job.progress", job_id: "j", step: 2, steps: 4 }))
    socket.receive("not json")
    expect(progress).toHaveBeenCalledWith({ type: "job.progress", job_id: "j", step: 2, steps: 4 })
    expect(all).toHaveBeenCalledTimes(1)
    off()
    socket.receive(JSON.stringify({ type: "job.progress", job_id: "j", step: 3, steps: 4 }))
    expect(progress).toHaveBeenCalledTimes(1)
    useWs.getState().disconnect()
  })

  it("turns preview frames into object URLs and revokes the previous one", async () => {
    const created: string[] = []
    const revoked: string[] = []
    vi.spyOn(URL, "createObjectURL").mockImplementation(() => {
      const url = `blob:test/${created.length}`
      created.push(url)
      return url
    })
    vi.spyOn(URL, "revokeObjectURL").mockImplementation((url) => void revoked.push(url))
    const { useWs, socket, encodeFrame, FrameFormat } = await setup()
    // A frame shows once decoded, in the next animation frame (api/paint-batch).
    const painted = () => vi.advanceTimersByTimeAsync(20)
    socket.open()
    socket.receive(encodeFrame(FrameFormat.Jpeg, 1, 5, new Uint8Array([1, 2, 3])))
    await painted()
    expect(useWs.getState().preview).toEqual({ jobSeq: 1, step: 5, url: "blob:test/0" })
    socket.receive(encodeFrame(FrameFormat.Webp, 1, 6, new Uint8Array([4])))
    await painted()
    expect(useWs.getState().preview?.url).toBe("blob:test/1")
    expect(revoked).toEqual(["blob:test/0"])
    useWs.getState().clearPreview()
    expect(revoked).toEqual(["blob:test/0", "blob:test/1"])
    // A frame still on its way when the preview is cleared (the job ended) never shows.
    socket.receive(encodeFrame(FrameFormat.Jpeg, 2, 1, new Uint8Array([5])))
    useWs.getState().clearPreview()
    await painted()
    expect(useWs.getState().preview).toBeNull()
    expect(revoked).toEqual(["blob:test/0", "blob:test/1", "blob:test/2"])
    useWs.getState().disconnect()
  })

  it("shows every image of a batch: the step starts again at 0 within one job", async () => {
    let created = 0
    vi.spyOn(URL, "createObjectURL").mockImplementation(() => `blob:test/${created++}`)
    vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {})
    const { useWs, socket, encodeFrame, FrameFormat } = await setup()
    socket.open()
    const shown: [number, number][] = []
    // batch_count 3, 3 steps per image: steps 0, 1, 2 of each image, one job (seq 7)
    for (let image = 0; image < 3; image++) {
      for (let step = 0; step < 3; step++) {
        socket.receive(encodeFrame(FrameFormat.Jpeg, 7, step, new Uint8Array([image, step])))
        await vi.advanceTimersByTimeAsync(20)
        const preview = useWs.getState().preview
        shown.push([preview!.jobSeq, preview!.step])
      }
    }
    expect(shown).toEqual([
      [7, 0], [7, 1], [7, 2],
      [7, 0], [7, 1], [7, 2],
      [7, 0], [7, 1], [7, 2],
    ])
    expect(useWs.getState().preview?.url).toBe("blob:test/8")
    useWs.getState().disconnect()
  })

  it("keeps the newest arrival when an earlier frame finishes decoding later", async () => {
    const revoked: string[] = []
    let created = 0
    vi.spyOn(URL, "createObjectURL").mockImplementation(() => `blob:test/${created++}`)
    vi.spyOn(URL, "revokeObjectURL").mockImplementation((url) => void revoked.push(url))
    // decoding resolves when the test says so, per URL
    const decodes = new Map<string, () => void>()
    vi.stubGlobal(
      "Image",
      class {
        src = ""
        decode() {
          return new Promise<void>((resolve) => decodes.set(this.src, resolve))
        }
      },
    )
    const { useWs, socket, encodeFrame, FrameFormat } = await setup()
    socket.open()
    socket.receive(encodeFrame(FrameFormat.Jpeg, 1, 4, new Uint8Array([1])))
    // the next image of the batch (step 0) arrives while step 4 is still decoding
    socket.receive(encodeFrame(FrameFormat.Jpeg, 1, 0, new Uint8Array([2])))
    decodes.get("blob:test/1")!()
    // the other frame still decoding holds the paint at most paint-batch's MAX_HOLD_MS
    await vi.advanceTimersByTimeAsync(150)
    expect(useWs.getState().preview).toEqual({ jobSeq: 1, step: 0, url: "blob:test/1" })
    // the older arrival finishes last: dropped, its URL revoked
    decodes.get("blob:test/0")!()
    await vi.advanceTimersByTimeAsync(20)
    expect(useWs.getState().preview).toEqual({ jobSeq: 1, step: 0, url: "blob:test/1" })
    expect(revoked).toEqual(["blob:test/0"])
    useWs.getState().disconnect()
  })

  it("reconnects with backoff after the connection drops", async () => {
    const { useWs, socket } = await setup()
    socket.open()
    socket.drop()
    expect(useWs.getState().status).toBe("reconnecting")
    expect(FakeSocket.instances).toHaveLength(1)
    await vi.advanceTimersByTimeAsync(700)
    expect(FakeSocket.instances).toHaveLength(2)
    FakeSocket.instances[1]!.open()
    expect(useWs.getState().status).toBe("open")
    expect(useWs.getState().attempt).toBe(0)
    useWs.getState().disconnect()
    expect(useWs.getState().status).toBe("closed")
  })

  it("keeps one socket when connect → disconnect → connect overlap (StrictMode)", async () => {
    const { useWs, onWsEvent } = await import("./ws")
    const pong = vi.fn()
    onWsEvent("pong", pong)
    useWs.getState().connect()
    useWs.getState().disconnect()
    useWs.getState().connect()
    await vi.advanceTimersByTimeAsync(0)
    const live = FakeSocket.instances.filter((s) => s.readyState !== 3)
    expect(FakeSocket.instances).toHaveLength(2)
    expect(live).toHaveLength(1)
    live[0]!.open()
    live[0]!.receive(JSON.stringify({ type: "pong" }))
    expect(pong).toHaveBeenCalledTimes(1)
    // The superseded socket's close does not schedule a reconnect.
    await vi.advanceTimersByTimeAsync(20_000)
    expect(FakeSocket.instances).toHaveLength(2)
    expect(useWs.getState().status).toBe("open")
    useWs.getState().disconnect()
  })
})
