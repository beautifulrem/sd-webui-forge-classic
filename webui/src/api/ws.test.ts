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
    expect(JSON.parse(socket.sent[0]!)).toEqual({ type: "hello", features: [...CLIENT_FEATURES] })
    socket.receive(JSON.stringify({ type: "hello", features: ["jobs", "preview.binary"] }))
    expect(useWs.getState().serverFeatures).toEqual(["jobs", "preview.binary"])
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
    // A late, older frame of the same job is ignored.
    socket.receive(encodeFrame(FrameFormat.Jpeg, 1, 3, new Uint8Array([9])))
    await painted()
    expect(useWs.getState().preview?.step).toBe(5)
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
    const toast = vi.fn()
    onWsEvent("toast", toast)
    useWs.getState().connect()
    useWs.getState().disconnect()
    useWs.getState().connect()
    await vi.advanceTimersByTimeAsync(0)
    const live = FakeSocket.instances.filter((s) => s.readyState !== 3)
    expect(FakeSocket.instances).toHaveLength(2)
    expect(live).toHaveLength(1)
    live[0]!.open()
    live[0]!.receive(JSON.stringify({ type: "toast", message: "once" }))
    expect(toast).toHaveBeenCalledTimes(1)
    // The superseded socket's close does not schedule a reconnect.
    await vi.advanceTimersByTimeAsync(20_000)
    expect(FakeSocket.instances).toHaveLength(2)
    expect(useWs.getState().status).toBe("open")
    useWs.getState().disconnect()
  })
})
