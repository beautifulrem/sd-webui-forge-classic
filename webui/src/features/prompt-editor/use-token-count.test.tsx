import { QueryClientProvider } from "@tanstack/react-query"
import { renderHook, waitFor } from "@testing-library/react"
import type { ReactNode } from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { api } from "@/api/endpoints"
import { bootstrap } from "@/api/mock/fixtures"
import { keys, queryClient } from "@/api/queries"
import { useWs } from "@/api/ws"
import { FakeSocket } from "@/test/fake-socket"
import { TOKEN_DEBOUNCE_MS, type TokenCountInput, useTokenCount } from "./use-token-count"

const wrapper = ({ children }: { children: ReactNode }) => <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>

async function socket() {
  useWs.getState().connect()
  await vi.waitFor(() => expect(FakeSocket.instances.length).toBeGreaterThan(0))
  const ws = FakeSocket.instances.at(-1)!
  ws.open()
  return (event: object) => ws.receive(JSON.stringify(event))
}

const settle = () => new Promise((resolve) => setTimeout(resolve, TOKEN_DEBOUNCE_MS + 150))

describe("useTokenCount", () => {
  beforeEach(() => {
    FakeSocket.instances = []
    vi.stubGlobal("WebSocket", FakeSocket)
    queryClient.setQueryData(keys.app, structuredClone(bootstrap))
  })
  afterEach(() => {
    useWs.getState().disconnect()
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
    queryClient.clear()
  })

  it("debounces steps together with the text", async () => {
    const count = vi.spyOn(api, "countTokens").mockResolvedValue({ tokens: 3, max_length: 75 })
    const { rerender, result } = renderHook((props: TokenCountInput) => useTokenCount(props), { wrapper, initialProps: { text: "a cat", steps: 20 } })
    await waitFor(() => expect(result.current.data?.tokens).toBe(3))
    expect(count).toHaveBeenCalledTimes(1)
    // Dragging the steps slider: one request once it rests, not one per tick.
    for (const steps of [21, 22, 23, 24]) rerender({ text: "a cat", steps })
    await settle()
    expect(count).toHaveBeenCalledTimes(2)
    expect(count.mock.calls[1]![0]).toMatchObject({ text: "a cat", steps: 24 })
  }, 5_000)

  it("recounts after a model load when the count was taken without a model", async () => {
    const send = await socket()
    const count = vi.spyOn(api, "countTokens").mockResolvedValueOnce({ tokens: null, max_length: null }).mockResolvedValue({ tokens: 5, max_length: 75 })
    const { result } = renderHook(() => useTokenCount({ text: "a dog" }), { wrapper })
    await waitFor(() => expect(count).toHaveBeenCalledTimes(1))
    await waitFor(() => expect(result.current.isFetching).toBe(false))
    expect(result.current.data?.tokens).toBeNull()
    send({ type: "model.loaded", model: "anima" })
    await waitFor(() => expect(result.current.data?.tokens).toBe(5))
  })

  it("recounts unknown counts after a finished job, keeps known ones", async () => {
    const send = await socket()
    const count = vi.spyOn(api, "countTokens").mockResolvedValue({ tokens: 7, max_length: 75 })
    const { result } = renderHook(() => useTokenCount({ text: "known" }), { wrapper })
    await waitFor(() => expect(result.current.data?.tokens).toBe(7))
    send({ type: "job.completed", job_id: "j", outputs: [] })
    await new Promise((resolve) => setTimeout(resolve, 50))
    expect(count).toHaveBeenCalledTimes(1)
  })
})
