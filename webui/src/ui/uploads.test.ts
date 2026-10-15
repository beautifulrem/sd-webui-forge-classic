import { afterEach, describe, expect, it, vi } from "vitest"
import { createFormStore } from "./form-store"
import { MAX_PREVIEWS, previewUrlOf, releaseUpload, uploadFile } from "./uploads"

let n = 0
function mockUploads() {
  vi.spyOn(globalThis, "fetch").mockImplementation(
    async () => new Response(JSON.stringify({ id: `up-${++n}` }), { headers: { "Content-Type": "application/json" } }),
  )
  const revoked: string[] = []
  vi.spyOn(URL, "createObjectURL").mockImplementation(() => `blob:t/${n}`)
  vi.spyOn(URL, "revokeObjectURL").mockImplementation((url) => void revoked.push(url))
  return revoked
}

const image = () => new File([new Uint8Array([1])], "a.png", { type: "image/png" })
const flush = () => new Promise((resolve) => setTimeout(resolve, 0))

describe("upload previews", () => {
  afterEach(() => vi.restoreAllMocks())

  it("revoke a released id only once no form references it", async () => {
    const revoked = mockUploads()
    const store = createFormStore("uploads", { persistDrafts: false })
    const a = await uploadFile(image())
    const b = await uploadFile(image())
    store.setState({ values: { core: { img: a.id, list: [b.id] } } })

    releaseUpload(b.id)
    await flush()
    expect(revoked).toEqual([]) // still in the list
    expect(previewUrlOf(b.id)).toBe(b.url)

    store.setState({ values: { core: { img: null, list: [] } } })
    releaseUpload(a.id)
    await flush()
    expect(revoked).toEqual([a.url])
    expect(previewUrlOf(a.id)).toBeUndefined()
  })

  it("keep at most MAX_PREVIEWS object URLs", async () => {
    const revoked = mockUploads()
    const store = createFormStore("uploads-cap", { persistDrafts: false })
    const ids: string[] = []
    for (let i = 0; i < MAX_PREVIEWS + 3; i++) {
      ids.push((await uploadFile(image())).id)
      store.setState({ values: { core: { list: [...ids] } } })
    }
    // Every id is referenced, so only the cap evicts the oldest ones.
    expect(revoked.length).toBeGreaterThanOrEqual(2)
    expect(previewUrlOf(ids.at(-1)!)).toBeDefined()
  })
})
