import { afterEach, describe, expect, it, vi } from "vitest"
import { api } from "@/api/endpoints"
import { isEditable, thumbWidth } from "./items"
import { networksFollowOptions, PAGE_SIZE, pagesFor } from "./queries"

describe("network queries", () => {
  afterEach(() => vi.restoreAllMocks())

  it("asks for the windows around what the grid shows", () => {
    expect(pagesFor({ start: 0, end: 30 }, 5000)).toEqual([0])
    expect(pagesFor({ start: 4970, end: 4999 }, 5000)).toEqual([49])
    expect(pagesFor({ start: PAGE_SIZE - 10, end: PAGE_SIZE + 20 }, 5000)).toEqual([0, 1])
    expect(pagesFor({ start: 0, end: 0 }, 0)).toEqual([0])
  })

  it("goes stale when an option the server reads changes", () => {
    expect(networksFollowOptions(["forge_preset"])).toBe(true)
    expect(networksFollowOptions(["ui_reorder_list", "extra_networks_default_multiplier"])).toBe(true)
    expect(networksFollowOptions(["lora_preferred_name"])).toBe(true)
    expect(networksFollowOptions(["ui_reorder_list"])).toBe(false)
    // unknown keys: anything may have changed
    expect(networksFollowOptions(undefined)).toBe(true)
  })

  it("snaps thumbnail widths to the server's sizes", () => {
    expect([thumbWidth(1), thumbWidth(150), thumbWidth(328), thumbWidth(329), thumbWidth(5000)]).toEqual([64, 192, 384, 384, 1024])
    expect(isEditable({ editable: false })).toBe(false)
    expect(isEditable({ editable: true })).toBe(true)
  })

  it("encodes a name with a slash as one path segment", async () => {
    const fetch = vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(JSON.stringify({}), { headers: { "Content-Type": "application/json" } }))
    await api.networks.meta("checkpoints", "anime/model v2.safetensors")
    const request = fetch.mock.calls[0]![0] as Request
    expect(new URL(request.url).pathname).toBe("/api/v2/networks/checkpoints/anime%2Fmodel%20v2.safetensors/meta")
  })
})
