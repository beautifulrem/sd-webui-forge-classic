import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import type { SettingsSchema } from "@/api/types"
import { downloadSettings } from "./settings-file"
import { useSettingsEditor } from "./settings-store"

/** Captures what `downloadSettings` hands to the browser. */
function captureDownload() {
  const blobs: Blob[] = []
  const links: { href: string; download: string; attached: boolean }[] = []
  vi.spyOn(URL, "createObjectURL").mockImplementation((blob) => {
    blobs.push(blob as Blob)
    return `blob:test/${blobs.length}`
  })
  const revoke = vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {})
  vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(function (this: HTMLAnchorElement) {
    links.push({ href: this.href, download: this.download, attached: this.isConnected })
  })
  return { blobs, links, revoke }
}

describe("downloadSettings", () => {
  beforeEach(() => {
    vi.useFakeTimers({ now: new Date("2026-10-19T12:00:00Z") })
    useSettingsEditor.setState({
      saved: { samples_format: "webp", jpeg_quality: 80, outdir_samples: "" },
      schema: { defaults: { samples_format: "png", jpeg_quality: 80, outdir_samples: "" } } as unknown as SettingsSchema,
    })
  })

  afterEach(() => {
    vi.useRealTimers()
    useSettingsEditor.setState({ saved: {}, schema: null })
  })

  it("downloads the changed settings as a dated JSON file and frees the URL afterwards", async () => {
    const { blobs, links, revoke } = captureDownload()
    downloadSettings(false)

    expect(links).toEqual([{ href: "blob:test/1", download: "remi-settings-2026-10-19.json", attached: true }])
    expect(document.querySelector("a[download]")).toBeNull() // the link is removed again
    expect(blobs[0]?.type).toBe("application/json")
    const text = await blobs[0]!.text()
    expect(text.endsWith("}\n")).toBe(true)
    const data = JSON.parse(text) as { version: number; exported_at: string; values: Record<string, unknown> }
    expect(data.version).toBe(1)
    expect(data.exported_at).toBe("2026-10-19T12:00:00.000Z")
    expect(data.values).toEqual({ samples_format: "webp" })

    expect(revoke).not.toHaveBeenCalled()
    vi.advanceTimersByTime(1000)
    expect(revoke).toHaveBeenCalledWith("blob:test/1")
  })

  it("includes every stored setting with `all`", async () => {
    const { blobs } = captureDownload()
    downloadSettings(true)
    const data = JSON.parse(await blobs[0]!.text()) as { values: Record<string, unknown> }
    expect(data.values).toEqual({ jpeg_quality: 80, outdir_samples: "", samples_format: "webp" })
  })
})
