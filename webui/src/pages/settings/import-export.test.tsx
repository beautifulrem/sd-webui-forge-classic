/** The settings import dialog: reads a file, validates the differences on the server, stages the valid ones. */
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { api } from "@/api/endpoints"
import { ApiError } from "@/api/http"
import type { SettingsSchema } from "@/ui/schema"
import { ImportDialog } from "./ImportExport"
import { EXPORT_FORMAT } from "./settings-model"
import { editorValues, pendingChanges, syncSchema, useSettingsEditor } from "./settings-store"

const schema: SettingsSchema = {
  version: 1,
  categories: [
    {
      id: "c",
      label: "C",
      sections: [
        {
          id: "s",
          label: "S",
          items: [
            { type: "toggle", name: "samples_save", label: "Save", default: true },
            { type: "slider", name: "jpeg_quality", label: "Quality", default: 80, minimum: 1, maximum: 100, step: 1, integer: true },
            { type: "text", name: "outdir_samples", label: "Folder", default: "", enabled_if: false, locked: "this setting is frozen" },
            { type: "text", name: "samples_format", label: "Format", default: "png" },
          ],
        },
      ],
    },
  ],
  quicksettings: [],
  values: { samples_save: true, jpeg_quality: 80, outdir_samples: "", samples_format: "png" },
  defaults: { samples_save: true, jpeg_quality: 80, outdir_samples: "", samples_format: "png" },
}

function upload(content: unknown) {
  const input = document.querySelector<HTMLInputElement>('input[type="file"]')!
  const file = new File([JSON.stringify(content)], "settings.json", { type: "application/json" })
  fireEvent.change(input, { target: { files: [file] } })
}

describe("settings import", () => {
  beforeEach(() => {
    useSettingsEditor.setState({ schema: null, saved: {}, inputs: new Map() })
    syncSchema(structuredClone(schema))
  })
  afterEach(() => vi.restoreAllMocks())

  it("validates the differences, skips rejected, locked and unknown keys, and stages the rest", async () => {
    const validate = vi.spyOn(api, "validateOptions").mockRejectedValue(
      new ApiError(422, "bad", { errors: [{ owner: "settings", param: "samples_format", message: "unknown format" }] }),
    )
    const onStaged = vi.fn()
    const onOpenChange = vi.fn()
    render(<ImportDialog open onOpenChange={onOpenChange} onStaged={onStaged} />)
    upload({ format: EXPORT_FORMAT, version: 1, exported_at: "", values: { samples_save: false, jpeg_quality: 80, outdir_samples: "/x", samples_format: "bmp", nope: 1 } })
    expect(await screen.findByText("1 settings will change")).toBeInTheDocument()
    expect(validate).toHaveBeenCalledWith({ samples_save: false, samples_format: "bmp" })
    expect(screen.getByText("1 already match, 2 unknown here (ignored)")).toBeInTheDocument()
    expect(screen.getByText(/unknown format/)).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Stage changes" }))
    expect(onStaged).toHaveBeenCalledWith(1)
    expect(editorValues().samples_save).toBe(false)
    expect(pendingChanges()).toEqual({ samples_save: false })
    expect(onOpenChange).toHaveBeenCalledWith(false)
  })

  it("explains a file that is not settings JSON", async () => {
    render(<ImportDialog open onOpenChange={() => {}} onStaged={() => {}} />)
    upload([1, 2])
    expect(await screen.findByRole("alert")).toHaveTextContent("The file does not contain settings.")
    await waitFor(() => expect(screen.getByRole("button", { name: "Stage changes" })).toBeDisabled())
  })
})
