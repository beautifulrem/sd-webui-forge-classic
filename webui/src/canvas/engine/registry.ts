/**
 * Editor documents by key (one per editor: `img2img`, a ControlNet unit, …).
 * A document outlives the editor component that shows it; its pending layer
 * uploads are flushed before the owning form submits (`ui/pending-writes`).
 */
import { api } from "@/api/endpoints"
import { registerPendingWrite } from "@/ui/pending-writes"
import { DEFAULT_MAX_BYTES } from "../core/history"
import { EditorDocument } from "./document"
import { createCanvasRaster, encodePng } from "./raster"

const documents = new Map<string, EditorDocument>()

const MiB = 1024 * 1024

/** Undo memory when the device does not say how much memory it has (Safari, Firefox). */
export const UNKNOWN_DEVICE_BUDGET = 160 * MiB
/** Undo memory at most on touch-first devices (phones and tablets share little memory with tabs). */
export const COARSE_POINTER_BUDGET = 96 * MiB

const coarsePointer = () => typeof matchMedia === "function" && matchMedia("(pointer: coarse)").matches

/**
 * Undo memory per document: 1/32 of the device's memory (Chromium reports up
 * to 8 GB), between 96 and 384 MiB; a conservative 160 MiB when the browser
 * does not tell (never more than a device that does), and at most 96 MiB on
 * touch-first devices. Strokes keep only the tiles they touch, compressed,
 * so even 96 MiB holds far more than 32 states of brush work.
 */
export function historyBudget(
  deviceMemoryGb: number | undefined = (navigator as { deviceMemory?: number }).deviceMemory,
  coarse: boolean = coarsePointer(),
): number {
  const known = Boolean(deviceMemoryGb && Number.isFinite(deviceMemoryGb))
  const budget = known ? Math.min(DEFAULT_MAX_BYTES, Math.max(96 * MiB, (deviceMemoryGb! * 1024 * MiB) / 32)) : UNKNOWN_DEVICE_BUDGET
  return coarse ? Math.min(budget, COARSE_POINTER_BUDGET) : budget
}

export function getDocument(key: string, pendingKey?: string | null): EditorDocument {
  let doc = documents.get(key)
  if (!doc) {
    doc = new EditorDocument({
      createRaster: createCanvasRaster,
      sync: {
        encode: encodePng,
        upload: async (blob) => (await api.upload(blob, "layer.png")).id,
      },
      history: { maxBytes: historyBudget() },
    })
    documents.set(key, doc)
    const created = doc
    if (pendingKey) registerPendingWrite(pendingKey, () => created.flush())
  }
  return doc
}

/** Loads an image (same origin: uploads, outputs, object URLs) fully decoded. */
export async function loadImage(url: string, signal?: AbortSignal): Promise<HTMLImageElement> {
  const image = new Image()
  image.decoding = "async"
  image.src = url
  await image.decode()
  if (signal?.aborted) throw new DOMException("aborted", "AbortError")
  return image
}
