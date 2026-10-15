/**
 * Uploads (POST /api/v2/uploads) and their local previews. Images are shown
 * from a local object URL right after upload; ids restored from elsewhere
 * (patches, infotext paste) fall back to the server's /files/{id}.
 *
 * Object URLs pin the file's bytes in memory, so they are revoked when no
 * form references the id any more (checked when an input drops an id and
 * before every new upload), and at most `MAX_PREVIEWS` are kept.
 */
import { useState } from "react"
import { create } from "zustand"
import { api } from "@/api/endpoints"
import type { Upload } from "@/api/types"
import { t } from "@/i18n"
import { showToast } from "@/lib/toast"
import { allFormStores } from "./form-store"

export type UploadInfo = Pick<Upload, "id"> & Partial<Omit<Upload, "id">> & { url?: string; /** ms timestamp of the upload */ at?: number }

type UploadState = { byId: Record<string, UploadInfo> }

const useUploads = create<UploadState>()(() => ({ byId: {} }))

/** Local previews kept at most; older ones fall back to the server URL. */
export const MAX_PREVIEWS = 24
/** A fresh upload is not yet in a form (its input stores the id after the whole batch). */
const GRACE_MS = 10_000

/** Whether any value of any form store contains the upload id (values are JSON). */
function referenced(id: string): boolean {
  const contains = (value: unknown): boolean =>
    value === id ||
    (Array.isArray(value) ? value.some(contains) : value !== null && typeof value === "object" && Object.values(value).some(contains))
  for (const store of allFormStores()) {
    if (Object.values(store.getState().values).some((owner) => Object.values(owner).some(contains))) return true
  }
  return false
}

/**
 * Revokes the previews of `ids` that no form references (all previews older
 * than the grace period when omitted), then caps the rest.
 */
function sweep(ids?: readonly string[]) {
  const byId = useUploads.getState().byId
  const now = Date.now()
  const candidates = ids ?? Object.keys(byId).filter((id) => now - (byId[id]?.at ?? 0) > GRACE_MS)
  const revoked = new Set(candidates.filter((id) => byId[id]?.url && !referenced(id)))
  // Insertion order is upload order: the oldest previews go first.
  const live = Object.keys(byId).filter((id) => byId[id]?.url && !revoked.has(id))
  for (const id of live.slice(0, Math.max(0, live.length - MAX_PREVIEWS))) revoked.add(id)
  if (revoked.size === 0) return
  const next = { ...byId }
  for (const id of revoked) {
    const info = next[id]
    if (!info?.url) continue
    URL.revokeObjectURL(info.url)
    // Keep the metadata (file name); the preview falls back to /files/{id}.
    next[id] = { ...info, url: undefined }
  }
  useUploads.setState({ byId: next })
}

/** Drops the local preview of an upload once no form uses it (after the caller's state update). */
export function releaseUpload(id: string) {
  queueMicrotask(() => sweep([id]))
}

export async function uploadFile(file: File): Promise<UploadInfo> {
  const result = await api.upload(file)
  sweep()
  const info: UploadInfo = {
    ...result,
    name: result.name ?? file.name,
    url: file.type.startsWith("image/") ? URL.createObjectURL(file) : undefined,
    at: Date.now(),
  }
  useUploads.setState((s) => ({ byId: { ...s.byId, [info.id]: info } }))
  return info
}

export function useUploadInfo(id: string | null | undefined): UploadInfo | null {
  const info = useUploads((s) => (id ? s.byId[id] : undefined))
  if (!id) return null
  return info?.url ? info : { ...info, id, url: api.fileUrl(id) }
}

export function imageFiles(list: FileList | readonly File[] | null | undefined): File[] {
  return Array.from(list ?? []).filter((file) => file.type.startsWith("image/"))
}

/** Uploads files, tracking busy state and reporting failures as toasts. */
export function useUploader() {
  const [busy, setBusy] = useState(0)
  const upload = async (files: readonly File[]): Promise<UploadInfo[]> => {
    if (files.length === 0) return []
    setBusy((n) => n + 1)
    try {
      const settled = await Promise.allSettled(files.map(uploadFile))
      const failed = settled.filter((r) => r.status === "rejected")
      if (failed.length > 0) {
        const reason = (failed[0] as PromiseRejectedResult).reason
        showToast("error", t("Upload failed"), reason instanceof Error ? reason.message : String(reason))
      }
      return settled.flatMap((r) => (r.status === "fulfilled" ? [r.value] : []))
    } finally {
      setBusy((n) => n - 1)
    }
  }
  return { upload, busy: busy > 0 }
}

/** Test hook: the preview URL currently held for an id. */
export const previewUrlOf = (id: string) => useUploads.getState().byId[id]?.url
