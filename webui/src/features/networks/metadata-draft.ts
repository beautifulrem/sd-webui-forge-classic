/** The user-metadata editor's form values and the sidecar patch they make (pure). */
import type { NetworkMeta } from "@/api/types"

export type Draft = {
  description: string
  sdVersion: string
  activation: string
  weight: number
  negative: string
  tags: string
  notes: string
}

/** `preferred weight` must lie in ±this (the server's `network_metadata.PREFERRED_WEIGHT_LIMIT`). */
export const PREFERRED_WEIGHT_LIMIT = 100

const text = (value: unknown) => (typeof value === "string" ? value : "")

export function draftFrom(meta: Pick<NetworkMeta, "user_metadata" | "description">): Draft {
  const user = meta.user_metadata
  const weight = Number(user["preferred weight"])
  const tags = user.tags
  return {
    description: text(user.description) || "",
    sdVersion: text(user["sd version"]) || "Unknown",
    activation: text(user["activation text"]),
    weight: Number.isFinite(weight) ? weight : 0,
    negative: text(user["negative text"]),
    tags: Array.isArray(tags) ? tags.map(String).join(", ") : text(tags),
    notes: text(user.notes),
  }
}

/** The sidecar keys to write for `fields` (empty text removes a key; weight 0 = the default multiplier). */
export function patchFrom(draft: Draft, fields: readonly string[]): Record<string, unknown> {
  const out: Record<string, unknown> = {}
  const set = (key: string, value: unknown) => {
    if (fields.includes(key)) out[key] = value
  }
  set("description", draft.description.trim() ? draft.description : null)
  set("sd version", draft.sdVersion === "Unknown" ? null : draft.sdVersion)
  set("activation text", draft.activation.trim() ? draft.activation.trim() : null)
  set("preferred weight", draft.weight ? draft.weight : null)
  set("negative text", draft.negative.trim() ? draft.negative.trim() : null)
  const tags = draft.tags.split(",").map((tag) => tag.trim()).filter(Boolean)
  set("tags", tags.length ? tags : null)
  set("notes", draft.notes.trim() ? draft.notes : null)
  return out
}
